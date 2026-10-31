"""ptychodus-stream: estimate the probe and object while a scan streams, and store each scan.

Runs one pvapy area-detector consumer whose processor, :class:`StreamProcessor`, feeds a
:class:`~ptychodus.cli._stream.ScanPipeline`: detector frames arrive on the input
channel, probe positions on their own channel, and a trigger PV marks the end of each
scan. Probe and object images are published for viewing, and each finished scan is
saved into the storage root the web service browses.

Needs the ``stream`` extra (pvapy). The PV contract it follows is documented in the
streaming guide, and ``scripts/replay_scan.py`` replays a recorded scan to it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final
import argparse
import json
import logging
import sys
import threading

import numpy
import pvaccess
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.hpc.dataConsumerController import DataConsumerController
from pvapy.utility.adImageUtility import AdImageUtility
from pydantic import ValidationError

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product
from ptychodus.cli import add_log_level_argument, configure_logging
from ptychodus.cli._stream import (
    PvProtocol,
    ScanPipeline,
    ScanReport,
    ScanSetup,
    StreamConfiguration,
    display_images,
    parse_positions,
)

__all__ = [
    'SCAN_REPORT_TYPE',
    'StreamProcessor',
    'TRIGGER_TYPE',
    'make_batched_positions_type',
    'make_sgsocket_event',
    'main',
]

logger = logging.getLogger(__name__)

# The processor's configuration travels under this key of pvapy's processor arguments,
# beside the keys pvapy adds itself.
_CONFIGURATION_KEY: Final[str] = 'ptychodus_stream'

TRIGGER_TYPE: Final[dict[str, Any]] = {'value': pvaccess.INT}
"""The trigger PV as the replay script hosts it: an integer ``value``."""

SCAN_REPORT_TYPE: Final[dict[str, Any]] = {
    'label': pvaccess.STRING,
    'diffraction_uuid': pvaccess.STRING,
    'product_uuid': pvaccess.STRING,
    'num_frames_paired': pvaccess.ULONG,
    'num_frames_dropped': pvaccess.ULONG,
    'written': pvaccess.BOOLEAN,
}
"""The ``<prefix>:scan`` structure; a uuid is empty when nothing was saved."""


def make_batched_positions_type(axis_names: list[str]) -> dict[str, Any]:
    """The batched position structure: per axis, ``{num_events, event_ids, positions}``."""
    axis = {
        'num_events': pvaccess.ULONG,
        'event_ids': [pvaccess.ULONG],
        'positions': [pvaccess.DOUBLE],
    }
    return {name: dict(axis) for name in axis_names}


def make_sgsocket_event(index: int, attributes: Mapping[str, float]) -> pvaccess.NtNdArray:
    """One sgsocket position event: an NTNDArray whose ``uniqueId`` is the scan index."""
    event = pvaccess.NtNdArray()
    event['uniqueId'] = index
    event['attribute'] = [
        pvaccess.NtAttribute(name, pvaccess.PvDouble(float(value)))
        for name, value in attributes.items()
    ]
    return event


def _protocol(configuration: StreamConfiguration) -> Any:
    return pvaccess.CA if configuration.pv_protocol is PvProtocol.CA else pvaccess.PVA


def _scalar(value: Any) -> float | None:
    # A CA enum (a binary output record, say) arrives as {'index': n, 'choices': [...]}.
    if isinstance(value, Mapping):
        value = value.get('index', value.get('value'))

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Monitor:
    """A PV monitor that ignores the value it connects with and keeps the latest."""

    def __init__(
        self, name: str, protocol: Any, on_update: Callable[[Any], None] | None = None
    ) -> None:
        self.channel = pvaccess.Channel(name, protocol)
        self._on_update = on_update
        self._connected = False
        self.latest: float | None = None
        self.channel.subscribe('ptychodus_stream', self._update)
        self.channel.startMonitor()

    def _update(self, pv: Any) -> None:
        value = pv.get()

        if not self._connected:
            # The value a monitor connects with is history, not an event.
            self._connected = True
            self.latest = _scalar(value.get('value')) if isinstance(value, Mapping) else None
            return

        if isinstance(value, Mapping) and 'value' in value:
            self.latest = _scalar(value['value'])

        if self._on_update is not None:
            self._on_update(value)

    def stop(self) -> None:
        try:
            self.channel.stopMonitor()
            self.channel.unsubscribe('ptychodus_stream')
        except Exception:
            logger.exception('Failed to stop a monitor.')


class _PvaPublisher:
    """Publishes the four images and the scan report on a PVA server."""

    def __init__(self, server: Any, prefix: str) -> None:
        self._server = server
        self._image_names = {
            'probe': (f'{prefix}:probe:amplitude', f'{prefix}:probe:phase'),
            'object': (f'{prefix}:object:amplitude', f'{prefix}:object:phase'),
        }
        self._scan_name = f'{prefix}:scan'
        self._counter = 0
        placeholder = numpy.zeros((1, 1), dtype=numpy.float32)

        for names in self._image_names.values():
            for name in names:
                server.addRecord(name, AdImageUtility.generateNtNdArray2D(0, placeholder))

        server.addRecord(self._scan_name, pvaccess.PvObject(SCAN_REPORT_TYPE))

    def _update(self, names: tuple[str, str], array: numpy.ndarray) -> None:
        for name, image in zip(names, display_images(array)):
            self._server.update(name, AdImageUtility.generateNtNdArray2D(self._counter, image))

    def publish_estimate(self, product: Product) -> None:
        self._counter += 1
        self._update(self._image_names['probe'], product.probes.get_probe_no_opr().get_array())
        self._update(self._image_names['object'], product.object_.get_array())

    def publish_scan(self, report: ScanReport) -> None:
        self._server.update(
            self._scan_name,
            pvaccess.PvObject(
                SCAN_REPORT_TYPE,
                {
                    'label': report.label,
                    'diffraction_uuid': str(report.diffraction_uuid or ''),
                    'product_uuid': str(report.product_uuid or ''),
                    'num_frames_paired': report.num_frames_paired,
                    'num_frames_dropped': report.num_frames_dropped,
                    'written': report.written,
                },
            ),
        )


# Configuration keys whose change needs a restart: they name channels the consumer and
# the monitors opened at start.
_RESTART_KEYS: Final[frozenset[str]] = frozenset(
    {
        'input_channel',
        'output_pv_prefix',
        'trigger_pv',
        'position_channel',
        'scan_number_pv',
        'tomography_angle_pv',
        'pv_protocol',
        'consumer_queue_size',
    }
)


class StreamProcessor(AdImageProcessor):
    """The pvapy processor behind ``ptychodus-stream``; see the module docstring.

    ``process`` and the PV monitors only enqueue; the pipeline's own threads do the work.
    pvapy calls every method positionally.
    """

    pvaServer: Any  # noqa: N815 (set by pvapy before start)

    def __init__(self, config_dict: dict[str, Any] | None = None) -> None:
        config_dict = {} if config_dict is None else config_dict
        super().__init__(config_dict)
        self._registry = PluginRegistry.load_plugins()
        self._configuration = StreamConfiguration.model_validate(config_dict[_CONFIGURATION_KEY])
        self._setup = ScanSetup.from_configuration(self._configuration, self._registry)
        self._lock = threading.Lock()
        self._pipeline: ScanPipeline | None = None
        self._monitors: list[_Monitor] = []
        self._trigger: _Monitor | None = None
        self._scan_number: _Monitor | None = None
        self._tomography_angle: _Monitor | None = None

    def start(self) -> None:
        configuration = self._configuration
        protocol = _protocol(configuration)
        server = self.pvaServer

        if server is None:
            server = pvaccess.PvaServer()
            self.pvaServer = server

        publisher = _PvaPublisher(server, configuration.output_pv_prefix)
        self._pipeline = ScanPipeline(
            self._setup,
            publisher,
            acknowledge_trigger=self._acknowledge_trigger,
            read_scan_number=self._read_scan_number,
            read_tomography_angle_deg=self._read_tomography_angle_deg,
        )
        self._pipeline.start()
        self._trigger = _Monitor(configuration.trigger_pv, protocol, self._on_trigger)
        positions = _Monitor(configuration.position_channel, pvaccess.PVA, self._on_positions)
        positions.channel.setMonitorMaxQueueLength(configuration.consumer_queue_size)
        self._monitors = [self._trigger, positions]

        if configuration.scan_number_pv is not None:
            self._scan_number = _Monitor(configuration.scan_number_pv, protocol)
            self._monitors.append(self._scan_number)

        if configuration.tomography_angle_pv is not None:
            self._tomography_angle = _Monitor(configuration.tomography_angle_pv, protocol)
            self._monitors.append(self._tomography_angle)

        logger.info(f'{ptychodus.VERSION_STRING} streaming from {configuration.input_channel}.')

    def stop(self) -> None:
        for monitor in self._monitors:
            monitor.stop()

        if self._pipeline is not None:
            # Finishes and stores the scan in progress before returning.
            self._pipeline.stop()

    def configure(self, config_dict: dict[str, Any]) -> None:
        # pvapy passes its own runtime settings through here too.
        pvapy_keys = {'skipInitialUpdates', 'objectIdOffset', 'nSequentialUpdates'}
        pvapy_keys |= {'receiverQueueSize'}
        overrides = {key: value for key, value in config_dict.items() if key not in pvapy_keys}

        if config_dict and not overrides:
            return

        with self._lock:
            configuration = self._configuration.merged(overrides)
            changed = {
                key
                for key in _RESTART_KEYS
                if getattr(configuration, key) != getattr(self._configuration, key)
            }

            if changed:
                raise ValueError(f'Restart to change {", ".join(sorted(changed))}.')

            setup = ScanSetup.from_configuration(configuration, self._registry)
            self._configuration = configuration

        if self._pipeline is not None:
            self._pipeline.configure(setup)

    def process(self, pv_object: Any) -> Any:
        image_id, image, nx, _, _, _, _ = self.reshapeNtNdArray(pv_object)

        if nx is not None and image is not None and self._pipeline is not None:
            self._pipeline.submit_frame(int(image_id), image)

        return pv_object

    def _on_positions(self, value: Any) -> None:
        if self._pipeline is None:
            return

        try:
            self._pipeline.submit_positions(parse_positions(value, self._configuration))
        except Exception:
            logger.exception('Could not read a position update.')

    def _on_trigger(self, value: Any) -> None:
        reading = _scalar(value.get('value')) if isinstance(value, Mapping) else None

        if reading and self._pipeline is not None:
            self._pipeline.trigger()

    def _acknowledge_trigger(self) -> None:
        if self._trigger is not None:
            self._trigger.channel.put(0)

    def _read_scan_number(self) -> int | None:
        monitor = self._scan_number
        return None if monitor is None or monitor.latest is None else int(monitor.latest)

    def _read_tomography_angle_deg(self) -> float | None:
        monitor = self._tomography_angle
        return None if monitor is None else monitor.latest

    def resetStats(self) -> None:  # noqa: N802
        pass

    def getStats(self) -> dict[str, Any]:  # noqa: N802
        if self._pipeline is None:
            return {}

        return {
            key: (int(value) if isinstance(value, bool) else value)
            for key, value in self._pipeline.get_counters().items()
        }

    def getStatsPvaTypes(self) -> dict[str, Any]:  # noqa: N802
        if self._pipeline is None:
            return {}

        return {
            key: pvaccess.DOUBLE if isinstance(value, float) else pvaccess.LONG
            for key, value in self._pipeline.get_counters().items()
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Estimate the probe and object while a scan streams, and store each scan.'
    )
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='YAML or JSON stream configuration; see the streaming guide for every key.',
    )
    parser.add_argument(
        '--report-period',
        type=float,
        default=0.0,
        help='Seconds between statistics reports in the log; zero reports none.',
    )
    add_log_level_argument(parser)
    return parser


def main() -> ExitCode:
    args = _build_parser().parse_args()
    configure_logging(args)

    try:
        configuration = StreamConfiguration.from_file(args.config)
        # Loads the maps, resolves the models and prepares the storage root, so a bad
        # configuration fails here rather than inside the consumer.
        ScanSetup.from_configuration(configuration, PluginRegistry.load_plugins())
    except (OSError, ValueError, ValidationError, LookupError) as exc:
        logger.error(f'Invalid configuration {args.config}: {exc}')
        return ExitCode.USAGE

    prefix = configuration.output_pv_prefix
    controller = DataConsumerController(
        configuration.input_channel,
        controlChannel=f'{prefix}:control',
        statusChannel=f'{prefix}:status',
        processorClass=f'{__name__}.StreamProcessor',
        processorArgs=json.dumps({_CONFIGURATION_KEY: configuration.model_dump(mode='json')}),
        serverQueueSize=configuration.consumer_queue_size,
        receiverQueueSize=configuration.consumer_queue_size,
        disableCurses=True,
        nConsumers=1,
    )
    controller.run(0, args.report_period)
    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
