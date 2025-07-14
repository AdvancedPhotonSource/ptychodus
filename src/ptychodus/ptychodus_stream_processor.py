from pathlib import Path
from typing import Any
import logging
import threading
import time

import numpy

from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import pvaccess
import pvapy

from ptychodus.model import ModelCore
import ptychodus


class ReconstructionThread(threading.Thread):
    def __init__(
        self,
        ptychodus: ModelCore,
        input_product_path: Path,
        output_product_path: Path,
        reconstruct_pv: str,
    ) -> None:
        super().__init__()
        self._ptychodus = ptychodus
        self._input_product_path = input_product_path
        self._output_product_path = output_product_path
        self._channel = pvapy.Channel(reconstruct_pv, pvapy.CA)
        self._reconstruct_event = threading.Event()
        self._stop_event = threading.Event()

        self._ptychodus_streaming_context = None

        self._channel.subscribe('reconstructor', self._monitor)
        self._channel.startMonitor()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._reconstruct_event.wait(timeout=1.0):
                logging.debug('ReconstructionThread: Begin assembling scan positions')

                if self._ptychodus_streaming_context is not None:
                    self._ptychodus_streaming_context.stop()

                logging.debug('ReconstructionThread: End assembling scan positions')
                self._ptychodus.batch_mode_execute(
                    'reconstruct', self._input_product_path, self._output_product_path
                )
                self._reconstruct_event.clear()
                # reconstruction done; indicate that results are ready
                self._channel.put(0)

    def _monitor(self, pv_object: pvaccess.PvObject) -> None:
        # NOTE caput bdpgp:gp:bit3 1
        logging.debug(f'ReconstructionThread::monitor {pv_object}')

        if pv_object['value']['index'] == 1:
            logging.debug('ReconstructionThread: Reconstruct PV triggered!')
            # start reconstructing
            self._reconstruct_event.set()
        else:
            logging.debug('ReconstructionThread: Reconstruct PV not triggered!')

    def stop(self) -> None:
        self._stop_event.set()


class PtychodusAdImageProcessor(AdImageProcessor):
    def __init__(self, config_dict: dict[str, Any] = {}) -> None:
        super().__init__(config_dict)

        self.logger.debug(f'{ptychodus.__name__.title()} ({ptychodus.__version__})')

        settings_file = config_dict.get('settingsFile')
        self._ptychodus = ModelCore(settings_file)
        self._reconstruction_thread = ReconstructionThread(
            self._ptychodus,
            Path(config_dict.get('inputProductPath', 'input.npz')),
            Path(config_dict.get('outputProductPath', 'output.npz')),
            config_dict.get('reconstructPV', 'bdpgp:gp:bit3'),
        )
        self._pos_x_pv = config_dict.get('pos_x_pv', 'bluesky:pos_x')
        self._pos_y_pv = config_dict.get('pos_y_pv', 'bluesky:pos_y')
        self._num_frames_processed = 0
        self._processing_time = 0.0

    def start(self) -> None:
        """Called at startup"""
        self._ptychodus.__enter__()
        self._reconstruction_thread.start()

    def stop(self) -> None:
        """Called at shutdown"""
        self._reconstruction_thread.stop()
        self._reconstruction_thread.join()
        self._ptychodus.__exit__(None, None, None)

    def configure(self, config_dict: dict[str, Any]) -> None:
        """Configures user processor"""
        num_arrays = config_dict['num_arrays']
        num_patterns_per_array = config_dict.get('num_patterns_per_array', 1)
        pattern_dtype = config_dict.get('pattern_dtype', 'uint16')

        metadata = ptychodus.api.diffraction.DiffractionMetadata(
            num_patterns_per_array=[int(num_patterns_per_array)] * int(num_arrays),
            pattern_dtype=numpy.dtype(pattern_dtype),
        )
        self._ptychodus_streaming_context = self._ptychodus.create_streaming_context(metadata)
        self._ptychodus_streaming_context.start()  # TODO clean up

    def process(self, pv_object: pvaccess.PvObject) -> pvaccess.PvObject:
        """Processes monitor update"""
        processing_begin_time = time.time()

        (frame_id, image, nx, ny, nz, color_mode, field_key) = self.reshapeNtNdArray(pv_object)
        frame_time_stamp = TimeUtility.getTimeStampAsFloat(pv_object['timeStamp'])

        if nx is None:
            self.logger.debug(f'Frame id {frame_id} contains an empty image.')
        else:
            self.logger.debug(f'Frame id {frame_id} time stamp {frame_time_stamp}')
            image3d = image[numpy.newaxis, :, :].copy()
            array = ptychodus.api.diffraction.SimpleDiffractionArray(
                label=f'Frame{frame_id}',
                indexes=numpy.array([frame_id]),
                patterns=image3d,
            )
            self._ptychodus_streaming_context.append_array(array)

        pos_x_queue = self.metadataQueueMap[self._pos_x_pv]

        while True:
            try:
                pos_x = pos_x_queue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                self._ptychodus_streaming_context.append_positions_x(
                    pos_x['values'],
                    [TimeUtility.getTimeStampAsFloat(ts) for ts in pos_x['t']],
                )

        pos_y_queue = self.metadataQueueMap[self._pos_y_pv]

        while True:
            try:
                pos_y = pos_y_queue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                self._ptychodus_streaming_context.append_positions_y(
                    pos_y['values'],
                    [TimeUtility.getTimeStampAsFloat(ts) for ts in pos_y['t']],
                )

        processing_end_time = time.time()
        self._processing_time += processing_end_time - processing_begin_time
        self._num_frames_processed += 1

        return pv_object

    def resetStats(self) -> None:  # noqa: N802
        """Resets statistics for user processor"""
        self._num_frames_processed = 0
        self._processing_time = 0.0

    def getStats(self) -> dict[str, Any]:  # noqa: N802
        """Retrieves statistics for user processor"""
        num_frames_queued = self._ptychodus_streaming_context.get_queue_size()
        processed_frame_rate = 0.0

        if self._processing_time > 0.0:
            processed_frame_rate = self._num_frames_processed / self._processing_time

        return {
            'num_frames_processed': self._num_frames_processed,
            'num_frames_queued': num_frames_queued,
            'processing_time': FloatWithUnits(self._processing_time, 's'),
            'processed_frame_rate': FloatWithUnits(processed_frame_rate, 'fps'),
        }

    def getStatsPvaTypes(self) -> dict[str, pvaccess.ScalarType]:  # noqa: N802
        """Defines PVA types for different stats variables"""
        return {
            'num_frames_processed': pvaccess.UINT,
            'num_frames_queued': pvaccess.UINT,
            'processing_time': pvaccess.DOUBLE,
            'processing_frame_rate': pvaccess.DOUBLE,
        }
