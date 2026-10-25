from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable, Sequence
import logging
import math

from ptychodus.api.object import (
    Object,
    ObjectFileReader,
    ObjectGeometryProvider,
    resample_object_to_pixel_size,
)
from ptychodus.api.simulate.object import generate_layers, pad_object
from ptychodus.api.parameters import ParameterGroup
from ptychodus.api.probe import Probe, ProbeEnsemble

from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectBuilder(ParameterGroup):
    def __init__(self, settings: ObjectSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

        self.extra_padding_x = settings.extra_padding_x.copy()
        self._add_parameter('extra_padding_x', self.extra_padding_x)
        self.extra_padding_y = settings.extra_padding_y.copy()
        self._add_parameter('extra_padding_y', self.extra_padding_y)

        self._probe_supplier: Callable[[], ProbeEnsemble] | None = None

    def get_name(self) -> str:
        return self._name.get_value()

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ObjectBuilder:
        pass

    @property
    def reads_diffraction_data(self) -> bool:
        """Whether the object this builds depends on the diffraction data.

        Such an object is rebuilt when the data changes under an unchanged detector
        geometry -- another dataset, or a reload -- which the geometry alone does not
        announce.
        """
        return False

    def bind_probe_supplier(self, supplier: Callable[[], ProbeEnsemble] | None) -> None:
        """Accept the source of the probe that shares this builder's product.

        The item holding a builder binds it, so a builder copied into another product
        reads that product's probe rather than the one it was copied from. A copy starts
        unbound. Builders that do not use the probe ignore it.
        """
        self._probe_supplier = supplier

    def _get_probe_on_object_pixels(
        self, geometry_provider: ObjectGeometryProvider, *, consequence: str
    ) -> Probe | None:
        """The bound probe, if there is one on the object's pixels, or None.

        A geometry edit can rebuild the object before the probe catches up, so a probe
        on other pixels degrades the object rather than failing it: the warning names
        *consequence*, what the builder does without the probe. A missing or still empty
        probe gives None without one.
        """
        if self._probe_supplier is None:
            return None

        probe = self._probe_supplier().get_probe_no_opr()

        if probe.get_array().size == 0:
            return None

        probe_pixel_geometry = probe.get_pixel_geometry()
        object_pixel_geometry = geometry_provider.get_object_geometry().get_pixel_geometry()

        if not (
            math.isclose(probe_pixel_geometry.width_m, object_pixel_geometry.width_m, rel_tol=1e-6)
            and math.isclose(
                probe_pixel_geometry.height_m, object_pixel_geometry.height_m, rel_tol=1e-6
            )
        ):
            logger.warning(
                f'Probe pixels ({probe_pixel_geometry}) differ from object pixels'
                f' ({object_pixel_geometry}); {consequence}.'
            )
            return None

        return probe

    @abstractmethod
    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        """Return the raw, unconditioned object.

        Implementations must NOT generate layers; `build` owns the conditioning
        pipeline. Generative implementations should return
        `self._pad_object(object_)` so the extra padding is applied to the canvas
        they just sized against the scan geometry.
        """
        pass

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        """Return the conditioned object: slice it to the requested layer spacing.

        Overriding this method is reserved for builders whose object is already
        conditioned; see `FromMemoryObjectBuilder`. Every builder that generates
        or ingests a raw object must leave it alone and implement `_build_raw`
        instead.
        """
        return self._condition_object(self._build_raw(geometry_provider), layer_spacing_m)

    def _condition_object(
        self,
        object_: Object,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        """Slice the object onto the requested depth grid.

        `generate_layers` resamples the whole stack and preserves the layer product, so
        reslicing a multislice object is no longer destructive and is done whenever a
        spacing is asked for.

        An empty spacing is the exception. It is the unset default, and it means one
        layer, so honoring it literally would collapse a multislice result loaded from
        file the moment nobody had configured a spacing. Absent a spacing this keeps
        whatever layers the object arrived with, which also makes one layer
        unrequestable for an object that has more -- the data model cannot tell "one
        layer" from "unspecified".

        Note the padding is applied earlier, in `_build_raw`, so the order here is
        pad-then-layers rather than the layers-then-pad of the original
        `_create_object`. The two do not commute, because a zero-amplitude border
        changes the resampled result near the edges.
        """
        if not layer_spacing_m and object_.num_layers > 1:
            logger.info(
                f'Object has {object_.num_layers} layer(s) and no layer spacing is set;'
                ' keeping them rather than collapsing to one.'
            )
            return object_

        return generate_layers(object_, layer_spacing_m)

    def _pad_object(self, object_: Object) -> Object:
        """Widen a freshly generated canvas by the extra padding.

        Only generative builders call this. `pad_object` is strictly additive --
        N applications add 2*N*pad pixels per dimension -- and it leaves no trace
        in the array, so there is no way to detect an already-padded object and
        skip it. The size of a file-supplied object is already fixed by the file,
        so the parameter has no coherent meaning there; applying it would grow
        every warm-start object on every load/save round trip, unbounded.
        """
        return pad_object(
            object_,
            self.extra_padding_x.get_value(),
            self.extra_padding_y.get_value(),
        )

    def _rescale_to_mean_transmission(
        self, object_: Object, geometry_provider: ObjectGeometryProvider
    ) -> Object:
        """Scale a freshly generated object to the sample's measured mean opacity.

        A unit-modulus guess asserts a transparent sample, so the exit wave carries the
        unobstructed beam and predicts more signal than the detector recorded. Scaling
        to the measured transmission puts the difference in the object, where absorption
        belongs, instead of leaving it to be absorbed into the probe -- which would make
        the probe stop meaning the incident illumination that `probe_photon_count`,
        and the dose derived from it, are stated in.

        Only generative builders call this. A file- or memory-supplied object already
        carries the transmission it was reconstructed at, and scaling it would decouple
        it from a matching probe, because the data constrains the product of the two and
        not either alone.
        """
        transmission = geometry_provider.mean_amplitude_transmission

        if not 0.0 < transmission <= 1.0:
            logger.warning(
                'Ignoring out-of-range mean amplitude transmission %r; using a transparent object.',
                transmission,
            )
            return object_

        return Object(
            array=object_.get_array() * transmission,
            layer_spacing_m=object_.layer_spacing_m,
            pixel_geometry=object_.get_pixel_geometry(),
            center=object_.get_center(),
        )


class FromMemoryObjectBuilder(ObjectBuilder):
    """An object that has already been conditioned.

    Two things produce these. Reconstruction output, which `ProcessingTaskMonitor`
    re-assigns to the output product item on every reconstructor iteration (see
    `model/processing/monitor.py`), and products loaded from HDF5/NPZ. In both
    cases the layer structure and the canvas size are already what the
    reconstructor solved for. `generate_layers` would truncate a multislice result
    back to whatever the item's `layer_spacing_m` parameter happens to say, and
    `pad_object` is strictly additive, so it would grow the array by twice the
    padding in each dimension on every iteration. `build` therefore deliberately
    bypasses the conditioning pipeline.

    The requested `layer_spacing_m` is likewise ignored in favor of the spacing
    the object actually has, which is why `ObjectRepositoryItem.set_num_layers`
    is inert for from-memory items.
    """

    def __init__(self, settings: ObjectSettings, object_: Object) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._object = object_.copy()

    def copy(self) -> FromMemoryObjectBuilder:
        builder = FromMemoryObjectBuilder(self._settings, self._object)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_geometry = geometry_provider.get_object_geometry()

        try:
            pixel_geometry = self._object.get_pixel_geometry()
        except ValueError:
            pixel_geometry = object_geometry.get_pixel_geometry()

        try:
            center = self._object.get_center()
        except ValueError:
            center = object_geometry.get_center()

        return Object(
            self._object.get_array(),
            pixel_geometry,
            center,
            self._object.layer_spacing_m,
        )

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        return self._build_raw(geometry_provider)


class FromFileObjectBuilder(ObjectBuilder):
    """An object read from file, conditioned on the way in.

    Unlike `FromMemoryObjectBuilder` this is an ingest path, so the layer spacing
    does apply -- slicing a two-dimensional object into a multislice warm start is
    a real workflow, and before the conditioning pipeline existed the setting was
    silently ignored here. `_condition_object` keeps whatever layers the file
    already carries.

    A file that records its own pixel size is reconciled to the run's, since an
    object sampled by another geometry is a misscaled initial guess rather than a
    warm start. None of the object formats currently record one, so today this
    only ever fires for a reader that supplies it.

    The extra padding is deliberately not applied; see `ObjectBuilder._pad_object`.
    """

    def __init__(
        self,
        settings: ObjectSettings,
        file_reader: ObjectFileReader,
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self._file_reader = file_reader

        self.file_path = settings.file_path.copy()
        self._add_parameter('file_path', self.file_path)

        self.file_type = settings.file_type.copy()
        self._add_parameter('file_type', self.file_type)

    def copy(self) -> FromFileObjectBuilder:
        builder = FromFileObjectBuilder(self._settings, self._file_reader)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            object_from_file = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        object_geometry = geometry_provider.get_object_geometry()

        try:
            pixel_geometry = object_from_file.get_pixel_geometry()
        except ValueError:
            # The format records no pixel size, so there is nothing to reconcile and the
            # array is taken to be sampled the way this run is.
            pixel_geometry = object_geometry.get_pixel_geometry()

        try:
            center = object_from_file.get_center()
        except ValueError:
            center = object_geometry.get_center()

        return resample_object_to_pixel_size(
            Object(
                object_from_file.get_array(),
                pixel_geometry,
                center,
                object_from_file.layer_spacing_m,
            ),
            object_geometry.get_pixel_geometry(),
        )
