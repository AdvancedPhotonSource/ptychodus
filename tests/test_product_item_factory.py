"""Unit tests for metadata passthrough in ProductRepositoryItemFactory.

``MetadataRepositoryItem`` applies a constructor override only when the keyword is
not None, and otherwise inherits the ``ProductSettings`` value. A field the factory
forgets to pass therefore does not fail -- it silently takes the value from
settings.ini. ``focus_object_distance_m`` was dropped that way on the load path, so a
product read from a file came back with the wrong magnification.

These assert the kwargs the factory builds rather than a finished item: the defect is
in the argument list, and building a real item would need the probe, object and
probe-position builder factories behind it.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ptychodus.api.diffraction import DiffractionMetadata, Polarization
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.workflow import WorkflowAPI
from ptychodus.model.product import item_factory as item_factory_module
from ptychodus.model.product.api import ProductAPI
from ptychodus.model.product.item_factory import (
    ProductRepositoryItemFactory,
    resolve_product_name,
)


# One value per field, each away from both the dataclass default and the
# ProductSettings default, so inheriting either fails the comparison.
DISTINCTIVE_METADATA = ProductMetadata(
    name='distinctive',
    comments='every field set away from its default',
    detector_distance_m=2.25,
    photon_energy_eV=8551.0,
    probe_photon_count=1234.5,
    exposure_time_s=0.05,
    mass_attenuation_m2_per_kg=3.75,
    tomography_angle_deg=-90.0461,
    focus_object_distance_m=-2.5e-3,
    tilt_angle_deg=61.0,
    polarization=Polarization.RIGHT_CIRCULAR,
    far_field=False,
)


def _make_factory(settings: Any = None) -> ProductRepositoryItemFactory:
    return ProductRepositoryItemFactory(
        settings=MagicMock() if settings is None else settings,
        scan_item_factory=MagicMock(),
        probe_item_factory=MagicMock(),
        object_item_factory=MagicMock(),
        repository=MagicMock(),
        file_reader_chooser=MagicMock(),
    )


def _capture_metadata_kwargs(build: Any) -> dict[str, Any]:
    """Run *build* with the item classes stubbed, returning the metadata kwargs.

    ``ProductGeometryProvider`` and ``ProductRepositoryItem`` are stubbed because both reject
    the mocked sub-items -- ``ProductRepositoryItem`` requires real ParameterGroups.
    """
    with (
        patch.object(item_factory_module, 'MetadataRepositoryItem') as metadata_cls,
        patch.object(item_factory_module, 'ProductGeometryProvider'),
        patch.object(item_factory_module, 'ProductRepositoryItem'),
    ):
        build()

    metadata_cls.assert_called_once()
    return dict(metadata_cls.call_args.kwargs)


def _settable_field_names() -> set[str]:
    return {field.name for field in fields(ProductMetadata)}


class TestCreateFromProduct:
    """The load path: ProductRepositoryItemFactory.create_from_settings routes a product
    read from a file through here, so a field dropped here is data loss on every open.
    """

    def test_forwards_every_metadata_field(self) -> None:
        product = Product(
            metadata=DISTINCTIVE_METADATA,
            probe_positions=MagicMock(),
            probes=MagicMock(),
            object_=MagicMock(),
            losses=[],
        )

        kwargs = _capture_metadata_kwargs(lambda: _make_factory().create_from_product(product))

        missing = _settable_field_names() - set(kwargs)
        assert not missing, f'create_from_product drops {sorted(missing)}'

        for name in _settable_field_names():
            assert kwargs[name] == getattr(DISTINCTIVE_METADATA, name), name

    def test_an_explicit_name_overrides_the_product(self) -> None:
        product = Product(
            metadata=DISTINCTIVE_METADATA,
            probe_positions=MagicMock(),
            probes=MagicMock(),
            object_=MagicMock(),
            losses=[],
        )

        kwargs = _capture_metadata_kwargs(
            lambda: _make_factory().create_from_product(product, name='renamed')
        )

        assert kwargs['name'] == 'renamed'
        assert kwargs['focus_object_distance_m'] == DISTINCTIVE_METADATA.focus_object_distance_m


class TestConstructionChainSignatures:
    """Every layer that builds a product from values must be able to express every
    metadata field. A missing parameter is invisible at runtime -- the field quietly
    takes its settings default -- so pin it structurally instead.
    """

    @pytest.mark.parametrize(
        ('label', 'function'),
        [
            ('create_from_values', ProductRepositoryItemFactory.create_from_values),
            ('insert_new_product', ProductAPI.insert_new_product),
            ('create_product', WorkflowAPI.create_product),
        ],
    )
    def test_accepts_every_metadata_field(self, label: str, function: Any) -> None:
        parameters = set(inspect.signature(function).parameters)

        missing = _settable_field_names() - parameters
        assert not missing, f'{label} cannot express {sorted(missing)}'


def _dataset_named(sample_name: str | None) -> MagicMock:
    dataset = MagicMock()
    dataset.get_metadata.return_value = replace(
        DiffractionMetadata.create_null(), sample_name=sample_name
    )
    return dataset


class TestResolveProductName:
    """A product with no name of its own takes the sample the data was measured from."""

    def test_an_explicit_name_wins(self) -> None:
        assert resolve_product_name('typed', _dataset_named('from_the_file')) == 'typed'

    def test_the_sample_name_is_the_fallback(self) -> None:
        assert resolve_product_name('', _dataset_named('from_the_file')) == 'from_the_file'

    @pytest.mark.parametrize('dataset', [None, _dataset_named(None), _dataset_named('')])
    def test_nothing_to_go_on_defers_to_the_settings(self, dataset: MagicMock | None) -> None:
        """An empty result is what lets ProductSettings.name keep deciding, as before."""
        assert resolve_product_name('', dataset) == ''


class TestSampleNameReachesTheMetadataItem:
    """Resolving in the factory is what makes every creation path inherit the rule."""

    def test_create_from_values_names_the_product_for_the_sample(self) -> None:
        kwargs = _capture_metadata_kwargs(
            lambda: _make_factory().create_from_values(dataset=_dataset_named('cRZP_star'))
        )

        assert kwargs['name'] == 'cRZP_star'

    def test_create_from_settings_names_the_product_for_the_sample(self) -> None:
        # Steer past the branch that reads a product file, which carries its own name.
        settings = MagicMock()
        settings.file_path.get_value.return_value.is_file.return_value = False
        factory = _make_factory(settings)

        kwargs = _capture_metadata_kwargs(
            lambda: factory.create_from_settings(dataset=_dataset_named('cRZP_star'))
        )

        assert kwargs['name'] == 'cRZP_star'
