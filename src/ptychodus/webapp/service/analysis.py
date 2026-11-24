"""The metrics and comparison reports of stored products, the bounded operations that compute them.

Each view loads the product files it is given and returns its report's ``scalars``,
synchronously, so each runs in a worker thread as a render does. A report that cannot
describe its product, one whose probe carries no power say, raises
:class:`~ptychodus.webapp.service.errors.InvalidRequestError` with the report's reason.
"""

from __future__ import annotations

from pathlib import Path

from ptychodus.api.analyze import (
    ObjectComparisonScalars,
    ObjectMetricsScalars,
    ProbeComparisonScalars,
    ProbeMetricsScalars,
    ProbePositionsComparisonScalars,
    ProbePositionsMetricsScalars,
    compute_object_comparison_report,
    compute_object_metrics_report,
    compute_probe_comparison_report,
    compute_probe_metrics_report,
    compute_probe_positions_comparison_report,
    compute_probe_positions_metrics_report,
)
from ptychodus.api.io import load_product
from ptychodus.api.metrics import SpectralSNRConvention

from ptychodus.webapp.service.errors import InvalidRequestError

__all__ = [
    'object_comparison_view',
    'object_metrics_view',
    'probe_comparison_view',
    'probe_metrics_view',
    'probe_positions_comparison_view',
    'probe_positions_metrics_view',
]


def probe_metrics_view(path: Path) -> ProbeMetricsScalars:
    product = load_product(path)

    try:
        return compute_probe_metrics_report(product).scalars
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


def object_metrics_view(path: Path, erosion_divisor: int) -> ObjectMetricsScalars:
    product = load_product(path)

    try:
        return compute_object_metrics_report(product, erosion_divisor=erosion_divisor).scalars
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


def probe_positions_metrics_view(path: Path) -> ProbePositionsMetricsScalars:
    product = load_product(path)

    try:
        return compute_probe_positions_metrics_report(product).scalars
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


def probe_comparison_view(
    reference_path: Path,
    test_path: Path,
    erosion_divisor: int,
    mad_threshold: float,
    resample_test: bool,
) -> ProbeComparisonScalars:
    reference = load_product(reference_path)
    test = load_product(test_path)

    try:
        report = compute_probe_comparison_report(
            reference,
            test,
            erosion_divisor=erosion_divisor,
            mad_threshold=mad_threshold,
            resample_test=resample_test,
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    return report.scalars


def object_comparison_view(
    reference_path: Path,
    test_path: Path,
    convention: SpectralSNRConvention,
    snr_threshold: float,
    erosion_divisor: int,
    resample_test: bool,
) -> ObjectComparisonScalars:
    reference = load_product(reference_path)
    test = load_product(test_path)

    try:
        report = compute_object_comparison_report(
            reference,
            test,
            convention=convention,
            snr_threshold=snr_threshold,
            erosion_divisor=erosion_divisor,
            resample_test=resample_test,
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    return report.scalars


def probe_positions_comparison_view(
    reference_path: Path, test_path: Path
) -> ProbePositionsComparisonScalars:
    reference = load_product(reference_path)
    test = load_product(test_path)

    try:
        return compute_probe_positions_comparison_report(reference, test).scalars
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
