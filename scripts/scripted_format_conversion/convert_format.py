#!/usr/bin/env python3

r"""Convert ptychography data format using Ptychodus.

Example usage:
python convert_ptychoshelves_to_ptychodus.py \
    --patterns data/ptychoshelves/data_roi0_dp.hdf5 \
    --scan data/ptychoshelves/data_roi0_para.hdf5 \
    --product-name ptychodus \
    --output-dir outputs \
    --diffraction-reader PtychoShelves \
    --scan-reader PtychoShelves
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

from ptychodus.model import ModelCore

DiffractionReaderChoices = (
    "APS_CSSI",
    "APS_HXN",
    "PtychoShelves",
    "CXI",
    "TIFF",
    "LCLS_XPP",
    "APS_2IDD",
    "APS_2IDE",
    "APS_BNP",
    "APS_LYNX",
    "NPZ",
    "SLAC_NPZ",
    "SLS_cSAXS",
    "APS_ISN",
    "APS_Velociprobe",
    "NPY",
    "MAX_IV_NanoMax",
    "APS_PtychoSAXS",
    "NSLS_II_1",
    "NSLS_II_2",
    "NSLS_II_MATLAB",
    "APS_Polar",
)
DiffractionWriterChoices = ("PtychoShelves", "NPZ", "NPY")
ScanReaderChoices = (
    "MDA",
    "APS_2IDD",
    "APS_2IDE",
    "APS_BNP",
    "APS_ISN_MDA",
    "CNM_APS_HXN",
    "APS_LYNX_Orchestra",
    "APS_PtychoSAXS",
    "APS_CSSI",
    "APS_Velociprobe_LI",
    "APS_Velociprobe_PE",
    "PtychoShelves",
    "APS_Polar",
    "APS_LYNX_SoftGlueZynq",
    "LCLS_XPP",
    "NSLS_II_1",
    "NSLS_II_2",
    "MAX_IV_NanoMAX",
    "TXT",
    "CSV",
    "CXI",
)
ProductWriterChoices = ("HDF5", "NPZ")

DiffractionReaderName = Literal[
    "APS_CSSI",
    "APS_HXN",
    "PtychoShelves",
    "CXI",
    "TIFF",
    "LCLS_XPP",
    "APS_2IDD",
    "APS_2IDE",
    "APS_BNP",
    "APS_LYNX",
    "NPZ",
    "SLAC_NPZ",
    "SLS_cSAXS",
    "APS_ISN",
    "APS_Velociprobe",
    "NPY",
    "MAX_IV_NanoMax",
    "APS_PtychoSAXS",
    "NSLS_II_1",
    "NSLS_II_2",
    "NSLS_II_MATLAB",
    "APS_Polar",
]
DiffractionWriterName = Literal["PtychoShelves", "NPZ", "NPY"]
ScanReaderName = Literal[
    "MDA",
    "APS_2IDD",
    "APS_2IDE",
    "APS_BNP",
    "APS_ISN_MDA",
    "CNM_APS_HXN",
    "APS_LYNX_Orchestra",
    "APS_PtychoSAXS",
    "APS_CSSI",
    "APS_Velociprobe_LI",
    "APS_Velociprobe_PE",
    "PtychoShelves",
    "APS_Polar",
    "APS_LYNX_SoftGlueZynq",
    "LCLS_XPP",
    "NSLS_II_1",
    "NSLS_II_2",
    "MAX_IV_NanoMAX",
    "TXT",
    "CSV",
    "CXI",
]
ProductWriterName = Literal["HDF5", "NPZ"]


def _metadata_kwargs(metadata) -> dict[str, float]:
    """Collect optional metadata fields that match create_product() parameters."""
    optional_fields = (
        "detector_distance_m",
        "probe_energy_eV",
        "probe_photon_count",
        "exposure_time_s",
        "mass_attenuation_m2_kg",
        "tomography_angle_deg",
    )

    values: dict[str, float] = {}

    for field in optional_fields:
        value = getattr(metadata, field, None)

        if value is not None:
            values[field] = value

    return values


def convert_data(
    patterns_path: Path,
    *,
    scan_path: Path | None,
    diffraction_output: Path,
    product_output: Path,
    product_name: str | None,
    diffraction_reader: DiffractionReaderName,
    scan_reader: ScanReaderName | None,
    diffraction_writer: DiffractionWriterName = "PtychoShelves",
    product_writer: ProductWriterName = "HDF5",
    settings_file: Optional[Path] = None,
    build_probe: Optional[bool] = True,
    build_object: Optional[bool] = True,
) -> tuple[Path, Path]:
    """Convert an external diffraction dataset into Ptychodus outputs.

    Parameters
    ----------
    patterns_path : Path
        Path to the diffraction file that should be opened via the specified
        `diffraction_reader` plugin.
    scan_path : Path or None
        Optional path to a scan or position file. When provided, it is opened
        with `scan_reader` before the probe/object builders are invoked.
    diffraction_output : Path
        Destination filename for the converted diffraction data.
    product_output : Path
        Destination filename for the converted product/parameter data.
    product_name : str or None
        Logical product name stored inside the Ptychodus workflow. Defaults to
        ``patterns_path.stem``.
    diffraction_reader : {DiffractionReaderChoices}
        Plugin simple name used to interpret the input diffraction data.
    diffraction_writer : {DiffractionWriterChoices} or None
        Plugin simple name used to write the converted diffraction data.
        ``None`` requests automatic selection (matching the reader when
        possible, otherwise falling back to ``'PtychoShelves'``).
    scan_reader : {ScanReaderChoices} or None
        Plugin simple name used to interpret the scan file. Ignored when
        `scan_path` is ``None``.
    product_writer : {ProductWriterChoices}
        Plugin simple name used to persist the converted product file.
    settings_file : Path or None
        Optional ``settings.ini`` loaded into :class:`ptychodus.model.ModelCore`
        before performing the conversion.
    build_probe : bool
        Whether to execute :meth:`build_probe` on the workflow product.
    build_object : bool
        Whether to execute :meth:`build_object` on the workflow product.

    Returns
    -------
    tuple of Path
        ``(diffraction_output, product_output)`` describing the saved paths.

    Raises
    ------
    FileNotFoundError
        Raised when either the diffraction file or the scan file is missing.
    """
    if not patterns_path.is_file():
        raise FileNotFoundError(f"Diffraction patterns file not found: {patterns_path}")

    if scan_path is not None and not scan_path.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_path}")

    diffraction_output.parent.mkdir(parents=True, exist_ok=True)
    product_output.parent.mkdir(parents=True, exist_ok=True)

    product_basename = product_name or patterns_path.stem

    with ModelCore(settings_file) as model:
        model.workflow_api.open_patterns(patterns_path, file_type=diffraction_reader)
        model.diffraction.diffraction_api.start_assembling_diffraction_patterns()
        model.diffraction.diffraction_api.finish_assembling_diffraction_patterns(block=True)

        metadata = model.diffraction.dataset.get_metadata()
        product_kwargs = _metadata_kwargs(metadata)

        workflow_product = model.workflow_api.create_product(product_basename, **product_kwargs)

        if scan_path is not None:
            workflow_product.open_scan(scan_path, file_type=scan_reader or None)

        if build_probe:
            workflow_product.build_probe()

        if build_object:
            workflow_product.build_object()

        model.diffraction.diffraction_api.save_patterns(diffraction_output, diffraction_writer)
        workflow_product.save_product(product_output, file_type=product_writer)

    return diffraction_output, product_output


def get_output_filenames(
    output_dir: Path, 
    base_name: str,
    diffraction_reader: str,
    diffraction_writer: str,
    product_writer: str,
):
    # Find output file names
    def _default_diffraction_suffix(plugin: str) -> str:
        return {
            "NPZ": ".npz",
            "NPY": ".npy",
        }.get(plugin, ".hdf5")

    def _default_product_suffix(plugin: str) -> str:
        return {"NPZ": ".npz"}.get(plugin, ".hdf5")

    resolved_diffraction_writer = diffraction_writer
    if resolved_diffraction_writer is None:
        if diffraction_reader in DiffractionWriterChoices:
            resolved_diffraction_writer = diffraction_reader
        else:
            resolved_diffraction_writer = "PtychoShelves"

    diffraction_output = (
        output_dir
        / f"{base_name}_dp{_default_diffraction_suffix(resolved_diffraction_writer)}"
    )
    product_output = (
        output_dir / f"{base_name}_para{_default_product_suffix(product_writer)}"
    )
    return diffraction_output, product_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert external diffraction data to Ptychodus HDF5 outputs using the Python API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=".",
        help=(
            "Path to the diffraction file to read. Provide an explicit path for non-demo datasets."
        ),
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=None,
        help="Optional scan/position file path required by many plugins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where converted HDF5 files should be written.",
    )
    parser.add_argument(
        "--diffraction-output",
        type=Path,
        default=None,
        help="Explicit path for the converted diffraction data file.",
    )
    parser.add_argument(
        "--product-output",
        type=Path,
        default=None,
        help="Explicit path for the converted product/parameter file.",
    )
    parser.add_argument(
        "--product-name",
        default=None,
        help="Name assigned to the resulting Ptychodus product.",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Optional settings.ini to load before running the conversion.",
    )
    parser.add_argument(
        "--diffraction-reader",
        default="PtychoShelves",
        choices=sorted(DiffractionReaderChoices),
        help="Diffraction file reader plugin to use when opening the source file.",
    )
    parser.add_argument(
        "--diffraction-writer",
        default="PtychoShelves",
        help="Diffraction file writer plugin to use when saving the converted dataset.",
    )
    parser.add_argument(
        "--scan-reader",
        default="PtychoShelves",
        choices=sorted(ScanReaderChoices),
        help="Scan reader plugin to use when a scan file is provided.",
    )
    parser.add_argument(
        "--product-writer",
        default="HDF5",
        choices=sorted(ProductWriterChoices),
        help="Product file writer plugin to use for the converted parameter file.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip building the probe after loading the scan.",
    )
    parser.add_argument(
        "--skip-object",
        action="store_true",
        help="Skip building the object after loading the scan.",
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available diffraction and scan plugins, then exit.",
    )

    args = parser.parse_args(argv)

    if args.list_plugins:
        print("Diffraction readers :", ", ".join(sorted(DiffractionReaderChoices)))
        print("Diffraction writers :", ", ".join(sorted(DiffractionWriterChoices)))
        print("Scan readers        :", ", ".join(sorted(ScanReaderChoices)))
        print("Product writers     :", ", ".join(sorted(ProductWriterChoices)))
        return 0

    if args.diffraction_writer is not None and args.diffraction_writer not in DiffractionWriterChoices:
        parser.error(
            f"--diffraction-writer must be one of {', '.join(sorted(DiffractionWriterChoices))}"
        )

    base_name = args.product_name or args.patterns.stem
    output_dir = args.output_dir or args.patterns.parent / f"{base_name}_ptychodus"
    
    (
        proposed_diffraction_output_filename, 
        proposed_product_output_filename
    ) = get_output_filenames(
        output_dir=output_dir,
        base_name=base_name,
        diffraction_reader=args.diffraction_reader,
        diffraction_writer=args.diffraction_writer,
        product_writer=args.product_writer,
    )
    diffraction_output = args.diffraction_output or proposed_diffraction_output_filename
    product_output = args.product_output or proposed_product_output_filename

    diffraction_path, product_path = convert_data(
        args.patterns,
        scan_path=args.scan,
        diffraction_output=diffraction_output,
        product_output=product_output,
        product_name=args.product_name,
        diffraction_reader=args.diffraction_reader,
        diffraction_writer=args.diffraction_writer,
        scan_reader=args.scan_reader,
        product_writer=args.product_writer,
        settings_file=args.settings,
        build_probe=not args.skip_probe,
        build_object=not args.skip_object,
    )

    print(f"Wrote diffraction data to {diffraction_path}")
    print(f"Wrote parameter file to {product_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
