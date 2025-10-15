#!/usr/bin/env python3

r"""Convert ptychography data format using Ptychodus.

Example usage:
python convert_format.py \
    --patterns "data/ptychoshelves/data_roi0_dp.hdf5" \
    --probe "data/ptychoshelves/Niter100.mat" \
    --probe-positions "data/ptychoshelves/data_roi0_para.hdf5" \
    --object "data/ptychoshelves/Niter100.mat" \
    --metadata "data/ptychoshelves/data_roi0_para.hdf5" \
    --product-name "ptychodus" \
    --output-dir "outputs" \
    --diffraction-reader "PtychoShelves" \
    --probe-reader "PtychoShelves" \
    --probe-position-reader "PtychoShelves" \
    --object-reader "PtychoShelves"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import h5py

from ptychodus.model import ModelCore
from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import ImageExtent


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
ProbeReaderName = Literal[
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
ProbePositionReaderName = Literal[
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
ObjectReaderName = ProbeReaderName
ProductWriterName = Literal["HDF5", "NPZ"]

DiffractionReaderChoices = list(DiffractionReaderName.__args__)
DiffractionWriterChoices = list(DiffractionWriterName.__args__)
ProbeReaderChoices = list(ProbeReaderName.__args__)
ProbePositionReaderChoices = list(ProbePositionReaderName.__args__)
ObjectReaderChoices = list(ObjectReaderName.__args__)
ProductWriterChoices = list(ProductWriterName.__args__)


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


def _get_pixel_size(
    metadata_file_path: Path | None = None,
    metadata_type: str | None = None,
    asserted_pixel_size: float | None = None,
) -> float | None:
    if asserted_pixel_size is not None:
        return asserted_pixel_size
    if metadata_file_path is not None:
        if metadata_type == "PtychoShelves":
            with h5py.File(metadata_file_path, "r") as f:
                pixel_size_m = float(f["dx"][0])
        else:
            raise ValueError(f"Unsupported metadata type: {metadata_type}")
    return pixel_size_m


def _get_diffraction_pattern_size(
    diffraction_pattern_path: Path,
    diffraction_reader: DiffractionReaderName,
) -> tuple[int, int]:
    if diffraction_reader == "PtychoShelves":
        with h5py.File(diffraction_pattern_path, "r") as f:
            return f["dp"].shape[-2:]
    else:
        raise ValueError(f"Unsupported diffraction reader: {diffraction_reader}")


def convert_data(
    diffraction_pattern_path: Path,
    *,
    probe_path: Path | None,
    probe_position_path: Path | None,
    object_path: Path | None,
    metadata_path: Path | None,
    diffraction_output: Path,
    product_output: Path,
    product_name: str | None,
    diffraction_reader: DiffractionReaderName,
    probe_reader: ProbeReaderName | None,
    probe_position_reader: ProbePositionReaderName | None,
    object_reader: ObjectReaderName | None,
    diffraction_writer: DiffractionWriterName = "PtychoShelves",
    product_writer: ProductWriterName = "HDF5",
    settings_file: Optional[Path] = None,
    asserted_pixel_size: Optional[float] = None,
) -> tuple[Path, Path]:
    """Convert an external diffraction dataset into Ptychodus outputs.

    Parameters
    ----------
    patterns_path : Path
        Path to the diffraction file that should be opened via the specified
        `diffraction_reader` plugin.
    probe_path : Path or None
        Optional path to a probe file. When provided, it is opened
        with `probe_reader` before the probe/object builders are invoked.
    probe_positions_path : Path or None
        Optional path to a scan or position file. When provided, it is opened
        with `probe_position_reader` before the probe/object builders are invoked.
    object_path : Path or None
        Optional path to a object file. When provided, it is opened
        with `object_reader` before the probe/object builders are invoked.
    metadata_path : Path or None
        Optional path to a metadata file. When provided, it is used to
        override the pixel size from data files.
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
    asserted_pixel_size : float or None
        Optional pixel size in meters. If provided, it is used to override
        the pixel size from data files. This parameter needs to be provided
        if pixel size is not available in the data files.

    Returns
    -------
    tuple of Path
        ``(diffraction_output, product_output)`` describing the saved paths.

    Raises
    ------
    FileNotFoundError
        Raised when either the diffraction file or the scan file is missing.
    """
    if not diffraction_pattern_path.is_file():
        raise FileNotFoundError(f"Diffraction patterns file not found: {diffraction_pattern_path}")

    if probe_position_path is not None and not probe_position_path.exists():
        raise FileNotFoundError(f"Scan file not found: {probe_position_path}")

    diffraction_output.parent.mkdir(parents=True, exist_ok=True)
    product_output.parent.mkdir(parents=True, exist_ok=True)

    product_basename = product_name or diffraction_pattern_path.stem

    with ModelCore(settings_file) as model:
        dp_size = _get_diffraction_pattern_size(diffraction_pattern_path, diffraction_reader)
        
        model.workflow_api.open_patterns(
            diffraction_pattern_path, 
            file_type=diffraction_reader,
            crop_center=CropCenter(position_x_px=dp_size[1] // 2, position_y_px=dp_size[0] // 2),
            crop_extent=ImageExtent(width_px=dp_size[1], height_px=dp_size[0]),
        )
        model.diffraction.diffraction_api.start_assembling_diffraction_patterns()
        model.diffraction.diffraction_api.finish_assembling_diffraction_patterns(block=True)

        metadata = model.diffraction.dataset.get_metadata()
        product_kwargs = _metadata_kwargs(metadata)

        workflow_product = model.workflow_api.create_product(product_basename, **product_kwargs)

        if probe_position_path is not None:
            workflow_product.open_scan(probe_position_path, file_type=probe_position_reader or None)

        if probe_path is not None:
            workflow_product.open_probe(probe_path, file_type=probe_reader or None)

        if object_path is not None:
            workflow_product.open_object(object_path, file_type=object_reader or None)

        model.diffraction.diffraction_api.save_patterns(diffraction_output, diffraction_writer)
        workflow_product.save_product(product_output, file_type=product_writer)
        
        # Supplement pixel size
        if metadata_path is not None:
            pixel_size_m = _get_pixel_size(
                metadata_path, 
                metadata_type=diffraction_reader, 
                asserted_pixel_size=asserted_pixel_size
            )
            if pixel_size_m is not None:
                with h5py.File(product_output, "r+") as f:
                    f["object"].attrs["pixel_height_m"] = pixel_size_m
                    f["object"].attrs["pixel_width_m"] = pixel_size_m

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
        "--probe",
        type=Path,
        default=None,
        help="Optional probe file path required by many plugins.",
    )
    parser.add_argument(
        "--probe-positions",
        type=Path,
        default=None,
        help="Optional probe positions file path required by many plugins.",
    )
    parser.add_argument(
        "--object",
        type=Path,
        default=None,
        help="Optional object file path required by many plugins.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata file path required by many plugins.",
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
        "--probe-reader",
        default="PtychoShelves",
        choices=sorted(ProbeReaderChoices),
        help="Probe reader plugin to use when a probe file is provided.",
    )
    parser.add_argument(
        "--probe-position-reader",
        default="PtychoShelves",
        choices=sorted(ProbePositionReaderChoices),
        help="Probe position reader plugin to use when a probe position file is provided.",
    )
    parser.add_argument(
        "--object-reader",
        default="PtychoShelves",
        choices=sorted(ObjectReaderChoices),
        help="Object reader plugin to use when a object file is provided.",
    )
    parser.add_argument(
        "--product-writer",
        default="HDF5",
        choices=sorted(ProductWriterChoices),
        help="Product file writer plugin to use for the converted parameter file.",
    )
    parser.add_argument(
        "--asserted-pixel-size",
        type=float,
        default=None,
        help="Optional pixel size in meters. If provided, it is used to override the pixel size from data files.",
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available diffraction and scan plugins, then exit.",
    )

    args = parser.parse_args(argv)

    if args.list_plugins:
        print("Diffraction readers   : ", ", ".join(sorted(DiffractionReaderChoices)))
        print("Diffraction writers   : ", ", ".join(sorted(DiffractionWriterChoices)))
        print("Probe readers         : ", ", ".join(sorted(ProbeReaderChoices)))
        print("Probe position readers: ", ", ".join(sorted(ProbePositionReaderChoices)))
        print("Object readers        : ", ", ".join(sorted(ObjectReaderChoices)))
        print("Product writers       : ", ", ".join(sorted(ProductWriterChoices)))
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
        probe_path=args.probe,
        probe_position_path=args.probe_positions,
        object_path=args.object,
        metadata_path=args.metadata,
        diffraction_output=diffraction_output,
        product_output=product_output,
        product_name=args.product_name,
        diffraction_reader=args.diffraction_reader,
        diffraction_writer=args.diffraction_writer,
        probe_reader=args.probe_reader,
        probe_position_reader=args.probe_position_reader,
        object_reader=args.object_reader,
        product_writer=args.product_writer,
        settings_file=args.settings,
        asserted_pixel_size=args.asserted_pixel_size,
    )

    print(f"Wrote diffraction data to {diffraction_path}")
    print(f"Wrote parameter file to {product_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
