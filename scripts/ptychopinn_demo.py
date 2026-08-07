#!/usr/bin/env python
"""Generate comparison montages of probeGuess, objectGuess, and mean diffraction
patterns across the experimental and synthetic ptychopinn_demo datasets.

Produces three PNG files in the output directory:

* ``probeGuess_montage.png`` and ``objectGuess_montage.png``: 2x4 grids that
  render the complex probe and object using ptychodus' HSV color model
  (hue=phase, value=amplitude). An HSV color-wheel legend occupies one of the
  empty top-row cells.
* ``diff3d_mean_montage.png``: 2x4 grid of the per-pattern mean diffraction
  intensity. Display is sqrt-compressed (matplotlib ``PowerNorm(gamma=0.5)``)
  with per-panel intensity ranges and a colorbar next to each panel.

Each montage places the (single) experimental dataset on the top-left and the
four synthetic datasets across the bottom row.
"""

from __future__ import annotations
from pathlib import Path
from typing import Protocol
import argparse
import sys

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.visualization import (  # noqa: E402
    CylindricalColorModel,
    ScalarTransformation,
    visualize_complex_values,
)

DEFAULT_DATA_DIR = Path('/home/beams0/SHENKE/workspace/ptychopinn_demo')
EXPERIMENTAL_SUBDIR = 'experimental_training_dataset_ic2'
SYNTHETIC_SUBDIR = 'synthetic_training_dataset_ic2'
COMPLEX_KEYS = ('probeGuess', 'objectGuess')
DIFFRACTION_KEY = 'diff3d'
DIFFRACTION_CMAP = 'inferno'
DIFFRACTION_GAMMA = 0.5  # sqrt scaling via PowerNorm
N_SYNTHETIC_COLS = 4
COLORWHEEL_COL = 1  # blank cell in row 0 to host the HSV legend
UNIT_PIXEL_GEOMETRY = PixelGeometry(width_m=1.0, height_m=1.0)


class Renderer(Protocol):
    def __call__(self, ax: plt.Axes, values: numpy.ndarray, caption: str) -> None: ...


def _render_complex(ax: plt.Axes, values: numpy.ndarray, caption: str) -> None:
    """Draw a complex 2D array onto ``ax`` using the ptychodus HSV color model."""
    product = visualize_complex_values(
        values,
        UNIT_PIXEL_GEOMETRY,
        CylindricalColorModel.HSV_VALUE,
        amplitude_transform=ScalarTransformation.IDENTITY,
    )
    ax.imshow(product.get_image_rgba(), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(caption, fontsize=10)


def _draw_colorwheel(ax: plt.Axes, *, size: int = 256) -> None:
    """Render the HSV_VALUE legend: a unit-disk color wheel with phase tick labels."""
    y, x = numpy.mgrid[-1.0 : 1.0 : size * 1j, -1.0 : 1.0 : size * 1j]
    r = numpy.hypot(x, y)
    theta = numpy.arctan2(y, x)
    z = (r * numpy.exp(1j * theta)).astype(numpy.complex64)
    product = visualize_complex_values(
        z,
        UNIT_PIXEL_GEOMETRY,
        CylindricalColorModel.HSV_VALUE,
        amplitude_transform=ScalarTransformation.IDENTITY,
    )
    rgba = product.get_image_rgba().copy()
    rgba[..., 3] = (r <= 1.0).astype(rgba.dtype)
    ax.imshow(rgba, interpolation='bilinear', extent=(-1.0, 1.0, -1.0, 1.0), origin='lower')

    for label, angle_deg in (
        ('0', 0),
        (r'$+\pi/2$', 90),
        (r'$\pm\pi$', 180),
        (r'$-\pi/2$', 270),
    ):
        angle = numpy.deg2rad(angle_deg)
        ax.text(
            1.18 * numpy.cos(angle),
            1.18 * numpy.sin(angle),
            label,
            ha='center',
            va='center',
            fontsize=9,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('HSV legend\nhue=phase, value=amplitude', fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_real(
    ax: plt.Axes, values: numpy.ndarray, caption: str, *, cmap: str = DIFFRACTION_CMAP
) -> None:
    """Draw a real-valued 2D array with a sqrt-compressed colormap and a colorbar."""
    vmin = float(values.min())
    vmax = float(values.max())
    norm = PowerNorm(gamma=DIFFRACTION_GAMMA, vmin=max(vmin, 0.0), vmax=vmax)
    img = ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(caption, fontsize=10)
    ax.figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def _load_key(path: Path, key: str) -> numpy.ndarray:
    with numpy.load(path) as npz:
        return numpy.asarray(npz[key])


def _mean_intensity(diff3d: numpy.ndarray) -> numpy.ndarray:
    """Mean intensity across the scan axis, clamped to non-negative values."""
    return numpy.maximum(diff3d.mean(axis=0), 0.0)


def _populate_row(
    axes: numpy.ndarray,
    row: int,
    datasets: list[tuple[str, numpy.ndarray]],
    caption_prefix: str,
    render: Renderer,
) -> None:
    """Render ``datasets`` along ``axes[row, :]``, turning off any leftover cells."""
    for col in range(N_SYNTHETIC_COLS):
        if col < len(datasets):
            name, array = datasets[col]
            render(axes[row, col], array, f'{caption_prefix}: {name}')
        else:
            axes[row, col].axis('off')


def _build_complex_montage(
    array_name: str,
    experimental: list[tuple[str, numpy.ndarray]],
    synthetic: list[tuple[str, numpy.ndarray]],
) -> plt.Figure:
    fig, axes = plt.subplots(2, N_SYNTHETIC_COLS, figsize=(16, 8), squeeze=False)
    fig.suptitle(f'{array_name}: experimental (top) vs synthetic (bottom)', fontsize=14)

    _populate_row(axes, 0, experimental, 'experimental', _render_complex)
    for col in range(len(experimental), N_SYNTHETIC_COLS):
        if col == COLORWHEEL_COL:
            _draw_colorwheel(axes[0, col])
        else:
            axes[0, col].axis('off')

    _populate_row(axes, 1, synthetic, 'synthetic', _render_complex)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _build_diffraction_montage(
    experimental: list[tuple[str, numpy.ndarray]],
    synthetic: list[tuple[str, numpy.ndarray]],
) -> plt.Figure:
    fig, axes = plt.subplots(2, N_SYNTHETIC_COLS, figsize=(16, 8), squeeze=False)
    fig.suptitle(
        f'{DIFFRACTION_KEY} mean intensity (sqrt-compressed, per-panel range): '
        'experimental (top) vs synthetic (bottom)',
        fontsize=14,
    )
    _populate_row(axes, 0, experimental, 'experimental', _render_real)
    _populate_row(axes, 1, synthetic, 'synthetic', _render_real)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate probeGuess, objectGuess, and mean diffraction-pattern '
            'comparison montages from the ptychopinn_demo training datasets.'
        )
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f'Root directory containing the dataset subfolders (default: {DEFAULT_DATA_DIR}).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Directory to write PNG figures into (default: current working directory).',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution of saved PNG files (default: 300).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    experimental_dir = args.data_dir / EXPERIMENTAL_SUBDIR
    synthetic_dir = args.data_dir / SYNTHETIC_SUBDIR

    experimental_paths = sorted(experimental_dir.glob('*.npz'))
    synthetic_paths = sorted(synthetic_dir.glob('synthetic_*.npz'))

    if not experimental_paths:
        print(f'No experimental .npz files found in {experimental_dir}', file=sys.stderr)
        return 1
    if not synthetic_paths:
        print(f'No synthetic .npz files found in {synthetic_dir}', file=sys.stderr)
        return 1

    synthetic_paths = synthetic_paths[:N_SYNTHETIC_COLS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key in COMPLEX_KEYS:
        experimental = [(p.stem, _load_key(p, key)) for p in experimental_paths]
        synthetic = [(p.stem, _load_key(p, key)) for p in synthetic_paths]
        fig = _build_complex_montage(key, experimental, synthetic)
        out_path = args.output_dir / f'{key}_montage.png'
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out_path}')

    experimental = [
        (p.stem, _mean_intensity(_load_key(p, DIFFRACTION_KEY))) for p in experimental_paths
    ]
    synthetic = [(p.stem, _mean_intensity(_load_key(p, DIFFRACTION_KEY))) for p in synthetic_paths]
    fig = _build_diffraction_montage(experimental, synthetic)
    out_path = args.output_dir / f'{DIFFRACTION_KEY}_mean_montage.png'
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
