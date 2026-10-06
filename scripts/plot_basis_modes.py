#!/usr/bin/env python
"""Render reference figures for the Zernike, Hermite and Legendre basis modes.

Each family gets the panel type that suits what it actually is: Zernike modes
are complex fields on the unit disk, Hermite modes are complex fields on a
Cartesian square, and Legendre modes are real surface-height profiles along a
single normalized aperture coordinate -- so the first two are rendered as 2-D
colour maps and the third as a set of 1-D curves.

Example usage:

    python scripts/plot_basis_modes.py --output-dir figures/
    python scripts/plot_basis_modes.py --which zernike --max-radial-degree 4

No packaged dependency -- this file is intentionally not shipped with the wheel.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use('Agg')

from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy

from ptychodus.api.geometry import HermiteMode, LegendreMode, ZernikeMode

# Panels are sized in inches and the figure is grown to fit them, rather than a
# fixed-size figure being subdivided until the titles collide.
PANEL_SIZE_IN = 1.4
# Panels hold their aspect ratio, so the room a title and the suptitle need has to
# be added to the figure rather than taken out of the panels -- otherwise the
# layout engine pushes the last row off the canvas.
TITLE_HEIGHT_IN = 0.3
SUPTITLE_HEIGHT_IN = 0.4
DIVERGING_COLORMAP = 'seismic'

# The conventional names of the low Legendre orders, per the LegendreMode docstring.
LEGENDRE_MEANINGS = {0: 'piston', 1: 'tilt', 2: 'defocus', 3: 'coma'}


def _grid_figsize(num_rows: int, num_columns: int) -> tuple[float, float]:
    """Size a figure to hold `num_rows` x `num_columns` panels plus their titles."""
    width_in = num_columns * PANEL_SIZE_IN
    height_in = num_rows * (PANEL_SIZE_IN + TITLE_HEIGHT_IN) + SUPTITLE_HEIGHT_IN
    return width_in, height_in


def _normalized_grid(num_pixels: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return (y, x) over the square [-1, 1] x [-1, 1], pixel centers included."""
    y, x = numpy.mgrid[:num_pixels, :num_pixels]
    half = num_pixels / 2
    return (y - (num_pixels - 1) / 2) / half, (x - (num_pixels - 1) / 2) / half


def plot_zernike_pyramid(max_radial_degree: int, num_pixels: int) -> Figure:
    """Lay the Zernike modes out in the conventional pyramid, row = radial degree.

    Points outside the unit disk are left undefined so they render blank rather
    than as a spurious zero ring around each mode.
    """
    y, x = _normalized_grid(num_pixels)
    distance = numpy.hypot(y, x)
    angle = numpy.arctan2(y, x)

    num_rows = max_radial_degree + 1
    figure = plt.figure(figsize=_grid_figsize(num_rows, num_rows), layout='constrained')
    # Half-width columns, so a mode can be centered on the pyramid's axis: each
    # panel spans two of them, and m shifts it one column per unit.
    grid = figure.add_gridspec(num_rows, 2 * num_rows)

    for radial_degree in range(num_rows):
        for angular_frequency in range(-radial_degree, radial_degree + 1, 2):
            mode = ZernikeMode(1.0, radial_degree, angular_frequency)
            values = mode(distance, angle, undefined_value=numpy.nan)

            column = max_radial_degree + angular_frequency
            axes = figure.add_subplot(grid[radial_degree, column : column + 2])
            axes.pcolormesh(x, y, values.real, norm=CenteredNorm(), cmap=DIVERGING_COLORMAP)
            axes.set_aspect('equal')
            axes.set_title(f'$Z_{{{radial_degree}}}^{{{angular_frequency:+d}}}$')
            axes.axis('off')

    figure.suptitle(f'Zernike modes to radial degree {max_radial_degree}')
    return figure


def plot_hermite_grid(max_order: int, num_pixels: int) -> Figure:
    """Lay the Hermite modes out as an (order_x, order_y) matrix.

    Each panel gets its own color normalization: Hermite amplitudes grow quickly
    with order, so a shared scale would flatten the low orders to a single tone.
    """
    y, x = _normalized_grid(num_pixels)

    num_rows = max_order + 1
    figure, axes_grid = plt.subplots(
        num_rows,
        num_rows,
        figsize=_grid_figsize(num_rows, num_rows),
        layout='constrained',
        squeeze=False,
    )

    for order_x in range(num_rows):
        for order_y in range(num_rows):
            mode = HermiteMode(1.0, order_x, order_y)
            values = mode(x, y)

            axes = axes_grid[order_x][order_y]
            axes.pcolormesh(x, y, values.real, norm=CenteredNorm(), cmap=DIVERGING_COLORMAP)
            axes.set_aspect('equal')
            axes.set_title(f'$H_{{{order_x},{order_y}}}$')
            axes.axis('off')

    figure.suptitle(f'Hermite modes to order {max_order}')
    return figure


def plot_legendre_profiles(max_order: int, num_samples: int) -> Figure:
    """Overlay the Legendre modes as height profiles against the aperture coordinate.

    These describe a surface height along one normalized coordinate, not a 2-D
    field, so they are drawn as curves. Overlaying them makes the defining
    property visible: order n crosses zero n times across the aperture.
    """
    u = numpy.linspace(-1.0, 1.0, num_samples)

    figure, axes = plt.subplots(figsize=(7.0, 4.5), layout='constrained')

    for order in range(max_order + 1):
        mode = LegendreMode(1.0, order)
        meaning = LEGENDRE_MEANINGS.get(order)
        label = f'$P_{{{order}}}$' if meaning is None else f'$P_{{{order}}}$ ({meaning})'
        axes.plot(u, mode(u), label=label)

    axes.axhline(0.0, color='gray', linewidth=0.8, zorder=0)
    axes.set_xlim(-1.0, 1.0)
    axes.set_xlabel('Normalized aperture coordinate $u$')
    axes.set_ylabel('Surface height / coefficient')
    axes.set_title(f'Legendre modes to order {max_order}')
    axes.legend(ncol=2, fontsize=8.0)
    return figure


def main() -> int:
    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Render reference figures for the Zernike, Hermite and Legendre basis modes.',
    )
    parser.add_argument(
        '--which',
        choices=('zernike', 'hermite', 'legendre', 'all'),
        default='all',
        help='Which family to render (default: all).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(),
        help='Directory to write the PNGs into (default: the current directory).',
    )
    parser.add_argument(
        '--max-radial-degree', type=int, default=6, help='Highest Zernike radial degree.'
    )
    parser.add_argument('--max-order', type=int, default=5, help='Highest Hermite order per axis.')
    parser.add_argument('--max-legendre-order', type=int, default=6, help='Highest Legendre order.')
    parser.add_argument(
        '--num-pixels', type=int, default=256, help='Samples per axis in the 2-D panels.'
    )
    parser.add_argument('--dpi', type=int, default=150, help='Output resolution.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Figure] = {}

    # Titles sit above ~1.4 in panels, so the sizes are set explicitly rather than
    # left at the ~10 pt default, which overflows a panel and runs into its neighbor.
    # Inlined rather than hoisted to a constant: matplotlib types rc_context's keys
    # as a Literal union, which a dict constant widens to str.
    with plt.rc_context({'font.size': 8.0, 'axes.titlesize': 8.0, 'axes.titlepad': 2.0}):
        if args.which in ('zernike', 'all'):
            figures['zernike_pyramid.png'] = plot_zernike_pyramid(
                args.max_radial_degree, args.num_pixels
            )

        if args.which in ('hermite', 'all'):
            figures['hermite_grid.png'] = plot_hermite_grid(args.max_order, args.num_pixels)

        if args.which in ('legendre', 'all'):
            figures['legendre_profiles.png'] = plot_legendre_profiles(
                args.max_legendre_order, args.num_pixels
            )

        for name, figure in figures.items():
            path = args.output_dir / name
            figure.savefig(path, dpi=args.dpi)
            plt.close(figure)
            print(path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
