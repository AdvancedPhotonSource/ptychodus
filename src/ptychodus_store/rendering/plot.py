"""Server-side plot rendering for quantities that are not colormapped arrays.

Scan positions are a scatter in continuous coordinates rather than a raster, so
``ptychodus.api.visualize`` -- which maps a 2-D value array through a colormap -- does
not apply. The desktop app draws them with matplotlib, and so does this, to the same
recipe, so the two panels read alike.

Figures are built through :class:`~matplotlib.figure.Figure` rather than ``pyplot``:
pyplot keeps a global registry that a long-lived server would have to remember to clear
after every request, while a bare Figure is owned by the caller and collected with it.
That also avoids imposing a global backend on a process that merely imports this.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ptychodus_store.rendering.schemas import PlotImage

_DPI = 100


@dataclass(frozen=True)
class ScanPath:
    """One scan's positions, labelled for the legend."""

    label: str
    x_m: Sequence[float]
    y_m: Sequence[float]


def render_scan_paths(
    scans: Sequence[ScanPath],
    *,
    connect_path: bool = True,
    width_px: int = 640,
    height_px: int = 640,
) -> PlotImage:
    """Draw one or more scan paths on shared axes, one color and legend entry each.

    Mirrors the desktop plot: dots joined by lines, y inverted, equal aspect, a grid
    and axes in meters.
    """
    figure = Figure(figsize=(width_px / _DPI, height_px / _DPI), dpi=_DPI, layout='constrained')
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)

    for scan in scans:
        axes.plot(
            scan.x_m,
            scan.y_m,
            '.-' if connect_path else '.',
            label=scan.label,
            linewidth=1.5,
        )

    axes.invert_yaxis()
    axes.axis('equal')
    axes.grid(True)
    axes.set_xlabel('X [m]')
    axes.set_ylabel('Y [m]')

    if axes.lines:
        axes.legend(loc='best')

    buffer = BytesIO()
    canvas.print_png(buffer)

    return PlotImage(
        png_base64=b64encode(buffer.getvalue()).decode('ascii'),
        shape_h_px=height_px,
        shape_w_px=width_px,
    )
