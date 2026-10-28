"""Probe position (scan pattern) generators: cartesian, concentric, Lissajous, and spiral grids.

Also home to :class:`ProbePositionsGenerator`, the interface a scan pattern is
offered through. It sits beside the functions because it is how they are
reached: a generator states what it accepts and turns one set of those values
into points, and the functions below are what it turns them with.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy
from pydantic import BaseModel

from ..probe_positions import ProbePosition


class ProbePositionsGenerator(ABC):
    """A scan pattern, and the options it accepts.

    Stateless and configured per call: a generator holds no settings of its
    own, so it costs nothing to construct and two callers can drive the same
    one with different values at the same time. It does not name itself
    either; the name a person selects it by is the one it was registered
    under.

    What it returns is raw: the points the pattern describes, in acquisition
    order, in meters, with no transform or jitter applied and nothing
    discarded. Conditioning a scan is a separate concern from describing one,
    and a generator that applied any of it would be applying it twice.
    """

    @abstractmethod
    def get_default_options(self) -> BaseModel:
        """Return this generator's options, populated with their defaults.

        A description, not live state: plain pydantic carrying values, types,
        ranges, enums and the words to put beside each field. A caller edits a
        copy and passes it back as ``options``, and its type is what an
        incoming ``options`` is validated against.

        An instance rather than the class, because a field declared with a
        ``default_factory`` reports its class-level default as undefined. Only
        an instance carries the value such a field actually starts at.
        """

    @abstractmethod
    def generate(self, options: BaseModel) -> Sequence[ProbePosition]:
        """Return the points *options* describes, in acquisition order."""


def generate_cartesian_probe_positions(
    num_points_x: int,
    num_points_y: int,
    step_size_x: float,
    step_size_y: float,
    *,
    snake: bool = False,
    stagger: bool = False,
) -> Iterator[ProbePosition]:
    """Yield probe positions on a regular rectangular grid, optionally in snake or staggered order."""
    for index in range(num_points_x * num_points_y):
        y, x = divmod(index, num_points_x)

        if snake:
            if y & 1:
                x = num_points_x - 1 - x

        cx = (num_points_x - 1) / 2
        cy = (num_points_y - 1) / 2

        xf = (x - cx) * step_size_x
        yf = (y - cy) * step_size_y

        if stagger:
            if y & 1:
                xf += step_size_x / 4
            else:
                xf -= step_size_x / 4

        yield ProbePosition(
            index=index,
            x_m=xf,
            y_m=yf,
        )


def generate_concentric_probe_positions(
    radial_step_size_m: float, num_shells: int, num_points_1st_shell: int
) -> Iterator[ProbePosition]:
    """https://doi.org/10.1088/1367-2630/12/3/035017"""
    triangle = (num_shells * (num_shells + 1)) // 2
    num_points = triangle * num_points_1st_shell

    for index in range(num_points):
        triangle = index // num_points_1st_shell
        shell_index = int((1 + numpy.sqrt(1 + 8 * triangle)) / 2) - 1  # see OEIS A002024
        shell_triangle = (shell_index * (shell_index + 1)) // 2
        first_index_in_shell = num_points_1st_shell * shell_triangle
        point_index_in_shell = index - first_index_in_shell

        radius_m = radial_step_size_m * (shell_index + 1)
        num_points_in_shell = num_points_1st_shell * (shell_index + 1)
        theta_rad = 2 * numpy.pi * point_index_in_shell / num_points_in_shell

        yield ProbePosition(
            index=index,
            x_m=radius_m * numpy.cos(theta_rad),
            y_m=radius_m * numpy.sin(theta_rad),
        )


def generate_lissajous_probe_positions(
    num_points: int,
    amplitude_x_m: float,
    amplitude_y_m: float,
    angular_step_x_tr: float,
    angular_step_y_tr: float,
    angular_shift_tr: float,
) -> Iterator[ProbePosition]:
    """Yield probe positions following a Lissajous figure parameterized by angular steps and amplitudes."""
    for index in range(num_points):
        two_pi = 2 * numpy.pi
        theta_x = two_pi * angular_step_x_tr * index + angular_shift_tr
        theta_y = two_pi * angular_step_y_tr * index

        yield ProbePosition(
            index=index,
            x_m=amplitude_x_m * numpy.sin(theta_x),
            y_m=amplitude_y_m * numpy.sin(theta_y),
        )


def generate_spiral_probe_positions(
    num_points: int, radius_scalar_m: float
) -> Iterator[ProbePosition]:
    """https://doi.org/10.1364/OE.22.012634"""
    for index in range(num_points):
        radius_m = radius_scalar_m * numpy.sqrt(index)
        divergence_angle_rad = (3.0 - numpy.sqrt(5)) * numpy.pi
        theta_rad = divergence_angle_rad * index

        yield ProbePosition(
            index=index,
            x_m=radius_m * numpy.cos(theta_rad),
            y_m=radius_m * numpy.sin(theta_rad),
        )
