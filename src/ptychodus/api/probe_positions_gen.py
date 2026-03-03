from collections.abc import Iterable, Iterator

import numpy

from .geometry import AffineTransform
from .probe_positions import ProbePosition


def generate_cartesian_probe_positions(
    num_points_x: int,
    num_points_y: int,
    step_size_x: float,
    step_size_y: float,
    *,
    snake: bool = False,
    stagger: bool = False,
) -> Iterator[ProbePosition]:
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
            coordinate_x_m=xf,
            coordinate_y_m=yf,
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
            coordinate_x_m=radius_m * numpy.cos(theta_rad),
            coordinate_y_m=radius_m * numpy.sin(theta_rad),
        )


def generate_lissajous_probe_positions(
    num_points: int,
    amplitude_x_m: float,
    amplitude_y_m: float,
    angular_step_x_tr: float,
    angular_step_y_tr: float,
    angular_shift_tr: float,
) -> Iterator[ProbePosition]:
    for index in range(num_points):
        two_pi = 2 * numpy.pi
        theta_x = two_pi * angular_step_x_tr * index + angular_shift_tr
        theta_y = two_pi * angular_step_y_tr * index

        yield ProbePosition(
            index=index,
            coordinate_x_m=amplitude_x_m * numpy.sin(theta_x),
            coordinate_y_m=amplitude_y_m * numpy.sin(theta_y),
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
            coordinate_x_m=radius_m * numpy.cos(theta_rad),
            coordinate_y_m=radius_m * numpy.sin(theta_rad),
        )


def transform_probe_positions(
    positions: Iterable[ProbePosition],
    transform: AffineTransform,
    rng: numpy.random.Generator | None = None,
    jitter_radius_m: float = 0.0,
) -> Iterator[ProbePosition]:
    for position in positions:
        x_m, y_m = transform(position.coordinate_x_m, position.coordinate_y_m)

        if rng is not None:
            angle_rad = 2 * numpy.pi * rng.uniform()
            radius_m = jitter_radius_m * numpy.sqrt(rng.uniform())

            x_m += radius_m * numpy.cos(angle_rad)
            y_m += radius_m * numpy.sin(angle_rad)

        yield ProbePosition(
            index=position.index,
            coordinate_x_m=x_m,
            coordinate_y_m=y_m,
        )
