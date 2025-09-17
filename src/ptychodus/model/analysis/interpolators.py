from typing import Generic, TypeVar
import logging

from numpy.typing import NDArray
import numpy

from ptychodus.api.typing import RealArrayType

__all__ = [
    'BarycentricArrayInterpolator',
    'BarycentricArrayStitcher',
    'NearestNeighborArrayInterpolator',
]

InexactDType = TypeVar('InexactDType', bound=numpy.inexact)

logger = logging.getLogger(__name__)


def calculate_support_frac(x: float, n: int) -> tuple[slice, float]:
    lower = x - n / 2
    whole = int(lower)
    return slice(whole, whole + n + 1), lower - whole


class NearestNeighborArrayInterpolator(Generic[InexactDType]):
    def __init__(self, array: NDArray[InexactDType]) -> None:
        super().__init__()
        self._array = array

    def get_patch(
        self, center_x: float, center_y: float, width: int, height: int
    ) -> NDArray[InexactDType]:
        y_lower = int(center_y - height / 2)
        y_support = slice(y_lower, y_lower + height)
        logger.debug(f'{y_support=}')

        x_lower = int(center_x - width / 2)
        x_support = slice(x_lower, x_lower + width)
        logger.debug(f'{x_support=}')

        return self._array[..., y_support, x_support]


class BarycentricArrayInterpolator(Generic[InexactDType]):
    def __init__(self, array: NDArray[InexactDType]) -> None:
        super().__init__()
        self._array = array

    def get_patch(
        self, center_x: float, center_y: float, width: int, height: int
    ) -> NDArray[InexactDType]:
        x_support, x_frac = calculate_support_frac(center_x, width)
        y_support, y_frac = calculate_support_frac(center_y, height)

        # reused quantities
        x_frac_c = 1.0 - x_frac
        y_frac_c = 1.0 - y_frac

        # barycentric interpolant weights
        weight00 = y_frac_c * x_frac_c
        weight01 = y_frac_c * x_frac
        weight10 = y_frac * x_frac_c
        weight11 = y_frac * x_frac

        support = self._array[..., y_support, x_support]
        patch = weight00 * support[..., :-1, :-1]
        patch = patch + weight01 * support[..., :-1, 1:]
        patch = patch + weight10 * support[..., 1:, :-1]
        patch = patch + weight11 * support[..., 1:, 1:]
        return patch  # type: ignore


class BarycentricArrayStitcher(Generic[InexactDType]):
    def __init__(self, upper: NDArray[InexactDType], lower: RealArrayType | None = None) -> None:
        super().__init__()
        self._upper = upper
        self._lower = lower

        if lower is not None and upper.shape != lower.shape:
            raise ValueError(f'Mismatched array shapes! ({upper.shape} != {lower.shape})')

    def add_patch(
        self,
        center_x: float,
        center_y: float,
        value: NDArray[InexactDType],
        weight: RealArrayType | None = None,
    ) -> None:
        if numpy.iscomplexobj(self._upper) != numpy.iscomplexobj(value):
            raise ValueError(f'Mismatched value dtypes! ({self._upper.dtype} != {value.dtype})')

        if weight is not None:
            if self._lower is None:
                raise ValueError('Provided weights without a lower array!')

            if value.shape != weight.shape:
                raise ValueError(f'Mismatched patch shapes! ({value.shape=} != {weight.shape=})')

        x_support, x_frac = calculate_support_frac(center_x, value.shape[-1])
        y_support, y_frac = calculate_support_frac(center_y, value.shape[-2])

        # reused quantities
        x_frac_c = 1.0 - x_frac
        y_frac_c = 1.0 - y_frac

        # barycentric interpolant weights
        weight00 = y_frac_c * x_frac_c
        weight01 = y_frac_c * x_frac
        weight10 = y_frac * x_frac_c
        weight11 = y_frac * x_frac

        # add patch update to upper array support
        uvalue = value if weight is None else weight * value
        usupport = self._upper[..., y_support, x_support]
        usupport[..., :-1, :-1] = usupport[..., :-1, :-1] + weight00 * uvalue
        usupport[..., :-1, 1:] = usupport[..., :-1, 1:] + weight01 * uvalue
        usupport[..., 1:, :-1] = usupport[..., 1:, :-1] + weight10 * uvalue
        usupport[..., 1:, 1:] = usupport[..., 1:, 1:] + weight11 * uvalue

        if self._lower is not None and weight is not None:
            # add patch update to lower array support
            lsupport = self._lower[..., y_support, x_support]
            lsupport[..., :-1, :-1] += weight00 * weight
            lsupport[..., :-1, 1:] += weight01 * weight
            lsupport[..., 1:, :-1] += weight10 * weight
            lsupport[..., 1:, 1:] += weight11 * weight

    def stitch(self) -> NDArray[InexactDType]:
        if self._lower is None:
            return self._upper

        return numpy.divide(
            self._upper, self._lower, out=numpy.zeros_like(self._upper), where=(self._lower > 0)
        )
