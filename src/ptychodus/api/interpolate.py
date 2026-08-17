"""Sub-pixel array interpolation and stitching utilities."""

from typing import Any, Generic, TypeVar, overload

import numpy.typing
import numpy

from ptychodus.api.typing import InexactArrayType, RealArrayType

__all__ = [
    'BarycentricArrayInterpolator',
    'BarycentricArrayStitcher',
    'lerp',
]

InexactDType = TypeVar('InexactDType', bound=numpy.inexact[Any])


@overload
def lerp(lower: float, upper: float, frac: float) -> float:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: complex, upper: complex, frac: float) -> complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: float) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: float, upper: float, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


def lerp(
    lower: InexactArrayType | complex,
    upper: InexactArrayType | complex,
    frac: RealArrayType | float,
) -> InexactArrayType | complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    return (1.0 - frac) * lower + frac * upper


def _calculate_support_frac(x: float, n: int) -> tuple[slice, float]:
    """Return the integer support slice and sub-pixel fractional offset for a width-n window centered at x.

    The returned slice spans ``n + 1`` samples so bilinear interpolation
    across the n-wide window has grid points at both endpoints.
    """
    lower = x - n / 2
    whole = int(lower)
    return slice(whole, whole + n + 1), lower - whole


class BarycentricArrayInterpolator(Generic[InexactDType]):
    """Extract patches from an array using bilinear (barycentric) sub-pixel interpolation."""

    def __init__(self, array: numpy.typing.NDArray[InexactDType]) -> None:
        super().__init__()
        self._array = array

    def get_patch(
        self, center_x: float, center_y: float, width: int, height: int
    ) -> numpy.typing.NDArray[InexactDType]:
        x_support, x_frac = _calculate_support_frac(center_x, width)
        y_support, y_frac = _calculate_support_frac(center_y, height)

        support = self._array[..., y_support, x_support]
        # separable bilinear as fused in-place lerp: lower + frac * (upper - lower)
        y_interp = numpy.subtract(support[..., 1:, :], support[..., :-1, :])
        y_interp *= y_frac
        y_interp += support[..., :-1, :]
        patch = numpy.subtract(y_interp[..., :, 1:], y_interp[..., :, :-1])
        patch *= x_frac
        patch += y_interp[..., :, :-1]
        return patch  # type: ignore

    def add_patch(
        self,
        center_x: float,
        center_y: float,
        patch: numpy.typing.NDArray[InexactDType],
    ) -> None:
        """Bilinear-scatter *patch* back into the underlying array (transpose of get_patch)."""
        x_support, x_frac = _calculate_support_frac(center_x, patch.shape[-1])
        y_support, y_frac = _calculate_support_frac(center_y, patch.shape[-2])

        # reused quantities
        x_frac_c = 1.0 - x_frac
        y_frac_c = 1.0 - y_frac

        # barycentric interpolant weights
        weight00 = y_frac_c * x_frac_c
        weight01 = y_frac_c * x_frac
        weight10 = y_frac * x_frac_c
        weight11 = y_frac * x_frac

        support = self._array[..., y_support, x_support]
        scratch = numpy.empty(patch.shape, dtype=patch.dtype)
        numpy.multiply(patch, weight00, out=scratch)
        support[..., :-1, :-1] += scratch
        numpy.multiply(patch, weight01, out=scratch)
        support[..., :-1, 1:] += scratch
        numpy.multiply(patch, weight10, out=scratch)
        support[..., 1:, :-1] += scratch
        numpy.multiply(patch, weight11, out=scratch)
        support[..., 1:, 1:] += scratch


class BarycentricArrayStitcher(Generic[InexactDType]):
    """Accumulate weighted patches into a canvas using bilinear sub-pixel spreading, then normalize."""

    def __init__(
        self, upper: numpy.typing.NDArray[InexactDType], lower: RealArrayType | None = None
    ) -> None:
        super().__init__()
        self._upper = upper
        self._lower = lower

        if lower is not None and upper.shape != lower.shape:
            raise ValueError(f'Mismatched array shapes! ({upper.shape} != {lower.shape})')

    def add_patch(
        self,
        center_x: float,
        center_y: float,
        value: numpy.typing.NDArray[InexactDType],
        weight: RealArrayType | None = None,
    ) -> None:
        if numpy.iscomplexobj(self._upper) != numpy.iscomplexobj(value):
            raise ValueError(f'Mismatched value dtypes! ({self._upper.dtype} != {value.dtype})')

        if weight is not None:
            if self._lower is None:
                raise ValueError('Provided weights without a lower array!')

            if value.shape != weight.shape:
                raise ValueError(f'Mismatched patch shapes! ({value.shape=} != {weight.shape=})')

        x_support, x_frac = _calculate_support_frac(center_x, value.shape[-1])
        y_support, y_frac = _calculate_support_frac(center_y, value.shape[-2])

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
        u_scratch = numpy.empty(uvalue.shape, dtype=uvalue.dtype)
        numpy.multiply(uvalue, weight00, out=u_scratch)
        usupport[..., :-1, :-1] += u_scratch
        numpy.multiply(uvalue, weight01, out=u_scratch)
        usupport[..., :-1, 1:] += u_scratch
        numpy.multiply(uvalue, weight10, out=u_scratch)
        usupport[..., 1:, :-1] += u_scratch
        numpy.multiply(uvalue, weight11, out=u_scratch)
        usupport[..., 1:, 1:] += u_scratch

        if self._lower is not None and weight is not None:
            # add patch update to lower array support
            lsupport = self._lower[..., y_support, x_support]
            l_scratch = numpy.empty(weight.shape, dtype=weight.dtype)
            numpy.multiply(weight, weight00, out=l_scratch)
            lsupport[..., :-1, :-1] += l_scratch
            numpy.multiply(weight, weight01, out=l_scratch)
            lsupport[..., :-1, 1:] += l_scratch
            numpy.multiply(weight, weight10, out=l_scratch)
            lsupport[..., 1:, :-1] += l_scratch
            numpy.multiply(weight, weight11, out=l_scratch)
            lsupport[..., 1:, 1:] += l_scratch

    def stitch(self) -> numpy.typing.NDArray[InexactDType]:
        """Return the accumulated canvas divided by the weight array; pixels with zero weight remain zero."""
        if self._lower is None:
            return self._upper

        return numpy.divide(
            self._upper, self._lower, out=numpy.zeros_like(self._upper), where=(self._lower > 0)
        )
