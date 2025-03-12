from ptychodus.api.typing import ComplexArrayType, RealArrayType

__all__ = ['BarycentricArrayStitcher']


# FIXME BEGIN
# For ObjectStitcher, add object patches with uniform(?) weights.
# For STXM, add pattern counts * normalized probe profile.
# For Exposure, add probe profile. No weights?
# FIXME END


def calculate_support_frac(x: float, n: int) -> tuple[slice, float]:
    lower = x - n / 2
    whole = int(lower)
    return slice(whole, whole + n + 1), lower - whole


class BarycentricArrayStitcher:  # FIXME use
    def __init__(
        self, upper: ComplexArrayType | RealArrayType, lower: RealArrayType | None = None
    ) -> None:
        self._upper = upper
        self._lower = lower

        if lower is not None and upper.shape != lower.shape:
            raise ValueError('Mismatched array shapes! ({upper.shape} != {lower.shape})')

    def add_patch(
        self,
        center_x: float,
        center_y: float,
        value: ComplexArrayType | RealArrayType,
        weight: RealArrayType | None = None,
    ) -> None:
        if self._upper.dtype != value.dtype:
            raise ValueError('Mismatched value dtypes! ({self._upper.dtype} != {value.dtype})')

        if weight is not None:
            if self._lower is None:
                raise ValueError('Provided weights without a lower array!')
            elif self._lower.dtype != weight.dtype:
                raise ValueError('Mismatched weight types! ({type(self._lower)} != {type(weight)})')

            if value.shape != weight.shape:
                raise ValueError('Mismatched patch shapes! ({value.shape=} != {weight.shape=})')

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
        usupport[..., :-1, :-1] += weight00 * uvalue
        usupport[..., :-1, 1:] += weight01 * uvalue
        usupport[..., 1:, :-1] += weight10 * uvalue
        usupport[..., 1:, 1:] += weight11 * uvalue

        if self._lower is not None and weight is not None:
            # add patch update to lower array support
            lsupport = self._lower[..., y_support, x_support]
            lsupport[..., :-1, :-1] += weight00 * weight
            lsupport[..., :-1, 1:] += weight01 * weight
            lsupport[..., 1:, :-1] += weight10 * weight
            lsupport[..., 1:, 1:] += weight11 * weight

    def stitch(self) -> ComplexArrayType | RealArrayType:
        return self._upper if self._lower is None else self._upper / self._lower
