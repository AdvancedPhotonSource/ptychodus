"""Unit tests for BiharmonicInpainter.

Behaviors verified:
  - it matches skimage's inpaint_biharmonic on the interior holes followed by zeroing the
    regions touching the border, for holes deep inside, 1-2 px from the border, and beside
    a saturated border region whose values skimage treats as known
  - single frames and stacks, integer and float dtypes, Fortran-ordered input
  - overwrite_input fills a float64 stack in place and leaves other input untouched
  - a mask with only border regions zeroes them, and frames of the wrong shape are rejected
"""

import numpy
import numpy.testing
import pytest
from scipy import ndimage
from skimage.restoration import inpaint_biharmonic

from ptychodus.api.diffraction import BadPixels
from ptychodus.api.inpaint import BiharmonicInpainter

_SHAPE = (40, 48)


def _reference(data: numpy.ndarray, bad_pixels: BadPixels) -> numpy.ndarray:
    labels, _ = ndimage.label(bad_pixels)
    border_labels = numpy.unique(
        numpy.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    edge_pixels = numpy.isin(labels, border_labels[border_labels != 0])
    interior_pixels = bad_pixels & ~edge_pixels

    if data.ndim == 2:
        filled = inpaint_biharmonic(data.astype(numpy.float64), interior_pixels)
    else:
        stacked = inpaint_biharmonic(
            numpy.moveaxis(data, 0, -1).astype(numpy.float64), interior_pixels, channel_axis=-1
        )
        filled = numpy.moveaxis(stacked, -1, 0)

    filled[..., edge_pixels] = 0.0
    return filled


def _patterns(num_patterns: int, *, seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    y = numpy.arange(_SHAPE[0]).reshape(-1, 1) - 20.0
    x = numpy.arange(_SHAPE[1]).reshape(1, -1) - 22.0
    beam = 5000.0 * numpy.exp(-(x**2 + y**2) / 30.0) + 2.0
    return rng.poisson(beam, size=(num_patterns, *_SHAPE)).astype(numpy.float64)


def _bad_pixels() -> BadPixels:
    bad_pixels: BadPixels = numpy.zeros(_SHAPE, dtype=bool)
    bad_pixels[20, 22] = True  # on the beam
    bad_pixels[10:13, 30:32] = True  # a cluster
    bad_pixels[1, 5:8] = True  # 1 px from the top
    bad_pixels[2, 40] = True  # 2 px from the top
    bad_pixels[38, 10] = True  # 1 px from the bottom
    bad_pixels[:, 44:46] = True  # a gap running off the frame
    bad_pixels[15:18, 42] = True  # beside the gap, diagonal only, so its own hole
    bad_pixels[14, 43] = True
    return bad_pixels


def test_a_stack_matches_skimage() -> None:
    patterns = _patterns(6)
    patterns[:, :, 44:46] = 4.0e9  # a saturated gap is known to skimage

    inpainted = BiharmonicInpainter(_bad_pixels())(patterns)

    assert inpainted.dtype == numpy.float64
    numpy.testing.assert_allclose(inpainted, _reference(patterns, _bad_pixels()), atol=1e-6)


def test_a_single_frame_matches_skimage() -> None:
    frame = _patterns(1)[0]

    inpainted = BiharmonicInpainter(_bad_pixels())(frame)

    assert inpainted.shape == _SHAPE
    numpy.testing.assert_allclose(inpainted, _reference(frame, _bad_pixels()), atol=1e-9)


@pytest.mark.parametrize('dtype', [numpy.uint16, numpy.int32, numpy.float32])
def test_the_dtype_is_kept_and_integers_are_rounded(dtype: type) -> None:
    patterns: numpy.ndarray = _patterns(3).astype(dtype)

    inpainted = BiharmonicInpainter(_bad_pixels())(patterns)
    expected = _reference(patterns, _bad_pixels())

    if numpy.issubdtype(dtype, numpy.integer):
        expected = numpy.rint(expected)

    assert inpainted.dtype == dtype
    numpy.testing.assert_allclose(inpainted, expected.astype(dtype), rtol=1e-6)


def test_fortran_ordered_input_is_filled() -> None:
    patterns = numpy.asfortranarray(_patterns(3))

    inpainted = BiharmonicInpainter(_bad_pixels())(patterns)

    numpy.testing.assert_allclose(inpainted, _reference(patterns, _bad_pixels()), atol=1e-9)


def test_overwrite_input_fills_in_place() -> None:
    patterns = _patterns(3)
    expected = _reference(patterns, _bad_pixels())

    inpainted = BiharmonicInpainter(_bad_pixels())(patterns, overwrite_input=True)

    assert inpainted is patterns
    numpy.testing.assert_allclose(patterns, expected, atol=1e-9)


def test_without_overwrite_the_input_is_untouched() -> None:
    patterns = _patterns(3)
    original = patterns.copy()
    integers = patterns.astype(numpy.uint32)

    BiharmonicInpainter(_bad_pixels())(patterns)
    BiharmonicInpainter(_bad_pixels())(integers, overwrite_input=True)

    numpy.testing.assert_array_equal(patterns, original)
    numpy.testing.assert_array_equal(integers, original.astype(numpy.uint32))


def test_border_regions_alone_are_zeroed() -> None:
    bad_pixels: BadPixels = numpy.zeros(_SHAPE, dtype=bool)
    bad_pixels[0, :] = True
    bad_pixels[:, -3:] = True
    patterns = _patterns(2)

    inpainted = BiharmonicInpainter(bad_pixels)(patterns)

    assert (inpainted[:, bad_pixels] == 0.0).all()
    numpy.testing.assert_array_equal(inpainted[:, ~bad_pixels], patterns[:, ~bad_pixels])


def test_frames_must_match_the_mask() -> None:
    inpainter = BiharmonicInpainter(_bad_pixels())

    with pytest.raises(ValueError):
        inpainter(numpy.zeros((2, 10, 10)))

    with pytest.raises(ValueError):
        inpainter(numpy.zeros((1, 1, *_SHAPE)))  # type: ignore[call-overload]
