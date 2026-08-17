#!/usr/bin/env python
"""Micro-benchmark for BarycentricArrayInterpolator / BarycentricArrayStitcher.

Times the per-call cost of the gather (``get_patch``), scatter (``add_patch``),
and stitcher scatter at sizes representative of the VSPI fluorescence enhancer
and PtychoNN training-data export inner loops.

Run this on both ``main`` and the optimization branch to quantify the speedup:

    python scripts/bench_interpolate.py

No packaged dependency — this file is intentionally not shipped with the wheel.
"""

from __future__ import annotations

import time

import numpy

from ptychodus.api.interpolate import (
    BarycentricArrayInterpolator,
    BarycentricArrayStitcher,
)


def _time(label: str, n_calls: int, fn):
    fn()  # warm-up: exclude first-touch page faults and dispatch caches
    tic = time.perf_counter()
    for _ in range(n_calls):
        fn()
    toc = time.perf_counter()
    per_call_us = (toc - tic) / n_calls * 1e6
    calls_per_sec = n_calls / (toc - tic)
    print(f'  {label:40s} {per_call_us:9.2f} us/call    {calls_per_sec:10.1f} calls/sec')


def _bench_case(
    *,
    array_shape: tuple[int, ...],
    patch_shape: tuple[int, int],
    n_calls: int,
    dtype: numpy.dtype,
    label: str,
) -> None:
    rng = numpy.random.default_rng(2026)
    array = rng.standard_normal(array_shape).astype(dtype, copy=False)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        array = array + 1j * rng.standard_normal(array_shape).astype(dtype, copy=False)

    height_px, width_px = patch_shape
    max_cy = array_shape[-2] - height_px - 2
    max_cx = array_shape[-1] - width_px - 2
    centers_y = rng.uniform(height_px, max_cy, size=n_calls)
    centers_x = rng.uniform(width_px, max_cx, size=n_calls)

    patch_full_shape = (*array_shape[:-2], *patch_shape)
    patches_real = rng.standard_normal(patch_full_shape).astype(dtype, copy=False)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        patches_real = patches_real + 1j * rng.standard_normal(patch_full_shape).astype(
            dtype, copy=False
        )

    print(f'{label}: array={array_shape} dtype={dtype} patch={patch_shape} N={n_calls}')

    interp = BarycentricArrayInterpolator(array)
    counter = [0]

    def do_get_patch() -> None:
        i = counter[0] % n_calls
        counter[0] += 1
        interp.get_patch(centers_x[i], centers_y[i], width_px, height_px)

    _time('get_patch', n_calls, do_get_patch)

    def do_add_patch() -> None:
        i = counter[0] % n_calls
        counter[0] += 1
        interp.add_patch(centers_x[i], centers_y[i], patches_real)

    _time('add_patch', n_calls, do_add_patch)

    if not numpy.issubdtype(dtype, numpy.complexfloating):
        return  # Stitcher-with-weights case exercised via complex path below

    upper = numpy.zeros(array_shape, dtype=dtype)
    lower = numpy.zeros(array_shape, dtype=numpy.float64)
    stitcher = BarycentricArrayStitcher(upper=upper, lower=lower)
    weight = rng.standard_normal(patch_full_shape).astype(numpy.float64, copy=False) ** 2

    def do_stitcher_add() -> None:
        i = counter[0] % n_calls
        counter[0] += 1
        stitcher.add_patch(centers_x[i], centers_y[i], patches_real, weight=weight)

    _time('Stitcher.add_patch (weighted)', n_calls, do_stitcher_add)


def main() -> None:
    print('=== VSPI-representative: 2048x2048 float64 object, 256x256 patches ===')
    _bench_case(
        array_shape=(2048, 2048),
        patch_shape=(256, 256),
        n_calls=200,
        dtype=numpy.dtype(numpy.float64),
        label='VSPI real',
    )

    print()
    print('=== PtychoNN-representative: 2048x2048 complex128 object, 128x128 patches ===')
    _bench_case(
        array_shape=(2048, 2048),
        patch_shape=(128, 128),
        n_calls=500,
        dtype=numpy.dtype(numpy.complex128),
        label='PtychoNN complex',
    )

    print()
    print('=== Broadcast leading axis: (4, 1024, 1024) complex64, 128x128 patches ===')
    _bench_case(
        array_shape=(4, 1024, 1024),
        patch_shape=(128, 128),
        n_calls=200,
        dtype=numpy.dtype(numpy.complex64),
        label='Multi-mode complex',
    )


if __name__ == '__main__':
    main()
