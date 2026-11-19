"""Summarize and plot the I0 monitor of an LCLS XPP smalldata file.

Prints the quartiles of the I0 readings and how many shots fall inside the window the
LCLS readers keep, then plots the readings shot by shot and their density, to help
choose that window for a run.

Usage: ``python scripts/plot_lcls_i0.py <smalldata.h5> [--i0-path /ipm2/sum]``
"""

from __future__ import annotations

from pathlib import Path
import argparse

from scipy.stats import gaussian_kde
import h5py
import matplotlib.pyplot as plt
import numpy

from ptychodus.plugins.lcls_file_readers import LCLSShotSelection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('file_path', type=Path, help='psana smalldata HDF5 file')
    parser.add_argument('--i0-path', default='/ipm2/sum', help='dataset holding I0')
    args = parser.parse_args()

    with h5py.File(args.file_path, 'r') as h5_file:
        i0 = numpy.asarray(h5_file[args.i0_path][()], dtype=numpy.float64)

    q0 = numpy.min(i0)
    q1, q2, q3 = numpy.quantile(i0, [0.25, 0.50, 0.75])
    q4 = numpy.max(i0)

    selection = LCLSShotSelection()
    low = selection.i0_low_threshold
    high = selection.i0_high_threshold
    num_in_window = numpy.count_nonzero((i0 > 0.0) & (low <= i0) & (i0 < high))

    print(f'I0: {i0.dtype}{i0.shape}')
    print(f'min = {q0}')
    print(f'Q1 = {q1}')
    print(f'Q2 = {q2}')
    print(f'Q3 = {q3}')
    print(f'max = {q4}')
    print(f'in [{low}, {high}) = {num_in_window} of {i0.size}')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(i0, '.', linewidth=1.5)
    ax.set_title('I0')
    ax.set_xlabel('Shot')
    ax.set_ylabel('Value')
    ax.grid(True)
    plt.show()

    values = numpy.linspace(q0, q4, 1000)
    density = gaussian_kde(i0)(values)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(values, density, '.-', linewidth=1.5)
    ax.set_title('I0')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
