#!/usr/bin/env python
"""Print a one-line summary (name, dtype, shape) of every array in an .npz file.

Example usage:

    npz-dump path/to/archive.npz
"""

from pathlib import Path
import argparse
import sys

import numpy


def main() -> int:
    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog,
        description='List the arrays inside an .npz file with their dtype and shape.',
    )
    parser.add_argument(
        'file',
        metavar='NPZ_FILE',
        type=argparse.FileType('rb'),
        help='Path to the .npz file.',
    )
    args = parser.parse_args()

    with numpy.load(args.file.name) as npz:
        names = list(npz.files)
        name_width = max((len(name) for name in names), default=0)
        for name in names:
            array = npz[name]
            line = f'{name:<{name_width}}  {array.dtype!s:<10}  {array.shape}'
            if array.ndim == 0 or array.size == 1:
                line += f' = {array.item()!r}'
            print(line)

    return 0


if __name__ == '__main__':
    sys.exit(main())
