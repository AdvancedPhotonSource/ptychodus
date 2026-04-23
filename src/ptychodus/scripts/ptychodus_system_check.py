#!/usr/bin/env python

import platform
import sys


def main() -> None:
    for key, value in platform.uname()._asdict().items():
        print(f'{key.title()}: {value}')

    try:
        import torch
    except ImportError:
        print('PyTorch is not installed.')
    else:
        print(f'PyTorch Version: {torch.__version__}')
        print(f'CUDA Available: {torch.cuda.is_available()}')
        print(f'CUDA Version: {torch.version.cuda}')


if __name__ == '__main__':
    sys.exit(main())
