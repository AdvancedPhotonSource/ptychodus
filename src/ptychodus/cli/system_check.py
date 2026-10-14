#!/usr/bin/env python

import argparse
import ctypes
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def read_driver_cuda_version() -> str | None:
    """The newest CUDA runtime the installed NVIDIA driver supports.

    Read from the ``nvidia-smi`` banner, which is where the driver reports its
    ceiling. This is a different number from ``torch.version.cuda`` -- the build
    torch was compiled against -- and it is the one that explains a `CUDA
    Available: False` on a machine that plainly has a GPU.

    Returns None when there is no usable ``nvidia-smi``, which is the normal
    answer on a CPU-only or non-NVIDIA host rather than an error.
    """
    try:
        completed = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=30.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    match = re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', completed.stdout)
    return None if match is None else match.group(1)


def check_nvrtc_builtins(torch_module) -> None:
    cuda_ver = torch_module.version.cuda
    if not cuda_ver:
        return

    soname = f'libnvrtc-builtins.so.{cuda_ver}'

    try:
        ctypes.CDLL(soname)
        print(f'NVRTC Builtins ({soname}): loadable')
        return
    except OSError:
        pass

    print(f'NVRTC Builtins ({soname}): NOT LOADABLE by dynamic linker')

    site_packages = Path(torch_module.__file__).parent.parent
    bundled = sorted(site_packages.glob(f'nvidia/*/lib/{soname}'))

    if bundled:
        lib_dir = bundled[0].parent
        print(f'  Bundled copy found at: {lib_dir}')
        print('  Fix: prepend this directory to LD_LIBRARY_PATH before launching ptychodus:')
        print(f'    export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"')
        current = os.environ.get('LD_LIBRARY_PATH', '')
        if str(lib_dir) not in current.split(':'):
            print(f'  (Current LD_LIBRARY_PATH does not contain {lib_dir})')
    else:
        print(f'  No bundled copy found under {site_packages}/nvidia/')
        print('  Fix: install a CUDA runtime matching the version above, or reinstall torch.')

    print('  Symptom if unfixed: ptychopinn_torch DDP (ddp_spawn) training fails with')
    print(f'    nvrtc: error: failed to open {soname}')


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='ptychodus-system-check',
        description='Report the interpreter, GPU driver and PyTorch build ptychodus sees.',
    )
    parser.add_argument(
        '--require-cuda',
        action='store_true',
        help='Exit non-zero if an NVIDIA GPU is present but PyTorch cannot use it. '
        'Without a GPU this has no effect, so it is safe in CI and on CPU hosts.',
    )
    args = parser.parse_args()

    for key, value in platform.uname()._asdict().items():
        print(f'{key.title()}: {value}')

    driver_cuda_version = read_driver_cuda_version()

    if driver_cuda_version is None:
        print('NVIDIA Driver: not detected')
    else:
        print(f'NVIDIA Driver CUDA Ceiling: {driver_cuda_version}')

    is_cuda_available = False

    try:
        import torch
    except ImportError:
        print('PyTorch is not installed.')
    else:
        is_cuda_available = torch.cuda.is_available()
        print(f'PyTorch Version: {torch.__version__}')
        print(f'CUDA Available: {is_cuda_available}')
        print(f'CUDA Version: {torch.version.cuda}')

        if is_cuda_available:
            check_nvrtc_builtins(torch)
        elif driver_cuda_version is not None and torch.version.cuda:
            print(
                f'  PyTorch is built for CUDA {torch.version.cuda}, but the driver supports '
                f'no higher than {driver_cuda_version}.'
            )
            print('  A CUDA major version bump is not covered by minor version compatibility.')
            print('  Fix: install the torch build matching the driver, e.g.')
            print('    uv sync --extra ptychi-cuda128')

    if args.require_cuda and driver_cuda_version is not None and not is_cuda_available:
        print(
            'ERROR: an NVIDIA GPU is present but PyTorch cannot use it.',
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
