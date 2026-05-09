#!/usr/bin/env python

import ctypes
import os
import platform
import sys
from pathlib import Path


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

        if torch.cuda.is_available():
            check_nvrtc_builtins(torch)

    return 0


if __name__ == '__main__':
    sys.exit(main())
