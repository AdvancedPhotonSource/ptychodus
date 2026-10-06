"""The driver-ceiling probe and the --require-cuda gate in ``ptychodus-system-check``.

Tested here rather than through a live GPU because the interesting cases are the ones
this machine cannot produce on demand: no NVIDIA driver at all, and a driver that is
present but older than the torch build.
"""

from __future__ import annotations

import subprocess

import pytest

from ptychodus.cli import system_check

# The banner nvidia-smi prints; the ceiling is on its third line.
_NVIDIA_SMI_BANNER = """Tue Oct  6 16:17:57 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
"""


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=['nvidia-smi'], returncode=returncode, stdout=stdout)


def test_driver_ceiling_is_read_from_the_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _completed(_NVIDIA_SMI_BANNER))

    assert system_check.read_driver_cuda_version() == '12.8'


def test_a_missing_nvidia_smi_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only host is a normal answer, not a failure."""

    def raise_not_found(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError('nvidia-smi')

    monkeypatch.setattr(subprocess, 'run', raise_not_found)

    assert system_check.read_driver_cuda_version() is None


def test_a_failing_nvidia_smi_reports_no_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _completed('', returncode=9))

    assert system_check.read_driver_cuda_version() is None


def test_a_banner_without_a_cuda_version_reports_no_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _completed('NVIDIA-SMI 570.195.03\n'))

    assert system_check.read_driver_cuda_version() is None


def test_require_cuda_fails_when_a_driver_is_present_but_torch_cannot_use_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this flag exists for: a silent fall back to CPU-speed reconstruction."""
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(system_check, 'read_driver_cuda_version', lambda: '12.8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('sys.argv', ['ptychodus-system-check', '--require-cuda'])

    assert system_check.main() == 1


def test_require_cuda_passes_when_torch_can_use_the_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(system_check, 'read_driver_cuda_version', lambda: '12.8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(system_check, 'check_nvrtc_builtins', lambda _torch: None)
    monkeypatch.setattr('sys.argv', ['ptychodus-system-check', '--require-cuda'])

    assert system_check.main() == 0


def test_require_cuda_is_inert_without_a_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe in CI and on CPU hosts: no GPU means nothing to require."""
    monkeypatch.setattr(system_check, 'read_driver_cuda_version', lambda: None)
    monkeypatch.setattr('sys.argv', ['ptychodus-system-check', '--require-cuda'])

    assert system_check.main() == 0


def test_the_default_invocation_never_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag this stays a report, whatever it finds."""
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(system_check, 'read_driver_cuda_version', lambda: '12.8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('sys.argv', ['ptychodus-system-check'])

    assert system_check.main() == 0
