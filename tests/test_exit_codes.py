"""The exit statuses are a contract with shells, schedulers and parent processes.

Each value is pinned here because changing one silently breaks every caller that branches
on it, and nothing else in the test suite would notice.
"""

from __future__ import annotations

import subprocess
import sys

from ptychodus.api.exit_codes import ExitCode


def test_values_match_the_conventions_they_follow() -> None:
    assert ExitCode.SUCCESS == 0
    assert ExitCode.FAILURE == 1
    assert ExitCode.USAGE == 2
    assert ExitCode.CANCELLED == 130


def test_cancelled_is_the_shell_signal_convention() -> None:
    import signal

    assert ExitCode.CANCELLED == 128 + signal.SIGINT


def test_sys_exit_writes_the_member_value() -> None:
    """The whole point of IntEnum here: the number reaches the process exit status."""
    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; from ptychodus.api.exit_codes import ExitCode; sys.exit(ExitCode.USAGE)',
        ],
    )
    assert completed.returncode == ExitCode.USAGE
