"""Process exit statuses for ptychodus commands."""

from __future__ import annotations

from enum import IntEnum

__all__ = ['ExitCode']


class ExitCode(IntEnum):
    """Process exit status for a ptychodus command.

    An :class:`~enum.IntEnum` because these values leave the process: a shell, a job
    scheduler and a parent process all read the number, so a member has to *be* the number
    :func:`sys.exit` writes rather than merely carry one.
    """

    SUCCESS = 0
    """The command did what it was asked to do. ``EXIT_SUCCESS`` in ``<stdlib.h>``."""

    FAILURE = 1
    """The work was attempted and did not finish. ``EXIT_FAILURE`` in ``<stdlib.h>``, and
    also what CPython exits with when an exception escapes, so a handled failure and an
    uncaught traceback agree."""

    USAGE = 2
    """The arguments were unusable and no work was attempted. The C standard library names
    no such status; 2 is the GNU convention, and is what :mod:`argparse` already exits with
    on its own errors. Deliberately not ``sysexits.h`` ``EX_USAGE``, which is 64 and would
    disagree with argparse."""

    CANCELLED = 130
    """The run stopped because it was asked to, by the shell convention of 128 + SIGINT. A
    SIGTERM cancellation reports this too rather than 143: what a caller acts on is that the
    run was interrupted and left a partial result, not which signal carried the request."""
