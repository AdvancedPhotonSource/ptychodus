"""Processing runs: the work that turns resources into new ones, and how each run went.

A processing run -- a reconstruction, a fluorescence enhancement, a training -- moves
through one lifecycle whatever produced it. :class:`RunState` names where a run is in that
lifecycle, in the vocabulary job schedulers use, and :class:`RunRecord` is what a result
keeps about the run that made it: its state and why, how far it got, and the command,
software and options behind it.

The module is named for the activity rather than its verb, because "process" already
means an operating-system process throughout ptychodus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any
import getpass

from pydantic import BaseModel, ConfigDict

__all__ = [
    'RunRecord',
    'RunState',
    'create_run_record',
]


class RunState(StrEnum):
    """Where a processing run is in its lifecycle.

    ``pending`` has not started, ``running`` is producing results, and the other three are
    terminal: once a run reaches one of them its record no longer changes. A finer account
    of a state -- why a run is pending, or what made it fail -- goes in
    :attr:`RunRecord.reason`.
    """

    PENDING = 'pending'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

    @property
    def is_terminal(self) -> bool:
        return self in (RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED)


class RunRecord(BaseModel):
    """What a result keeps about the run that produced it.

    ``progress`` counts towards ``progress_goal`` in whatever unit the run steps through --
    epochs for a reconstruction, elements for an enhancement -- and a goal of zero means
    unknown. ``options`` names the options document, in the result's folder, that the run
    was configured by; it is ``None`` for a run configured by no such document.
    ``updated_at`` is refreshed whenever the record changes, so a ``running`` record that
    has stopped advancing can be told from one that is still at work.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    state: RunState
    reason: str | None = None
    progress: int = 0
    progress_goal: int = 0
    command: str
    ptychodus_version: str
    user: str
    options: str | None = None
    updated_at: datetime

    def revise(self, **changes: Any) -> RunRecord:
        """This record with `changes` applied and validated, and ``updated_at`` set to now."""
        fields = {**self.model_dump(), **changes, 'updated_at': datetime.now(timezone.utc)}
        return RunRecord.model_validate(fields)


def _ptychodus_version() -> str:
    try:
        return version('ptychodus')
    except PackageNotFoundError:
        return 'unknown'


def create_run_record(
    state: RunState,
    *,
    command: str,
    options: str | None = None,
    progress: int = 0,
    progress_goal: int = 0,
    reason: str | None = None,
) -> RunRecord:
    """A record for a run of `command`, made now by this process's user and software."""
    return RunRecord(
        state=state,
        reason=reason,
        progress=progress,
        progress_goal=progress_goal,
        command=command,
        ptychodus_version=_ptychodus_version(),
        user=getpass.getuser(),
        options=options,
        updated_at=datetime.now(timezone.utc),
    )
