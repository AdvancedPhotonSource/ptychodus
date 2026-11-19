"""Tests for the processing-run vocabulary and record."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ptychodus.api.processing import RunRecord, RunState, create_run_record


def test_run_states_are_the_lowercase_lifecycle() -> None:
    assert [state.value for state in RunState] == [
        'pending',
        'running',
        'succeeded',
        'failed',
        'cancelled',
    ]
    assert {state for state in RunState if state.is_terminal} == {
        RunState.SUCCEEDED,
        RunState.FAILED,
        RunState.CANCELLED,
    }


def test_a_run_record_round_trips_and_forbids_undeclared_fields() -> None:
    record = create_run_record(
        RunState.RUNNING, command='ptychodus-reconstruct', options='reconstruct_options.yaml'
    )

    assert RunRecord.model_validate_json(record.model_dump_json()) == record
    assert record.user and record.ptychodus_version

    with pytest.raises(ValidationError):
        RunRecord.model_validate({**record.model_dump(), 'epoch': 3})


def test_revising_a_record_refreshes_its_time_and_validates() -> None:
    record = create_run_record(RunState.RUNNING, command='ptychodus-enhance-xrf')
    revised = record.revise(progress=3, progress_goal=4)

    assert (revised.progress, revised.progress_goal) == (3, 4)
    assert revised.updated_at >= record.updated_at
    assert record.progress == 0

    with pytest.raises(ValidationError):
        record.revise(state='paused')
