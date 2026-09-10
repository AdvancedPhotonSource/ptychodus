from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptychodus.model.processing._subprocess_protocol import ChildError
from ptychodus.model.processing.subprocess_reconstructor import SubprocessReconstructor


_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

FIXTURES_MODULE = 'subprocess_child_fixtures'


def test_reconstruct_passes_configured_gridsize_to_spawned_child() -> None:
    payload = SimpleNamespace(
        inference_config=SimpleNamespace(model=SimpleNamespace(N=64, gridsize=2)),
        model_bundle_dir=Path('/unused'),
        n_nearest_neighbors=7,
        n_samples=3,
        reconstruct_input=object(),
    )
    adapter = SubprocessReconstructor(
        name='PtychoPINN grouping probe',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:capture_ptychopinn_grouping_call',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=lambda _parameters, _loaded: payload,
    )

    with pytest.raises(ChildError) as excinfo:
        list(adapter.reconstruct(SimpleNamespace()))  # type: ignore[arg-type]

    assert excinfo.value.child_exception_type == 'GroupingCallCapturedError'
    assert excinfo.value.child_exception is not None
    assert excinfo.value.child_exception.args == ({'N': 64, 'K': 7, 'nsamples': 3, 'gridsize': 2},)
