"""Restricting a child to chosen devices, and the renumbering that makes it hard.

The mask a child is given is read against the physical devices, while the
ordinals a probe reports are positions within whatever mask this process was
started under. Composing the two wrongly sends work to a device the scheduler
did not grant -- quite possibly one already running someone else's job -- and
nothing about the result looks wrong from inside.

Needs no GPU: every case here is the string arithmetic, driven against an
explicit environment rather than the real one.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from ptychodus.api.devices import Device, visibility_environment


class TestWithNothingInherited:
    """The probe's ordinals and the physical ones coincide."""

    def test_one_device_masks_to_its_own_ordinal(self) -> None:
        assert visibility_environment(['cuda:1'], environ={}) == {'CUDA_VISIBLE_DEVICES': '1'}

    def test_several_devices_of_one_family_share_a_variable(self) -> None:
        assert visibility_environment(['cuda:0', 'cuda:1'], environ={}) == {
            'CUDA_VISIBLE_DEVICES': '0,1'
        }

    def test_selecting_nothing_restricts_nothing(self) -> None:
        """An empty selection leaves the child seeing whatever the parent does."""
        assert visibility_environment([], environ={}) == {}

    def test_each_family_sets_the_variable_its_own_runtime_reads(self) -> None:
        assert visibility_environment(['rocm:2'], environ={}) == {'HIP_VISIBLE_DEVICES': '2'}
        assert visibility_environment(['xpu:0'], environ={}) == {'ZE_AFFINITY_MASK': '0'}

    def test_two_families_at_once_each_get_their_own_variable(self) -> None:
        assert visibility_environment(['cuda:0', 'xpu:1'], environ={}) == {
            'CUDA_VISIBLE_DEVICES': '0',
            'ZE_AFFINITY_MASK': '1',
        }


class TestWithAMaskInherited:
    """The probe counts within the inherited mask; the child's value does not."""

    def test_the_first_visible_device_is_not_physical_device_zero(self) -> None:
        """The trap: a bare 0 here would name a GPU this process was not given."""
        assert visibility_environment(['cuda:0'], environ={'CUDA_VISIBLE_DEVICES': '1'}) == {
            'CUDA_VISIBLE_DEVICES': '1'
        }

    def test_ordinals_index_back_into_the_inherited_list(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '2,3,5'}
        assert visibility_environment(['cuda:0', 'cuda:2'], environ=environ) == {
            'CUDA_VISIBLE_DEVICES': '2,5'
        }

    def test_a_uuid_mask_composes_the_same_way(self) -> None:
        """Entries are copied, not parsed, so a mask need not be numeric."""
        environ = {'CUDA_VISIBLE_DEVICES': 'GPU-abc,GPU-def'}
        assert visibility_environment(['cuda:1'], environ=environ) == {
            'CUDA_VISIBLE_DEVICES': 'GPU-def'
        }

    def test_whitespace_in_the_inherited_mask_is_tolerated(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': ' 4 , 7 '}
        assert visibility_environment(['cuda:1'], environ=environ) == {'CUDA_VISIBLE_DEVICES': '7'}

    def test_one_family_s_mask_does_not_renumber_another_s(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '3'}
        assert visibility_environment(['xpu:1'], environ=environ) == {'ZE_AFFINITY_MASK': '1'}


class TestASelectionThatNoLongerFits:
    """A selection is persisted and outlives the machine it was made on."""

    def test_an_ordinal_past_the_inherited_mask_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '1'}
        assert visibility_environment(['cuda:4'], environ=environ) == {}
        assert 'cuda:4' in caplog.text

    def test_a_dropped_device_does_not_take_its_siblings_with_it(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '6,7'}
        assert visibility_environment(['cuda:0', 'cuda:9'], environ=environ) == {
            'CUDA_VISIBLE_DEVICES': '6'
        }

    def test_a_family_with_no_mask_of_its_own_contributes_nothing(self) -> None:
        """Nothing hides a CPU, so naming one restricts nothing rather than raising."""
        assert visibility_environment(['cpu'], environ={}) == {}

    def test_an_unparsable_ordinal_is_dropped_rather_than_raising(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '0,1'}
        assert visibility_environment(['cuda:all'], environ=environ) == {}


def test_a_device_carries_a_name_a_person_can_choose_between() -> None:
    device = Device('cuda:0', 'NVIDIA RTX A4500')
    assert device.identifier == 'cuda:0'
    assert device.name == 'NVIDIA RTX A4500'


# --- against real hardware ---------------------------------------------------


def _probed_devices() -> list[Device]:
    from ptychodus.api.devices import probe_devices

    devices = list(probe_devices())

    if len(devices) < 2:
        pytest.skip(f'Needs two or more devices to verify a restriction; probed {len(devices)}.')

    return devices


def test_the_probe_reports_devices_a_child_can_actually_see() -> None:
    """The parent asks in a child, so it never holds the context it is describing."""
    import torch

    devices = _probed_devices()
    assert [d.identifier for d in devices] == [f'cuda:{n}' for n in range(len(devices))]
    assert all(d.name for d in devices)
    assert not torch.cuda.is_initialized()


def _cuda_uuids(environment: dict[str, str]) -> list[str]:
    """Ask a child spawned under *environment* which physical devices it can see."""
    import pickle

    from ptychodus.api.reconstruct import TAG_OUTPUT
    from ptychodus.api.spawn import run_subprocess

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        with run_subprocess(
            'subprocess_child_fixtures:probe_cuda_uuids', None, environment=environment
        ) as events:
            for event in events:
                if event[0] == TAG_OUTPUT:
                    uuids: list[str] = pickle.loads(event[1])
                    return uuids
    finally:
        sys.path.pop(0)

    return []


def test_a_restriction_reaches_the_child_and_not_merely_the_environment() -> None:
    """Checked by UUID, because two cards of one model are not told apart by name.

    Asking the restricted child what it sees is the point: that the variable was
    set says nothing about whether the runtime read it, and it is read once.
    """
    from ptychodus.api.devices import visibility_environment

    devices = _probed_devices()
    everything = _cuda_uuids({})
    assert len(everything) == len(devices)

    for index, device in enumerate(devices):
        environment = visibility_environment([device.identifier], environ={})
        seen = _cuda_uuids(environment)
        assert seen == [everything[index]], f'{device.identifier} under {environment} saw {seen}.'
