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

from ptychodus.api.devices import CPU, Device, visibility_environment


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
        """Dropped, not ignored: the family was named, so it is still masked."""
        environ = {'CUDA_VISIBLE_DEVICES': '1'}
        assert visibility_environment(['cuda:4'], environ=environ) == {'CUDA_VISIBLE_DEVICES': ''}
        assert 'cuda:4' in caplog.text

    def test_a_dropped_device_does_not_take_its_siblings_with_it(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '6,7'}
        assert visibility_environment(['cuda:0', 'cuda:9'], environ=environ) == {
            'CUDA_VISIBLE_DEVICES': '6'
        }

    def test_a_cpu_alone_is_not_an_empty_selection(self) -> None:
        """Nothing hides a CPU, but choosing it has to hide everything else."""
        assert visibility_environment(['cpu'], ['cpu', 'cuda:0'], environ={}) == {
            'CUDA_VISIBLE_DEVICES': ''
        }

    def test_an_unparsable_ordinal_is_dropped_rather_than_raising(self) -> None:
        environ = {'CUDA_VISIBLE_DEVICES': '0,1'}
        assert visibility_environment(['cuda:all'], environ=environ) == {'CUDA_VISIBLE_DEVICES': ''}


class TestChoosingTheCpu:
    """Where to run and whether to use an accelerator are one question here."""

    def test_the_probe_always_offers_a_cpu(self) -> None:
        assert CPU.identifier == 'cpu'

    def test_an_unnamed_family_that_exists_is_masked_to_nothing(self) -> None:
        available = ['cpu', 'cuda:0', 'cuda:1']
        assert visibility_environment(['cpu'], available, environ={}) == {
            'CUDA_VISIBLE_DEVICES': ''
        }

    def test_naming_one_gpu_also_hides_a_second_family(self) -> None:
        """'Run on exactly these' means the others are gone, not merely unpreferred."""
        available = ['cpu', 'cuda:0', 'xpu:0']
        assert visibility_environment(['cuda:0'], available, environ={}) == {
            'CUDA_VISIBLE_DEVICES': '0',
            'ZE_AFFINITY_MASK': '',
        }

    def test_a_family_this_machine_does_not_have_is_left_alone(self) -> None:
        """HIP reads CUDA's variable too, so writing one it never probed could
        hide the device it did."""
        assert visibility_environment(['cuda:0'], ['cpu', 'cuda:0'], environ={}) == {
            'CUDA_VISIBLE_DEVICES': '0'
        }

    def test_selecting_everything_still_names_every_device(self) -> None:
        available = ['cpu', 'cuda:0', 'cuda:1']
        assert visibility_environment(available, available, environ={}) == {
            'CUDA_VISIBLE_DEVICES': '0,1'
        }


def test_a_device_carries_a_name_a_person_can_choose_between() -> None:
    device = Device('cuda:0', 'NVIDIA RTX A4500')
    assert device.identifier == 'cuda:0'
    assert device.name == 'NVIDIA RTX A4500'


# --- the probe itself --------------------------------------------------------


def _probed_devices() -> list[Device]:
    """Every probed device, CPU first."""
    from ptychodus.api.devices import probe_devices

    return list(probe_devices())


def test_a_machine_without_torch_still_has_one_device(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """torch is optional, and its absence is an answer rather than a failure.

    None in sys.modules is what makes ``import torch`` raise, so this exercises
    the real branch without needing an install that lacks it.
    """
    from ptychodus.api.devices import enumerate_devices

    monkeypatch.setitem(sys.modules, 'torch', None)

    assert enumerate_devices() == [CPU]
    assert 'fail' not in caplog.text.lower()


def test_the_probe_offers_a_cpu_whatever_the_machine_has() -> None:
    """So a bare install can still say where a run should go."""
    assert _probed_devices()[0] == CPU


def test_a_probe_that_fails_still_reports_the_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken GPU stack is when forcing the CPU matters most."""
    import ptychodus.api.devices as devices

    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError('probe is broken')

    monkeypatch.setattr(devices, 'run_subprocess', explode)

    assert devices.probe_devices() == [CPU]
    assert 'raised in the parent' in caplog.text


def _probed_gpus() -> list[Device]:
    pytest.importorskip('torch')
    gpus = [d for d in _probed_devices() if d.identifier != CPU.identifier]

    if len(gpus) < 2:
        pytest.skip(f'Needs two or more GPUs to verify a restriction; probed {len(gpus)}.')

    return gpus


def test_the_probe_reports_devices_a_child_can_actually_see() -> None:
    """The parent asks in a child, so it never holds the context it is describing."""
    torch = pytest.importorskip('torch')

    devices = _probed_devices()
    assert [d.identifier for d in devices[1:]] == [f'cuda:{n}' for n in range(len(devices) - 1)]
    assert all(d.name for d in devices)
    assert not torch.cuda.is_initialized()


def _cuda_uuids(environment: dict[str, str]) -> list[str]:
    """Ask a child spawned under *environment* which physical devices it can see."""
    from ptychodus.api.reconstruct import ReconstructorMessage
    from ptychodus.api.spawn import run_subprocess

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        with run_subprocess(
            'subprocess_child_fixtures:probe_cuda_uuids', None, environment=environment
        ) as events:
            for event in events:
                if event[0] == ReconstructorMessage.OUTPUT:
                    uuids: list[str] = event[1]
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

    gpus = _probed_gpus()
    available = [d.identifier for d in _probed_devices()]
    everything = _cuda_uuids({})
    assert len(everything) == len(gpus)

    for index, device in enumerate(gpus):
        environment = visibility_environment([device.identifier], available, environ={})
        seen = _cuda_uuids(environment)
        assert seen == [everything[index]], f'{device.identifier} under {environment} saw {seen}.'


def test_choosing_the_cpu_leaves_a_child_no_gpu_at_all() -> None:
    """The whole point of listing the CPU: it is how a run says 'not on a GPU'."""
    from ptychodus.api.devices import visibility_environment

    _probed_gpus()
    available = [d.identifier for d in _probed_devices()]
    environment = visibility_environment([CPU.identifier], available, environ={})

    assert _cuda_uuids(environment) == []
