from unittest.mock import MagicMock

from ptychodus.model.processing.api import _build_output_product_name
from ptychodus.model.product.repository import ProductRepository

ALGORITHM = 'ptychi_dm'


def _make_named_item(name: str) -> MagicMock:
    item = MagicMock()
    item.get_name.return_value = name
    return item


def test_output_name_tags_the_input_with_the_algorithm() -> None:
    assert _build_output_product_name('scan', ALGORITHM, '') == 'scan_ptychi_dm'


def test_output_name_does_not_repeat_the_algorithm_tag() -> None:
    # Continuing a reconstruction: the repository resolves the collision with a counter
    # rather than the name growing a second copy of the tag.
    assert _build_output_product_name('scan_ptychi_dm', ALGORITHM, '') == 'scan_ptychi_dm'


def test_output_name_ignores_an_existing_counter_suffix() -> None:
    assert _build_output_product_name('scan_ptychi_dm-2', ALGORITHM, '') == 'scan_ptychi_dm'


def test_output_name_appends_the_split_suffix() -> None:
    assert _build_output_product_name('scan', ALGORITHM, 'odd') == 'scan_ptychi_dm_odd'


def test_output_name_continues_a_split_half() -> None:
    assert (
        _build_output_product_name('scan_ptychi_dm_odd-2', ALGORITHM, 'odd') == 'scan_ptychi_dm_odd'
    )


def test_output_name_records_a_change_of_algorithm() -> None:
    # Switching reconstructors keeps the provenance of the earlier one.
    assert _build_output_product_name('scan_ptychi_dm', 'ptychi_pie', '') == (
        'scan_ptychi_dm_ptychi_pie'
    )


def test_continuing_a_reconstruction_only_advances_the_counter() -> None:
    """The user-visible chain: name building composed with collision resolution."""
    repository = ProductRepository()
    name = 'scan'
    repository.insert_product(_make_named_item(name))
    names = []

    for _ in range(3):
        name = repository.create_unique_name(_build_output_product_name(name, ALGORITHM, ''))
        repository.insert_product(_make_named_item(name))
        names.append(name)

    assert names == ['scan_ptychi_dm', 'scan_ptychi_dm-1', 'scan_ptychi_dm-2']
