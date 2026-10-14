from ptychodus.model.naming import create_unique_name, split_name_counter


def test_split_name_counter_without_suffix() -> None:
    assert split_name_counter('foo') == ('foo', 0)


def test_split_name_counter_with_suffix() -> None:
    assert split_name_counter('foo-1') == ('foo', 1)


def test_split_name_counter_tolerates_leading_zeros() -> None:
    assert split_name_counter('foo-007') == ('foo', 7)


def test_split_name_counter_requires_digits() -> None:
    assert split_name_counter('foo-') == ('foo-', 0)


def test_split_name_counter_takes_only_the_last_counter() -> None:
    # Greedy stem: only the trailing counter is peeled, so a name that legitimately
    # contains a dash-number survives as part of the stem.
    assert split_name_counter('a-1-2') == ('a-1', 2)


def test_create_unique_name_returns_input_when_free() -> None:
    assert create_unique_name('foo', set()) == 'foo'


def test_create_unique_name_keeps_a_free_counter_suffix() -> None:
    assert create_unique_name('foo-7', {'foo'}) == 'foo-7'


def test_create_unique_name_suffixes_collisions() -> None:
    assert create_unique_name('foo', {'foo', 'foo-1'}) == 'foo-2'


def test_create_unique_name_increments_rather_than_nests() -> None:
    assert create_unique_name('foo-1', {'foo', 'foo-1'}) == 'foo-2'


def test_create_unique_name_increments_repeatedly_without_growing() -> None:
    reserved = {'foo'}
    name = 'foo'

    for _ in range(5):
        name = create_unique_name(name, reserved)
        reserved.add(name)

    assert name == 'foo-5'


def test_create_unique_name_maps_empty_to_unnamed() -> None:
    assert create_unique_name('', set()) == 'Unnamed'


def test_create_unique_name_suffixes_the_unnamed_fallback() -> None:
    assert create_unique_name('', {'Unnamed'}) == 'Unnamed-1'
