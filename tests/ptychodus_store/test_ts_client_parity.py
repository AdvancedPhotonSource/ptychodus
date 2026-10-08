"""The TypeScript client must mirror the pydantic response models.

`ui/src/api.ts` is a hand-maintained copy of `routers/schemas.py`, and nothing links
the two: they are different languages in different directories, and the TypeScript
compiles happily against a stale interface. That is exactly how it drifted -- three
commits added fields to the schemas and the client was never updated, so metadata the
service returned could not reach the browser.

Checked by name only. Types are left to tsc and to whoever reads the diff; what matters
here is that no field is silently unreachable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ptychodus_store.routers import schemas

_API_TS = Path(__file__).parents[2] / 'src' / 'ptychodus_store' / 'ui' / 'src' / 'api.ts'

# Mirrored interface -> the pydantic model it copies. CampaignRead and LineageRead
# are absent on purpose: campaigns and lineage are a store-only grouping with no
# desktop counterpart, served to agents and scripts but not browsed.
_MIRRORED = {
    'DiffractionRead': schemas.DiffractionRead,
    'ProductRead': schemas.ProductRead,
    'FluorescenceRead': schemas.FluorescenceRead,
}


def _interface_fields(source: str, name: str) -> set[str]:
    """Field names declared on one `export interface`, excluding anything inherited."""
    match = re.search(rf'export interface {name}\b[^{{]*\{{(.*?)\n\}}', source, re.S)

    if match is None:
        return set()

    return set(re.findall(r'^\s*(\w+)\??:', match.group(1), re.M))


@pytest.mark.parametrize('interface', sorted(_MIRRORED))
def test_ts_interface_declares_every_model_field(interface: str) -> None:
    if not _API_TS.is_file():
        pytest.skip(f'{_API_TS} is absent; ui/src ships only in a source checkout')

    source = _API_TS.read_text()
    # RowBase holds the bookkeeping every row carries, so a mirror may inherit it.
    declared = _interface_fields(source, interface) | _interface_fields(source, 'RowBase')
    expected = set(_MIRRORED[interface].model_fields)
    missing = sorted(expected - declared)

    assert not missing, (
        f'{interface} in api.ts is missing {missing}; a schemas.py field the browser cannot reach'
    )
