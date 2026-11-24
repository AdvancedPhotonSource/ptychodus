"""The TypeScript client must mirror the pydantic response models.

`ui/src/api.ts` is a hand-maintained copy of `service/schemas.py`, and nothing links
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
from pydantic import BaseModel

from ptychodus.webapp.service import schemas

_API_TS = Path(__file__).parents[2] / 'src' / 'ptychodus' / 'webapp' / 'ui' / 'src' / 'api.ts'

# Mirrored interface -> the pydantic model it copies. CampaignRead and LineageRead
# are absent on purpose: campaigns and lineage are a webapp-only grouping with no
# GUI counterpart, served to agents and scripts but not browsed.
_MIRRORED: dict[str, type[BaseModel]] = {
    'DiffractionRead': schemas.DiffractionRead,
    'ProductRead': schemas.ProductRead,
    'FluorescenceRead': schemas.FluorescenceRead,
    'ProbeMetricsRead': schemas.ProbeMetricsRead,
    'ObjectMetricsRead': schemas.ObjectMetricsRead,
    'ProbePositionsMetricsRead': schemas.ProbePositionsMetricsRead,
    'ProbeComparisonRead': schemas.ProbeComparisonRead,
    'ObjectComparisonRead': schemas.ObjectComparisonRead,
    'ProbePositionsComparisonRead': schemas.ProbePositionsComparisonRead,
}


def _interface_fields(source: str, name: str) -> set[str]:
    """Field names an `export interface` declares, with those of the interfaces it extends."""
    match = re.search(
        rf'export interface {name}\b(?:\s+extends\s+([\w\s,]+?))?\s*\{{(.*?)\}}\n', source, re.S
    )

    if match is None:
        return set()

    bases, body = match.groups()
    fields = set(re.findall(r'^\s*(\w+)\??:', body, re.M))

    for base in (bases or '').split(','):
        if base.strip():
            fields |= _interface_fields(source, base.strip())

    return fields


@pytest.mark.parametrize('interface', sorted(_MIRRORED))
def test_ts_interface_declares_every_model_field(interface: str) -> None:
    if not _API_TS.is_file():
        pytest.skip(f'{_API_TS} is absent; ui/src ships only in a source checkout')

    declared = _interface_fields(_API_TS.read_text(), interface)
    expected = set(_MIRRORED[interface].model_fields)
    missing = sorted(expected - declared)

    assert not missing, (
        f'{interface} in api.ts is missing {missing}; a schemas.py field the browser cannot reach'
    )
