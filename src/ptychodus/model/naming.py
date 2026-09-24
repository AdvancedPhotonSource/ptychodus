from __future__ import annotations
from collections.abc import Container
import re

__all__ = [
    'create_unique_name',
    'split_name_counter',
]

_COUNTER_RE = re.compile(r'^(?P<stem>.*)-(?P<counter>\d+)$')

DEFAULT_NAME = 'Unnamed'


def split_name_counter(name: str) -> tuple[str, int]:
    """Split a trailing "-<digits>" collision counter off a name.

    Returns the stem and the counter; the counter is zero when the name carries no
    such suffix. This is the inverse of the "<stem>-<n>" form that create_unique_name
    produces, so that resolving a collision on an already-suffixed name increments the
    counter instead of nesting another suffix.
    """
    match = _COUNTER_RE.match(name)

    if match is None:
        return name, 0

    return match.group('stem'), int(match.group('counter'))


def create_unique_name(candidate_name: str, reserved_names: Container[str]) -> str:
    """Return the candidate name, or the first free "<stem>-<n>" variant of it.

    An empty candidate becomes DEFAULT_NAME before the stem is taken, so a collision on
    the fallback yields "Unnamed-1" rather than a bare "-1".
    """
    candidate = candidate_name or DEFAULT_NAME

    if candidate not in reserved_names:
        return candidate

    stem, counter = split_name_counter(candidate)

    while True:
        counter += 1
        name = f'{stem}-{counter}'

        if name not in reserved_names:
            return name
