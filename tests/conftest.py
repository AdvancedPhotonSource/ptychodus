"""Skip optional-extra test packages when their dependencies are absent."""

from importlib.util import find_spec

# tests/ptychodus_store/ needs the "store" extra plus pytest-asyncio. Its own conftest
# imports sqlalchemy eagerly, so the directory has to be dropped before pytest descends
# into it -- pytest.importorskip in a conftest is reported as an error, not a skip.
_STORE_TEST_DEPS = (
    'aiosqlite',
    'fastapi',
    'fastmcp',
    'pydantic_settings',
    'pytest_asyncio',
    'sqlalchemy',
)

collect_ignore = []

if any(find_spec(name) is None for name in _STORE_TEST_DEPS):
    collect_ignore.append('ptychodus_store')

if find_spec('PyQt5') is None:
    collect_ignore.append('view')
