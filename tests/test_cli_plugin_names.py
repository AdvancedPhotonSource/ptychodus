"""Guard that every plugin a command-line driver names is still registered.

The drivers name plugins by string -- reader simple names in each
``InstrumentProfile``, option defaults, PEAR's instrument table, the batch-LamNI
settings, zone-plate presets -- and ``PluginCatalog.get_strategy_by_name``
resolves them only when a driver runs. A plugin renamed under them therefore
survives every import, every type check and every test that does not run a
reconstruction -- it surfaces as a ``LookupError`` on the first real run at the
beamline, and never in CI.

The names are read from the source rather than imported, because most drivers
import pty-chi at module scope and this must pass on a bare install. Each
name-bearing site is keyed by where it sits -- ``Class.field`` for a field
default or a constructor argument, the option string for an argparse default,
the bare name for a module constant -- and a table maps each key to the
catalogs that resolve it. The sites are discovered rather than enumerated: a key
whose name says it carries a plugin name must be in the table, and every table
key must still be found, so neither a new site nor a stale table entry slips past.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import ast
import re

import pytest

import ptychodus
from ptychodus.api.plugins import PluginCatalog, PluginRegistry

_CLI_ROOT = Path(ptychodus.__file__).parent / 'cli'

# Matched against a key's last component, an option string spelled as its dest.
_NAME_BEARING = re.compile(r'(file_type|_preset|_reader|_readers_by_suffix)$', re.IGNORECASE)

# Every catalog a key's names must resolve in. A key whose default is not a literal
# still belongs here: the guard cannot tell that the name it carries is checked elsewhere.
_CATALOGS: dict[str, tuple[str, ...]] = {
    # argparse options
    '--background-file-type': ('background_file_readers',),
    '--bad-pixels-file-type': ('bad_pixels_file_readers',),
    '--file-type': ('fluorescence_file_readers', 'fluorescence_file_writers'),
    '--flat-field-file-type': ('flat_field_file_readers',),
    '--fzp-preset': ('fresnel_zone_plates',),
    '--probe-file-type': ('probe_file_readers',),
    # module constants
    'DIFFRACTION_READER': ('diffraction_file_readers',),
    'POSITION_READER': ('probe_position_file_readers',),
    '_DEFAULT_FILE_TYPE': ('fluorescence_file_readers', 'fluorescence_file_writers'),
    # the standard drivers' profiles
    'InstrumentProfile.default_bad_pixels_reader': ('bad_pixels_file_readers',),
    'InstrumentProfile.default_fzp_preset': ('fresnel_zone_plates',),
    'InstrumentProfile.diffraction_reader': ('diffraction_file_readers',),
    'InstrumentProfile.diffraction_readers_by_suffix': ('diffraction_file_readers',),
    'InstrumentProfile.position_reader': ('probe_position_file_readers',),
    # PEAR's parameter file and instrument table; the table's fields carry no telling name
    'PearParameters.diffraction_file_type': ('diffraction_file_readers',),
    'PearParameters.fzp_preset': ('fresnel_zone_plates',),
    'PearParameters.init_object_file_type': ('object_file_readers',),
    'PearParameters.init_probe_file_type': ('probe_file_readers',),
    'PearParameters.position_file_type': ('probe_position_file_readers',),
    '_Readers.diffraction': ('diffraction_file_readers',),
    '_Readers.positions': ('probe_position_file_readers',),
    # the batch-LamNI settings
    'Diffraction.bad_pixels_file_type': ('bad_pixels_file_readers',),
    'ObjectFromFile.file_type': ('object_file_readers',),
    'ProbeFromFile.file_type': ('probe_file_readers',),
    'SimulatedProbe.fzp_preset': ('fresnel_zone_plates',),
    # ptychodus-stream's configuration; one source type serves every detector map
    'PixelMapSource.file_type': (
        'bad_pixels_file_readers',
        'background_file_readers',
        'flat_field_file_readers',
    ),
    # calls into the model
    'save_product.file_type': ('product_file_writers',),
}


@dataclass(frozen=True)
class _Site:
    """One place in cli/ that names plugins, and the catalogs they must resolve in."""

    path: Path
    lineno: int
    key: str
    names: tuple[str, ...]
    catalogs: tuple[str, ...]


def _is_name_bearing(key: str) -> bool:
    return bool(_NAME_BEARING.search(key.rsplit('.', 1)[-1].lstrip('-').replace('-', '_')))


def _literal_names(node: ast.expr | None) -> tuple[str, ...]:
    """The plugin names *node* spells literally; an empty string names none."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,) if node.value else ()

    if isinstance(node, ast.Dict):
        return tuple(name for value in node.values for name in _literal_names(value))

    return ()


def _callee_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr

    if isinstance(node.func, ast.Name):
        return node.func.id

    return ''


def _keyed_sites(path: Path, tree: ast.Module, fields: dict[str, list[str]]) -> Iterator[_Site]:
    """Every site in *tree* with a key, whether or not the key bears a plugin name."""

    def site(lineno: int, key: str, node: ast.expr | None) -> _Site:
        return _Site(path, lineno, key, _literal_names(node), _CATALOGS.get(key, ()))

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    yield site(statement.lineno, target.id, statement.value)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            yield site(statement.lineno, statement.target.id, statement.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    key = f'{node.name}.{statement.target.id}'
                    yield site(statement.lineno, key, statement.value)
        elif isinstance(node, ast.Call):
            callee = _callee_name(node)

            if callee == 'add_argument':
                options = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]

                if options:
                    option = next((o for o in options if o.startswith('--')), options[0])
                    default = next((k.value for k in node.keywords if k.arg == 'default'), None)
                    yield site(node.lineno, option, default)
            elif callee in ('get_strategy_by_name', 'find_plugin'):
                # A literal looked up directly names its catalog in the attribute chain.
                if node.args and isinstance(node.func, ast.Attribute):
                    catalog = node.func.value
                    names = _literal_names(node.args[0])

                    if names and isinstance(catalog, ast.Attribute):
                        yield _Site(path, node.lineno, callee, names, (catalog.attr,))
            else:
                for field_name, arg in zip(fields.get(callee, ()), node.args):
                    yield site(arg.lineno, f'{callee}.{field_name}', arg)

                for keyword in node.keywords:
                    if keyword.arg is not None:
                        yield site(keyword.value.lineno, f'{callee}.{keyword.arg}', keyword.value)


def _discover_sites() -> list[_Site]:
    """Return every site in cli/ that names plugins or whose key says it should."""
    trees = {
        path: ast.parse(path.read_text(), filename=str(path))
        for path in sorted(_CLI_ROOT.glob('*.py'))
    }
    # Field names in declaration order, so a positional constructor argument has a key.
    fields: dict[str, list[str]] = dict()

    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                fields[node.name] = [
                    statement.target.id
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                ]

    return [
        site
        for path, tree in trees.items()
        for site in _keyed_sites(path, tree, fields)
        if site.catalogs or _is_name_bearing(site.key)
    ]


_SITES = _discover_sites()
_NAMED = [
    pytest.param(site, name, catalog, id=f'{site.path.name}:{site.lineno}:{name}:{catalog}')
    for site in _SITES
    for name in site.names
    for catalog in site.catalogs
]


@pytest.fixture(scope='module')
def registry() -> PluginRegistry:
    return PluginRegistry.load_plugins()


def test_sites_are_found() -> None:
    assert _NAMED


@pytest.mark.parametrize(('site', 'name', 'catalog'), _NAMED)
def test_every_named_plugin_is_registered(
    registry: PluginRegistry, site: _Site, name: str, catalog: str
) -> None:
    plugins: PluginCatalog[object] = getattr(registry, catalog)
    assert plugins.find_plugin(name) is not None, (
        f'{site.path.name}:{site.lineno}: {site.key} names "{name}", which {catalog} '
        f'does not register. Registered: {plugins.stringify_plugin_names()}.'
    )


def test_every_name_bearing_site_is_mapped() -> None:
    unmapped = sorted(
        f'{site.path.name}:{site.lineno}: {site.key}' for site in _SITES if not site.catalogs
    )
    assert not unmapped, f'Add these to _CATALOGS: {unmapped}'


def test_every_mapped_key_is_still_found() -> None:
    stale = sorted(set(_CATALOGS) - {site.key for site in _SITES})
    assert not stale, f'Nothing in cli/ is keyed {stale} any more; update _CATALOGS.'


def test_every_mapped_catalog_exists(registry: PluginRegistry) -> None:
    for key, catalogs in _CATALOGS.items():
        for catalog in catalogs:
            assert isinstance(getattr(registry, catalog, None), PluginCatalog), (key, catalog)
