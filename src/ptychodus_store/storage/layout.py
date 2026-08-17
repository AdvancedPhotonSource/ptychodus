from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ptychodus_store.storage.manifest import MANIFEST_FILENAME, ResourceKind

_KIND_VALUES: frozenset[str] = frozenset(k.value for k in ResourceKind)


class LayoutError(ValueError):
    """Raised when a path cannot be mapped to the store layout."""


@dataclass(frozen=True)
class ResourceLocation:
    """A resolved <store_root>/<kind>/<uuid>/ folder."""

    kind: str
    uuid: UUID
    folder: Path

    @property
    def manifest_path(self) -> Path:
        return self.folder / MANIFEST_FILENAME


class StoreLayout:
    """Resolves and validates paths within the store root."""

    def __init__(self, store_root: Path) -> None:
        self._root = store_root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def kind_dir(self, kind: str) -> Path:
        if kind not in _KIND_VALUES:
            raise LayoutError(f'unknown kind {kind!r}')
        return self._root / kind

    def resource_folder(self, kind: str, uuid: UUID) -> Path:
        return self.kind_dir(kind) / str(uuid)

    def manifest_path(self, kind: str, uuid: UUID) -> Path:
        return self.resource_folder(kind, uuid) / MANIFEST_FILENAME

    def ensure_kind_dirs(self) -> None:
        """Create the four top-level kind directories if they do not exist."""
        for kind in ResourceKind:
            (self._root / kind).mkdir(parents=True, exist_ok=True)

    def parse_manifest_path(self, manifest_path: Path) -> ResourceLocation:
        """Map a `manifest.json` path to its (kind, uuid, folder).

        Expected shape: `<store_root>/<kind>/<uuid>/manifest.json`.
        """
        path = manifest_path.resolve()
        if path.name != MANIFEST_FILENAME:
            raise LayoutError(f'not a manifest file: {manifest_path}')

        folder = path.parent
        kind_dir = folder.parent

        try:
            kind_dir.relative_to(self._root)
        except ValueError as exc:
            raise LayoutError(f'{manifest_path} is outside store root {self._root}') from exc

        if kind_dir.parent != self._root:
            raise LayoutError(
                f'{manifest_path} is not at the expected depth '
                f'<store_root>/<kind>/<uuid>/manifest.json'
            )

        kind = kind_dir.name
        if kind not in _KIND_VALUES:
            raise LayoutError(f'{manifest_path} sits under unknown kind {kind!r}')

        try:
            uuid = UUID(folder.name)
        except ValueError as exc:
            raise LayoutError(
                f'{manifest_path} parent folder name {folder.name!r} is not a UUID'
            ) from exc

        return ResourceLocation(kind=kind, uuid=uuid, folder=folder)

    def iter_manifest_paths(self) -> list[Path]:
        """Return all `manifest.json` paths under the store, campaigns first."""
        paths: list[Path] = []
        for kind in (
            ResourceKind.CAMPAIGN,
            ResourceKind.DIFFRACTION,
            ResourceKind.PRODUCT,
            ResourceKind.FLUORESCENCE,
        ):
            kind_dir = self._root / kind
            if not kind_dir.is_dir():
                continue
            for child in sorted(kind_dir.iterdir()):
                if not child.is_dir():
                    continue
                m = child / MANIFEST_FILENAME
                if m.is_file():
                    paths.append(m)
        return paths
