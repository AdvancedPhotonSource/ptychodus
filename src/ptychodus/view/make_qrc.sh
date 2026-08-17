#!/bin/sh
# Regenerate resources.py from resources.qrc.
# All referenced icons live in ../../ptychodus_store/ui/icons/ (see the README there
# for how to add or update icons).
set -e
pyrcc5 resources.qrc -o resources.py
# pyrcc5 emits camelCase function names that ruff flags as N802 and formatting that
# ruff would rewrite. Apply the same fixups the tracked file uses so lint stays clean.
sed -i \
    -e 's/^def qInitResources():$/def qInitResources() -> None:  # noqa: N802/' \
    -e 's/^def qCleanupResources():$/def qCleanupResources() -> None:  # noqa: N802/' \
    resources.py
ruff format resources.py >/dev/null
