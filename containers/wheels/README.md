# Staged backend wheels

`scripts/podman/build` builds wheels for the backends published on no package index (PtychoPINN, PtychoFM) from their sibling checkouts and stages them here for the duration of a build, removing them again on the way out. The images install every `*.whl` they find here. Nothing else belongs in this directory; it is tracked only so that `COPY wheels/` has a source when nothing is staged, as in a plain `podman build`.
