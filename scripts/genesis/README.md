# Genesis Facility Scripts

14 April 2025

These scripts are not packaged; run them from a checkout with `ptychodus` installed:

```sh
python scripts/genesis/ptychodus_iri_tokens.py --help       # IRI facility API tokens
python scripts/genesis/ptychodus_transfer_tokens.py --help  # Globus transfer tokens
python scripts/genesis/<facility>/submit_job.py             # per-facility job submission
```

Per-facility setup is documented in [alcf/README.md](alcf/README.md), [nersc/README.md](nersc/README.md), and [olcf/README.md](olcf/README.md).

## AmSC Data Transfer API

The [Demo Data Transfer APIs for AmSC website](https://amsc-data-api.nersc.gov/docs) links to a [script (generate_token.py)](https://gist.github.com/tylern4/924b19e58d75046e593e0db2d87f6c5c) that gets a Globus bearer token for testing:

```sh
python generate_token.py login \
    --mapped-collections 05d2c76a-e867-4f67-aa57-76edeb0beda0 \
    --mapped-collections 9d6d994a-6d04-11e5-ba46-22000b92c6ec
```

## Ptychodus Installation

Install uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python

```sh
uv python install 3.11
```

Install Ptychodus

```sh
uv tool install ptychodus[globus,gui,ptychi]
```
