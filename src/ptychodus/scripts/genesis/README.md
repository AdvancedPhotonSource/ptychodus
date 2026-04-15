14 April 2025

AmSC Data Transfer API
======================

The Demo Data Transfer APIs for AmSC website (https://amsc-data-api.nersc.gov/docs)
links to a script (generate_token.py) that gets a Globus bearer token for testing
(https://gist.github.com/tylern4/924b19e58d75046e593e0db2d87f6c5c):

```bash
python generate_token.py login \
    --mapped-collections 9032dd3a-e841-4687-a163-2720da731b5b \
    --mapped-collections 05d2c76a-e867-4f67-aa57-76edeb0beda0 \
    --mapped-collections 3caddd4a-bb35-4c3d-9101-d9a0ad7f3a30 \
    --mapped-collections 9d6d994a-6d04-11e5-ba46-22000b92c6ec
```

AmSC IRI Compute API
====================

ALCF
----

Get ALCF IRI API (https://api.alcf.anl.gov) access tokens using instructions and scripts here:

https://github.com/argonne-lcf/alcf-facility-api-token

NERSC
-----

The script (get_globus_token.py) and instructions (get_globus_token.md) are from here:

https://gist.github.com/dingp/347b99840d9b3ff2553ee53f47f0bf07

Run the script and follow instructions to input the auth code:
```bash
python get_globus_token.py
```
Token JSON is saved to `~/.globus/auth_tokens.json`.

It is possible that this procedures will eventually be standardized here:
https://github.com/doe-iri/iri-facility-api-examples


Ptychodus Installation on ALCF/NERSC
====================================

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python
```bash
uv python install 3.11
```

Install Ptychodus

```bash
uv tool install ptychodus[globus,gui,ptychi]
```
