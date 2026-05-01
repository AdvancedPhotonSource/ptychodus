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

Ptychodus Installation
======================

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
