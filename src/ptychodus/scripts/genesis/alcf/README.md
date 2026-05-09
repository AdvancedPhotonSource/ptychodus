Get Tokens
==========

Get ALCF IRI API (https://api.alcf.anl.gov) access tokens using instructions and scripts here:

https://github.com/argonne-lcf/alcf-facility-api-token

The `globus_access_token.py` script helps you authenticate with Globus and obtain access tokens for filesystem operations. Authenticate with your ALCF account:
```bash
python globus_access_token.py authenticate
```

You can view your access token with:
```bash
python globus_access_token.py get_access_token
```

Access Polaris
==============

ssh USERNAME@polaris.alcf.anl.gov

Use passcode only.
