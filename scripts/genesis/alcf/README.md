# ALCF

## Get Tokens

Get [ALCF IRI API](https://api.alcf.anl.gov) access tokens using the instructions and scripts at <https://github.com/argonne-lcf/alcf-facility-api-token>.

The `globus_access_token.py` script helps you authenticate with Globus and obtain access tokens for filesystem operations. Authenticate with your ALCF account:

```sh
python globus_access_token.py authenticate
```

You can view your access token with:

```sh
python globus_access_token.py get_access_token
```

## Access Polaris

```sh
ssh USERNAME@polaris.alcf.anl.gov
```

Use passcode only.
