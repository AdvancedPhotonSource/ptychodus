# NERSC

## Get Tokens

The script (`get_globus_token.py`) and instructions ([get_globus_token.md](get_globus_token.md)) are from <https://github.com/NERSC/iri-api-get-globus-token>.

Run the script and follow instructions to input the auth code:

```sh
python get_globus_token.py
```

Token JSON is saved to `~/.globus/auth_tokens.json`.

## Provided Instructions

Use account "amsc013". qos name for GPU "express_amsc_g" and for CPU "express_amsc".

## Access Perlmutter

```sh
ssh USERNAME@perlmutter.nersc.gov
```

Use password + passcode.
