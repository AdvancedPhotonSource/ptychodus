#!/bin/sh

# test token validity
curl -s -H "Authorization: $S3M_TOKEN" https://s3m.olcf.ornl.gov/olcf/v1/token/ctls/introspect | jq

# test resource availability
curl -s -H "Authorization: $S3M_TOKEN" 'https://amsc-open.s3m.olcf.ornl.gov/api/v1/status/resources/odo' | jq
