#!/bin/bash

# NOTE: Replace OLCF_ALLOCATION below with your own OLCF project allocation
# before running this example.
OLCF_ALLOCATION="OLCF_ALLOCATION"

echo "JWT Decode"
jq -R 'split(".") | .[0],.[1] | @base64d | fromjson' <<< "${S3M_TOKEN}"

# Login nodes are shared, and anything on the curl command line is visible to every user on
# the node via `ps`. Pass the credential on stdin with --config instead of -H.
auth_config() {
    printf 'header = "Authorization: Bearer %s"\n' "${S3M_TOKEN}"
}

echo "Token Introspection"
auth_config | curl --config - https://s3m.olcf.ornl.gov/olcf/v1/token/ctls/introspect -s | jq

echo "Job Submission"
auth_config | curl -i -s --config - \
    -H "Content-Type: application/json" \
    -X POST \
    -d "{
\"executable\": \"/bin/bash\",
\"arguments\": [\"-lc\", \"echo hello\\nhostname\"],
\"name\": \"s3m-echo-test\",
\"directory\": \"/gpfs/wolf2/olcf/${OLCF_ALLOCATION}/proj-shared\",
\"environment\": {
\"DUMMY\": \"1\"
},
\"resources\": {
\"node_count\": 1,
\"process_count\": 1,
\"cpu_cores_per_process\": 1
},
\"attributes\": {
\"account\": \"${OLCF_ALLOCATION}\",
\"queue_name\": \"batch\",
\"duration\": 300
}
}" "https://amsc-open.s3m.olcf.ornl.gov/api/v1/compute/job/odo"
