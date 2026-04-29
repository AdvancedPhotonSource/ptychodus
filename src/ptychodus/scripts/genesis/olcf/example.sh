#!/bin/bash

echo "JWT Decode"
jq -R 'split(".") | .[0],.[1] | @base64d | fromjson' <<< "${S3M_TOKEN}"

echo "Token Introspection"
curl -H "Authorization: Bearer $S3M_TOKEN" https://s3m.olcf.ornl.gov/olcf/v1/token/ctls/introspect -s | jq

echo "Job Submission"
curl -i -s -H "Authorization: Bearer $S3M_TOKEN" \
-H "Content-Type: application/json" \
-X POST \
-d '{
"executable": "/bin/bash",
"arguments": ["-lc", "echo hello\nhostname"],
"name": "s3m-echo-test",
"directory": "/gpfs/wolf2/olcf/csc682/proj-shared",
"environment": {
"DUMMY": "1"
},
"resources": {
"node_count": 1,
"process_count": 1,
"cpu_cores_per_process": 1
},
"attributes": {
"account": “csc682",
"queue_name": "batch",
"duration": 300
}
}' "https://amsc-open.s3m.olcf.ornl.gov/api/v1/compute/job/odo"
