import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Targeted resource
# resource_id = "7f7d0593-162e-43b9-8476-07d7d137d6ab" # Edith
resource_id = '55c1c993-1124-47f9-b823-514ba3849a9a'  # Polaris

# Build commands (everything the qsub would have, besides the #PBS instructions)
commands = """
echo Start
sleep 10
echo "slept for 10 secs"
sleep 60
echo "using following python executable"
which python
echo End
"""

# Convert commands into a single-line string separated with ";" and escaped quotes
commands = commands.strip()
commands = '; '.join(line.strip() for line in commands.splitlines() if line.strip())
commands = commands.replace('"', '\\"')

# Build input data
data = {
    'executable': '/bin/bash',
    'arguments': ['-c', commands],
    'name': 'TEST',
    'stdout_path': '/home/bcote/qsub',
    'stderr_path': '/home/bcote/qsub',
    'resources': {'memory': 2222, 'node_count': 1},
    'attributes': {
        'duration': 300,
        'queue_name': 'debug',
        'account': 'datascience',
        'custom_attributes': {'filesystems': 'eagle'},
    },
}

# Build headers
headers = {
    'Authorization': f'Bearer {os.getenv("ACCESS_TOKEN", None)}',
    'Content-Type': 'application/json',
}

# Build URL
url = f'{os.getenv("BASE_URL")}/compute/job/{resource_id}'

# Send request to Facility API
response = requests.post(url, json=data, headers=headers)

# Print response
print(response.status_code)
print(json.dumps(response.json(), indent=2))
