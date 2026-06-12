from collections.abc import Mapping
from typing import Any

PTYCHODUS_FLOW_INPUT_SCHEMA: Mapping[str, Any] = {
    'required': ['transfer_input_data', 'compute', 'transfer_output_data'],
    'properties': {
        'transfer_input_data': {
            'type': 'object',
            'required': ['source', 'destination', 'sync_level', 'recursive'],
            'properties': {
                'source': {
                    'type': 'object',
                    'title': 'Select source collection and path',
                    'description': 'The raw data collection and path (e.g., on the instrument).',
                    'format': 'globus-collection',
                    'required': ['id', 'path'],
                    'properties': {
                        'id': {'type': 'string', 'format': 'uuid'},
                        'path': {'type': 'string'},
                    },
                    'additionalProperties': False,
                },
                'destination': {
                    'type': 'object',
                    'title': 'Select destination collection and path',
                    'description': 'The collection and path on the Globus Compute endpoint.',
                    'format': 'globus-collection',
                    'required': ['id', 'path'],
                    'properties': {
                        'id': {'type': 'string', 'format': 'uuid'},
                        'path': {'type': 'string'},
                    },
                    'additionalProperties': False,
                },
                'sync_level': {
                    'description': 'Must have one of the values 0, 1, 2, 3 as defined in the Globus Transfer API',
                    'enum': [None, '0', '1', '2', '3', 0, 1, 2, 3],
                },
                'recursive': {
                    'type': 'boolean',
                    'title': 'Recursive transfer',
                    'description': 'Whether or not to transfer recursively, must be True when transferring a directory.',
                    'default': True,
                },
            },
            'additionalProperties': False,
        },
        'compute': {
            'type': 'object',
            'required': ['endpoint_id', 'function_id', 'function_kwargs'],
            'properties': {
                'endpoint_id': {
                    'type': 'string',
                    'format': 'uuid',
                    'title': 'Globus Compute Endpoint ID',
                    'description': 'The UUID of the Globus Compute endpoint where the function will run.',
                },
                'function_id': {
                    'type': 'string',
                    'format': 'uuid',
                    'title': 'Globus Compute Function ID',
                    'description': 'The UUID of the function to invoke; must be registered with the Globus Compute service.',
                },
                'function_kwargs': {
                    'type': 'object',
                    'title': 'Function Inputs',
                    'description': 'Inputs to pass to the function.',
                    'required': ['action', 'input_directory', 'output_directory'],
                    'properties': {
                        'action': {
                            'type': 'string',
                            'title': 'Ptychodus Action',
                            'description': 'The ptychodus action to perform.',
                            'enum': ['reconstruct', 'train'],
                        },
                        'input_directory': {
                            'type': 'string',
                            'title': 'Input Directory',
                            'description': 'POSIX path to the input data directory on the compute resource.',
                        },
                        'output_directory': {
                            'type': 'string',
                            'title': 'Output Directory',
                            'description': 'POSIX path to the output data directory on the compute resource.',
                        },
                    },
                    'additionalProperties': False,
                },
            },
            'additionalProperties': False,
        },
        'transfer_output_data': {
            'type': 'object',
            'required': ['source', 'destination', 'sync_level', 'recursive'],
            'properties': {
                'source': {
                    'type': 'object',
                    'title': 'Select source collection and path',
                    'description': 'The collection and path on the Globus Compute endpoint where output data resides.',
                    'format': 'globus-collection',
                    'required': ['id', 'path'],
                    'properties': {
                        'id': {'type': 'string', 'format': 'uuid'},
                        'path': {'type': 'string'},
                    },
                    'additionalProperties': False,
                },
                'destination': {
                    'type': 'object',
                    'title': 'Select destination collection and path',
                    'description': 'The collection and path where processed output data will be stored (e.g., archive or analysis system).',
                    'format': 'globus-collection',
                    'required': ['id', 'path'],
                    'properties': {
                        'id': {'type': 'string', 'format': 'uuid'},
                        'path': {'type': 'string'},
                    },
                    'additionalProperties': False,
                },
                'sync_level': {
                    'description': 'Must have one of the values 0, 1, 2, 3 as defined in the Globus Transfer API',
                    'enum': [None, '0', '1', '2', '3', 0, 1, 2, 3],
                },
                'recursive': {
                    'type': 'boolean',
                    'title': 'Recursive transfer',
                    'description': 'Whether or not to transfer recursively, must be True when transferring a directory.',
                    'default': True,
                },
            },
            'additionalProperties': False,
        },
    },
    'additionalProperties': False,
}
