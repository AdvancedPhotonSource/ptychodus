from collections.abc import Mapping
from typing import Any

PTYCHODUS_FLOW_DEFINITION: Mapping[str, Any] = {
    'Comment': 'Transfer input data, process with ptychodus, and transfer output data.',
    'StartAt': 'TransferPtychodusInputData',
    'States': {
        'TransferPtychodusInputData': {
            'Comment': 'Transfer processing input to the compute resource',
            'Type': 'Action',
            'ActionUrl': 'https://transfer.actions.globus.org/transfer',
            'Parameters': {
                'DATA': [
                    {
                        'source_path.$': '$.transfer_input_data.source.path',
                        'destination_path.$': '$.transfer_input_data.destination.path',
                        'recursive.$': '$.transfer_input_data.recursive',
                    }
                ],
                'source_endpoint.$': '$.transfer_input_data.source.id',
                'destination_endpoint.$': '$.transfer_input_data.destination.id',
                'sync_level.$': '$.transfer_input_data.sync_level',
            },
            'ResultPath': '$.TransferPtychodusInputData',
            'WaitTime': 600,
            'Next': 'ProcessWithPtychodus',
        },
        'ProcessWithPtychodus': {
            'Comment': 'Process input data with Ptychodus to generate output data',
            'Type': 'Action',
            'ActionUrl': 'https://compute.actions.globus.org/v3',
            'Parameters': {
                'endpoint_id.$': '$.compute.endpoint_id',
                'tasks': [
                    {
                        'function_id.$': '$.compute.function_id',
                        'kwargs.$': '$.compute.function_kwargs',
                    }
                ],
            },
            'ResultPath': '$.ProcessWithPtychodus',
            'ExceptionOnActionFailure': True,
            'WaitTime': 3600,
            'Next': 'TransferPtychodusOutputData',
        },
        'TransferPtychodusOutputData': {
            'Comment': 'Transfer processing output from the compute resource',
            'Type': 'Action',
            'ActionUrl': 'https://transfer.actions.globus.org/transfer',
            'Parameters': {
                'DATA': [
                    {
                        'source_path.$': '$.transfer_output_data.source.path',
                        'destination_path.$': '$.transfer_output_data.destination.path',
                        'recursive.$': '$.transfer_output_data.recursive',
                    }
                ],
                'source_endpoint.$': '$.transfer_output_data.source.id',
                'destination_endpoint.$': '$.transfer_output_data.destination.id',
                'sync_level.$': '$.transfer_output_data.sync_level',
            },
            'ResultPath': '$.TransferPtychodusOutputData',
            'WaitTime': 600,
            'End': True,
        },
    },
}
