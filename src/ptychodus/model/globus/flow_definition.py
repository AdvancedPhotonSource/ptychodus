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
                        'source_path.$': '$.input_data_transfer.source.path',
                        'destination_path.$': '$.input_data_transfer.destination.path',
                        'recursive.$': '$.input_data_transfer.recursive',
                    }
                ],
                'source_endpoint.$': '$.input_data_transfer.source.id',
                'destination_endpoint.$': '$.input_data_transfer.destination.id',
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
                        'source_path.$': '$.output_data_transfer.source.path',
                        'destination_path.$': '$.output_data_transfer.destination.path',
                        'recursive.$': '$.output_data_transfer.recursive',
                    }
                ],
                'source_endpoint.$': '$.output_data_transfer.source.id',
                'destination_endpoint.$': '$.output_data_transfer.destination.id',
            },
            'ResultPath': '$.TransferPtychodusOutputData',
            'WaitTime': 600,
            'End': True,
        },
    },
}
