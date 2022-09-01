import sys

from gladier import GladierBaseClient, GladierBaseTool, generate_flow_definition


def ptychodus_reconstruct(**data):
    from pathlib import Path
    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        settingsFilePath=Path(data['settings_file']),
        replacementPathPrefix=data['replacement_path_prefix'],
    )

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct()


@generate_flow_definition
class PtychodusReconstruct(GladierBaseTool):
    required_input = [
        'settings_file',
        'replacement_path_prefix',
    ]
    funcx_functions = [ptychodus_reconstruct]


@generate_flow_definition
class PtychodusClient(GladierBaseClient):
    # TODO client_id = GlobusWorkflowClient.CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:InputData',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:OutputData',
        #'gladier_tools.publish.Publish',
    ]


def main() -> int:
    client = PtychodusClient()
    flow_id = client.get_flow_id()
    print(f'Flow ID: {flow_id}')
    return 0


sys.exit(main())
