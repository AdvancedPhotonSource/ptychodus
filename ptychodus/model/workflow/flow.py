from gladier import GladierBaseClient, GladierBaseTool, generate_flow_definition

from .authorizerRepository import GlobusAuthorizerRepository


def ptychodus_reconstruct(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        restartFilePath=Path(data['restart_file']),
        settingsFilePath=Path(data['settings_file']),
        replacementPathPrefix=data['replacement_path_prefix'],
    )

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct()


@generate_flow_definition
class PtychodusReconstruct(GladierBaseTool):
    funcx_functions = [ptychodus_reconstruct]
    required_input = [
        'restart_file',
        'settings_file',
        'replacement_path_prefix',
    ]


@generate_flow_definition
class PtychodusClient(GladierBaseClient):
    client_id = GlobusAuthorizerRepository.CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:Settings',
        'gladier_tools.globus.transfer.Transfer:State',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:Results',
        # TODO 'gladier_tools.publish.Publish',
    ]
