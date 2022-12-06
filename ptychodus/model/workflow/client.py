from importlib.metadata import version
import json
import logging

from gladier.managers import BaseLoginManager
import gladier

from .api import WorkflowClient, WorkflowRun
from .flow import PtychodusClient
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class GlobusWorkflowClient(WorkflowClient):

    def __init__(self, settings: WorkflowSettings, loginManager: BaseLoginManager) -> None:
        logger.info('\tGladier ' + version('gladier'))

        super().__init__()
        self._settings = settings
        self._client = PtychodusClient(login_manager=loginManager)

    def listFlowRuns(self) -> list[WorkflowRun]:
        runList: list[WorkflowRun] = list()

        response = self._client.list_flow_runs(
            flow_id=self._client.flows_manager.get_flow_id(),
            orderings={'start_time': 'desc'},  # order by start_time (descending)
        )
        logger.debug(f'Flow Run List: {response}')

        for runDict in response['runs']:
            action = runDict.get('display_status', '')  # TODO display_status -> current action
            runID = runDict.get('run_id', '')
            run = WorkflowRun(
                label=runDict.get('label', ''),
                startTime=runDict.get('start_time', ''),
                completionTime=runDict.get('completion_time', ''),
                status=runDict.get('status', ''),
                action=action,
                runID=runID,
                runURL=f'https://app.globus.org/runs/{runID}/logs',
            )
            runList.append(run)

        return runList

    def runFlow(self, label: str) -> None:
        flowInput = {
            'settings_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'settings_transfer_source_path':
            str(self._settings.inputDataPath.value),
            'settings_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'settings_transfer_destination_path':
            str(self._settings.computeDataPath.value),
            'settings_transfer_recursive':
            False,
            'state_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'state_transfer_source_path':
            str(self._settings.inputDataPath.value),
            'state_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'state_transfer_destination_path':
            str(self._settings.computeDataPath.value),
            'state_transfer_recursive':
            False,
            'funcx_endpoint_compute':
            str(self._settings.computeEndpointID.value),
            'results_transfer_source_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'results_transfer_source_path':
            str(self._settings.computeDataPath.value),
            'results_transfer_destination_endpoint_id':
            str(self._settings.outputDataEndpointID.value),
            'results_transfer_destination_path':
            str(self._settings.outputDataPath.value),
            'results_transfer_recursive':
            False,
        }

        response = self._client.run_flow(
            flow_input={'input': flowInput},
            label=label,
            tags=['aps', 'ptychography'],
        )
        logger.info(f'Run Flow Response: {json.dumps(response.data, indent=4)}')
