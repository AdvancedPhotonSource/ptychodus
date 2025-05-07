from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from importlib.metadata import version
from pprint import pformat
from typing import Final, TypeAlias
import json
import logging
import os
import queue
import threading

import fair_research_login
import gladier
import gladier.managers
import globus_sdk

from .authorizer import WorkflowAuthorizer
from .executor import WorkflowExecutor
from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)

AuthorizerTypes: TypeAlias = globus_sdk.AccessTokenAuthorizer | globus_sdk.RefreshTokenAuthorizer
ScopeAuthorizerMapping: TypeAlias = Mapping[str, AuthorizerTypes]

PTYCHODUS_CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

# TODO add review consents fix


def ptychodus_reconstruct(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelCore

    action = data['ptychodus_action']
    input_file = Path(data['ptychodus_input_file'])
    output_file = Path(data['ptychodus_output_file'])
    settings_file = Path(data['ptychodus_settings_file'])
    patterns_file = Path(data['ptychodus_patterns_file'])

    with ModelCore(settings_file) as model:
        model.workflow_api.import_assembled_patterns(patterns_file)
        model.batch_mode_execute(action, input_file, output_file)


@gladier.generate_flow_definition
class PtychodusReconstruct(gladier.GladierBaseTool):
    compute_functions = [ptychodus_reconstruct]
    required_input = [
        'ptychodus_action',
        'ptychodus_input_file',
        'ptychodus_output_file',
        'ptychodus_patterns_file',
        'ptychodus_settings_file',
    ]


@gladier.generate_flow_definition
class PtychodusClient(gladier.GladierBaseClient):
    client_id = PTYCHODUS_CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:InputData',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:OutputData',
        # TODO 'gladier_tools.publish.Publish',
    ]


class CustomCodeHandler(fair_research_login.CodeHandler):
    def __init__(self, authorizer: WorkflowAuthorizer) -> None:
        super().__init__()
        self._authorizer = authorizer
        self.set_browser_enabled(False)

    def authenticate(self, url: str) -> str:
        self._authorizer.authenticate(url)
        return self.get_code()

    def get_code(self) -> str:
        return self._authorizer.get_code_from_authorize_url()


class PtychodusClientBuilder(ABC):
    @abstractmethod
    def build(self) -> gladier.GladierBaseClient:
        pass


class NativePtychodusClientBuilder(PtychodusClientBuilder):
    def __init__(self, authorizer: WorkflowAuthorizer) -> None:
        super().__init__()
        self._auth_client = fair_research_login.NativeClient(
            client_id=PTYCHODUS_CLIENT_ID,
            app_name='Ptychodus',
            code_handlers=[CustomCodeHandler(authorizer)],
        )

    def _request_authorization(self, scopes: list[str]) -> ScopeAuthorizerMapping:
        logger.debug(f'Requested authorization scopes: {pformat(scopes)}')

        # 'force' is used for any underlying scope changes. For example, if a flow adds transfer
        # functionality since it was last run, running it again would require a re-login.
        self._auth_client.login(requested_scopes=scopes, force=True, refresh_tokens=True)
        return self._auth_client.get_authorizers_by_scope()

    def build(self) -> gladier.GladierBaseClient:
        initial_authorizers: dict[str, AuthorizerTypes] = dict()

        try:
            # Try to use a previous login to avoid a new login flow
            initial_authorizers = self._auth_client.get_authorizers_by_scope()
        except fair_research_login.LoadError:
            pass

        login_manager = gladier.managers.CallbackLoginManager(
            authorizers=initial_authorizers,
            callback=self._request_authorization,
        )

        return PtychodusClient(login_manager=login_manager)


class ConfidentialPtychodusClientBuilder(PtychodusClientBuilder):
    def __init__(self, client_id: str, client_secret: str, flow_id: str | None) -> None:
        super().__init__()
        self._auth_client = globus_sdk.ConfidentialAppAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            app_name='Ptychodus',
        )
        self._flow_id = flow_id

    def _request_authorization(self, scopes: list[str]) -> ScopeAuthorizerMapping:
        logger.debug(f'Requested authorization scopes: {pformat(scopes)}')

        response = self._auth_client.oauth2_client_credentials_tokens(requested_scopes=scopes)
        return {
            scope: globus_sdk.AccessTokenAuthorizer(access_token=tokens['access_token'])
            for scope, tokens in response.by_scopes.scope_map.items()
        }

    def build(self) -> gladier.GladierBaseClient:
        initial_authorizers: dict[str, AuthorizerTypes] = dict()
        login_manager = gladier.managers.CallbackLoginManager(
            authorizers=initial_authorizers,
            callback=self._request_authorization,
        )
        flows_manager = gladier.managers.FlowsManager(flow_id=self._flow_id)
        return PtychodusClient(login_manager=login_manager, flows_manager=flows_manager)


class GlobusWorkflowThread(threading.Thread):
    def __init__(
        self,
        authorizer: WorkflowAuthorizer,
        status_repository: WorkflowStatusRepository,
        executor: WorkflowExecutor,
        client_builder: PtychodusClientBuilder,
    ) -> None:
        super().__init__()
        self._authorizer = authorizer
        self._status_repository = status_repository
        self._executor = executor
        self._client_builder = client_builder

        logger.info('\tGlobus SDK ' + version('globus-sdk'))
        logger.info('\tFair Research Login ' + version('fair-research-login'))
        logger.info('\tGladier ' + version('gladier'))

        self.__gladier_client: gladier.GladierBaseClient | None = None

    @classmethod
    def create_instance(
        cls,
        authorizer: WorkflowAuthorizer,
        status_repository: WorkflowStatusRepository,
        executor: WorkflowExecutor,
    ) -> GlobusWorkflowThread:
        try:
            client_id = os.environ['CLIENT_ID']
        except KeyError:
            client_builder: PtychodusClientBuilder = NativePtychodusClientBuilder(authorizer)
            return cls(authorizer, status_repository, executor, client_builder)

        try:
            client_secret = os.environ['CLIENT_SECRET']
        except KeyError as ex:
            raise KeyError('CLIENT_ID requires a CLIENT_SECRET environment variable.') from ex

        try:
            flow_id = os.environ['FLOW_ID']
        except KeyError:
            # This isn't necessarily bad, but CCs like regular users only get one flow
            # to play with. They probably don't need more than one, but this will ensure
            # there aren't errors due to tracking mismatch in the Glaider config
            flow_id = ''
            logger.warning('No flow ID enforced. Recommend setting FLOW_ID environment variable.')

        client_builder = ConfidentialPtychodusClientBuilder(client_id, client_secret, flow_id)
        return cls(authorizer, status_repository, executor, client_builder)

    @property
    def _gladier_client(self) -> gladier.GladierBaseClient:
        if self.__gladier_client is None:
            self.__gladier_client = self._client_builder.build()

        return self.__gladier_client

    def _get_current_action(self, run_id: str) -> str:
        status = self._gladier_client.get_status(run_id)
        action = status.get('state_name')

        if not action:
            try:
                det = status['details']
            except Exception:
                logger.exception('Unexpected flow status!')
                logger.error(pformat(status))
            else:
                if det.get('details') and det['details'].get('state_name'):
                    action = det['details']['state_name']
                elif det.get('details') and det['details'].get('output'):
                    action = list(det['details']['output'].keys())[0]
                elif det.get('action_statuses'):
                    action = det['action_statuses'][0].get('state_name')
                elif det.get('code') == 'FlowStarting':
                    pass

        return action

    def _refresh_status(self) -> None:
        status_list: list[WorkflowStatus] = list()
        flows_manager = self._gladier_client.flows_manager
        flow_id = flows_manager.get_flow_id()
        flows_client = flows_manager.flows_client
        response = flows_client.list_runs(filter_flow_id=flow_id)
        run_dict_list = response['runs']

        while response['has_next_page']:
            response = flows_client.list_runs(filter_flow_id=flow_id, marker=response['marker'])
            run_dict_list.extend(response['runs'])

        for run_dict in run_dict_list:
            run_id = run_dict.get('run_id', '')
            action = self._get_current_action(run_id)
            start_time_str = run_dict.get('start_time', '')
            completion_time_str = run_dict.get('completion_time', '')

            try:
                start_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                logger.warning(f'Failed to parse startTime "{start_time_str}"!')
                start_time = datetime(1, 1, 1)

            try:
                completion_time = datetime.fromisoformat(completion_time_str)
            except ValueError:
                completion_time = None

            run = WorkflowStatus(
                label=run_dict.get('label', ''),
                start_time=start_time,
                completion_time=completion_time,
                status=run_dict.get('status', ''),
                action=action,
                run_id=run_id,
                run_url=f'https://app.globus.org/runs/{run_id}/logs',
            )

            status_list.append(run)

        self._status_repository.update(status_list)

    def run(self) -> None:
        while not self._authorizer.shutdown_event.is_set():
            if self._status_repository.refresh_status_event.is_set():
                self._refresh_status()
                self._status_repository.refresh_status_event.clear()

            try:
                input_ = self._executor.job_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                response = self._gladier_client.run_flow(
                    flow_input={'input': input_.flow_input},
                    label=input_.flow_label,
                    tags=['aps', 'ptychography'],
                )
            except Exception:
                logger.exception('Error running flow!')
            else:
                logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')
            finally:
                self._executor.job_queue.task_done()
