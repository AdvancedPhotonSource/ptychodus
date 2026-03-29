from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Final
import hashlib
import inspect
import json
import logging
import os
import platform
import queue
import stat
import threading
import uuid

from globus_sdk.flows import SpecificFlowClient
from globus_sdk.gare import GlobusAuthorizationParameters
from globus_sdk.globus_app import GlobusAppConfig
from globus_sdk.login_flows import LoginFlowManager
from globus_sdk.tokenstorage import SimpleJSONFileAdapter
import globus_compute_sdk
import globus_sdk

from ptychodus.api.common import get_ptychodus_dir

from ..task_manager import TaskManager
from .actions import process_with_ptychodus
from .authorizer import GlobusAuthorizer
from .client import GlobusClient, GlobusJob, GlobusStatus
from .flow_definition import PTYCHODUS_FLOW_DEFINITION
from .input_schema import PTYCHODUS_FLOW_INPUT_SCHEMA
from .settings import GlobusSettings

__all__ = ['RealGlobusClient']

logger = logging.getLogger(__name__)

PTYCHODUS_APP_NAME: Final[str] = 'Ptychodus'
PTYCHODUS_CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'
PTYCHODUS_FLOW_TITLE: Final[str] = 'Ptychodus Flow'


def ensure_owner_only_read_write(file_path: Path) -> Path:
    OWNER_ONLY_RW_MODE: Final[int] = 0o600  # noqa: N806

    if file_path.is_file():
        file_stat = file_path.stat()

        if stat.S_IMODE(file_stat.st_mode) != OWNER_ONLY_RW_MODE:
            raise RuntimeError(f'Token storage path "{file_path} must have 0o600 permissions!"')
    else:
        file_path.touch(mode=OWNER_ONLY_RW_MODE)

    return file_path


def uses_data_access(transfer_client: globus_sdk.TransferClient, collection_id: str) -> bool:
    """
    Use the provided `transfer_client` to lookup a collection by ID.

    Based on the record, return `True` if it uses a `data_access` scope and `False`
    otherwise.

    Example usage:

        with globus_sdk.TransferClient(app=app) as transfer_client:
            # check if either collection needs data_access, and if so add the relevant requirement
            if uses_data_access(transfer_client, SRC_COLLECTION):
                transfer_client.add_app_data_access_scope(SRC_COLLECTION)
            if uses_data_access(transfer_client, DST_COLLECTION):
                transfer_client.add_app_data_access_scope(DST_COLLECTION)

            transfer_request = globus_sdk.TransferData(SRC_COLLECTION, DST_COLLECTION)
            transfer_request.add_item(SRC_PATH, DST_PATH)

            task = transfer_client.submit_transfer(transfer_request)
            print(f'Submitted transfer. Task ID: {task["task_id"]}.')

    Copied from here:
    https://globus-sdk-python.readthedocs.io/en/latest/user_guide/usage_patterns/data_transfer/detecting_data_access/index.html

    """
    doc = transfer_client.get_endpoint(collection_id)

    if doc['entity_type'] != 'GCSv5_mapped_collection':
        return False

    if doc['high_assurance']:
        return False

    return True


class PtychodusLoginFlowManager(LoginFlowManager):
    """
    Based on this example:
    https://globus-sdk-python.readthedocs.io/en/stable/_modules/globus_sdk/login_flows/command_line_login_flow_manager.html#CommandLineLoginFlowManager
    """

    def __init__(
        self,
        login_client: globus_sdk.AuthLoginClient,
        authorize_url_q: queue.Queue[str],
        auth_code_q: queue.Queue[str],
        stop_event: threading.Event,
        *,
        redirect_uri: str | None = None,
        request_refresh_tokens: bool = False,
        native_prefill_named_grant: str | None = None,
    ) -> None:
        super().__init__(
            login_client,
            request_refresh_tokens=request_refresh_tokens,
            native_prefill_named_grant=native_prefill_named_grant,
        )
        self._authorize_url_q = authorize_url_q
        self._auth_code_q = auth_code_q
        self._stop_event = stop_event

        if redirect_uri is None:
            # Confidential clients must always define their own custom redirect URI.
            if isinstance(login_client, globus_sdk.ConfidentialAppAuthClient):
                msg = 'Use of a Confidential client requires an explicit redirect_uri.'
                raise globus_sdk.GlobusSDKUsageError(msg)

            # Native clients may infer the globus-provided helper page if omitted.
            redirect_uri = login_client.base_url + 'v2/web/auth-code'

        self.redirect_uri = redirect_uri

    def run_login_flow(
        self,
        auth_parameters: GlobusAuthorizationParameters,
    ) -> globus_sdk.OAuthTokenResponse:
        """
        Run an interactive login flow to get tokens for the user.

        :param auth_parameters: ``GlobusAuthorizationParameters`` passed through
            to the authentication flow to control how the user will authenticate.
        """
        authorize_url = self._get_authorize_url(auth_parameters, self.redirect_uri)
        auth_code: str | None = None

        logger.info(f'Please authenticate with Globus here: {authorize_url}')
        self._authorize_url_q.put(authorize_url)

        while not self._stop_event.is_set():
            try:
                auth_code = self._auth_code_q.get(block=True, timeout=1.0)
            except queue.Empty:
                pass
            else:
                break

        if auth_code is None:
            raise RuntimeError('Missing auth code!')

        return self.login_client.oauth2_exchange_code_for_tokens(auth_code)


def create_globus_app(
    authorize_url_q: queue.Queue[str],
    auth_code_q: queue.Queue[str],
    stop_event: threading.Event,
    request_refresh_tokens: bool = True,
) -> globus_sdk.GlobusApp:
    token_storage_path = get_ptychodus_dir() / 'globus_tokens.json'
    token_storage = SimpleJSONFileAdapter(ensure_owner_only_read_write(token_storage_path))

    hostname = platform.node()
    prefill = f'{PTYCHODUS_APP_NAME} on {hostname}' if hostname else PTYCHODUS_APP_NAME

    try:
        client_id = os.environ['CLIENT_ID']
        client_secret = os.environ['CLIENT_SECRET']
    except KeyError:
        logger.info(
            'Environment variables "CLIENT_ID" and "CLIENT_SECRET" not set. '
            'Creating native user app.'
        )
        login_client = globus_sdk.NativeAppAuthClient(client_id=PTYCHODUS_CLIENT_ID)
        login_flow_manager = PtychodusLoginFlowManager(
            login_client,
            authorize_url_q,
            auth_code_q,
            stop_event,
            request_refresh_tokens=request_refresh_tokens,
            native_prefill_named_grant=prefill,
        )
        config = GlobusAppConfig(
            login_flow_manager=login_flow_manager,
            token_storage=token_storage,
            request_refresh_tokens=request_refresh_tokens,
            native_prefill_named_grant=prefill,
        )
        return globus_sdk.UserApp(
            PTYCHODUS_APP_NAME,
            client_id=PTYCHODUS_CLIENT_ID,
            client_secret=None,
            scope_requirements=None,
            config=config,
            login_client=login_client,
        )
    else:
        logger.info(
            'Environment variables "CLIENT_ID" and "CLIENT_SECRET" set. '
            'Creating confidential client app.'
        )
        config = GlobusAppConfig(
            token_storage=token_storage,
            request_refresh_tokens=request_refresh_tokens,
            native_prefill_named_grant=prefill,
        )
        return globus_sdk.ClientApp(
            PTYCHODUS_APP_NAME, client_id=client_id, client_secret=client_secret, config=config
        )


def get_process_with_ptychodus_action_checksum() -> str:
    """
    Get the SHA256 checksum of the current compute function.

    :return: sha256 hex string of compute function
    """
    source = inspect.getsource(process_with_ptychodus)
    return hashlib.sha256(source.encode()).hexdigest()


def get_ptychodus_flow_checksum() -> str:
    """
    Get the SHA256 checksum of the current flow definition.

    :return: sha256 hex string of flow definition
    """
    definition_json = json.dumps(PTYCHODUS_FLOW_DEFINITION, sort_keys=True)
    input_schema_json = json.dumps(PTYCHODUS_FLOW_INPUT_SCHEMA, sort_keys=True)
    data = (definition_json + input_schema_json).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RegisteredPtychodusFlow:
    process_action_id: uuid.UUID
    process_action_checksum: str
    flow_id: uuid.UUID
    flow_checksum: str

    @classmethod
    def read_from_file(cls, file_path: Path) -> RegisteredPtychodusFlow:
        data = json.loads(file_path.read_text())
        return cls(
            process_action_id=uuid.UUID(data['process_action_id']),
            process_action_checksum=data['process_action_checksum'],
            flow_id=uuid.UUID(data['flow_id']),
            flow_checksum=data['flow_checksum'],
        )

    def write_to_file(self, file_path: Path) -> None:
        data = {
            'process_action_id': str(self.process_action_id),
            'process_action_checksum': self.process_action_checksum,
            'flow_id': str(self.flow_id),
            'flow_checksum': self.flow_checksum,
        }
        file_path.write_text(json.dumps(data))


def sync_flow(
    compute_client: globus_compute_sdk.Client, flows_client: globus_sdk.FlowsClient
) -> RegisteredPtychodusFlow:
    file_path = get_ptychodus_dir() / 'globus_flows.json'
    expected_action_checksum = get_process_with_ptychodus_action_checksum()
    expected_flow_checksum = get_ptychodus_flow_checksum()

    try:
        registered_flow = RegisteredPtychodusFlow.read_from_file(file_path)
    except FileNotFoundError:
        # First run: register compute function and create flow
        new_action = compute_client.register_function(process_with_ptychodus)
        process_action_id = uuid.UUID(new_action)
        logger.info(f'Process action registered with ID: {process_action_id}')

        new_flow = flows_client.create_flow(
            title=PTYCHODUS_FLOW_TITLE,
            definition=dict(PTYCHODUS_FLOW_DEFINITION),
            input_schema=dict(PTYCHODUS_FLOW_INPUT_SCHEMA),
        )
        flow_id = uuid.UUID(new_flow['id'])
        logger.info(f'Flow created with ID: {flow_id}')
    else:
        if registered_flow.process_action_checksum == expected_action_checksum:
            process_action_id = registered_flow.process_action_id
        else:
            new_action = compute_client.register_function(process_with_ptychodus)
            process_action_id = uuid.UUID(new_action)
            logger.info(f'Process action re-registered with ID: {process_action_id}')

        if registered_flow.flow_checksum != expected_flow_checksum:
            flows_client.update_flow(
                registered_flow.flow_id,
                title=PTYCHODUS_FLOW_TITLE,
                definition=dict(PTYCHODUS_FLOW_DEFINITION),
                input_schema=dict(PTYCHODUS_FLOW_INPUT_SCHEMA),
            )
            logger.info(f'Flow updated with ID: {registered_flow.flow_id}')

        flow_id = registered_flow.flow_id

    result = RegisteredPtychodusFlow(
        process_action_id=process_action_id,
        process_action_checksum=expected_action_checksum,
        flow_id=flow_id,
        flow_checksum=expected_flow_checksum,
    )
    result.write_to_file(file_path)
    return result


class GlobusClientThread(threading.Thread):
    def __init__(
        self,
        authorize_url_q: queue.Queue[str],
        auth_code_q: queue.Queue[str],
        job_q: queue.Queue[GlobusJob],
        status_q: queue.Queue[GlobusStatus],
        refresh_status_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._job_q = job_q
        self._status_q = status_q
        self._refresh_status_event = refresh_status_event
        self._stop_event = stop_event

        self._globus_app = create_globus_app(
            authorize_url_q,
            auth_code_q,
            stop_event,
        )
        self._flows_client = globus_sdk.FlowsClient(app=self._globus_app)

    def _list_flows(self) -> None:
        for flow in self._flows_client.list_flows(filter_roles='flow_owner'):
            logger.info(f'title={flow["title"]} id={flow["id"]}')

    def _delete_flow(self, flow_id: uuid.UUID):
        self._flows_client.delete_flow(flow_id)

    def _get_current_action(self, run_id: str) -> str:
        status = self._flows_client.get_run(run_id).data
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
        logger.debug('Refreshing status.')

        flow_id = uuid.UUID(int=0)  # FIXME

        response = self._flows_client.list_runs(filter_flow_id=flow_id)
        run_dict_list = response['runs']

        while response['has_next_page']:
            response = self._flows_client.list_runs(
                filter_flow_id=flow_id, marker=response['marker']
            )
            run_dict_list.extend(response['runs'])

        for run_dict in run_dict_list:
            run_id = run_dict.get('run_id', '')
            action = self._get_current_action(run_id)
            start_time_str = run_dict.get('start_time', '')

            try:
                start_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                logger.warning(f'Failed to parse startTime "{start_time_str}"!')
                start_time = datetime.min

            completion_time_str = run_dict.get('completion_time', '')

            try:
                completion_time = datetime.fromisoformat(completion_time_str)
            except ValueError:
                completion_time = None

            status = GlobusStatus(
                label=run_dict.get('label', ''),
                start_time=start_time,
                completion_time=completion_time,
                status=run_dict.get('status', ''),
                action=action,
                run_id=run_id,
            )
            self._status_q.put(status)

    def run(self) -> None:
        flow_id = uuid.UUID(int=0)  # FIXME
        specific_flow_client = SpecificFlowClient(flow_id, app=self._globus_app)

        while not self._stop_event.is_set():
            if self._refresh_status_event.is_set():
                self._refresh_status_event.clear()
                self._refresh_status()

            try:
                job = self._job_q.get(block=True, timeout=1)
            except queue.Empty:
                pass
            else:
                try:
                    response = specific_flow_client.run_flow(
                        flow_input=job.flow_input,
                        label=job.label,
                        tags=job.tags,
                    )
                except Exception:
                    logger.exception('Error running flow!')
                else:
                    logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')
                finally:
                    self._job_q.task_done()

        self._globus_app.close()


class RealGlobusClient(GlobusClient):
    def __init__(
        self,
        task_manager: TaskManager,
        settings: GlobusSettings,
        authorizer: GlobusAuthorizer,
        status_q: queue.Queue[GlobusStatus],
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._authorizer = authorizer
        self._status_q = status_q

        self._stop_event = threading.Event()
        self._refresh_status_event = threading.Event()
        self._job_q: queue.Queue[GlobusJob] = queue.Queue()
        self._worker: GlobusClientThread | None = None

    @property
    def is_supported(self) -> bool:
        return True

    def start(self) -> None:  # FIXME start as needed
        if self._worker is None:
            logger.debug('Starting Globus thread...')
            self._stop_event.clear()
            self._worker = GlobusClientThread(
                authorize_url_q=self._authorizer._auth_url_q,
                auth_code_q=self._authorizer._auth_code_q,
                job_q=self._job_q,
                status_q=self._status_q,
                refresh_status_event=self._refresh_status_event,
                stop_event=self._stop_event,
            )
            self._worker.start()
            logger.debug('Globus thread started.')
        else:
            logger.warning('Worker already started!')

    def stop(self) -> None:
        if self._stop_event.is_set():
            logger.debug('Globus thread already stopped.')
        else:
            logger.debug('Finishing tasks...')
            self._job_q.join()
            logger.debug('Tasks finished.')

            if self._worker is None:
                logger.warning('Worker is None!')
            else:
                logger.debug('Stopping Globus thread...')
                self._stop_event.set()
                self._worker.join()
                self._worker = None
                logger.debug('Globus thread stopped.')

    def run_flow(self, job: GlobusJob) -> None:
        self._job_q.put(job)

    def refresh_status(self) -> None:
        self._refresh_status_event.set()
