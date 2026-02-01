from __future__ import annotations
from datetime import datetime, timezone
from importlib.metadata import version
from pprint import pformat
import json
import logging
import queue
import threading

import fair_research_login
import gladier

from ptychodus.api.observer import Observable, Observer

from ..task_manager import TaskManager
from ._gladier_client import PtychodusClient
from .authorizer import AuthorizeWithGlobus, GlobusAuthorizer
from .client import GlobusClient, GlobusJob
from .settings import GlobusSettings
from .status import GlobusStatus, GlobusStatusRepository, UpdateGlobusStatusRepository

__all__ = ['RealGlobusClient']

logger = logging.getLogger(__name__)


class CustomCodeHandler(fair_research_login.CodeHandler):
    def __init__(
        self, task_manager: TaskManager, stop_event: threading.Event, authorizer: GlobusAuthorizer
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._stop_event = stop_event
        self._authorizer = authorizer

        self._user_provided_code = threading.Event()
        self._code_lock = threading.Lock()
        self._code = ''

        self.set_browser_enabled(False)

    def authenticate(self, url: str) -> str:
        self._user_provided_code.clear()

        with self._code_lock:
            self._code = ''

        task = AuthorizeWithGlobus(self._authorizer, url)
        self._task_manager.put_foreground_task(task)

        while not self._stop_event.is_set():
            if self._user_provided_code.wait(timeout=1.0):
                break

        return self.get_code()

    def set_code_from_user(self, code: str) -> None:
        with self._code_lock:
            self._code = code

        self._user_provided_code.set()

    def get_code(self) -> str:
        with self._code_lock:
            return self._code


class RunGlobusFlow:
    def __init__(self, gladier_client: gladier.GladierBaseClient, job: GlobusJob) -> None:
        self._gladier_client = gladier_client
        self._job = job

    def __call__(self) -> None:
        try:
            response = self._gladier_client.run_flow(
                flow_input={'input': self._job.flow_input},
                label=self._job.label,
                tags=self._job.tags,
            )
        except Exception:
            logger.exception('Error running flow!')
        else:
            logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')


class RealGlobusClient(GlobusClient, Observer):
    def __init__(
        self,
        task_manager: TaskManager,
        settings: GlobusSettings,
        authorizer: GlobusAuthorizer,
        status_repository: GlobusStatusRepository,
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._authorizer = authorizer
        self._status_repository = status_repository

        logger.info('\tGlobus SDK ' + version('globus-sdk'))
        logger.info('\tFair Research Login ' + version('fair-research-login'))
        logger.info('\tGladier ' + version('gladier'))

        self._stop_event = threading.Event()
        self._job_queue: queue.Queue[GlobusJob] = queue.Queue()
        self._worker: threading.Thread | None = None
        self.__gladier_client: gladier.GladierBaseClient | None = None

        self._code_handler = CustomCodeHandler(task_manager, self._stop_event, authorizer)
        authorizer.add_observer(self)

        self._status_date_time = datetime.min
        self._status_lock = threading.Lock()
        self._status_auto_refresh = False
        self._status_refresh_interval_s = 0

        self._sync_status_refresh_interval()
        settings.status_auto_refresh.add_observer(self)
        settings.status_refresh_interval_s.add_observer(self)

    @property
    def _gladier_client(self) -> gladier.GladierBaseClient:
        if self.__gladier_client is None:
            self.__gladier_client = PtychodusClient.create_client([self._code_handler])

        return self.__gladier_client

    @property
    def is_supported(self) -> bool:
        return True

    def start(self) -> None:
        if self._worker is None:
            logger.info('Starting Globus thread...')
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._run_tasks)
            self._worker.start()
            logger.info('Globus thread started.')
        else:
            logger.warning('Worker already started!')

    def stop(self) -> None:
        if self._stop_event.is_set():
            logger.info('Globus thread already stopped.')
        else:
            logger.info('Finishing tasks...')
            self._job_queue.join()
            logger.info('Tasks finished.')

            if self._worker is None:
                logger.warning('Worker is None!')
            else:
                logger.info('Stopping Globus thread...')
                self._stop_event.set()
                self._worker.join()
                self._worker = None
                logger.info('Globus thread stopped.')

    def run_flow(self, job: GlobusJob) -> None:
        self._job_queue.put(job)

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
        status_list: list[GlobusStatus] = list()

        logger.debug('Refreshing status.')

        flows_manager = self._gladier_client.flows_manager
        flow_id = flows_manager.get_flow_id()
        flows_client = flows_manager.flows_client
        response = flows_client.list_runs(filter_flow_id=flow_id)
        run_dict_list = response['runs']

        while response['has_next_page']:
            response = flows_client.list_runs(filter_flow_id=flow_id, marker=response['marker'])
            run_dict_list.extend(response['runs'])

        self._status_date_time = datetime.now(timezone.utc)

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
            status_list.append(status)

        status_update = UpdateGlobusStatusRepository(self._status_repository, status_list)
        self._task_manager.put_foreground_task(status_update)

    def _run_tasks(self) -> None:
        while not self._stop_event.is_set():
            with self._status_lock:
                if self._status_auto_refresh:
                    status_time_delta = datetime.now(timezone.utc) - self._status_date_time
                    status_age_s = status_time_delta.total_seconds()

                    if status_age_s >= self._status_refresh_interval_s:
                        self._status_repository.refresh_status()

            if self._status_repository.needs_status_refresh():
                self._status_repository._status_refreshed()
                self._refresh_status()

            try:
                job = self._job_queue.get(block=True, timeout=1)
            except queue.Empty:
                pass
            else:
                task = RunGlobusFlow(self._gladier_client, job)
                self._task_manager.put_background_task(task)
                self._job_queue.task_done()

    def _sync_status_refresh_interval(self) -> None:
        with self._status_lock:
            self._status_auto_refresh = self._settings.status_auto_refresh.get_value()
            self._status_refresh_interval_s = self._settings.status_refresh_interval_s.get_value()

    def _update(self, observable: Observable) -> None:
        if observable is self._authorizer:
            if self._authorizer.has_authorize_code:
                self._code_handler.set_code_from_user(self._authorizer.get_authorize_code())
        elif observable in (
            self._settings.status_auto_refresh,
            self._settings.status_refresh_interval_s,
        ):
            self._sync_status_refresh_interval()
