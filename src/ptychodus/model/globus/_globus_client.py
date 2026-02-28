from __future__ import annotations
from datetime import datetime
from pprint import pformat
import json
import logging
import queue
import threading

import fair_research_login


from ..task_manager import TaskManager
from ._gladier_client import PtychodusClient
from .authorizer import GlobusAuthorizer
from .client import GlobusClient, GlobusJob, GlobusStatus
from .settings import GlobusSettings

__all__ = ['RealGlobusClient']

logger = logging.getLogger(__name__)


class CustomCodeHandler(fair_research_login.CodeHandler):
    def __init__(
        self,
        auth_url_q: queue.Queue[str],
        auth_code_q: queue.Queue[str],
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._auth_url_q = auth_url_q
        self._auth_code_q = auth_code_q
        self._stop_event = stop_event
        self._code = ''
        self.set_browser_enabled(False)

    def authenticate(self, url: str) -> str:
        self._auth_url_q.put(url)

        while not self._stop_event.is_set():
            try:
                self._code = self._auth_code_q.get(block=True, timeout=1.0)
            except queue.Empty:
                pass

        return self._code

    def get_code(self) -> str:
        return self._code


class GlobusClientThread(threading.Thread):
    def __init__(
        self,
        auth_url_q: queue.Queue[str],
        auth_code_q: queue.Queue[str],
        job_q: queue.Queue[GlobusJob],
        status_q: queue.Queue[GlobusStatus],
        refresh_status_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._auth_url_q = auth_url_q
        self._auth_code_q = auth_code_q
        self._job_q = job_q
        self._status_q = status_q
        self._refresh_status_event = refresh_status_event
        self._stop_event = stop_event
        self._gladier_client = PtychodusClient.create_client(
            [CustomCodeHandler(auth_url_q, auth_code_q, stop_event)]
        )

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
        logger.debug('Refreshing status.')

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
                    response = self._gladier_client.run_flow(
                        flow_input={'input': job.flow_input},
                        label=job.label,
                        tags=job.tags,
                    )
                except Exception:
                    logger.exception('Error running flow!')
                else:
                    logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')
                finally:
                    self._job_q.task_done()


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
                auth_url_q=self._authorizer._auth_url_q,
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
