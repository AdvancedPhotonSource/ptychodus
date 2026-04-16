from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time

import requests

from .iri import IRIComputeClient, JobSpecification, JobState
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenesisStatus:
    label: str
    start_time: datetime
    completion_time: datetime | None
    status: str
    action: str


def transfer_task(
    client: AmSCGlobusTransferClient,
    inputs: GlobusTransferInputs,
    stop_event: threading.Event,
    status_interval_s: float,
) -> Iterator[GenesisStatus]:
    start_time = datetime.now()

    for details in client.start_transfer(inputs):
        if details.transfer_uuid is None:
            continue

        while not stop_event.is_set():
            result = client.get_transfer(details.transfer_uuid)
            status = result.status or 'UNKNOWN'
            logger.info(f'Transfer {details.name!r} [{details.transfer_uuid}]: {status}')

            if status == 'SUCCEEDED':
                yield GenesisStatus(
                    label=details.name,
                    start_time=start_time,
                    completion_time=result.completion_time or datetime.now(),
                    status=status,
                    action='Transfer',
                )
                break
            elif status in {'FAILED', 'CANCELED'}:
                raise RuntimeError(
                    f'Transfer {details.name!r} [{details.transfer_uuid}] '
                    f'ended with status {status!r}: {result.reason}'
                )
            else:
                yield GenesisStatus(
                    label=details.name,
                    start_time=start_time,
                    completion_time=None,
                    status=status,
                    action='Transfer',
                )
                stop_event.wait(status_interval_s)


def compute_task(
    client: IRIComputeClient,
    resource_id: str,
    spec: JobSpecification,
    stop_event: threading.Event,
    status_interval_s: float,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> Iterator[GenesisStatus]:
    start_time = datetime.now()

    for attempt in range(1, max_retries + 1):
        try:
            job_response = client.submit_job(resource_id, spec)
        except requests.HTTPError as exc:
            if exc.response.status_code == 404:
                if attempt == max_retries:
                    logger.warning('Exceeded max retries.')
                    raise

                time.sleep(retry_delay)
            else:
                raise
        else:
            break
    else:
        raise RuntimeError('submit_job_with_retry exited loop without returning')

    job_id = job_response.job_id

    while not stop_event.is_set():
        response = client.get_job_status(resource_id, job_id)

        if response.status is None:
            logger.info(f'Job [{job_id}]: status unknown, waiting...')
            stop_event.wait(status_interval_s)
            continue

        state = response.status.state
        message = response.status.message or ''
        logger.info(f'Job [{job_id}]: {state} {message}'.rstrip())

        if state == JobState.COMPLETED:
            yield GenesisStatus(
                label=job_id,
                start_time=start_time,
                completion_time=datetime.now(),
                status=state,
                action='Compute',
            )
            break
        elif state in {JobState.FAILED, JobState.CANCELED}:
            raise RuntimeError(f'Job [{job_id}] ended with state {state!r}: {message}')
        else:
            yield GenesisStatus(
                label=job_id,
                start_time=start_time,
                completion_time=None,
                status=state,
                action='Compute',
            )
            stop_event.wait(status_interval_s)
