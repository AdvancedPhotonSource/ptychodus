from __future__ import annotations
from collections.abc import Iterator
from datetime import datetime
import logging
import threading
import time

import httpx

from .iri import IRIComputeClient, JobSpecification, JobState
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs, TransferStatus
from .status import GenesisStatus

logger = logging.getLogger(__name__)

__all__ = ['compute_task', 'transfer_task']


def transfer_task(
    client: AmSCGlobusTransferClient,
    inputs: GlobusTransferInputs,
    stop_event: threading.Event,
    status_interval_s: float,
    facility: str,
    flow_label: str,
) -> Iterator[GenesisStatus]:
    start_time = datetime.now()

    yield GenesisStatus(
        facility=facility,
        label=flow_label,
        action=inputs.label,
        status='Starting',
        start_time=start_time,
        completion_time=None,
    )

    transfer = client.start_transfer(inputs)
    transfer_uuid = transfer.transfer_uuid
    transfer_label = transfer.label or transfer_uuid

    while not stop_event.is_set():
        result = client.get_transfer(transfer_uuid)
        status = result.status or TransferStatus.UNKNOWN
        logger.info(f'Transfer {transfer_label!r} [{transfer_uuid}]: {status}')

        match status:
            case TransferStatus.SUCCEEDED | TransferStatus.FAILED | TransferStatus.CANCELED:
                yield GenesisStatus(
                    facility=facility,
                    label=flow_label,
                    action=transfer_label,
                    status=str(status).title(),
                    start_time=start_time,
                    completion_time=result.completion_time or datetime.now(),
                )

                if status == TransferStatus.SUCCEEDED:
                    break
                else:
                    raise RuntimeError(
                        f'Transfer {transfer_label!r} [{transfer_uuid}] ended with status {status!r}'
                    )
            case _:
                yield GenesisStatus(
                    facility=facility,
                    label=flow_label,
                    action=transfer_label,
                    status=str(status).title(),
                    start_time=start_time,
                    completion_time=None,
                )
                stop_event.wait(status_interval_s)


def compute_task(
    client: IRIComputeClient,
    resource_id: str,
    spec: JobSpecification,
    stop_event: threading.Event,
    status_interval_s: float,
    facility: str,
    flow_label: str,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> Iterator[GenesisStatus]:
    start_time = datetime.now()

    for attempt in range(1, max_retries + 1):
        yield GenesisStatus(
            facility=facility,
            label=flow_label,
            action='Submit Job',
            status=f'Attempt {attempt}',
            start_time=start_time,
            completion_time=None,
        )

        try:
            job_response = client.submit_job(resource_id, spec)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                if attempt == max_retries:
                    yield GenesisStatus(
                        facility=facility,
                        label=flow_label,
                        action='Submit Job',
                        status='Failed',
                        start_time=start_time,
                        completion_time=None,
                    )
                    logger.warning('Exceeded max retries.')
                    raise

                time.sleep(retry_delay)
            else:
                raise
        else:
            yield GenesisStatus(
                facility=facility,
                label=flow_label,
                action='Submit Job',
                status='Succeeded',
                start_time=start_time,
                completion_time=None,
            )
            break
    else:
        raise RuntimeError('submit_job_with_retry exited loop without returning')

    start_time = datetime.now()
    job_id = job_response.job_id
    action = f'Job {job_id}'

    while not stop_event.is_set():
        try:
            response = client.get_job_status(resource_id, job_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                logger.warning('Invalid request parameters.')
                break
            else:
                raise

        if response.status is None:
            logger.info(f'{action}: status unknown, waiting...')
            stop_event.wait(status_interval_s)
            continue

        state = response.status.state
        message = response.status.message or ''
        logger.info(f'{action}: {state} {message}'.rstrip())

        match state:
            case JobState.COMPLETED | JobState.FAILED | JobState.CANCELED:
                yield GenesisStatus(
                    facility=facility,
                    label=flow_label,
                    action=action,
                    status=str(state).title(),
                    start_time=start_time,
                    completion_time=datetime.now(),
                )

                if state == JobState.COMPLETED:
                    break
                else:
                    raise RuntimeError(f'{action} ended with state {state!r}: {message}')
            case _:
                yield GenesisStatus(
                    facility=facility,
                    label=flow_label,
                    action=action,
                    status=str(state).title(),
                    start_time=start_time,
                    completion_time=None,
                )
                stop_event.wait(status_interval_s)
