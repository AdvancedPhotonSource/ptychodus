#!/usr/bin/env python

import json
import logging

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.genesis.core import create_facility_adapters
from ptychodus.model.genesis.iri import (
    JobAttributes,
    JobSpecification,
    ResourceSpecification,
)
from ptychodus.model.genesis.settings import GenesisSettings

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings_registry = SettingsRegistry()
    settings = GenesisSettings(settings_registry)
    adapters = create_facility_adapters(settings)

    for name, adapter in adapters.items():
        if name != 'OLCF':
            continue

        logger.info(f'Checking IRI access token for facility "{name}"...')

        client = adapter.get_iri_client()
        resource_id = 'odo'
        commands = 'echo BEGIN; module load miniforge3; conda activate ptychodus; which python; which ptychodus; ptychodus --version; echo END'
        job_spec = JobSpecification(
            executable='/bin/bash',
            arguments=['-c', commands],
            name='Ptychodus',
            directory='/gpfs/wolf2/olcf/csc682/proj-shared',
            environment={'DUMMY': '1'},
            resources=ResourceSpecification(
                node_count=1,
                process_count=1,
                cpu_cores_per_process=1,
            ),
            attributes=JobAttributes(
                duration=300,
                queue_name='batch',
                account='csc682',
            ),
        )
        response = client.compute.submit_job(resource_id, job_spec)

        print(f'Job ID: {response.job_id}')

        if response.status is not None:
            print(f'State:  {response.status.state}')

            if response.status.message:
                print(f'Message: {response.status.message}')

        print(json.dumps(response.model_dump(mode='json', by_alias=True), indent=2))


if __name__ == '__main__':
    main()
