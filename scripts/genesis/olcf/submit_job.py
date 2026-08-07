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

    # NOTE: Replace OLCF_ALLOCATION below with your own OLCF project allocation
    # before running this example.
    olcf_allocation = 'OLCF_ALLOCATION'

    for name, adapter in adapters.items():
        if name != 'OLCF':
            continue

        logger.info(f'Checking IRI access token for facility "{name}"...')

        client = adapter.get_iri_client()
        resource_id = '70e0dde0-88e4-52e3-89f3-4849760f2e87'
        job_spec = JobSpecification(
            executable='ptychodus',
            arguments=['-v'],
            name='ptychodus',
            directory=f'/gpfs/wolf2/olcf/{olcf_allocation}/proj-shared',
            resources=ResourceSpecification(
                node_count=1,
                process_count=1,
                processes_per_node=1,
                cpu_cores_per_process=1,
            ),
            attributes=JobAttributes(
                duration=300,
                queue_name='batch',
                account=olcf_allocation,
            ),
            pre_launch=f'source /etc/bash.bashrc; module load miniforge3; conda activate /ccsopen/proj/{olcf_allocation}/ptychodus-env',
            post_launch='echo POST_LAUNCH',
            launcher='srun',
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
