import json
import logging

from ptychodus.model.genesis.iri import (
    JobAttributes,
    JobSpecification,
    ResourceSpecification,
)
from ptychodus.model.genesis.core import create_facility_adapters

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    adapters = create_facility_adapters()

    for name, adapter in adapters.items():
        if name != 'NERSC':
            continue

        logger.info(f'Checking IRI access token for facility "{name}"...')

        client = adapter.get_iri_client()
        resource_id = '94351904-6dba-4c16-b5cd-fbd280d8615b'  # Perlmutter
        # FIXME commands = 'source $HOME/.local/bin/env; which python; which ptychodus; ptychodus --version; nvidia-smi'
        commands = 'echo BEGIN; nvidia-smi; echo END'
        job_spec = JobSpecification(
            executable='/bin/bash',
            arguments=['-c', commands],
            directory='/global/homes/s/shenke',
            name='Ptychodus',
            stdout_path='/global/homes/s/shenke/outputs',
            stderr_path='/global/homes/s/shenke/outputs',
            resources=ResourceSpecification(
                node_count=1,
                process_count=1,
                processes_per_node=1,
                cpu_cores_per_process=1,
                gpu_cores_per_process=4,
            ),
            attributes=JobAttributes(
                duration=1800,
                queue_name='debug',
                # FIXME account='m5074',
                account='m5074_g',
            ),
            # FIXME launcher='single',
            # FIXME launcher='srun',
        )
        print(job_spec.model_dump_json(by_alias=True, indent=2))
        response = client.compute.submit_job(resource_id, job_spec)

        print(f'Job ID: {response.job_id}')

        if response.status is not None:
            print(f'State:  {response.status.state}')

            if response.status.message:
                print(f'Message: {response.status.message}')

        print(json.dumps(response.model_dump(mode='json', by_alias=True), indent=2))


if __name__ == '__main__':
    main()
