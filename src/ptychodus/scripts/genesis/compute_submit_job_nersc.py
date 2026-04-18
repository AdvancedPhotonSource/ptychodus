import json
import logging

from ptychodus.model.genesis.compute import (
    GenesisComputeClient,
    JobAttributes,
    JobSpecification,
    ResourceSpecification,
)
from ptychodus.model.genesis.facility_adapters import get_genesis_facility_adapters
from ptychodus.model.genesis.tokens import GenesisAccessTokens, get_iri_tokens_file, read_tokens


def get_access_token(api_base_url: str) -> GenesisAccessTokens:
    tokens_file = get_iri_tokens_file()
    tokens = read_tokens(tokens_file)

    for token in tokens:
        if token.api_base_url == api_base_url:
            return token

    raise RuntimeError(f'No access token found for {api_base_url!r} in {tokens_file}.')


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    facilities = get_genesis_facility_adapters()
    facility_name = 'NERSC'
    resource_name = 'Perlmutter'
    facility = facilities[facility_name]
    resource_id = str(facility.compute_resource_ids[resource_name])

    token = get_access_token(facility.iri_api_base_url)
    client = GenesisComputeClient(
        api_base_url=token.api_base_url,
        access_token=token.access_token,
    )

    commands = 'echo BEGIN; which python; which ptychodus; ptychodus --version; echo END'
    job_spec = JobSpecification(
        # executable='/bin/bash',
        # arguments=['-c', commands],
        executable='/bin/hostname',
        resources=ResourceSpecification(
            gpu_cores_per_process=4,
            node_count=1,
            process_count=1,
            processes_per_node=1,
            cpu_cores_per_process=4,
        ),
        attributes=JobAttributes(
            duration=300,
            queue_name='debug',
            account='m5074_g',
        ),
        pre_launch='echo PRE_LAUNCH',
        post_launch='echo POST_LAUNCH',
        launcher='srun',
    )

    logger.info(
        'Submitting job to %s / %s (resource %s)', facility_name, resource_name, resource_id
    )
    response = client.submit_job_with_retry(resource_id, job_spec)

    print(f'Job ID: {response.job_id}')

    if response.status is not None:
        print(f'State:  {response.status.state}')

        if response.status.message:
            print(f'Message: {response.status.message}')

    print(json.dumps(response.model_dump(mode='json', by_alias=True), indent=2))


if __name__ == '__main__':
    main()
