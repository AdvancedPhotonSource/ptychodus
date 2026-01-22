from __future__ import annotations
from collections.abc import Mapping, Sequence
from pprint import pformat
from typing import Final, TypeAlias
import logging
import os

import fair_research_login
import gladier
import gladier.managers
import globus_sdk

AuthorizerTypes: TypeAlias = globus_sdk.AccessTokenAuthorizer | globus_sdk.RefreshTokenAuthorizer
ScopeAuthorizerMapping: TypeAlias = Mapping[str, AuthorizerTypes]

PTYCHODUS_CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

logger = logging.getLogger(__name__)


def process_with_ptychodus(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelCore

    action = data['ptychodus_action']
    input_directory = Path(data['ptychodus_input_directory'])
    output_directory = Path(data['ptychodus_output_directory'])

    with ModelCore() as model:
        model.batch_mode_execute(action, input_directory, output_directory)


@gladier.generate_flow_definition
class PtychodusProcessingTool(gladier.GladierBaseTool):
    compute_functions = [process_with_ptychodus]
    required_input = [
        'ptychodus_action',
        'ptychodus_input_directory',
        'ptychodus_output_directory',
    ]


@gladier.generate_flow_definition
class PtychodusClient(gladier.GladierBaseClient):
    client_id = PTYCHODUS_CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:InputData',
        PtychodusProcessingTool,
        'gladier_tools.globus.transfer.Transfer:OutputData',
        # TODO 'gladier_tools.publish.Publish',
    ]

    @classmethod
    def _create_confidential_client(cls, client_id: str, client_secret: str) -> PtychodusClient:
        auth_client = globus_sdk.ConfidentialAppAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            app_name='Ptychodus',
        )

        def _request_auth(scopes: list[str]) -> ScopeAuthorizerMapping:
            logger.debug(f'Requested authorization scopes: {pformat(scopes)}')

            response = auth_client.oauth2_client_credentials_tokens(requested_scopes=scopes)
            return {
                scope: globus_sdk.AccessTokenAuthorizer(access_token=tokens['access_token'])
                for scope, tokens in response.by_scopes.scope_map.items()
            }

        initial_authorizers: dict[str, AuthorizerTypes] = dict()
        login_manager = gladier.managers.CallbackLoginManager(
            authorizers=initial_authorizers,
            callback=_request_auth,
        )

        flow_id = os.environ.get('FLOW_ID', '')

        if not flow_id:
            # This isn't necessarily bad, but confidential clients (like regular
            # users) only get one flow to play with. They probably don't need
            # more than one, but this will ensure there aren't errors due to
            # tracking mismatch in the Glaider config.
            logger.warning('No flow ID enforced. Recommend setting FLOW_ID environment variable.')

        flows_manager = gladier.managers.FlowsManager(flow_id=flow_id)
        return cls(login_manager=login_manager, flows_manager=flows_manager)

    @classmethod
    def _create_native_client(
        cls, code_handlers: Sequence[fair_research_login.CodeHandler]
    ) -> PtychodusClient:
        auth_client = fair_research_login.NativeClient(
            client_id=PTYCHODUS_CLIENT_ID,
            app_name='Ptychodus',
            code_handlers=code_handlers,
        )

        def _request_auth(self, requested_scopes: list[str]) -> ScopeAuthorizerMapping:
            logger.debug(f'Requested authorization scopes: {pformat(requested_scopes)}')

            # 'force' is used for any underlying scope changes. For example, if a flow adds transfer
            # functionality since it was last run, running it again would require a re-login.
            auth_client.login(requested_scopes=requested_scopes, force=True, refresh_tokens=True)
            return auth_client.get_authorizers_by_scope()  # type: ignore

        initial_authorizers: dict[str, AuthorizerTypes] = dict()

        try:
            # Try to use a previous login to avoid a new login flow
            initial_authorizers = auth_client.get_authorizers_by_scope()  # type: ignore
        except fair_research_login.LoadError:
            pass

        login_manager = gladier.managers.CallbackLoginManager(
            authorizers=initial_authorizers,
            callback=_request_auth,
        )
        return cls(login_manager=login_manager)

    @classmethod
    def create_client(
        cls, code_handlers: Sequence[fair_research_login.CodeHandler]
    ) -> PtychodusClient:
        try:
            client_id = os.environ['CLIENT_ID']
            client_secret = os.environ['CLIENT_SECRET']
        except KeyError:
            logger.info(
                'Environment variables "CLIENT_ID" and "CLIENT_SECRET" not set. '
                'Creating native PtychodusClient.'
            )
            return cls._create_native_client(code_handlers)
        else:
            logger.info(
                'Environment variables "CLIENT_ID" and "CLIENT_SECRET" set. '
                'Creating confidential PtychodusClient.'
            )
            return cls._create_confidential_client(client_id, client_secret)
