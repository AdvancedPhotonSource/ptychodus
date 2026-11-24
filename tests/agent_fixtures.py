"""What the agent service's tests share: a product to serve, a scripted language model, and
the AG-UI request and response as the web UI sees them. pydantic-ai is imported here, so
only tests/agent/ imports this module."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata

USER_HEADER = 'X-Forwarded-User'
"""The header the agent's tests configure as the proxy's user header."""


@dataclass(frozen=True)
class Webapp:
    """The webapp a test's agent uses: its MCP server, the agent's token, and the one
    product its storage root holds."""

    mcp_url: str
    token: str
    product_uuid: UUID


def _tool_results_this_turn(messages: Sequence[ModelMessage]) -> int:
    count = 0

    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            count += sum(
                isinstance(part, ToolReturnPart | RetryPromptPart) for part in message.parts
            )

            if any(isinstance(part, UserPromptPart) for part in message.parts):
                break

    return count


class ScriptedLanguageModel:
    """A language model that, every turn, calls `tool_calls` in order, one per response,
    then streams `text_chunks`. It keeps the history it was shown on each request."""

    def __init__(
        self, tool_calls: Sequence[tuple[str, dict[str, Any]]], text_chunks: Sequence[str]
    ) -> None:
        self.tool_calls = list(tool_calls)
        self.text_chunks = list(text_chunks)
        self.requests: list[list[ModelMessage]] = []

    async def stream(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        self.requests.append(list(messages))
        step = _tool_results_this_turn(messages)

        if step < len(self.tool_calls):
            name, args = self.tool_calls[step]
            yield {
                0: DeltaToolCall(name, json.dumps(args), tool_call_id=f'call-{len(self.requests)}')
            }
        else:
            for chunk in self.text_chunks:
                yield chunk

    def model(self) -> FunctionModel:
        return FunctionModel(stream_function=self.stream, model_name='scripted')


def sample_product() -> Product:
    """A small product with a Gaussian probe, for the webapp to serve."""
    pixel_geometry = PixelGeometry(width_m=1e-8, height_m=1e-8)
    yy, xx = numpy.mgrid[:16, :16] - 7.5
    probe_array = numpy.exp(-(yy**2 + xx**2) / 8.0).astype(complex)
    positions = [
        ProbePosition(index=index, x_m=1e-8 * (index % 3), y_m=1e-8 * (index // 3))
        for index in range(9)
    ]
    return Product(
        metadata=ProductMetadata(
            name='agent-test',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(
            array=probe_array[numpy.newaxis, :, :], opr_weights=None, pixel_geometry=pixel_geometry
        ),
        object_=Object(
            array=numpy.ones((32, 32), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def run_input(conversation_id: str, prompt: str, *, message_id: str = 'u1') -> dict[str, Any]:
    """An AG-UI run input carrying one new user message, as the web UI sends it."""
    return {
        'threadId': conversation_id,
        'runId': f'run-{message_id}',
        'messages': [{'id': message_id, 'role': 'user', 'content': prompt}],
        'tools': [],
        'context': [],
    }


def read_events(body: str) -> list[dict[str, Any]]:
    """The AG-UI events of an SSE body."""
    return [
        json.loads(line.removeprefix('data:'))
        for line in body.splitlines()
        if line.startswith('data:')
    ]


def headers(user: str = 'jdoe') -> dict[str, str]:
    return {USER_HEADER: user}
