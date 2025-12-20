import time
from collections.abc import AsyncGenerator
from typing import cast

import anyio
from anyio import create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteInstanceResponse,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ChatCompletion,
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    PlaceInstance,
    TaskFinished,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import ChunkGenerated, Event, ForwarderEvent, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.dashboard_path import find_dashboard
from exo.utils.event_buffer import OrderedBuffer

"""REST API server for EXO cluster.

This module provides the FastAPI-based REST API for interacting with the
EXO cluster, including chat completions, instance management, and state queries.
"""

import time
from collections.abc import AsyncGenerator
from typing import cast

import anyio
from anyio import create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteInstanceResponse,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ChatCompletion,
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    PlaceInstance,
    TaskFinished,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import ChunkGenerated, Event, ForwarderEvent, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.dashboard_path import find_dashboard
from exo.utils.event_buffer import OrderedBuffer

HIDE_THINKING = False


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a token chunk to a chat completion response.

    Args:
        chunk: Token chunk from generation.
        command_id: Command ID for this completion.

    Returns:
        ChatCompletionResponse formatted for streaming.
    """
    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def resolve_model_meta(model_id: str) -> ModelMetadata:
    """Resolve model metadata from model ID.

    Checks model cards first, then falls back to fetching from Hugging Face.

    Args:
        model_id: Model identifier.

    Returns:
        ModelMetadata for the model.
    """
    if model_id in MODEL_CARDS:
        model_card = MODEL_CARDS[model_id]
        return model_card.metadata
    else:
        return await get_model_meta(model_id)


class API:
    """REST API server for EXO cluster interactions.

    Provides HTTP endpoints for:
    - Chat completions (OpenAI-compatible)
    - Instance management (create, delete, placement)
    - Model listing
    - State queries
    - Dashboard static files

    Maintains local state by applying events from the master, and can pause
    during elections to avoid serving stale state.

    Attributes:
        state: Local cluster state (derived from events).
        node_id: Node ID of this node.
        session_id: Current session ID.
        port: Port number for the HTTP server.
        paused: Whether the API is currently paused (during elections).
        app: FastAPI application instance.
        _event_log: Local event log for debugging/querying.
        _chat_completion_queues: Queues for streaming chat completions.
        _tg: Task group for background operations.
    """

    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        port: int,
        global_event_receiver: Receiver[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
        election_receiver: Receiver[ElectionMessage],
    ) -> None:
        """Initialize the API server.

        Args:
            node_id: Node ID of this node.
            session_id: Initial session ID.
            port: Port number to bind the HTTP server to.
            global_event_receiver: Channel to receive indexed events from master.
            command_sender: Channel to send commands to master.
            election_receiver: Channel to receive election messages (for pausing).
        """
        self.state = State()
        self._event_log: list[Event] = []
        self.command_sender = command_sender
        self.global_event_receiver = global_event_receiver
        self.election_receiver = election_receiver
        self.event_buffer: OrderedBuffer[Event] = OrderedBuffer[Event]()
        self.node_id = node_id
        self.session_id = session_id
        self.last_completed_election: int = 0
        self.port = port

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=find_dashboard(),
                html=True,
            ),
            name="dashboard",
        )

        self._chat_completion_queues: dict[CommandId, Sender[TokenChunk]] = {}
        self._tg: TaskGroup | None = None

    def reset(self, new_session_id: SessionId, result_clock: int) -> None:
        """Reset API state for a new session.

        Clears state, event log, and chat queues, then unpauses with the
        new session ID. Called when a new master is elected.

        Args:
            new_session_id: New session ID.
            result_clock: Election clock value.
        """
        logger.info("Resetting API State")
        self.state = State()
        self.session_id = new_session_id
        self.event_buffer = OrderedBuffer[Event]()
        self._chat_completion_queues = {}
        self.unpause(result_clock)

    def unpause(self, result_clock: int) -> None:
        """Unpause the API after an election completes.

        Args:
            result_clock: Election clock value for the completed election.
        """
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_cors(self) -> None:
        """Configure CORS middleware to allow all origins."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Register all API endpoints with FastAPI.

        Sets up routes for:
        - GET /node_id: Get this node's ID
        - POST /instance: Create instance with explicit config
        - POST /place_instance: Place instance with automatic placement
        - GET /instance/placement: Get placement preview
        - GET /instance/previews: Get placement previews for all configs
        - GET /instance/{instance_id}: Get instance details
        - DELETE /instance/{instance_id}: Delete instance
        - GET /models, /v1/models: List available models
        - POST /v1/chat/completions: Generate chat completion
        - GET /state: Get cluster state
        - GET /events: Get event log
        """
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/state")(lambda: self.state)
        self.app.get("/events")(lambda: self._event_log)

    async def place_instance(self, payload: PlaceInstanceParams) -> CreateInstanceResponse:
        """Place an instance with automatic placement calculation.

        Args:
            payload: Placement parameters including model ID, sharding, etc.

        Returns:
            Response with command ID for tracking.
        """
        command = PlaceInstance(
            model_meta=await resolve_model_meta(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        """Create an instance with explicit configuration.

        Args:
            payload: Instance configuration to create.

        Returns:
            Response with command ID for tracking.
        """
        command = CreateInstance(instance=payload.instance)
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
        )

    async def get_placement(
        self,
        model_id: str,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        """Get placement preview for a model without creating the instance.

        Calculates where an instance would be placed but doesn't actually
        create it. Useful for previewing placement decisions.

        Args:
            model_id: Model identifier.
            sharding: Sharding strategy to use.
            instance_meta: Instance metadata/type.
            min_nodes: Minimum nodes required.

        Returns:
            Instance configuration that would be created.

        Raises:
            HTTPException: If placement fails or produces unexpected results.
        """
        model_meta = await resolve_model_meta(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_meta=model_meta,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                topology=self.state.topology,
                current_instances=self.state.instances,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self, model_id: ModelId
    ) -> PlacementPreviewResponse:
        """Get placement previews for all sharding/instance combinations.

        Calculates placements for a model across all combinations of:
        - Sharding strategies (Pipeline, Tensor)
        - Instance types (MlxRing, MlxJaccl)
        - Minimum node counts (1 to number of nodes)

        Returns previews showing where each configuration would be placed,
        including memory deltas per node and any placement errors.

        Args:
            model_id: Model identifier to get previews for.

        Returns:
            PlacementPreviewResponse with all placement previews.

        Raises:
            HTTPException: If model not found (404) or no nodes in topology.
        """
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        cards = [card for card in MODEL_CARDS.values() if card.short_id == model_id]
        if not cards:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for card in cards:
            model_meta = card.metadata
            for sharding, instance_meta, min_nodes in instance_combinations:
                try:
                    placements = get_instance_placements(
                        PlaceInstance(
                            model_meta=model_meta,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            min_nodes=min_nodes,
                        ),
                        topology=self.state.topology,
                        current_instances=self.state.instances,
                    )
                except ValueError as exc:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error=str(exc),
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                current_ids = set(self.state.instances.keys())
                new_instances = [
                    instance
                    for instance_id, instance in placements.items()
                    if instance_id not in current_ids
                ]

                if len(new_instances) != 1:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error="Expected exactly one new instance from placement",
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                instance = new_instances[0]
                shard_assignments = instance.shard_assignments
                node_ids = list(shard_assignments.node_to_runner.keys())

                memory_delta_by_node: dict[str, int] = {}
                if node_ids:
                    total_bytes = model_meta.storage_size.in_bytes
                    per_node = total_bytes // len(node_ids)
                    remainder = total_bytes % len(node_ids)
                    for index, node_id in enumerate(sorted(node_ids, key=str)):
                        extra = 1 if index < remainder else 0
                        memory_delta_by_node[str(node_id)] = per_node + extra

                if (
                    card.model_id,
                    sharding,
                    instance_meta,
                    len(node_ids),
                ) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=instance,
                            memory_delta_by_node=memory_delta_by_node or None,
                            error=None,
                        )
                    )
                seen.add((card.model_id, sharding, instance_meta, len(node_ids)))

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        """Get instance details by ID.

        Args:
            instance_id: Instance identifier.

        Returns:
            Instance configuration.

        Raises:
            HTTPException: If instance not found (404).
        """
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        """Delete an instance.

        Args:
            instance_id: Instance identifier to delete.

        Returns:
            Response with command ID for tracking.

        Raises:
            HTTPException: If instance not found (404).
        """
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def _generate_chat_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as Server-Sent Events (SSE).

        Receives token chunks from the worker and formats them as SSE messages.
        Handles thinking/reasoning tokens if HIDE_THINKING is enabled.

        Args:
            command_id: Command ID for this completion.

        Yields:
            SSE-formatted JSON strings (data: {...}\n\n).

        Note:
            Sends TaskFinished command when stream completes or is cancelled.
        """
        try:
            self._chat_completion_queues[command_id], recv = channel[TokenChunk]()

            is_thinking = False
            with recv as token_chunks:
                async for chunk in token_chunks:
                    if HIDE_THINKING:
                        if chunk.text == "<think>":
                            is_thinking = True
                        if chunk.text == "</think>":
                            is_thinking = False
                    chunk_response: ChatCompletionResponse = chunk_to_response(
                        chunk, command_id
                    )
                    if not (is_thinking and HIDE_THINKING):
                        logger.debug(f"chunk_response: {chunk_response}")
                        yield f"data: {chunk_response.model_dump_json()}\n\n"

                    if chunk.finish_reason is not None:
                        yield "data: [DONE]\n\n"
                        break

        except anyio.get_cancelled_exc_class():
            raise
        finally:
            command = TaskFinished(finished_command_id=command_id)
            await self._send(command)
            del self._chat_completion_queues[command_id]

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        """Trigger notification to user about missing model.

        Args:
            model_id: Model ID that is not available.

        Note:
            Currently just logs a warning. TODO: Implement actual notification.
        """
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def chat_completions(
        self, payload: ChatCompletionTaskParams
    ) -> StreamingResponse:
        """Handle chat completion request (OpenAI-compatible endpoint).

        Validates that an instance exists for the requested model, sends
        a ChatCompletion command, and returns a streaming SSE response.

        Args:
            payload: Chat completion parameters (messages, temperature, etc.).

        Returns:
            StreamingResponse with Server-Sent Events containing token chunks.

        Raises:
            HTTPException: If no instance exists for the requested model (404).
        """
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        if not any(
            instance.shard_assignments.model_id == payload.model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {payload.model}"
            )

        command = ChatCompletion(
            request_params=payload,
        )
        await self._send(command)
        return StreamingResponse(
            self._generate_chat_stream(command.command_id),
            media_type="text/event-stream",
        )

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes.

        Sums up RAM available from all nodes in the topology.

        Returns:
            Total available memory as a Memory object.
        """
        total_available = Memory()

        for node in self.state.topology.list_nodes():
            if node.node_profile is not None:
                total_available += node.node_profile.memory.ram_available

        return total_available

    async def get_models(self) -> ModelList:
        """Get list of available models.

        Returns:
            ModelList containing all models from model cards.
        """
        return ModelList(
            data=[
                ModelListModel(
                    id=card.short_id,
                    hugging_face_id=card.model_id,
                    name=card.name,
                    description=card.description,
                    tags=card.tags,
                )
                for card in MODEL_CARDS.values()
            ]
        )

    async def run(self) -> None:
        """Run the API server.

        Starts background tasks for state application and election monitoring,
        then starts the Hypercorn HTTP server. Blocks until shutdown.
        """
        cfg = Config()
        cfg.bind = f"0.0.0.0:{self.port}"
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        async with create_task_group() as tg:
            self._tg = tg
            logger.info("Starting API")
            tg.start_soon(self._applystate)
            tg.start_soon(self._pause_on_new_election)
            print_startup_banner(self.port)
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=lambda: anyio.sleep_forever(),
            )

        self.command_sender.close()
        self.global_event_receiver.close()

    async def _applystate(self) -> None:
        """Apply indexed events to local state.

        Receives indexed events from the master, orders them using OrderedBuffer,
        and applies them to the local state. Also routes ChunkGenerated events
        to chat completion queues for streaming responses.
        """
        with self.global_event_receiver as events:
            async for f_event in events:
                if f_event.origin != self.session_id.master_node_id:
                    continue
                self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                for idx, event in self.event_buffer.drain_indexed():
                    self._event_log.append(event)
                    self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                    if (
                        isinstance(event, ChunkGenerated)
                        and event.command_id in self._chat_completion_queues
                    ):
                        assert isinstance(event.chunk, TokenChunk)
                        await self._chat_completion_queues[event.command_id].send(
                            event.chunk
                        )

    async def _pause_on_new_election(self) -> None:
        """Monitor election messages and pause API during new elections.

        Pauses the API when a new election starts (clock > last_completed_election)
        to avoid serving stale state during master transitions.
        """
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _send(self, command: Command) -> None:
        """Send a command to the master, waiting if API is paused.

        Blocks until the API is unpaused (after election completes), then
        sends the command wrapped in a ForwarderCommand.

        Args:
            command: Command to send to the master.
        """
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self.node_id, command=command)
        )
