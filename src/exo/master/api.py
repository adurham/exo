import time
from collections.abc import AsyncGenerator
from typing import cast

import anyio
from anyio import create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException, Request
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
from exo.shared.types.worker.shards import PipelineShardMetadata
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
    CancelTask,
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
    if model_id in MODEL_CARDS:
        model_card = MODEL_CARDS[model_id]
        return model_card.metadata
    else:
        return await get_model_meta(model_id)


class API:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        port: int,
        # Ideally this would be a MasterForwarderEvent but type system says no :(
        global_event_receiver: Receiver[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
        # Election receiver is optional (not used in static setup)
        election_receiver: Receiver[ElectionMessage] | None = None,
    ) -> None:
        self.state = State()
        self._event_log: list[Event] = []
        self.command_sender = command_sender
        self.global_event_receiver = global_event_receiver
        self.election_receiver = election_receiver
        self.event_buffer: OrderedBuffer[Event] = OrderedBuffer[Event]()
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id
        self.port = port
        # Reference to Master for direct state access (will be set in master_app)
        self._master_ref = None

        self.app = FastAPI()
        self._setup_cors()
        self._setup_routes()

        # Mount dashboard if available (optional for static setup)
        try:
            dashboard_dir = find_dashboard()
            self.app.mount(
                "/",
                StaticFiles(
                    directory=dashboard_dir,
                    html=True,
                ),
                name="dashboard",
            )
        except FileNotFoundError:
            # Dashboard not available - API will still work, just no web UI
            logger.warning("Dashboard not found - web UI will not be available. API endpoints are still functional.")

        self._chat_completion_queues: dict[CommandId, Sender[TokenChunk]] = {}
        self._tg: TaskGroup | None = None

    @property
    def _state(self) -> State:
        """Get the current state, preferring master's state if available."""
        return self._master_ref.state if self._master_ref else self.state

    def reset(self, new_session_id: SessionId, result_clock: int):
        logger.info("Resetting API State")
        self.state = State()
        self.session_id = new_session_id
        self.event_buffer = OrderedBuffer[Event]()
        self._chat_completion_queues = {}

    def unpause(self, result_clock: int):
        """No-op in static setup (no elections)."""
        pass

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
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
        self.app.post("/v1/chat/completions/{command_id}/cancel")(self.cancel_chat_completion)
        self.app.get("/state")(lambda: self._state)
        self.app.get("/events")(lambda: self._event_log)
        # Add endpoint for workers to send events via HTTP
        self.app.post("/events")(self.receive_event)

    async def place_instance(self, payload: PlaceInstanceParams):
        # CRITICAL: Only allow one instance at a time
        # Delete all existing instances first before creating a new one
        existing_instances = list(self._state.instances.keys())
        if existing_instances:
            logger.info(f"Deleting {len(existing_instances)} existing instance(s) before creating new one")
            for instance_id in existing_instances:
                delete_command = DeleteInstance(
                    instance_id=instance_id,
                )
                await self._send(delete_command)
                logger.info(f"Sent delete command for instance {instance_id}")
            
            # Wait for deletion to complete by polling state (up to 30 seconds)
            # This ensures memory is freed before creating new instance
            import asyncio
            max_wait = 30
            waited = 0
            while waited < max_wait and len(self._state.instances) > 0:
                await asyncio.sleep(1)
                waited += 1
                if len(self._state.instances) == 0:
                    logger.info(f"All instances deleted after {waited}s, proceeding with new instance creation")
                    break
            if len(self._state.instances) > 0:
                logger.warning(f"Some instances still exist after {waited}s wait, proceeding anyway")
        
        # Always use all available nodes - override min_nodes from payload
        total_nodes = len(list(self._state.topology.list_nodes()))
        min_nodes_to_use = total_nodes if total_nodes > 0 else payload.min_nodes
        
        command = PlaceInstance(
            model_meta=await resolve_model_meta(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=min_nodes_to_use,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
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
        min_nodes: int | None = None,
    ) -> Instance:
        model_meta = await resolve_model_meta(model_id)
        
        # Always use all available nodes if min_nodes not specified
        if min_nodes is None:
            min_nodes = len(list(self._state.topology.list_nodes()))

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_meta=model_meta,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                topology=self._state.topology,
                current_instances=self._state.instances,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self._state.instances.keys())
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
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        if len(list(self._state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        cards = [card for card in MODEL_CARDS.values() if card.short_id == model_id]
        if not cards:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        total_nodes = len(list(self._state.topology.list_nodes()))
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                # Only generate one preview per (sharding, instance_meta) combination
                # Use max min_nodes to get the greedy allocation across all nodes
                instance_combinations.append((sharding, instance_meta, total_nodes))
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
                        topology=self._state.topology,
                        current_instances=self._state.instances,
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

                current_ids = set(self._state.instances.keys())
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
                
                logger.info(
                    f"Placement preview: model={model_meta.model_id}, "
                    f"nodes_in_instance={len(node_ids)}, node_ids={node_ids}"
                )

                memory_delta_by_node: dict[str, int] = {}
                if node_ids:
                    total_bytes = model_meta.storage_size.in_bytes
                    total_layers = model_meta.n_layers
                    
                    if total_layers > 0:
                        bytes_per_layer = total_bytes / total_layers
                        
                        for node_id in node_ids:
                            runner_id = shard_assignments.node_to_runner.get(node_id)
                            if runner_id is None:
                                # Node is in instance but has no runner (shouldn't happen, but handle gracefully)
                                memory_delta_by_node[str(node_id)] = 0
                                logger.warning(f"Node {node_id} has no runner_id, setting memory_delta to 0")
                                continue
                                
                            shard_meta = shard_assignments.runner_to_shard.get(runner_id)
                            
                            if shard_meta is not None:
                                if isinstance(shard_meta, PipelineShardMetadata):
                                    layers_per_node = shard_meta.end_layer - shard_meta.start_layer
                                    # CRITICAL: If layers_per_node is 0, node_bytes MUST be 0 (KV cache only)
                                    if layers_per_node == 0:
                                        node_bytes = 0
                                    else:
                                        node_bytes = int(layers_per_node * bytes_per_layer)
                                    logger.info(
                                        f"Node {node_id}: start={shard_meta.start_layer}, end={shard_meta.end_layer}, "
                                        f"layers={layers_per_node}, bytes={node_bytes} ({node_bytes / (1024**3):.2f} GB), "
                                        f"KV_CACHE={layers_per_node == 0}"
                                    )
                                else:
                                    node_bytes = total_bytes // len(node_ids)
                            else:
                                node_bytes = 0
                                logger.warning(f"Node {node_id} has no shard_meta, setting memory_delta to 0")
                            
                            # EXPLICITLY ensure 0 is stored as integer 0, not float or None
                            memory_delta_by_node[str(node_id)] = int(node_bytes) if node_bytes is not None else 0
                            
                        # Log final memory_delta_by_node for debugging
                        logger.info(
                            f"Final memory_delta_by_node: {memory_delta_by_node}"
                        )
                    else:
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
        if instance_id not in self._state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self._state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self._state.instances:
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
        """Generate chat completion stream as JSON strings."""

        try:
            self._chat_completion_queues[command_id], recv = channel[TokenChunk]()
            logger.info(f"API: Created streaming queue for command_id={command_id}")

            is_thinking = False
            logger.info(f"API: Starting to consume chunks for command_id={command_id}")
            chunk_count = 0
            with recv as token_chunks:
                async for chunk in token_chunks:
                    chunk_count += 1
                    logger.info(
                        f"API: Received chunk #{chunk_count} for command_id={command_id}, "
                        f"text='{chunk.text[:50] if chunk.text else 'N/A'}', "
                        f"finish_reason={chunk.finish_reason}"
                    )
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
                        response_str = f"data: {chunk_response.model_dump_json()}\n\n"
                        logger.info(f"API: Yielding chunk #{chunk_count} for command_id={command_id}")
                        yield response_str

                    if chunk.finish_reason is not None:
                        logger.info(f"API: Finished streaming for command_id={command_id}, total chunks={chunk_count}")
                        yield "data: [DONE]\n\n"
                        break

        except anyio.get_cancelled_exc_class():
            # TODO: TaskCancelled
            """
            self.command_sender.send_nowait(
                ForwarderCommand(origin=self.node_id, command=command)
            )
            """
            raise
        finally:
            command = TaskFinished(finished_command_id=command_id)
            await self._send(command)
            del self._chat_completion_queues[command_id]

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def chat_completions(
        self, payload: ChatCompletionTaskParams
    ) -> StreamingResponse:
        """Handle chat completions with proper streaming response."""
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        if not any(
            instance.shard_assignments.model_id == payload.model
            for instance in self._state.instances.values()
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

    async def cancel_chat_completion(self, command_id: str):
        """Cancel a running chat completion task."""
        from exo.shared.types.common import CommandId

        target_command_id = CommandId(command_id)

        if target_command_id not in self._chat_completion_queues:
            raise HTTPException(
                status_code=404,
                detail=f"No active chat completion found for command_id {command_id}",
            )

        # Close the queue to stop streaming
        queue_sender = self._chat_completion_queues[target_command_id]
        await queue_sender.aclose()
        del self._chat_completion_queues[target_command_id]

        # Send cancel command to worker
        command = CancelTask(target_command_id=target_command_id)
        await self._send(command)

        return {"message": "Chat completion cancelled", "command_id": command_id}

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for node in self._state.topology.list_nodes():
            if node.node_profile is not None:
                total_available += node.node_profile.memory.ram_available

        return total_available

    async def get_models(self) -> ModelList:
        """Returns list of available models."""
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

    async def run(self):
        cfg = Config()
        cfg.bind = f"0.0.0.0:{self.port}"
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        async with create_task_group() as tg:
            self._tg = tg
            logger.info("Starting API")
            tg.start_soon(self._applystate)
            # No election pausing in static setup
            print_startup_banner(self.port)
            try:
                await serve(
                    cast(ASGIFramework, self.app),
                    cfg,
                    shutdown_trigger=lambda: anyio.sleep_forever(),
                )
            except OSError as exc:
                if exc.errno == 48 or "Address already in use" in str(exc):
                    logger.error(
                        f"Port {self.port} is already in use. "
                        f"Another process may be using this port, or a previous EXO instance may not have shut down cleanly. "
                        f"Please choose a different port with --api-port or stop the process using port {self.port}."
                    )
                raise

        self.command_sender.close()
        self.global_event_receiver.close()
    
    async def receive_event(self, request: Request):
        """Receive an event from a worker via HTTP."""
        from exo.shared.types.events import ForwarderEvent, IndexedEvent
        from exo.shared.apply import apply
        
        try:
            # Parse the event from the request body
            event_data = await request.json()
            forwarder_event = ForwarderEvent.model_validate(event_data)
            
            # Apply the event to Master's state directly
            if self._master_ref is not None:
                # Use origin_idx from ForwarderEvent, but map to master's event log sequence
                # For HTTP events, we use a per-worker counter to ensure sequential ordering
                if not hasattr(self._master_ref, '_http_event_counters'):
                    self._master_ref._http_event_counters: dict[NodeId, int] = {}
                worker_counter = self._master_ref._http_event_counters.get(forwarder_event.origin, -1) + 1
                self._master_ref._http_event_counters[forwarder_event.origin] = worker_counter
                
                # Map to master's global event index
                master_idx = len(self._master_ref._event_log)
                indexed = IndexedEvent(event=forwarder_event.event, idx=master_idx)
                self._master_ref.state = apply(self._master_ref.state, indexed)
                self._master_ref._event_log.append(forwarder_event.event)
            else:
                # Fallback: apply to API's own state
                indexed = IndexedEvent(event=forwarder_event.event, idx=len(self._event_log))
                if not hasattr(self, '_master_state'):
                    from exo.shared.static_config import create_static_topology
                    static_topology = create_static_topology()
                    self._master_state = State(topology=static_topology)
                self._master_state = apply(self._master_state, indexed)
                self._event_log.append(forwarder_event.event)
            
            logger.info(f"Received event {forwarder_event.event.__class__.__name__} from {forwarder_event.origin}")
            
            return {"status": "ok", "event_id": str(forwarder_event.event.event_id)}
        except Exception as e:
            logger.error(f"Error processing event from worker: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    async def _applystate(self):
        with self.global_event_receiver as events:
            async for f_event in events:
                # Log ChunkGenerated events to debug streaming
                if isinstance(f_event.event, ChunkGenerated):
                    logger.info(
                        f"API received ChunkGenerated: origin={f_event.origin}, "
                        f"master_node_id={self.session_id.master_node_id}, "
                        f"command_id={f_event.event.command_id}, "
                        f"queues={list(self._chat_completion_queues.keys())}"
                    )
                
                if f_event.origin != self.session_id.master_node_id:
                    continue
                
                # For ChunkGenerated events, process immediately to enable streaming
                # Don't wait for event ordering - tokens need to be streamed as soon as they're generated
                if isinstance(f_event.event, ChunkGenerated):
                    chunk_event = f_event.event
                    logger.info(
                        f"API processing ChunkGenerated: command_id={chunk_event.command_id}, "
                        f"text='{chunk_event.chunk.text[:50] if chunk_event.chunk.text else 'N/A'}'"
                    )
                    if chunk_event.command_id in self._chat_completion_queues:
                        assert isinstance(chunk_event.chunk, TokenChunk)
                        logger.info(f"API sending chunk to stream for command_id={chunk_event.command_id}")
                        # Send chunk immediately to stream
                        await self._chat_completion_queues[chunk_event.command_id].send(
                            chunk_event.chunk
                        )
                        # Still process through normal event flow for state management
                        self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                        for idx, event in self.event_buffer.drain_indexed():
                            self._event_log.append(event)
                            self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                    else:
                        logger.warning(
                            f"API: ChunkGenerated for command_id={chunk_event.command_id} "
                            f"but queue not found. Available queues: {list(self._chat_completion_queues.keys())}"
                        )
                        # Not a streaming request, process normally
                        self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                        for idx, event in self.event_buffer.drain_indexed():
                            self._event_log.append(event)
                            self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                else:
                    # Non-ChunkGenerated events: process normally through buffer
                    self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                    for idx, event in self.event_buffer.drain_indexed():
                        self._event_log.append(event)
                        self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                        # Handle ChunkGenerated from buffer (shouldn't happen if above works, but keep for safety)
                        if (
                            isinstance(event, ChunkGenerated)
                            and event.command_id in self._chat_completion_queues
                        ):
                            assert isinstance(event.chunk, TokenChunk)
                            await self._chat_completion_queues[event.command_id].send(
                                event.chunk
                            )

    async def _send(self, command: Command):
        # No pausing in static setup (no elections)
        await self.command_sender.send(
            ForwarderCommand(origin=self.node_id, command=command)
        )
