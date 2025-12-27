"""gRPC server for Master node - handles bidirectional streaming with workers."""
import asyncio
import json
from datetime import datetime
from typing import AsyncIterator

import grpc
from loguru import logger

from exo.generated import cluster_pb2, cluster_pb2_grpc
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, ForwarderEvent
from exo.shared.types.state import State


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class MasterGrpcServicer(cluster_pb2_grpc.MasterServiceServicer):
    """gRPC servicer for Master - handles worker connections."""
    
    def __init__(self, master_ref):
        """Initialize with reference to Master instance."""
        self.master_ref = master_ref
        self._worker_streams: dict[NodeId, asyncio.Queue] = {}
        self._event_counters: dict[NodeId, int] = {}
    
    async def StreamEvents(
        self,
        request_iterator: AsyncIterator[cluster_pb2.EventMessage],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[cluster_pb2.EventMessage]:
        """Bidirectional event streaming with a worker."""
        worker_id = None
        outgoing_queue: asyncio.Queue = asyncio.Queue()
        
        try:
            # Process incoming events from worker
            async for event_msg in request_iterator:
                if worker_id is None:
                    worker_id = NodeId(event_msg.origin_node_id)
                    self._worker_streams[worker_id] = outgoing_queue
                    logger.info(f"Worker {worker_id} connected via gRPC stream")
                
                # Deserialize event from JSON
                try:
                    event_dict = json.loads(event_msg.event_json)
                    event = ForwarderEvent.model_validate(event_dict)
                    
                    # Apply event to master state
                    if not hasattr(self.master_ref, '_http_event_counters'):
                        self.master_ref._http_event_counters: dict[NodeId, int] = {}
                    
                    worker_counter = self.master_ref._http_event_counters.get(event.origin, -1) + 1
                    self.master_ref._http_event_counters[event.origin] = worker_counter
                    
                    # Add to master's event log and apply to state
                    master_idx = len(self.master_ref._event_log)
                    from exo.shared.apply import apply
                    from exo.shared.types.events import IndexedEvent
                    
                    indexed = IndexedEvent(event=event.event, idx=master_idx)
                    self.master_ref.state = apply(self.master_ref.state, indexed)
                    self.master_ref._event_log.append(event.event)
                    
                    logger.info(
                        f"gRPC: Received {event.event.__class__.__name__} from {event.origin} "
                        f"(idx={event.origin_idx})"
                    )
                except Exception as e:
                    logger.error(f"Error processing event from worker: {e}", exc_info=True)
            
            # Stream events to worker from queue
            while True:
                try:
                    event_msg = await outgoing_queue.get()
                    yield event_msg
                except asyncio.CancelledError:
                    break
                    
        except Exception as e:
            logger.error(f"Error in StreamEvents for worker {worker_id}: {e}", exc_info=True)
        finally:
            if worker_id:
                self._worker_streams.pop(worker_id, None)
                logger.info(f"Worker {worker_id} disconnected from gRPC stream")
    
    async def send_event_to_worker(self, worker_id: NodeId, event: Event):
        """Send an event to a specific worker's stream."""
        if worker_id in self._worker_streams:
            try:
                # Serialize event to JSON
                event_dict = event.model_dump(by_alias=True)
                event_json = json.dumps(event_dict, cls=DateTimeEncoder)
                
                # Create protobuf message
                event_msg = cluster_pb2.EventMessage(
                    event_json=event_json,
                    origin_node_id=str(self.master_ref.node_id),
                    origin_idx=0,  # Master events don't need indexing for now
                )
                
                await self._worker_streams[worker_id].put(event_msg)
                logger.debug(f"Queued {event.__class__.__name__} for worker {worker_id}")
            except Exception as e:
                logger.error(f"Error sending event to worker {worker_id}: {e}")
    
    async def GetState(
        self,
        request: cluster_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> cluster_pb2.StateMessage:
        """Worker requests current state snapshot."""
        try:
            state_dict = self.master_ref.state.model_dump(by_alias=True)
            state_json = json.dumps(state_dict, cls=DateTimeEncoder)
            return cluster_pb2.StateMessage(state_json=state_json)
        except Exception as e:
            logger.error(f"Error in GetState: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cluster_pb2.StateMessage()
    
    async def HealthCheck(
        self,
        request: cluster_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> cluster_pb2.HealthCheckResponse:
        """Health check endpoint."""
        return cluster_pb2.HealthCheckResponse(status="healthy")


async def serve_grpc(master_ref, port: int = 50051):
    """Start gRPC server for master."""
    server = grpc.aio.server()
    servicer = MasterGrpcServicer(master_ref)
    cluster_pb2_grpc.add_MasterServiceServicer_to_server(servicer, server)
    
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting Master gRPC server on {listen_addr}")
    await server.start()
    logger.info(f"✓ Master gRPC server listening on {listen_addr}")
    
    # Store servicer reference for pushing events
    master_ref._grpc_servicer = servicer
    
    try:
        await server.wait_for_termination()
    except Exception as e:
        logger.error(f"gRPC server error: {e}", exc_info=True)
    finally:
        await server.stop(grace=5)
