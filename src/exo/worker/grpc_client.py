"""gRPC client for Worker to connect to Master."""
import asyncio
import json
from datetime import datetime
from typing import AsyncIterator

import grpc
from loguru import logger

from exo.generated import cluster_pb2, cluster_pb2_grpc
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import Event, ForwarderEvent


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class WorkerGrpcClient:
    """gRPC client for worker to communicate with master."""
    
    def __init__(self, master_url: str, node_id: NodeId, session_id: SessionId):
        self.master_url = master_url.replace("http://", "").replace("https://", "")
        # gRPC expects host:port format, extract from URL
        if ":" in self.master_url:
            parts = self.master_url.split(":")
            self.master_host = parts[0]
            # Use gRPC port instead of HTTP port
            self.master_grpc_url = f"{self.master_host}:50051"
        else:
            self.master_grpc_url = f"{self.master_url}:50051"
            
        self.node_id = node_id
        self.session_id = session_id
        self._channel = None
        self._stub = None
        self._stream = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_counter = 0
        self._running = False
    
    async def __aenter__(self):
        """Initialize gRPC connection."""
        self._channel = grpc.aio.insecure_channel(self.master_grpc_url)
        self._stub = cluster_pb2_grpc.MasterServiceStub(self._channel)
        self._running = True
        
        # Start bidirectional streaming
        asyncio.create_task(self._stream_events())
        
        logger.info(f"Worker gRPC client connected to master at {self.master_grpc_url}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close gRPC connection."""
        self._running = False
        if self._channel:
            await self._channel.close()
        logger.info("Worker gRPC client disconnected")
    
    async def _event_generator(self) -> AsyncIterator[cluster_pb2.EventMessage]:
        """Generate events from queue to send to master."""
        while self._running:
            try:
                event_msg = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event_msg
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event generator: {e}")
                break
    
    async def _stream_events(self):
        """Maintain bidirectional event stream with master."""
        while self._running:
            try:
                logger.info(f"Establishing gRPC event stream to master at {self.master_grpc_url}")
                self._stream = self._stub.StreamEvents(self._event_generator())
                
                # Process incoming events from master
                async for event_msg in self._stream:
                    try:
                        event_dict = json.loads(event_msg.event_json)
                        event = Event.model_validate(event_dict)
                        logger.debug(f"Received {event.__class__.__name__} from master via gRPC")
                        # Events from master would be processed here (e.g., state updates via events)
                    except Exception as e:
                        logger.error(f"Error processing event from master: {e}")
                        
            except grpc.aio.AioRpcError as e:
                logger.warning(f"gRPC stream error: {e.code()} - {e.details()}")
                if self._running:
                    logger.info("Reconnecting to master in 5 seconds...")
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in event stream: {e}", exc_info=True)
                if self._running:
                    await asyncio.sleep(5)
    
    async def send_event(self, event: Event):
        """Send an event to master via gRPC stream."""
        try:
            # Create ForwarderEvent wrapper
            forwarder_event = ForwarderEvent(
                origin_idx=self._event_counter,
                origin=self.node_id,
                session=self.session_id,
                event=event,
            )
            self._event_counter += 1
            
            # Serialize to JSON
            event_dict = forwarder_event.model_dump(by_alias=True)
            event_json = json.dumps(event_dict, cls=DateTimeEncoder)
            
            # Create protobuf message
            event_msg = cluster_pb2.EventMessage(
                event_json=event_json,
                origin_node_id=str(self.node_id),
                origin_idx=forwarder_event.origin_idx,
            )
            
            await self._event_queue.put(event_msg)
            logger.info(f"Queued {event.__class__.__name__} for master via gRPC")
            
        except Exception as e:
            logger.error(f"Error sending event to master: {e}", exc_info=True)
    
    async def get_state(self):
        """Request current state snapshot from master."""
        try:
            request = cluster_pb2.Empty()
            response = await self._stub.GetState(request)
            state_dict = json.loads(response.state_json)
            from exo.shared.types.state import State
            return State.model_validate(state_dict)
        except Exception as e:
            logger.error(f"Error getting state from master: {e}", exc_info=True)
            return None
