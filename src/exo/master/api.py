# ... existing code ...
        self.app.get("/state")(lambda: self.state)
        self.app.get("/events")(lambda: self._event_log)
        # Add endpoint for workers to send events via HTTP
        self.app.post("/events")(self.receive_event)
        # Add endpoint for workers to send events via HTTP
        self.app.post("/events")(self.receive_event)
        
        # Add endpoint for workers to send events via HTTP
        self.app.post("/events")(self.receive_event)

    async def receive_event(self, event_data: dict):
        """Receive an event from a worker via HTTP."""
        from exo.shared.types.events import ForwarderEvent, Event
        
        try:
            # Parse the event from the request body
            # Workers send events wrapped in ForwarderEvent format
            forwarder_event = ForwarderEvent.model_validate(event_data)
            
            # Apply the event to our state immediately
            from exo.shared.apply import apply
            from datetime import datetime, timezone
            
            # Update event index if needed
            if forwarder_event.origin_idx > self.state.last_event_applied_idx:
                self.state = self.state.model_copy(
                    update={"last_event_applied_idx": forwarder_event.origin_idx}
                )
            
            # Apply the event to state
            self.state = apply(forwarder_event.event, self.state)
            
            # Update API state (since API maintains its own copy)
            self._event_log.append(forwarder_event.event)
            
            logger.debug(f"Received event {forwarder_event.event.__class__.__name__} from {forwarder_event.origin}")
            
            return {"status": "ok", "event_id": str(forwarder_event.event.event_id)}
        except Exception as e:
            logger.error(f"Error processing event from worker: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

# ... existing code ...
