from exo.master.api import API
from exo.shared.types.common import NodeId, SessionId
from unittest.mock import MagicMock

def test_delete_model_route_uses_path_converter():
    """
    Verify that the delete model route is configured to match paths (slashes) in the model_id.
    """
    # Initialize minimal API to inspect routes
    api = API(
        node_id=NodeId("master"),
        session_id=SessionId(master_node_id=NodeId("master"), election_clock=0),
        port=8080,
        global_event_receiver=MagicMock(),
        command_sender=MagicMock(),
        election_receiver=MagicMock(),
    )
    
    # Check the routes in the FastAPI app
    found_route = False
    for route in api.app.routes:
        if route.path == "/models/{model_id}":
            # In FastAPI/Starlette, the path attribute shows the definition.
            # If we used {model_id:path}, the path string in the route object typically reflects that 
            # OR we check the regex pattern.
            
            # Starlette stores the compiled regex in .path_regex
            # We want to ensure it matches strings with slashes.
            match = route.path_regex.match("/models/org/model-name")
            if match:
                found_route = True
                break
    
    # If not found directly by string check (sometimes path is normalized), let's iterate and test matching.
    if not found_route:
        for route in api.app.routes:
            # We are looking for the DELETE route to /models/...
            if "DELETE" in route.methods:
                match = route.path_regex.match("/models/org/model-name")
                if match:
                    # found it!
                    found_route = True
                    # Validate it captures the right group
                    # route.path_regex.match returns a match object, key should be 'model_id'
                    assert match.group("model_id") == "org/model-name"
                    break
    
    assert found_route, "Could not find a DELETE route for /models/{model_id} that accepts slashes"
