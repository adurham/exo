# Deployed Integration Tests

This directory contains integration tests that verify the deployed EXO cluster works end-to-end.

## Running the Tests

### Using just (recommended)

```bash
# Run tests against default API URL (http://100.93.253.67:52415)
just test-deployed

# Run tests against a custom API URL
just test-deployed http://your-api-url:52415
```

### Using pytest directly

```bash
# Run all tests
uv run pytest src/exo/tests/test_deployed_integration.py -v -s

# Run with custom API URL via environment variable
EXO_API_URL=http://your-api-url:52415 uv run pytest src/exo/tests/test_deployed_integration.py -v -s

# Run a specific test
uv run pytest src/exo/tests/test_deployed_integration.py::TestDeployedIntegration::test_chat_completion_basic -v -s
```

## Test Coverage

The integration tests verify:

1. **API Connectivity** (`test_api_connectivity`)
   - API endpoint is reachable
   - Returns expected status codes

2. **Node Status** (`test_node_status`)
   - All expected nodes are connected
   - Topology information is available

3. **Models Endpoint** (`test_models_endpoint`)
   - Models list is accessible
   - Returns valid model information

4. **Instance Placement Preview** (`test_instance_placement_preview`)
   - Placement preview API works
   - Returns valid placement options

5. **Instance Creation and Deletion** (`test_instance_creation_and_deletion`)
   - Can create instances via API
   - Instances appear in state after creation
   - Can delete instances via API

6. **Chat Completion - Basic** (`test_chat_completion_basic`)
   - Can make chat completion requests
   - Streaming responses work correctly
   - Response content is non-empty and sensible
   - Validates specific answers (e.g., "What is the capital of France?" should contain "Paris")

7. **Chat Completion - Non-Gibberish** (`test_chat_completion_non_gibberish`)
   - Verifies responses are not gibberish
   - Checks for reasonable character distribution
   - Validates word length and structure

8. **Chat Completion - Streaming** (`test_chat_completion_streaming`)
   - Verifies streaming works correctly
   - Tracks chunk reception
   - Validates incremental token delivery

## Prerequisites

1. **Deployed Cluster**: The cluster must be deployed and running on the nodes
2. **Instance Available**: At least one model instance should be running for chat completion tests
3. **Network Access**: The test runner must be able to reach the API endpoint

## Expected Behavior

- Tests will skip if prerequisites aren't met (e.g., no models available, no instances running)
- Tests have reasonable timeouts (2 minutes for instance creation, 2 minutes for chat completions)
- Tests print progress information with ✅ markers for successful steps

## Troubleshooting

### Tests fail with connection errors
- Verify the API URL is correct
- Check that the cluster is deployed and running
- Verify network connectivity to the API endpoint

### Instance creation tests timeout
- Check that models are available
- Verify worker nodes are connected
- Check logs on worker nodes for errors

### Chat completion tests fail
- Ensure an instance is running for the model being tested
- Check that the model is loaded and ready
- Verify worker logs for inference errors

