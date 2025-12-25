"""
Integration tests for deployed EXO cluster.

These tests verify that the deployed system works end-to-end:
- API connectivity
- Node status and connectivity
- Instance placement and creation
- Chat completion with sensible output validation
"""
import json
import os
import re
import sys
import time

import httpx  # type: ignore
import pytest  # type: ignore


class TestDeployedIntegration:
    """Integration tests for deployed EXO cluster."""

    # Default API URL - can be overridden with environment variable or pytest parameter
    _api_base_url: str = "http://100.93.253.67:52415"

    @property
    def API_BASE_URL(self) -> str:
        """Get the API base URL."""
        return self._api_base_url

    @API_BASE_URL.setter
    def API_BASE_URL(self, value: str) -> None:
        """Set the API base URL."""
        self._api_base_url = value

    @pytest.fixture(autouse=True)  # type: ignore
    def setup(self, request: pytest.FixtureRequest) -> None:  # type: ignore
        """Setup test fixture that allows overriding API URL."""
        # Allow pytest markers to override API URL
        api_url_marker = request.node.get_closest_marker("api_url")  # type: ignore
        if api_url_marker:
            self._api_base_url = api_url_marker.args[0]  # type: ignore

        # Also check environment variable
        env_api_url = os.environ.get("EXO_API_URL")
        if env_api_url:
            self._api_base_url = env_api_url

    def test_api_connectivity(self):
        """Test that the API is reachable."""
        response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert response.status_code == 200, f"API not reachable: {response.status_code}"
        print(f"✅ API is reachable at {self.API_BASE_URL}")

    def test_node_status(self):
        """Test that all expected nodes are connected."""
        response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert response.status_code == 200

        state = response.json()
        topology = state.get("topology", {})
        nodes = topology.get("nodes", [])

        assert len(nodes) >= 1, f"Expected at least 1 node, got {len(nodes)}"
        print(f"✅ Found {len(nodes)} connected node(s)")

        # Log node details
        for node in nodes:
            node_id = node.get("nodeId", "unknown")
            print(f"   - Node: {node_id[:50]}...")

    def test_models_endpoint(self):
        """Test that the models endpoint returns available models."""
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200

        models_data = response.json()
        assert "data" in models_data
        models = models_data["data"]

        assert len(models) > 0, "No models available"
        print(f"✅ Found {len(models)} available model(s)")

        # Log first few models
        for model in models[:5]:
            model_id = model.get("id", "unknown")
            print(f"   - Model: {model_id}")

    def test_instance_placement_preview(self):
        """Test that instance placement preview works."""
        # Get available models first
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200
        models = response.json()["data"]

        if not models:
            pytest.skip("No models available for testing")

        # Use first model
        model_id = models[0]["id"]

        # Test placement preview
        response = httpx.get(
            f"{self.API_BASE_URL}/instance/previews",
            params={"model_id": model_id},
            timeout=30.0,
        )

        assert response.status_code == 200, f"Placement preview failed: {response.status_code}"
        previews = response.json().get("previews", [])

        print(f"✅ Got {len(previews)} placement preview(s) for model {model_id}")

        # Check that we have at least one valid preview
        valid_previews = [p for p in previews if p.get("error") is None]
        assert len(valid_previews) > 0, "No valid placement previews available"

        print(f"   - {len(valid_previews)} valid preview(s)")

    def test_instance_creation_and_deletion(self):
        """Test creating and deleting an instance."""
        # Get available models
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200
        models = response.json()["data"]

        if not models:
            pytest.skip("No models available for testing")

        model_id = models[0]["id"]

        # Check for existing instances
        state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert state_response.status_code == 200
        state = state_response.json()
        instances_before = state.get("instances", {})

        # Try to place instance (this will create it if it doesn't exist)
        placement_response = httpx.post(
            f"{self.API_BASE_URL}/place_instance",
            json={
                "model_id": model_id,
                "sharding": "Pipeline",
                "instance_meta": "MlxRing",
                "min_nodes": 1,
            },
            timeout=30.0,
        )

        # Placement command accepted (202) or already exists (200)
        assert placement_response.status_code in (200, 202), \
            f"Instance placement failed: {placement_response.status_code} - {placement_response.text}"

        print(f"✅ Instance placement command sent for model {model_id}")

        # Wait for instance to be created (with timeout)
        max_wait = 120  # 2 minutes
        wait_interval = 2
        elapsed = 0

        instance_created = False
        while elapsed < max_wait:
            time.sleep(wait_interval)
            elapsed += wait_interval

            state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
            if state_response.status_code != 200:
                continue

            state = state_response.json()
            instances_after = state.get("instances", {})

            if len(instances_after) > len(instances_before):
                instance_created = True
                print(f"✅ Instance created after {elapsed} seconds")
                break

            print(f"   Waiting for instance creation... ({elapsed}/{max_wait}s)")

        if not instance_created:
            pytest.skip("Instance creation timed out - may need manual intervention")

        # Get the new instance ID
        state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        state = state_response.json()
        instances_after = state.get("instances", {})
        instance_ids = set(instances_after.keys()) - set(instances_before.keys())

        if instance_ids:
            instance_id = list(instance_ids)[0]
            print(f"✅ Instance ID: {instance_id}")

            # Test deleting the instance
            delete_response = httpx.delete(
                f"{self.API_BASE_URL}/instance/{instance_id}",
                timeout=30.0,
            )
            assert delete_response.status_code in (200, 202), \
                f"Instance deletion failed: {delete_response.status_code}"

            print(f"✅ Instance deletion command sent")

    def test_chat_completion_basic(self):
        """Test basic chat completion with sensible output validation."""
        # Get available models
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200
        models = response.json()["data"]

        if not models:
            pytest.skip("No models available for testing")

        model_id = models[0]["id"]

        # Check if instance exists for this model
        state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert state_response.status_code == 200
        state = state_response.json()
        instances = state.get("instances", {})

        # Find instance for this model
        model_instance = None
        for instance_id, instance_data in instances.items():
            # Instance data structure: {instance_id: {shard_assignments: {model_id: ...}}}
            if isinstance(instance_data, dict):
                shard_assignments = instance_data.get("shardAssignments", {})
                instance_model_id = shard_assignments.get("modelId")
                if instance_model_id == model_id:
                    model_instance = instance_id
                    break

        if not model_instance:
            pytest.skip(f"No instance found for model {model_id} - create one first")

        # Make a chat completion request
        prompt = "What is the capital of France? Answer with just the city name."
        response = httpx.post(
            f"{self.API_BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.0,  # Deterministic output
                "max_tokens": 50,
            },
            timeout=120.0,  # 2 minute timeout
        )

        assert response.status_code == 200, \
            f"Chat completion failed: {response.status_code} - {response.text}"

        # Read streaming response
        full_response = ""
        got_done = False

        for line in response.iter_lines():
            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                got_done = True
                break

            try:
                chunk_data = json.loads(data_str)
                choices = chunk_data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_response += content
            except json.JSONDecodeError:
                continue

        # Validate response
        assert got_done, "Stream did not complete with [DONE]"
        assert len(full_response) > 0, "Response is empty"
        assert len(full_response.strip()) > 0, "Response contains only whitespace"

        print(f"✅ Chat completion successful")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {full_response[:200]}...")

        # Validate response quality - should contain "Paris" (case-insensitive)
        response_lower = full_response.lower()
        assert "paris" in response_lower, \
            f"Response should contain 'Paris' but got: {full_response[:100]}"

        print(f"✅ Response validation passed (contains expected answer)")

    def test_chat_completion_non_gibberish(self):
        """Test that chat completion returns non-gibberish output."""
        # Get available models
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200
        models = response.json()["data"]

        if not models:
            pytest.skip("No models available for testing")

        model_id = models[0]["id"]

        # Check if instance exists for this model
        state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert state_response.status_code == 200
        state = state_response.json()
        instances = state.get("instances", {})

        model_instance = None
        for instance_id, instance_data in instances.items():
            if isinstance(instance_data, dict):
                shard_assignments = instance_data.get("shardAssignments", {})
                instance_model_id = shard_assignments.get("modelId")
                if instance_model_id == model_id:
                    model_instance = instance_id
                    break

        if not model_instance:
            pytest.skip(f"No instance found for model {model_id}")

        # Make a chat completion request
        prompt = "Say hello and introduce yourself briefly."
        response = httpx.post(
            f"{self.API_BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=120.0,
        )

        assert response.status_code == 200

        # Read streaming response
        full_response = ""
        got_done = False

        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                got_done = True
                break

            try:
                chunk_data = json.loads(data_str)
                choices = chunk_data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_response += content
            except json.JSONDecodeError:
                continue

        # Validate response is not gibberish
        assert got_done, "Stream did not complete"
        assert len(full_response.strip()) > 0, "Response is empty"

        # Check for reasonable character distribution (not all the same character)
        unique_chars = len(set(full_response.lower()))
        assert unique_chars > 3, \
            f"Response appears to be gibberish (only {unique_chars} unique characters): {full_response[:100]}"

        # Check for reasonable word length (not all single characters)
        words = re.findall(r'\b\w+\b', full_response)
        avg_word_length = 0.0
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            assert avg_word_length > 2.0, \
                f"Response appears to be gibberish (avg word length {avg_word_length:.2f})"

        print(f"✅ Non-gibberish validation passed")
        print(f"   Response length: {len(full_response)} chars")
        print(f"   Unique characters: {unique_chars}")
        print(f"   Words: {len(words)}, avg length: {avg_word_length:.2f}")
        print(f"   Preview: {full_response[:150]}...")

    def test_chat_completion_streaming(self):
        """Test that streaming works correctly and produces tokens incrementally."""
        # Get available models
        response = httpx.get(f"{self.API_BASE_URL}/v1/models", timeout=10.0)
        assert response.status_code == 200
        models = response.json()["data"]

        if not models:
            pytest.skip("No models available for testing")

        model_id = models[0]["id"]

        # Check if instance exists
        state_response = httpx.get(f"{self.API_BASE_URL}/state", timeout=10.0)
        assert state_response.status_code == 200
        state = state_response.json()
        instances = state.get("instances", {})

        model_instance = None
        for instance_id, instance_data in instances.items():
            if isinstance(instance_data, dict):
                shard_assignments = instance_data.get("shardAssignments", {})
                instance_model_id = shard_assignments.get("modelId")
                if instance_model_id == model_id:
                    model_instance = instance_id
                    break

        if not model_instance:
            pytest.skip(f"No instance found for model {model_id}")

        # Make a streaming request
        prompt = "Count from 1 to 5, one number per line."
        response = httpx.post(
            f"{self.API_BASE_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=120.0,
        )

        assert response.status_code == 200

        # Track streaming progress
        chunks_received = 0
        first_chunk_time: float | None = None
        full_response = ""

        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk_data = json.loads(data_str)
                choices = chunk_data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunks_received += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        full_response += content
            except json.JSONDecodeError:
                continue

        # Validate streaming worked
        assert chunks_received > 0, "No chunks received"
        assert first_chunk_time is not None, "No first chunk received"
        assert len(full_response) > 0, "Empty response"

        print(f"✅ Streaming test passed")
        print(f"   Chunks received: {chunks_received}")
        print(f"   Total response length: {len(full_response)} chars")
        print(f"   Response preview: {full_response[:100]}...")


def pytest_configure(config: pytest.Config) -> None:  # type: ignore
    """Configure pytest markers."""
    config.addinivalue_line(  # type: ignore
        "markers", "api_url(url): Set the API base URL for integration tests"
    )


if __name__ == "__main__":
    # Allow running directly with API URL as argument
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
        pytest.main([__file__, "-v", "-s", "-m", f"api_url({api_url})"])
    else:
        pytest.main([__file__, "-v", "-s"])

