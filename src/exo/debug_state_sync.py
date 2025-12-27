#!/usr/bin/env python3
"""
Diagnostic tool to test master-worker state synchronization.

Usage:
    python -m exo.debug_state_sync

This tool tests HTTP connectivity from master to workers and verifies
state push functionality.
"""
import asyncio
import sys

import aiohttp
from loguru import logger

from exo.shared.static_config import get_static_config
from exo.shared.types.state import State


async def test_worker_connectivity(worker_url: str, worker_id: str) -> bool:
    """Test basic HTTP connectivity to a worker's state endpoint."""
    logger.info(f"Testing connectivity to {worker_id} at {worker_url}...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            # Try a simple POST with minimal state
            minimal_state = State().model_dump(mode="json")
            async with session.post(
                f"{worker_url}/state/update",
                json=minimal_state,
            ) as response:
                if response.status == 200:
                    logger.info(f"✓ {worker_id}: SUCCESS (status {response.status})")
                    return True
                else:
                    logger.error(f"✗ {worker_id}: HTTP {response.status}")
                    return False
    except aiohttp.ClientConnectorError as e:
        logger.error(f"✗ {worker_id}: Connection failed - {e}")
        return False
    except aiohttp.ClientError as e:
        logger.error(f"✗ {worker_id}: Client error - {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {worker_id}: Unexpected error - {e}")
        return False


async def main() -> int:
    """Main diagnostic routine."""
    logger.info("=== Master-Worker State Sync Diagnostic Tool ===\n")
    
    # Get static configuration
    config = get_static_config()
    logger.info(f"Master: {config.master.node_id} at {config.master.tailscale_ip}:{config.master.port}")
    logger.info(f"Workers: {len(config.workers)}\n")
    
    # Test each worker
    results = {}
    for worker in config.workers:
        worker_url = f"http://{worker.tailscale_ip}:8080"
        logger.info(f"Worker: {worker.node_id}")
        logger.info(f"  Hostname: {worker.hostname}")
        logger.info(f"  Tailscale IP: {worker.tailscale_ip}")
        logger.info(f"  HTTP URL: {worker_url}")
        
        success = await test_worker_connectivity(worker_url, worker.node_id)
        results[worker.node_id] = success
        logger.info("")
    
    # Summary
    logger.info("=== Summary ===")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    logger.info(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        logger.info("✓ All workers are reachable!")
        return 0
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"✗ Failed workers: {', '.join(failed)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check that workers are running (ps aux | grep worker_app)")
        logger.error("2. Check worker logs (~/.exo/worker.log)")
        logger.error("3. Verify port 8080 is listening (lsof -i :8080)")
        logger.error("4. Check Tailscale connectivity (tailscale status)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
