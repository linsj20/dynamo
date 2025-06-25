# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp
from pydantic import BaseModel

from dynamo.sdk import async_on_start, endpoint, service

logger = logging.getLogger(__name__)

class SLOLevel(Enum):
    """SLO requirement levels"""
    LOW = "low"       # Best effort, >10s response time
    MEDIUM = "medium" # Standard, 1-10s response time  
    HIGH = "high"     # Premium, <1s response time

@dataclass
class PoolConfig:
    """Configuration for a pool (namespace with router + SLA planner)"""
    pool_id: str
    slo_level: SLOLevel
    base_url: str  # HTTP URL for the pool's Frontend service
    namespace: str  # Dynamo namespace for service discovery fallback
    model_name: str = "auto"  # Will be auto-discovered from pool
    description: str = ""

@service(
    dynamo={
        "namespace": "dynamo",
    },
    dependencies=[]
)
class GlobalScheduler:
    """
    Global Scheduler that routes requests across multiple pools based on SLO requirements.
    Uses HTTP requests to communicate with pool Frontend services.
    """
    
    def __init__(self):
        """Initialize Global Scheduler with HTTP-based pool communication."""
        self.runtime = None
        self.pools: Dict[str, PoolConfig] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("Global Scheduler initialized")
    
    async def _get_pool_endpoint(self, namespace: str, service_prefix: str, default_port: int) -> str:
        """
        Dynamically discover pool endpoint URLs for multi-node/multi-region deployments.
        
        Tries multiple discovery methods in priority order:
        1. Environment variables (for production deployments)
        2. Kubernetes service discovery patterns
        3. Docker Compose service names
        4. Localhost fallback (for development)
        
        Args:
            namespace: Pool namespace (e.g., "high_slo")
            service_prefix: Service name prefix (e.g., "high-slo")
            default_port: Default port number
            
        Returns:
            Full HTTP URL for the pool's Frontend service
        """
        # Method 1: Environment variables (highest priority)
        env_var = f"{namespace.upper()}_POOL_URL"
        if env_var in os.environ:
            url = os.environ[env_var]
            logger.info(f"Found {namespace} pool URL from environment: {url}")
            return url
        
        # Method 2: Kubernetes service discovery patterns
        k8s_patterns = [
            f"http://{namespace}-frontend:8000",
            f"http://{service_prefix}-frontend:8000", 
            f"http://{namespace}:8000",
            f"http://{service_prefix}:8000"
        ]
        
        # Method 3: Docker Compose service names
        docker_patterns = [
            f"http://{namespace}_frontend:8000",
            f"http://{service_prefix}_frontend:8000"
        ]
        
        # Method 4: Localhost fallback (development)
        localhost_url = f"http://localhost:{default_port}"
        
        all_patterns = k8s_patterns + docker_patterns + [localhost_url]
        
        # Test each pattern to see if it's reachable
        if self.http_session:
            for pattern in all_patterns[:-1]:  # Skip localhost for testing
                try:
                    async with self.http_session.get(f"{pattern}/health", timeout=2) as response:
                        if response.status == 200:
                            logger.info(f"✓ Discovered {namespace} pool at: {pattern}")
                            return pattern
                except:
                    continue
        
        # Final fallback to localhost
        logger.info(f"Using localhost fallback for {namespace}: {localhost_url}")
        return localhost_url

    async def _discover_pool_model(self, pool_id: str, pool_config: PoolConfig) -> Optional[str]:
        """
        Discover the model name served by a pool by querying its /v1/models endpoint.
        
        Args:
            pool_id: ID of the pool
            pool_config: Pool configuration
            
        Returns:
            Model name if discovered, None if failed
        """
        try:
            logger.debug(f"Discovering model name for {pool_id}...")
            async with self.http_session.get(f"{pool_config.base_url}/v1/models", timeout=10) as response:
                if response.status == 200:
                    models_data = await response.json()
                    # Extract the first available model
                    if "data" in models_data and len(models_data["data"]) > 0:
                        model_name = models_data["data"][0]["id"]
                        logger.info(f"✓ Discovered model for {pool_id}: {model_name}")
                        return model_name
                    else:
                        logger.debug(f"No models found in {pool_id} response: {models_data}")
                else:
                    logger.debug(f"{pool_id} /v1/models returned HTTP {response.status}")
        except Exception as e:
            logger.debug(f"Model discovery failed for {pool_id}: {e}")
        
        # Fallback: try to infer from a test request
        try:
            logger.debug(f"Trying fallback model discovery for {pool_id}...")
            test_request = {
                "model": "auto",  # Try with auto first
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1
            }
            
            async with self.http_session.post(
                f"{pool_config.base_url}/v1/chat/completions",
                json=test_request,
                timeout=10
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if "model" in response_data:
                        model_name = response_data["model"]
                        logger.info(f"✓ Inferred model for {pool_id} from response: {model_name}")
                        return model_name
                elif response.status == 422:  # Unprocessable entity - model not found
                    logger.debug(f"{pool_id} doesn't support 'auto', will need explicit model name")
        except Exception as e:
            logger.debug(f"Fallback model discovery failed for {pool_id}: {e}")
        
        logger.debug(f"Could not discover model name for {pool_id}, keeping 'auto'")
        return None

    async def _periodic_model_discovery(self):
        """
        Background task that periodically discovers model names for pools.
        Runs every second until all pools have their models discovered or shutdown.
        """
        logger.debug("Starting periodic model discovery task...")
        
        while not self._shutdown_event.is_set():
            try:
                # Check if any pools still need model discovery
                pools_to_discover = [
                    (pool_id, pool_config) 
                    for pool_id, pool_config in self.pools.items() 
                    if pool_config.model_name == "auto"
                ]
                
                if not pools_to_discover:
                    logger.info("All pools have discovered models, stopping periodic discovery")
                    break
                
                # Try to discover models for pools that still need it
                for pool_id, pool_config in pools_to_discover:
                    logger.debug(f"Attempting model discovery for {pool_id}...")
                    discovered_model = await self._discover_pool_model(pool_id, pool_config)
                    if discovered_model:
                        pool_config.model_name = discovered_model
                        logger.info(f"✓ Pool {pool_id} model discovered: {discovered_model}")
                
                # Wait 1 second before next discovery attempt
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue discovery
                    
            except Exception as e:
                logger.error(f"Error in periodic model discovery: {e}")
                # Wait before retrying
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue
        
        logger.debug("Periodic model discovery task stopped")

    @async_on_start
    async def async_init(self):
        """Initialize the Global Scheduler with HTTP session and pool discovery"""
        from dynamo.sdk import dynamo_context
        self.runtime = dynamo_context["runtime"]
        
        logger.info("Global Scheduler async initialization starting...")
        
        # Create HTTP session for pool communication
        self.http_session = aiohttp.ClientSession()
        
        # Configure pools with dynamic discovery (models will be discovered lazily)
        # High SLO pool
        high_slo_url = await self._get_pool_endpoint("high_slo", "high-slo", 8000)
        self.pools["high_slo"] = PoolConfig(
            pool_id="high_slo",
            slo_level=SLOLevel.HIGH,
            base_url=high_slo_url,
            namespace="high_slo",
            model_name="auto",  # Will be discovered by background task
            description="High priority pool - <1s response time, high throughput"
        )
        
        # Low SLO pool
        low_slo_url = await self._get_pool_endpoint("low_slo", "low-slo", 8002)  
        self.pools["low_slo"] = PoolConfig(
            pool_id="low_slo",
            slo_level=SLOLevel.LOW,
            base_url=low_slo_url,
            namespace="low_slo",
            model_name="auto",  # Will be discovered by background task
            description="Standard priority pool - <5s response time, cost-optimized"
        )
        
        # Start background model discovery task
        self._discovery_task = asyncio.create_task(self._periodic_model_discovery())
        
        logger.info("Global Scheduler async initialization complete")
        logger.info("Pool configuration (models will be discovered lazily):")
        for pool_id, pool_config in self.pools.items():
            logger.info(f"  {pool_id}: {pool_config.base_url} (model: {pool_config.model_name})")
        logger.info("Background model discovery task started")
    
    @endpoint()
    async def generate(self, request: dict):
        """
        Generate endpoint that accepts a request object and routes it to the appropriate pool.
        
        Args:
            request: Request object containing request_id, slo_requirement, and other parameters
            
        Yields:
            Response items from the appropriate pool
        """
        # Handle different request formats
        if isinstance(request, dict):
            request_data = request
        elif hasattr(request, '__dict__'):
            request_data = request.__dict__
        elif hasattr(request, 'model_dump'):
            request_data = request.model_dump()
        else:
            # Yield error for invalid request format
            logger.error(f"Invalid request format: {type(request)}")
            yield {
                "success": False,
                "error": f"Invalid request format: {type(request)}. Expected dict or object with request_id and slo_requirement",
                "assigned_pool": None
            }
            return
        
        # Extract required parameters
        request_id = request_data.get("request_id")
        slo_requirement = request_data.get("slo_requirement")
        
        if not request_id:
            yield {
                "success": False,
                "error": "Missing required parameter: request_id",
                "assigned_pool": None
            }
            return
            
        if not slo_requirement:
            yield {
                "success": False,
                "error": "Missing required parameter: slo_requirement", 
                "assigned_pool": None
            }
            return
        
        # Convert SLO requirement to level
        try:
            slo_level = SLOLevel(slo_requirement.lower())
        except ValueError:
            logger.error(f"Invalid SLO requirement: {slo_requirement}")
            yield {
                "success": False,
                "error": f"Invalid SLO requirement: {slo_requirement}. Must be 'high', 'medium', or 'low'",
                "assigned_pool": None
            }
            return

        # Find appropriate pool
        pool_id = self._get_pool_for_slo(slo_level)
        pool_config = self.pools[pool_id]
        
        logger.info(f"Routing {request_id} to {pool_id} (model: {pool_config.model_name})")
        
        try:
            # Prepare the request payload for the pool's chat/completions endpoint
            chat_request = {
                "model": pool_config.model_name,  # Use discovered model name
                "messages": [
                    {
                        "role": "user", 
                        "content": request_data.get("prompt", "Hello")
                    }
                ],
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "stream": request_data.get("stream", False)
            }
            
            # Make HTTP request to pool's Frontend service
            url = f"{pool_config.base_url}/v1/chat/completions"
            logger.info(f"Making HTTP request to {url}")
            
            async with self.http_session.post(url, json=chat_request, timeout=30) as response:
                if response.status == 200:
                    response_data = await response.json()
                    logger.info(f"✓ Successfully routed {request_id} to {pool_id}")
                    
                    # Yield the complete response
                    yield {
                        "success": True,
                        "assigned_pool": pool_id,
                        "pool_url": pool_config.base_url,
                        "model_used": pool_config.model_name,
                        "response": response_data
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Pool {pool_id} returned HTTP {response.status}: {error_text}")
                    yield {
                        "success": False,
                        "error": f"Pool {pool_id} returned HTTP {response.status}: {error_text}",
                        "assigned_pool": pool_id
                    }
                    
        except Exception as e:
            logger.error(f"Failed to route {request_id} to {pool_id}: {e}")
            yield {
                "success": False,
                "error": f"Routing failed: {str(e)}",
                "assigned_pool": pool_id
            }
    
    @endpoint()
    async def get_pool_status(self):
        """
        Get status of all configured pools.
        
        Returns:
            Dict with pool configuration and connection status
        """
        pool_statuses = {}
        
        for pool_id, pool_config in self.pools.items():
            try:
                # Test pool connectivity
                async with self.http_session.get(f"{pool_config.base_url}/health", timeout=5) as response:
                    connected = response.status == 200
                    
                pool_statuses[pool_id] = {
                    "pool_config": {
                        "slo_level": pool_config.slo_level.value,
                        "base_url": pool_config.base_url,
                        "namespace": pool_config.namespace,
                        "model_name": pool_config.model_name,
                        "description": pool_config.description
                    },
                    "connection_status": "connected" if connected else "disconnected",
                    "connected": connected
                }
                
            except Exception as e:
                logger.warning(f"Failed to get status from pool {pool_id}: {e}")
                pool_statuses[pool_id] = {
                    "pool_config": {
                        "slo_level": pool_config.slo_level.value,
                        "base_url": pool_config.base_url,
                        "namespace": pool_config.namespace, 
                        "model_name": pool_config.model_name,
                        "description": pool_config.description
                    },
                    "connection_status": f"error: {str(e)}",
                    "connected": False
                }
        
        return {
            "timestamp": time.time(),
            "total_pools": len(self.pools),
            "connected_pools": len([p for p in pool_statuses.values() if p["connected"]]),
            "pools": pool_statuses
        }
    
    def _get_pool_for_slo(self, slo_level: SLOLevel) -> str:
        """
        Map SLO level to pool ID.
        
        Args:
            slo_level: SLO requirement level
            
        Returns:
            Pool ID to use for this SLO level
        """
        mapping = {
            SLOLevel.HIGH: "high_slo",
            SLOLevel.MEDIUM: "high_slo",  # Route medium to high pool since we only have 2 pools
            SLOLevel.LOW: "low_slo"
        }
        return mapping[slo_level] 

    async def cleanup(self):
        """Cleanup resources when shutting down"""
        logger.info("Global Scheduler cleanup starting...")
        
        # Signal shutdown to background task
        self._shutdown_event.set()
        
        # Wait for discovery task to complete
        if self._discovery_task and not self._discovery_task.done():
            try:
                await asyncio.wait_for(self._discovery_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Discovery task didn't complete within timeout, cancelling")
                self._discovery_task.cancel()
                try:
                    await self._discovery_task
                except asyncio.CancelledError:
                    pass
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        
        logger.info("Global Scheduler cleanup complete") 