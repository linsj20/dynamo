# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import random
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
    namespace: str  # Dynamo namespace
    model_name: str = "auto"  # Model name served by the pool
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
    Pools register themselves with the Global Scheduler when they start up.
    """
    
    def __init__(self):
        """Initialize Global Scheduler with HTTP-based pool communication."""
        self.runtime = None
        self.pools: Dict[str, PoolConfig] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._shutdown_event = asyncio.Event()
        
        # HTTP connection pool configuration (configurable via environment variables)
        # Set to 0 for unlimited connections
        # WARNING: Unlimited connections can lead to resource exhaustion under extreme load
        self.connection_pool_limit = int(os.getenv("GLOBAL_SCHEDULER_CONNECTION_LIMIT", "0"))
        self.connection_per_host_limit = int(os.getenv("GLOBAL_SCHEDULER_CONNECTION_PER_HOST_LIMIT", "0"))
        
        # Communication with planners
        self.pool_metrics: Dict[str, dict] = {}
        self._communication_task = None
        
        logger.info("Global Scheduler initialized")

    @async_on_start
    async def async_init(self):
        """Initialize the Global Scheduler with HTTP session"""
        from dynamo.sdk import dynamo_context
        self.runtime = dynamo_context["runtime"]
        
        logger.info("Global Scheduler starting...")
        
        # Log connection limits (0 means unlimited)
        total_limit_str = "unlimited" if self.connection_pool_limit == 0 else str(self.connection_pool_limit)
        per_host_limit_str = "unlimited" if self.connection_per_host_limit == 0 else str(self.connection_per_host_limit)
        logger.info(f"HTTP connection pool configuration: total_limit={total_limit_str}, per_host_limit={per_host_limit_str}")
        
        # Warn about unlimited connections
        if self.connection_pool_limit == 0 or self.connection_per_host_limit == 0:
            logger.warning("WARNING: Unlimited HTTP connections enabled. Monitor resource usage to prevent exhaustion.")
        
        # Create HTTP session for pool communication with connection isolation
        # Note: limit=0 means unlimited in aiohttp
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_limit,  # Total connection pool size (0 = unlimited)
            limit_per_host=self.connection_per_host_limit,  # Max connections per host (0 = unlimited)
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,  # Enable DNS caching
            keepalive_timeout=30,  # Connection keepalive
            enable_cleanup_closed=True,  # Clean up closed connections
            force_close=False,  # Don't force close connections
        )
        
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'GlobalScheduler/1.0'}
        )
        
        logger.info("Global Scheduler initialized - waiting for pool registrations...")
        
        # Start communication loop with planners
        self._communication_task = asyncio.create_task(self._planner_communication_loop())

    @endpoint()
    async def register_pool(self, request: dict):
        """
        Pool registration endpoint. Pools call this when they start up.
        
        Args:
            request: Pool registration info containing:
                - pool_id: Unique identifier for the pool
                - slo_level: SLO level (high/medium/low)
                - base_url: HTTP URL of the pool's Frontend service
                - namespace: Dynamo namespace
                - model_name: Model name served by the pool (optional)
                - description: Human-readable description (optional)
                
        Yields:
            Registration confirmation
        """
        # Handle different request formats
        if isinstance(request, str):
            import json
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
        
        # Extract registration parameters
        pool_id = request_data.get("pool_id")
        slo_level_str = request_data.get("slo_level")
        base_url = request_data.get("base_url")
        namespace = request_data.get("namespace")
        model_name = request_data.get("model_name", "auto")
        description = request_data.get("description", "")
        
        # Validate required parameters
        if not pool_id:
            yield {"success": False, "error": "Missing required parameter: pool_id"}
            return
        if not slo_level_str:
            yield {"success": False, "error": "Missing required parameter: slo_level"}
            return
        if not base_url:
            yield {"success": False, "error": "Missing required parameter: base_url"}
            return
        if not namespace:
            yield {"success": False, "error": "Missing required parameter: namespace"}
            return
        
        # Validate SLO level
        if slo_level_str.lower() not in ["high", "medium", "low"]:
            yield {
                "success": False,
                "error": f"Invalid SLO level: {slo_level_str}. Must be 'high', 'medium', or 'low'"
            }
            return
        
        slo_level = SLOLevel(slo_level_str.lower())
        
        # Create pool configuration
        pool_config = PoolConfig(
            pool_id=pool_id,
            slo_level=slo_level,
            base_url=base_url,
            namespace=namespace,
            model_name=model_name,
            description=description or f"{slo_level.value.title()} priority pool"
        )
        
        # Assert that pool_id is unique - it should always be different
        assert pool_id not in self.pools, f"Pool ID '{pool_id}' already exists! Pool IDs must be unique."
        
        # Register the pool
        self.pools[pool_id] = pool_config
        
        # Count pools by SLO level for load balancing status
        slo_pool_counts = {}
        for slo in SLOLevel:
            slo_pool_counts[slo.value] = len([p for p in self.pools.values() if p.slo_level == slo])
        
        logger.info(f"Registered pool: {pool_id} at {base_url} (SLO: {slo_level.value}, model: {model_name})")
        logger.info(f"Pool counts by SLO level: {slo_pool_counts} (random load balancing enabled)")
        
        yield {
            "success": True,
            "message": f"Pool {pool_id} registered successfully",
            "pool_id": pool_id,
            "registered_at": time.time()
        }

    @endpoint()
    async def unregister_pool(self, request: dict):
        """
        Pool unregistration endpoint. Pools call this when they shut down.
        
        Args:
            request: Pool unregistration info containing:
                - pool_id: Unique identifier for the pool to unregister
                
        Yields:
            Unregistration confirmation
        """
        # Handle different request formats
        if isinstance(request, str):
            import json
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
            
        pool_id = request_data.get("pool_id")
        
        if not pool_id:
            yield {"success": False, "error": "Missing required parameter: pool_id"}
            return
        
        if pool_id in self.pools:
            del self.pools[pool_id]
            
            # Count pools by SLO level after removal
            slo_pool_counts = {}
            for slo in SLOLevel:
                slo_pool_counts[slo.value] = len([p for p in self.pools.values() if p.slo_level == slo])
            
            logger.info(f"Unregistered pool: {pool_id}")
            logger.info(f"Pool counts by SLO level: {slo_pool_counts} (random load balancing)")
            yield {
                "success": True,
                "message": f"Pool {pool_id} unregistered successfully",
                "pool_id": pool_id
            }
        else:
            yield {
                "success": False,
                "message": f"Pool {pool_id} was not registered",
                "pool_id": pool_id
            }
    
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
        if slo_requirement.lower() not in ["high", "medium", "low"]:
            yield {
                "success": False,
                "error": f"Invalid SLO requirement: {slo_requirement}. Must be 'high', 'medium', or 'low'",
                "assigned_pool": None
            }
            return
        
        slo_level = SLOLevel(slo_requirement.lower())

        # Find appropriate pool
        pool_config = self._get_pool_for_slo(slo_level)
        if not pool_config:
            yield {
                "success": False,
                "error": f"No pools available for SLO level: {slo_requirement}. No pools have registered yet.",
                "assigned_pool": None
            }
            return
        
        # Prepare the request payload for the pool's chat/completions endpoint
        chat_request = {
            "model": pool_config.model_name,  # Use registered model name
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
        
        try:
            # Handle streaming vs non-streaming responses
            if chat_request.get('stream', False):
                # Streaming request - process SSE stream
                async with self.http_session.post(url, json=chat_request) as response:
                    
                    if not response.status == 200:
                        error_text = await response.text()
                        logger.error(f"ERROR: Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}")
                        yield {
                            "success": False,
                            "error": f"Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}",
                            "assigned_pool": pool_config.pool_id
                        }
                        return
                    
                    # Process Server-Sent Events stream
                    is_first_chunk = True
                    accumulated_content = ""
                    final_response = None
                    
                    async for line in response.content:
                        if not line:
                            continue
                            
                        line_str = line.decode('utf-8').strip()
                        
                        # Skip empty lines and comments
                        if not line_str or line_str.startswith(':'):
                            continue
                        
                        # Process SSE data lines
                        if line_str.startswith('data: '):
                            data_content = line_str[6:]  # Remove 'data: ' prefix
                            
                            # Check for end of stream
                            if data_content == '[DONE]':
                                break
                                
                            try:
                                # Parse JSON chunk
                                chunk_data = json.loads(data_content)
                                
                                # Extract content from streaming chunk
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    choice = chunk_data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        accumulated_content += content
                                        
                                    
                                    # Check if this is the final chunk with finish_reason
                                    if 'finish_reason' in choice and choice['finish_reason'] is not None:
                                        
                                        # Create final response in OpenAI format
                                        final_response = {
                                            "id": chunk_data.get("id", f"chatcmpl-{request_id}"),
                                            "object": "chat.completion",
                                            "created": int(time.time()),
                                            "model": pool_config.model_name,
                                            "choices": [{
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": accumulated_content
                                                },
                                                "finish_reason": choice['finish_reason']
                                            }]
                                        }
                                        
                                        # Add usage information if available from the last chunk
                                        if 'usage' in chunk_data:
                                            final_response['usage'] = chunk_data['usage']
                                        break
                                
                                # Yield intermediate streaming status (if first chunk)
                                if is_first_chunk:
                                    is_first_chunk = False
                                    yield {
                                        "success": True,
                                        "assigned_pool": pool_config.pool_id,
                                        "pool_url": pool_config.base_url,
                                        "model_used": pool_config.model_name,
                                        "streaming": True,
                                        "first_token_received": True
                                    }
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"WARNING: Failed to parse streaming chunk: {e}, data: {data_content}")
                                continue
                    
                    # Yield final complete response
                    if final_response:
                        yield {
                            "success": True,
                            "assigned_pool": pool_config.pool_id,
                            "pool_url": pool_config.base_url,
                            "model_used": pool_config.model_name,
                            "response": final_response,
                            "streaming": True,
                            "total_content_length": len(accumulated_content)
                        }
                    elif accumulated_content:
                        # Create final response even if we didn't get explicit finish_reason
                        final_response = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": pool_config.model_name,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": accumulated_content
                                },
                                "finish_reason": "stop"  # Default finish reason
                            }]
                        }
                        yield {
                            "success": True,
                            "assigned_pool": pool_config.pool_id,
                            "pool_url": pool_config.base_url,
                            "model_used": pool_config.model_name,
                            "response": final_response,
                            "streaming": True,
                            "total_content_length": len(accumulated_content)
                        }
                    else:
                        logger.warning(f"WARNING: No content received for streaming request {request_id}")
                        yield {
                            "success": False,
                            "error": "Stream ended without final response",
                            "assigned_pool": pool_config.pool_id
                        }
            else:
                # Non-streaming request (legacy support)
                async with self.http_session.post(url, json=chat_request) as response:
                    
                    if not response.status == 200:
                        error_text = await response.text()
                        logger.error(f"ERROR: Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}")
                        yield {
                            "success": False,
                            "error": f"Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}",
                            "assigned_pool": pool_config.pool_id
                        }
                        return
                    
                    response_data = await response.json()
                    
                    # Yield the complete response
                    yield {
                        "success": True,
                        "assigned_pool": pool_config.pool_id,
                        "pool_url": pool_config.base_url,
                        "model_used": pool_config.model_name,
                        "response": response_data
                    }
                    
        except Exception as e:
            logger.error(f"ERROR: HTTP request to {url} failed: {e}")
            yield {
                "success": False,
                "error": f"HTTP request to pool {pool_config.pool_id}: {e}",
                "assigned_pool": pool_config.pool_id
            }
    
    @endpoint()
    async def get_pool_status(self, request: dict = None):
        """
        Get status of all registered pools.
        
        Args:
            request: Optional request data (not used)
            
        Yields:
            Dict with pool configuration and connection status
        """
        pool_statuses = {}
        
        for pool_id, pool_config in self.pools.items():
            # Test current connectivity - let errors crash immediately
            async with self.http_session.get(f"{pool_config.base_url}/health", timeout=30) as response:  # Increased from 5s to 30s
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
        
        yield {
            "timestamp": time.time(),
            "total_pools": len(self.pools),
            "connected_pools": len([p for p in pool_statuses.values() if p["connected"]]),
            "pools": pool_statuses
        }
    
    def _get_pool_for_slo(self, slo_level: SLOLevel) -> Optional[PoolConfig]:
        """
        Find the best available pool for the requested SLO level using random selection.
        
        Args:
            slo_level: SLO requirement level
            
        Returns:
            Pool configuration if available, None if no suitable pool found
        """
        # Filter pools by SLO level
        matching_pools = [pool for pool in self.pools.values() if pool.slo_level == slo_level]
        
        if matching_pools:
            # Simple random selection among pools of the same SLO level
            selected_pool = random.choice(matching_pools)
            
            logger.info(f"RANDOM: Selected pool {selected_pool.pool_id} for {slo_level.value} SLO - {selected_pool.base_url}")
            
            return selected_pool
        
        # Fallback: if no exact match, try to find a higher SLO level pool using round-robin
        if slo_level == SLOLevel.LOW:
            # Low can use medium or high
            fallback_pools = [pool for pool in self.pools.values() 
                            if pool.slo_level in [SLOLevel.MEDIUM, SLOLevel.HIGH]]
        elif slo_level == SLOLevel.MEDIUM:
            # Medium can use high
            fallback_pools = [pool for pool in self.pools.values() 
                            if pool.slo_level == SLOLevel.HIGH]
        else:
            # High SLO requires exact match
            fallback_pools = []
        
        if fallback_pools:
            # Simple random selection among fallback pools
            selected_pool = random.choice(fallback_pools)
            
            logger.info(f"RANDOM FALLBACK: Using pool {selected_pool.pool_id} (SLO: {selected_pool.slo_level.value}) "
                       f"for requested {slo_level.value} SLO - {selected_pool.base_url}")
            
            return selected_pool
        
        logger.warning(f"No suitable pool found for SLO level: {slo_level.value}")
        return None

    async def _planner_communication_loop(self):
        """Periodically communicate with planners"""
        logger.info("GLOBAL SCHEDULER COMMUNICATION: Starting communication loop")
        while True:
            await asyncio.sleep(20)  # Every 20 seconds
            logger.info("GLOBAL SCHEDULER COMMUNICATION: Requesting metrics from planners")
            await self._request_metrics_from_planners()
            await asyncio.sleep(10)
            logger.info("GLOBAL SCHEDULER COMMUNICATION: Sending instructions to planners")
            await self._send_instructions_to_planners()

    async def _request_metrics_from_planners(self):
        """Request metrics from all registered planners"""
        for pool_id, pool_config in self.pools.items():
            try:
                logger.info(f"GLOBAL SCHEDULER: Attempting to connect to planner in namespace {pool_config.namespace}")
                planner_component = self.runtime.namespace(pool_config.namespace).component("Planner")
                metrics_endpoint = planner_component.endpoint("get_planner_metrics")
                client = await metrics_endpoint.client()
                logger.info(f"GLOBAL SCHEDULER: Successfully connected to planner endpoint in {pool_config.namespace}")
                
                request_data = {"requester_id": "global_scheduler"}
                response = await client.generate(request_data)
                
                async for response_item in response:
                    # Handle Dynamo SDK Annotated response format
                    data = response_item.data if hasattr(response_item, 'data') else response_item
                    if data.get("success", False):
                        metrics = data.get("metrics", {})
                        logger.info("=" * 60)
                        logger.info(f"GLOBAL SCHEDULER - RECEIVED METRICS FROM PLANNER: {pool_id}")
                        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
                        logger.info("=" * 60)
            except Exception as e:
                logger.info(f"Could not request metrics from planner {pool_id}: {e}")

    async def _send_instructions_to_planners(self):
        """Send coordination instructions to planners"""
        for pool_id, pool_config in self.pools.items():
            try:
                planner_component = self.runtime.namespace(pool_config.namespace).component("Planner")
                instructions_endpoint = planner_component.endpoint("receive_coordination_instructions")
                client = await instructions_endpoint.client()
                logger.info(f"GLOBAL SCHEDULER: Successfully connected to instructions endpoint in {pool_config.namespace}")
                
                # Generate random instructions
                instructions = {
                    "scaling_suggestion": random.choice(["scale_up", "scale_down", "maintain"]),
                    "priority_level": random.choice(["high", "medium", "low"]),
                    "resource_limit": random.randint(1, 8),
                    "load_balancing": random.choice(["enable", "disable"])
                }
                
                request_data = {
                    "sender_id": "global_scheduler",
                    "instructions": instructions
                }
                response = await client.generate(request_data)
                
                async for response_item in response:
                    # Handle Dynamo SDK Annotated response format
                    data = response_item.data if hasattr(response_item, 'data') else response_item
                    if data.get("success", False):
                        logger.info("=" * 60)
                        logger.info(f"GLOBAL SCHEDULER - SENT INSTRUCTIONS TO PLANNER: {pool_id}")
                        logger.info(f"Instructions: {json.dumps(instructions, indent=2)}")
                        logger.info("=" * 60)
            except Exception as e:
                logger.info(f"Could not send instructions to planner {pool_id}: {e}")

    @endpoint()
    async def receive_planner_metrics(self, request: dict):
        """
        Receive metrics from pool planners.
        
        Args:
            request: Metrics data containing:
                - planner_id: Unique identifier for the sending planner
                - metrics: Dictionary of metrics data
                
        Yields:
            Acknowledgment of received metrics
        """
        # Handle different request formats
        if isinstance(request, str):
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
        
        # Extract required parameters
        planner_id = request_data.get("planner_id")
        metrics = request_data.get("metrics", {})
        
        # Validate required parameters
        if not planner_id:
            yield {"success": False, "error": "Missing required parameter: planner_id"}
            return
        
        logger.info("=" * 60)
        logger.info(f"RECEIVED METRICS FROM PLANNER: {planner_id}")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        logger.info("=" * 60)
        
        yield {
            "success": True,
            "status": "received", 
            "planner_id": planner_id,
            "metrics_count": len(metrics) if isinstance(metrics, dict) else 0
        }

    @endpoint()
    async def send_coordination_data(self, request: dict):
        """
        Send coordination data to requesting planner.
        
        Args:
            request: Coordination request containing:
                - planner_id: Unique identifier for the requesting planner
                
        Yields:
            Random coordination data for demonstration
        """
        # Handle different request formats
        if isinstance(request, str):
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
        
        # Extract required parameters
        planner_id = request_data.get("planner_id")
        
        # Validate required parameters
        if not planner_id:
            yield {"success": False, "error": "Missing required parameter: planner_id"}
            return
        
        # Generate random coordination data
        coordination_data = {
            "global_load": random.uniform(0.3, 0.8),
            "recommended_action": random.choice(["scale_up", "scale_down", "maintain"]),
            "priority_boost": random.choice([True, False]),
            "resource_allocation": random.randint(1, 8)
        }
        
        logger.info("=" * 60)
        logger.info(f"SENDING COORDINATION DATA TO PLANNER: {planner_id}")
        logger.info(f"Data: {json.dumps(coordination_data, indent=2)}")
        logger.info("=" * 60)
        
        yield {
            "success": True,
            "planner_id": planner_id, 
            "coordination_data": coordination_data
        }

    async def cleanup(self):
        """Cleanup resources when shutting down"""
        logger.info("Global Scheduler cleanup starting...")
        
        # Cancel communication task
        if self._communication_task:
            self._communication_task.cancel()
            try:
                await self._communication_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        
        logger.info("Global Scheduler cleanup complete")