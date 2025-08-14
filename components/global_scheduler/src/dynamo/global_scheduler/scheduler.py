# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import aiohttp
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dynamo.sdk import api, async_on_start, endpoint, service

logger = logging.getLogger(__name__)

class SLOLevel(Enum):
    """SLO requirement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

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
    """Global Scheduler that routes requests across multiple pools based on SLO requirements."""
    
    def __init__(self):
        """Initialize Global Scheduler with HTTP connection pool and metrics tracking."""
        self.runtime = None
        self.pools: Dict[str, PoolConfig] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._shutdown_event = asyncio.Event()
        self.connection_pool_limit = int(os.getenv("GLOBAL_SCHEDULER_CONNECTION_LIMIT", "0"))
        self.connection_per_host_limit = int(os.getenv("GLOBAL_SCHEDULER_CONNECTION_PER_HOST_LIMIT", "0"))
        self.pool_metrics: Dict[str, dict] = {}
        self._communication_task = None
        logger.info("Global Scheduler initialized")

    def _parse_request_data(self, request: Union[str, dict, Any]) -> dict:
        """Parse request data from various formats into a dictionary."""
        if isinstance(request, str):
            return json.loads(request)
        elif isinstance(request, dict):
            return request
        elif hasattr(request, 'model_dump'):
            return request.model_dump()
        elif hasattr(request, '__dict__'):
            return request.__dict__
        else:
            return request

    def _extract_response_data(self, response_item: Any) -> Optional[dict]:
        """Extract response data from Dynamo SDK response items."""
        if hasattr(response_item, 'data'):
            return response_item.data
        elif isinstance(response_item, dict):
            return response_item
        else:
            return None

    @async_on_start
    async def async_init(self):
        """Initialize HTTP session and start planner communication loop."""
        from dynamo.sdk import dynamo_context
        self.runtime = dynamo_context["runtime"]
        logger.info("Global Scheduler starting...")
        total_limit_str = "unlimited" if self.connection_pool_limit == 0 else str(self.connection_pool_limit)
        per_host_limit_str = "unlimited" if self.connection_per_host_limit == 0 else str(self.connection_per_host_limit)
        logger.info(f"HTTP connection pool configuration: total_limit={total_limit_str}, per_host_limit={per_host_limit_str}")
        if self.connection_pool_limit == 0 or self.connection_per_host_limit == 0:
            logger.warning("WARNING: Unlimited HTTP connections enabled. Monitor resource usage to prevent exhaustion.")
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_limit,
            limit_per_host=self.connection_per_host_limit,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
        )
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'GlobalScheduler/1.0'}
        )
        logger.info("Global Scheduler initialized - waiting for pool registrations...")
        self._communication_task = asyncio.create_task(self._planner_communication_loop())

    @endpoint()
    async def register_pool(self, request: dict):
        """Register a new pool with the global scheduler."""
        request_data = self._parse_request_data(request)
        pool_id = request_data.get("pool_id")
        slo_level_str = request_data.get("slo_level")
        base_url = request_data.get("base_url")
        namespace = request_data.get("namespace")
        model_name = request_data.get("model_name", "auto")
        description = request_data.get("description", "")
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
        if slo_level_str.lower() not in ["high", "medium", "low"]:
            yield {
                "success": False,
                "error": f"Invalid SLO level: {slo_level_str}. Must be 'high', 'medium', or 'low'"
            }
            return
        slo_level = SLOLevel(slo_level_str.lower())
        pool_config = PoolConfig(
            pool_id=pool_id,
            slo_level=slo_level,
            base_url=base_url,
            namespace=namespace,
            model_name=model_name,
            description=description or f"{slo_level.value.title()} priority pool"
        )
        assert pool_id not in self.pools, f"Pool ID '{pool_id}' already exists! Pool IDs must be unique."
        self.pools[pool_id] = pool_config
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
        """Unregister a pool from the global scheduler."""
        request_data = self._parse_request_data(request)
        pool_id = request_data.get("pool_id")
        if not pool_id:
            yield {"success": False, "error": "Missing required parameter: pool_id"}
            return
        if pool_id in self.pools:
            del self.pools[pool_id]
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
        """Legacy generate endpoint - deprecated in favor of v1/chat/completions."""
        pass

    @endpoint()
    @api(name="v1/chat/completions")
    async def v1_chat_completions(self, request: dict):
        """OpenAI-compatible chat completions endpoint with automatic SLO assignment."""
        try:
            request_data = self._parse_request_data(request)
            if not isinstance(request_data, dict):
                error_response = {
                    "error": {
                        "message": f"Invalid request format: {type(request)}",
                        "type": "invalid_request_error",
                        "code": "invalid_request"
                    }
                }
                logger.error(f"GLOBAL SCHEDULER ERROR: Invalid request format: {type(request)}")
                return error_response
            messages = request_data.get("messages", [])
            if not messages:
                error_response = {
                    "error": {
                        "message": "Missing required parameter: messages",
                        "type": "invalid_request_error", 
                        "code": "invalid_request"
                    }
                }
                logger.error("GLOBAL SCHEDULER ERROR: Missing required parameter: messages")
                return error_response

            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            slo_requirement = request_data.get("slo_requirement")
            if not slo_requirement:
                if not hasattr(self, '_request_counter'):
                    self._request_counter = 0
                slo_levels = ["high", "low"]
                slo_requirement = slo_levels[self._request_counter % len(slo_levels)]
                self._request_counter += 1
            prompt = ""
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    prompt = last_message["content"]

            logger.info(f"GLOBAL SCHEDULER: Processing v1/chat/completions request {request_id} with SLO {slo_requirement}")

            # Find appropriate pool using SLO requirement
            slo_level = SLOLevel(slo_requirement.lower())
            pool_config = self._get_pool_for_slo(slo_level)
            if not pool_config:
                error_response = {
                    "error": {
                        "message": f"No pools available for SLO level: {slo_requirement}. No pools have registered yet.",
                        "type": "server_error",
                        "code": "no_pools_available"
                    }
                }
                logger.error(f"GLOBAL SCHEDULER ERROR: No pools available for SLO level: {slo_requirement}")
                return error_response

            # Prepare the request payload for the pool's chat/completions endpoint
            chat_request = {
                "model": pool_config.model_name,  # Use registered model name
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "stream": request_data.get("stream", False)
            }
            
            url = f"{pool_config.base_url}/v1/chat/completions"

            is_streaming = request_data.get("stream", False)
            
            if is_streaming:
                async def stream_generator():
                    try:
                        async with self.http_session.post(url, json=chat_request) as response:
                            if not response.status == 200:
                                error_text = await response.text()
                                logger.error(f"ERROR: Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}")
                                error_response = {
                                    "error": {
                                        "message": f"Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}",
                                        "type": "server_error",
                                        "code": "pool_error"
                                    }
                                }
                                yield f"data: {json.dumps(error_response)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                            
                            # Process Server-Sent Events stream line by line
                            buffer = ""
                            async for chunk in response.content.iter_chunked(8192):
                                if not chunk:
                                    continue
                                    
                                # Decode chunk and add to buffer
                                buffer += chunk.decode('utf-8')
                                
                                # Process complete lines
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    line = line.strip()
                                    
                                    if not line:
                                        continue
                                    
                                    # Forward properly formatted SSE lines
                                    if line.startswith('data: '):
                                        yield f"{line}\n\n"
                                    elif line == 'data: [DONE]':
                                        yield f"{line}\n\n"
                                    elif line.startswith('{') and line.endswith('}'):
                                        # Raw JSON chunk - wrap in SSE format
                                        yield f"data: {line}\n\n"
                                    elif line == '[DONE]':
                                        yield f"data: [DONE]\n\n"
                                        
                            # Process any remaining buffer content
                            if buffer.strip():
                                line = buffer.strip()
                                if line.startswith('{') and line.endswith('}'):
                                    yield f"data: {line}\n\n"
                                    
                    except Exception as e:
                        logger.error(f"GLOBAL SCHEDULER ERROR in streaming: {str(e)}", exc_info=True)
                        error_response = {
                            "error": {
                                "message": f"Internal server error: {str(e)}",
                                "type": "server_error",
                                "code": "internal_error"
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                try:
                    async with self.http_session.post(url, json=chat_request) as response:
                        if not response.status == 200:
                            error_text = await response.text()
                            logger.error(f"ERROR: Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}")
                            error_response = {
                                "error": {
                                    "message": f"Pool {pool_config.pool_id} returned HTTP {response.status}: {error_text}",
                                    "type": "server_error",
                                    "code": "pool_error"
                                }
                            }
                            return error_response
                        
                        response_data = await response.json()
                        logger.info(f"GLOBAL SCHEDULER: Successfully processed non-streaming request {request_id} from pool {pool_config.pool_id}")
                        return response_data
                        
                except Exception as e:
                    logger.error(f"GLOBAL SCHEDULER ERROR in non-streaming processing: {str(e)}", exc_info=True)
                    error_response = {
                        "error": {
                            "message": f"Internal server error: {str(e)}",
                            "type": "server_error",
                            "code": "internal_error"
                        }
                    }
                    return error_response
        
        except Exception as e:
            logger.error(f"GLOBAL SCHEDULER CRITICAL ERROR in v1_chat_completions: {str(e)}", exc_info=True)
            error_response = {
                "error": {
                    "message": f"Critical internal error: {str(e)}",
                    "type": "server_error",
                    "code": "critical_error"
                }
            }
            return error_response
    
    @endpoint()
    async def get_pool_status(self, request: dict = None):
        """Get status of all registered pools."""
        pool_statuses = {}
        
        for pool_id, pool_config in self.pools.items():
            async with self.http_session.get(f"{pool_config.base_url}/health", timeout=30) as response:
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
        """Find the best available pool for the requested SLO level using random selection."""
        matching_pools = [pool for pool in self.pools.values() if pool.slo_level == slo_level]
        if matching_pools:
            selected_pool = random.choice(matching_pools)
            logger.info(f"RANDOM: Selected pool {selected_pool.pool_id} for {slo_level.value} SLO - {selected_pool.base_url}")
            return selected_pool
        if slo_level == SLOLevel.LOW:
            fallback_pools = [pool for pool in self.pools.values() 
                            if pool.slo_level in [SLOLevel.MEDIUM, SLOLevel.HIGH]]
        elif slo_level == SLOLevel.MEDIUM:
            fallback_pools = [pool for pool in self.pools.values() 
                            if pool.slo_level == SLOLevel.HIGH]
        else:
            fallback_pools = []
        if fallback_pools:
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
                    try:
                        # Handle Dynamo SDK Annotated response format with proper type checking
                        if hasattr(response_item, 'data'):
                            data = response_item.data
                        elif isinstance(response_item, dict):
                            data = response_item
                        else:
                            logger.warning(f"GLOBAL SCHEDULER: Unexpected response type from planner {pool_id}: {type(response_item)}")
                            continue
                        
                        # Ensure data is a dictionary before calling .get()
                        if not isinstance(data, dict):
                            logger.warning(f"GLOBAL SCHEDULER: Response data is not a dictionary for planner {pool_id}: {type(data)}")
                            continue
                            
                        if data.get("success", False):
                            metrics = data.get("metrics", {})
                            logger.info("=" * 60)
                            logger.info(f"GLOBAL SCHEDULER - RECEIVED METRICS FROM PLANNER: {pool_id}")
                            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
                            logger.info("=" * 60)
                        else:
                            logger.warning(f"GLOBAL SCHEDULER: Planner {pool_id} returned unsuccessful response: {data}")
                    except Exception as response_error:
                        logger.error(f"GLOBAL SCHEDULER ERROR: Failed to process response from planner {pool_id}: {response_error}", exc_info=True)
                        
            except Exception as e:
                logger.error(f"GLOBAL SCHEDULER ERROR: Could not request metrics from planner {pool_id}: {e}", exc_info=True)

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
                    try:
                        # Handle Dynamo SDK Annotated response format with proper type checking
                        if hasattr(response_item, 'data'):
                            data = response_item.data
                        elif isinstance(response_item, dict):
                            data = response_item
                        else:
                            logger.warning(f"GLOBAL SCHEDULER: Unexpected response type from planner {pool_id}: {type(response_item)}")
                            continue
                        
                        # Ensure data is a dictionary before calling .get()
                        if not isinstance(data, dict):
                            logger.warning(f"GLOBAL SCHEDULER: Response data is not a dictionary for planner {pool_id}: {type(data)}")
                            continue
                            
                        if data.get("success", False):
                            logger.info("=" * 60)
                            logger.info(f"GLOBAL SCHEDULER - SENT INSTRUCTIONS TO PLANNER: {pool_id}")
                            logger.info(f"Instructions: {json.dumps(instructions, indent=2)}")
                            logger.info("=" * 60)
                        else:
                            logger.warning(f"GLOBAL SCHEDULER: Planner {pool_id} returned unsuccessful response: {data}")
                    except Exception as response_error:
                        logger.error(f"GLOBAL SCHEDULER ERROR: Failed to process response from planner {pool_id}: {response_error}", exc_info=True)
                        
            except Exception as e:
                logger.error(f"GLOBAL SCHEDULER ERROR: Could not send instructions to planner {pool_id}: {e}", exc_info=True)

    @endpoint()
    async def receive_planner_metrics(self, request: dict):
        """Receive metrics from pool planners."""
        request_data = self._parse_request_data(request)
        planner_id = request_data.get("planner_id")
        metrics = request_data.get("metrics", {})
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
        """Send coordination data to requesting planner."""
        request_data = self._parse_request_data(request)
        planner_id = request_data.get("planner_id")
        if not planner_id:
            yield {"success": False, "error": "Missing required parameter: planner_id"}
            return
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