# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set up deployment target FIRST, before any decorators are imported
import sys
import os

# Ensure we have the deployment target set up before importing any decorators
try:
    from dynamo.sdk.core.lib import get_target
    # Check if target is already set
    try:
        target = get_target()
        print(f"PASS: Target already set: {target}")
    except (NameError, RuntimeError) as e:
        print(f"WARNING: Target not set, setting up: {e}")
        from dynamo.sdk.core.lib import set_target
        from dynamo.sdk.core.runner.dynamo import LocalDeploymentTarget
        
        target = LocalDeploymentTarget()
        set_target(target)
        print(f"PASS: Target set successfully: {target}")
except Exception as e:
    print(f"FAIL: Failed to set up target: {e}")
    # Continue anyway - maybe we're in a different context

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from dynamo.sdk import service, endpoint, async_on_start

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
    namespace: str
    router_component: str  # Usually "Router" or "Frontend"
    router_endpoint: str   # Usually "generate" or "chat_completions"
    description: str

@dataclass
class Request:
    """A request with SLO requirements"""
    request_id: str
    slo_level: SLOLevel
    timestamp: float
    assigned_pool: Optional[str] = None

@service(
    dynamo={
        "namespace": "global-scheduler",
    },
    resources={"cpu": "2", "memory": "4Gi"},
    workers=1,
)
class SimpleGlobalScheduler:
    """
    Simple Global Scheduler that routes requests to different SLO-based pools.
    
    Each pool is a separate namespace with:
    - Router component (handles requests)
    - SLA Planner (manages auto-scaling)
    - Workers (process requests)
    
    The global scheduler just routes incoming requests to the appropriate
    pool's router based on SLO requirements.
    """
    
    def __init__(self):
        # Configure available pools (each pool is a separate namespace)
        self.pools: Dict[str, PoolConfig] = {}
        self._configure_pools()
        
        # Track active requests for metrics
        self.active_requests: Dict[str, Request] = {}
        
        # Pool router clients (will be initialized in async_init)
        self.pool_clients: Dict[str, Any] = {}
        
        # Metrics
        self.total_requests = 0
        self.successful_routes = 0
        self.failed_routes = 0
        
        logger.info("Simple Global Scheduler initialized")
    
    def _configure_pools(self):
        """Configure the available pool namespaces"""
        # NOTE: Due to namespace override issues, both pools are currently in 'dynamo' namespace
        # This is a temporary workaround until proper namespace isolation is implemented
        self.pools["high_slo"] = PoolConfig(
            pool_id="high_slo",
            slo_level=SLOLevel.HIGH,
            namespace="dynamo",  # Temporarily using dynamo namespace
            router_component="Router",
            router_endpoint="generate",
            description="High priority pool - <1s response time"
        )
        
        self.pools["low_slo"] = PoolConfig(
            pool_id="low_slo",
            slo_level=SLOLevel.LOW,
            namespace="dynamo",  # Temporarily using dynamo namespace
            router_component="Router",
            router_endpoint="generate",
            description="Low priority pool - >10s response time"
        )
        
        logger.info(f"Configured {len(self.pools)} pool namespaces")
    
    @async_on_start
    async def async_init(self):
        """Initialize connections to pool routers"""
        from dynamo.sdk import dynamo_context
        self.runtime = dynamo_context["runtime"]
        
        # Initialize clients for each pool's router
        for pool_id, pool_config in self.pools.items():
            try:
                self.pool_clients[pool_id] = (
                    await self.runtime
                    .namespace(pool_config.namespace)
                    .component(pool_config.router_component)
                    .endpoint(pool_config.router_endpoint)
                    .client()
                )
                logger.info(f"Connected to pool router: {pool_id} -> {pool_config.namespace}/{pool_config.router_component}")
            except Exception as e:
                logger.error(f"Failed to connect to pool {pool_id}: {e}")
        
        logger.info("Global Scheduler async initialization complete")
    
    async def _do_route_request(self, request_id: str, slo_requirement: str, **kwargs):
        """
        Internal method to route a request to appropriate pool router.
        This is the actual implementation without endpoint decoration.
        
        Args:
            request_id: Unique identifier for the request
            slo_requirement: "low", "medium", or "high" SLO level
            **kwargs: Additional request parameters (prompt, max_tokens, etc.)
            
        Returns:
            Response from the pool router
        """
        # Convert individual parameters to request dict for internal processing
        request_data = {
            "request_id": request_id,
            "slo_requirement": slo_requirement,
            **kwargs
        }
        
        if not request_id:
            self.failed_routes += 1
            return {
                "success": False,
                "error": "Missing required parameter: request_id",
                "assigned_pool": None
            }
            
        if not slo_requirement:
            self.failed_routes += 1
            return {
                "success": False,
                "error": "Missing required parameter: slo_requirement",
                "assigned_pool": None
            }
        
        logger.info(f"Routing request {request_id} with SLO: {slo_requirement}")
        self.total_requests += 1
        
        try:
            # Parse SLO level
            slo_level = SLOLevel(slo_requirement.lower())
        except ValueError:
            self.failed_routes += 1
            return {
                "success": False,
                "error": f"Invalid SLO requirement: {slo_requirement}. Must be 'low', 'medium', or 'high'",
                "assigned_pool": None
            }
        
        # Create request object for tracking
        request = Request(
            request_id=request_id,
            slo_level=slo_level,
            timestamp=time.time()
        )
        
        # Find appropriate pool
        pool_id = self._get_pool_for_slo(slo_level)
        pool_config = self.pools[pool_id]
        
        # Check if we have a client for this pool
        if pool_id not in self.pool_clients:
            self.failed_routes += 1
            return {
                "success": False,
                "error": f"No connection to pool: {pool_id}",
                "assigned_pool": pool_id,
                "pool_namespace": pool_config.namespace
            }
        
        try:
            # Route request to pool's router
            pool_client = self.pool_clients[pool_id]
            
            # Create a copy of request data for forwarding (excluding our routing parameters)
            forwarded_request = dict(request_data)
            
            # Add routing metadata to the request
            forwarded_request["_routing_info"] = {
                "request_id": request_id,
                "slo_requirement": slo_requirement,
                "routed_by": "global-scheduler",
                "assigned_pool": pool_id,
                "timestamp": request.timestamp
            }
            
            # Track active request
            request.assigned_pool = pool_id
            self.active_requests[request_id] = request
            
            # Forward request to pool router
            logger.info(f"Forwarding request {request_id} to pool {pool_id} ({pool_config.namespace})")
            
            # Convert prompt to tokens for the Router's generate endpoint
            # For now, use a simple tokenization (in real system, would use proper tokenizer)
            prompt = forwarded_request.get("prompt", "Hello")
            # Simple token approximation: 1 token per 4 characters
            tokens = list(range(len(prompt) // 4 + 1))
            
            # Create a simple dict that matches the Tokens structure
            tokens_dict = {"tokens": tokens}
            
            # Call the pool router's generate endpoint (pass dict as JSON)
            # The Router's generate endpoint returns AsyncIterator[Tuple[WorkerId, float]]
            # We need to disable annotation to get raw tuples instead of Annotated objects
            import json
            pool_response_stream = await pool_client.generate(json.dumps(tokens_dict), annotated=False)
            
            # Consume the async iterator to get the routing decision
            worker_id = None
            score = 0.0
            async for worker_id, score in pool_response_stream:
                # Router yields exactly one result
                break
            
            self.successful_routes += 1
            logger.info(f"Successfully routed request {request_id} to pool {pool_id} -> worker {worker_id} (score: {score})")
            
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            # Create a proper response object
            pool_response = {
                "success": True,
                "assigned_worker": worker_id,
                "routing_score": score,
                "request_id": request_id,
                "_routing_metadata": {
                    "assigned_pool": pool_id,
                    "pool_namespace": pool_config.namespace,
                    "routing_timestamp": request.timestamp,
                    "slo_requirement": slo_requirement
                }
            }
            
            return pool_response
            
        except Exception as e:
            self.failed_routes += 1
            logger.error(f"Failed to route request {request_id} to pool {pool_id}: {e}")
            
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            return {
                "success": False,
                "error": f"Routing failed: {str(e)}",
                "assigned_pool": pool_id,
                "pool_namespace": pool_config.namespace
            }
    
    @endpoint()
    async def route_request(self, request_id: str, slo_requirement: str, **kwargs):
        """
        Route a request to appropriate pool router based on SLO requirement.
        
        Args:
            request_id: Unique identifier for the request
            slo_requirement: "low", "medium", or "high" SLO level
            **kwargs: Additional request parameters (prompt, max_tokens, etc.)
            
        Returns:
            Response from the pool router
        """
        return await self._do_route_request(request_id, slo_requirement, **kwargs)
    
    @endpoint()
    async def generate(self, request: dict):
        """
        Generate endpoint that accepts a request object and routes it appropriately.
        This provides a generate interface similar to other components.
        
        Args:
            request: Request object containing request_id, slo_requirement, and other parameters
            
        Yields:
            Response from the appropriate pool router
        """
        # Handle different request formats
        if isinstance(request, dict):
            request_data = request
        elif hasattr(request, '__dict__'):
            request_data = request.__dict__
        elif hasattr(request, 'model_dump'):
            request_data = request.model_dump()
        else:
            # Assume it's a request with request_id and slo_requirement as parameters
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
        
        # Use the private routing method to avoid endpoint wrapper issues
        result = await self._do_route_request(
            request_id,
            slo_requirement,
            **{k: v for k, v in request_data.items() if k not in ["request_id", "slo_requirement"]}
        )
        
        # Yield the result
        yield result
    
    async def _cleanup_request(self, request_id: str, delay: int = 60):
        """Clean up completed request after a delay"""
        await asyncio.sleep(delay)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
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
                # Try to get basic status from pool router
                connected = pool_id in self.pool_clients
                
                pool_statuses[pool_id] = {
                    "pool_config": {
                        "slo_level": pool_config.slo_level.value,
                        "namespace": pool_config.namespace,
                        "router_component": pool_config.router_component,
                        "router_endpoint": pool_config.router_endpoint,
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
                        "namespace": pool_config.namespace,
                        "router_component": pool_config.router_component,
                        "router_endpoint": pool_config.router_endpoint,
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
    
    @endpoint()
    async def get_metrics(self):
        """
        Get global scheduler metrics.
        
        Returns:
            Dict with routing metrics and active request info
        """
        # Calculate active requests per pool
        active_by_pool = {}
        for request in self.active_requests.values():
            pool = request.assigned_pool or "unassigned"
            active_by_pool[pool] = active_by_pool.get(pool, 0) + 1
        
        return {
            "timestamp": time.time(),
            "routing_metrics": {
                "total_requests": self.total_requests,
                "successful_routes": self.successful_routes,
                "failed_routes": self.failed_routes,
                "success_rate": (self.successful_routes / self.total_requests) if self.total_requests > 0 else 0.0
            },
            "active_requests": {
                "total": len(self.active_requests),
                "by_pool": active_by_pool
            },
            "pool_connections": {
                "total_pools": len(self.pools),
                "connected_pools": len(self.pool_clients)
            }
        }
    
    @endpoint()
    async def update_pool_config(self, pool_id: str, **config_updates):
        """
        Update configuration for a specific pool.
        
        Args:
            pool_id: ID of the pool to update
            **config_updates: Configuration parameters to update
            
        Returns:
            Dict with update status
        """
        if pool_id not in self.pools:
            return {
                "success": False,
                "error": f"Pool {pool_id} not found. Available pools: {list(self.pools.keys())}"
            }
        
        try:
            # Update pool configuration
            pool_config = self.pools[pool_id]
            
            # Apply configuration updates (be careful to only update allowed fields)
            allowed_updates = ["description"]  # Add more fields as needed
            
            updated_fields = []
            for field, value in config_updates.items():
                if field in allowed_updates:
                    setattr(pool_config, field, value)
                    updated_fields.append(field)
                else:
                    logger.warning(f"Ignoring update to restricted field: {field}")
            
            logger.info(f"Updated pool {pool_id} configuration: {updated_fields}")
            
            return {
                "success": True,
                "pool_id": pool_id,
                "updated_fields": updated_fields,
                "current_config": {
                    "pool_id": pool_config.pool_id,
                    "slo_level": pool_config.slo_level.value,
                    "namespace": pool_config.namespace,
                    "router_component": pool_config.router_component,
                    "router_endpoint": pool_config.router_endpoint,
                    "description": pool_config.description
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to update pool {pool_id} configuration: {e}")
            return {
                "success": False,
                "error": f"Configuration update failed: {str(e)}"
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