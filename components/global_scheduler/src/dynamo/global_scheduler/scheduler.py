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
        
        logger.info("Global Scheduler initialized")

    @async_on_start
    async def async_init(self):
        """Initialize the Global Scheduler with HTTP session"""
        from dynamo.sdk import dynamo_context
        self.runtime = dynamo_context["runtime"]
        
        logger.info("Global Scheduler starting...")
        
        # Create HTTP session for pool communication with connection isolation
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=10,  # Max connections per host
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
        
        # Register the pool
        self.pools[pool_id] = pool_config
        
        logger.info(f"Registered pool: {pool_id} at {base_url} (SLO: {slo_level.value}, model: {model_name})")
        
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
            logger.info(f"Unregistered pool: {pool_id}")
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
        
        # CRITICAL: Add comprehensive logging to debug routing issues
        logger.info(f"Routing {request_id} to {pool_config.pool_id}")
        logger.info(f"DEBUG: Pool config - ID: {pool_config.pool_id}, SLO: {pool_config.slo_level.value}, URL: {pool_config.base_url}")
        
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
        
        # CRITICAL: Add pre-request logging to verify URL
        logger.info(f"DEBUG: Making HTTP request to URL: {url}")
        logger.info(f"DEBUG: Request ID: {request_id}, Pool: {pool_config.pool_id}")
        
        try:
            # Use the existing HTTP session for stable connections
            async with self.http_session.post(url, json=chat_request) as response:
                    logger.info(f"DEBUG: Received response from {url} with status {response.status}")
                    
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
                "error": f"HTTP request to pool {pool_config.pool_id} failed: {e}",
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
        
        yield {
            "timestamp": time.time(),
            "total_pools": len(self.pools),
            "connected_pools": len([p for p in pool_statuses.values() if p["connected"]]),
            "pools": pool_statuses
        }
    
    def _get_pool_for_slo(self, slo_level: SLOLevel) -> Optional[PoolConfig]:
        """
        Find the best available pool for the requested SLO level.
        
        Args:
            slo_level: SLO requirement level
            
        Returns:
            Pool configuration if available, None if no suitable pool found
        """
        # DEBUG: Log current pool state
        logger.info(f"DEBUG: Looking for pool with SLO level: {slo_level.value}")
        logger.info(f"DEBUG: Available pools: {[(p.pool_id, p.slo_level.value, p.base_url) for p in self.pools.values()]}")
        
        # Filter pools by SLO level
        matching_pools = [pool for pool in self.pools.values() if pool.slo_level == slo_level]
        
        if matching_pools:
            selected_pool = matching_pools[0]
            logger.info(f"DEBUG: Selected pool: {selected_pool.pool_id} (SLO: {selected_pool.slo_level.value}, URL: {selected_pool.base_url})")
            return selected_pool
        
        # Fallback: if no exact match, try to find a higher SLO level pool
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
            selected_pool = fallback_pools[0]
            logger.info(f"DEBUG: Using fallback pool: {selected_pool.pool_id} (SLO: {selected_pool.slo_level.value}, URL: {selected_pool.base_url})")
            return selected_pool
        
        logger.warning(f"DEBUG: No suitable pool found for SLO level: {slo_level.value}")
        return None

    async def cleanup(self):
        """Cleanup resources when shutting down"""
        logger.info("Global Scheduler cleanup starting...")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        
        logger.info("Global Scheduler cleanup complete")