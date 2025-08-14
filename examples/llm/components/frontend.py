# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Frontend service that automatically registers with Global Scheduler.

Multi-node Configuration:
- GLOBAL_SCHEDULER_URL: Full URL to Global Scheduler (highest priority)
- GLOBAL_SCHEDULER_HOST: Hostname/IP of Global Scheduler (default: localhost)
- GLOBAL_SCHEDULER_PORT: Port of Global Scheduler (default: 3999)
- POOL_BASE_URL: Full URL for this pool (highest priority)
- POOL_HOST: Hostname/IP for this pool (default: NODE_IP, HOSTNAME, or localhost)
- NODE_IP: Node's external IP address for multi-node deployments
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

from components.planner_service import Planner
from components.processor import Processor
from components.worker import VllmWorker
from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import api, async_on_start, depends, on_shutdown, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE
from dynamo.sdk import dynamo_context

logger = logging.getLogger(__name__)

# TODO: temp workaround to avoid port conflict with subprocess HTTP server; remove this once ingress is fixed
# TEMPORARILY DISABLED for Global Scheduler testing - let YAML configs control ports
# os.environ["DYNAMO_PORT"] = "3999"


def get_http_binary_path():
    """Find the HTTP binary path in SDK or fallback to 'http' command."""
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    """Configuration for the Frontend service including model and HTTP server settings."""

    served_model_name: str
    endpoint: str
    port: int = 8080


def determine_pool_info(namespace: str, hostname: str = None):
    """Determine pool ID and SLO level from namespace, with optional hostname for multinode deployments."""
    # Determine SLO level from namespace
    if "high" in namespace.lower():
        slo_level = "high"
    elif "low" in namespace.lower():
        slo_level = "low"
    elif "medium" in namespace.lower():
        slo_level = "medium"
    else:
        raise RuntimeError(f"Invalid pool namespace: {namespace}")
    
    # For multinode deployments, include hostname in pool_id to ensure uniqueness
    if hostname and hostname != "localhost":
        # Clean hostname to be safe for pool_id (remove dots, etc)
        hostname_safe = hostname.replace(".", "_").replace("-", "_")
        pool_id = f"{slo_level}_slo_pool_{hostname_safe}"
    else:
        # Single node or local deployment - use simple pool_id
        pool_id = f"{slo_level}_slo_pool"
    
    return pool_id, slo_level


async def register_with_global_scheduler(pool_id: str, slo_level: str, base_url: str, namespace: str, model_name: str):
    """Register this pool with the Global Scheduler using Dynamo runtime - CRITICAL operation that must succeed."""
    runtime = dynamo_context["runtime"]
    
    registration_data = {
        "pool_id": pool_id,
        "slo_level": slo_level,
        "base_url": base_url,
        "namespace": namespace,
        "model_name": model_name,
        "description": f"{slo_level.title()} priority pool in {namespace} namespace"
    }
    
    logger.info(f"Registering pool {pool_id} with Global Scheduler using Dynamo runtime")
    logger.info(f"Registration data: {registration_data}")
    
    scheduler_component = runtime.namespace("dynamo").component("GlobalScheduler")
    
    register_endpoint = scheduler_component.endpoint("register_pool")
    register_client = await register_endpoint.client()
    
    response_stream = await register_client.generate(registration_data)
    
    async for result in response_stream:
        logger.info(f"Registration result: {result}")
        
        if isinstance(result, dict) and not result.get("success", False):
            raise RuntimeError(f"Pool registration failed: {result.get('error', 'Unknown error')}")
        
        logger.info(f"Successfully registered pool {pool_id}")
        break  # Only expect one response for registration

async def unregister_with_global_scheduler(pool_id: str):
    """Unregister this pool from the Global Scheduler using Dynamo runtime."""
    runtime = dynamo_context["runtime"]
    
    unregistration_data = {"pool_id": pool_id}
    
    logger.info(f"Unregistering pool {pool_id} from Global Scheduler using Dynamo runtime")
    
    scheduler_component = runtime.namespace("dynamo").component("GlobalScheduler")
    
    unregister_endpoint = scheduler_component.endpoint("unregister_pool")
    unregister_client = await unregister_endpoint.client()
    
    response_stream = await unregister_client.generate(unregistration_data)
    
    async for result in response_stream:
        logger.info(f"Unregistration result: {result}")
        
        # Check if unregistration was successful
        if isinstance(result, dict) and not result.get("success", False):
            raise RuntimeError(f"Pool unregistration failed: {result.get('error', 'Unknown error')}")
        
        logger.info(f"Successfully unregistered pool {pool_id}")
        break  # Only expect one response for unregistration


# todo this should be called ApiServer
@service(
    dynamo={},  # Empty dynamo config allows dynamic namespace from command line
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Frontend:
    planner = depends(Planner)
    worker = depends(VllmWorker)
    processor = depends(Processor)

    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        frontend_config = FrontendConfig(**ServiceConfig.get_parsed_config("Frontend"))
        self.frontend_config = frontend_config
        self.process = None
        self.pool_id = None
        self.slo_level = None
        self.setup_model()
        self.start_http_server()
        
        self.namespace = dynamo_context["namespace"]
        self.base_url = self._construct_pool_base_url()
        
        import urllib.parse
        parsed_url = urllib.parse.urlparse(self.base_url)
        hostname = parsed_url.hostname or "localhost"
        self.pool_id, self.slo_level = determine_pool_info(self.namespace, hostname)

    def _construct_pool_base_url(self):
        """Construct the base URL for this pool, supporting multi-node deployments."""
        pool_url = os.environ.get("POOL_BASE_URL")
        if pool_url:
            return pool_url
        
        pool_host = os.environ.get("POOL_HOST")
        if not pool_host:
            pool_host = os.environ.get("NODE_IP") or os.environ.get("HOSTNAME") or "localhost"
        return f"http://{pool_host}:{self.frontend_config.port}"

    def setup_model(self):
        """Configure the model for HTTP service using llmctl."""
        namespace = dynamo_context.get("namespace")
        if namespace and namespace != "dynamo":
            endpoint = f"{namespace}.Processor.chat/completions"
        else:
            endpoint = self.frontend_config.endpoint
        
        # Remove existing model registration (with namespace parameter)
        subprocess.run(
            [
                "llmctl",
                "-n", namespace,
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ],
            check=False,
        )
        
        # Add model registration with the correct namespace and endpoint
        subprocess.run(
            [
                "llmctl",
                "-n", namespace,
                "http",
                "add",
                "chat-models",
                self.frontend_config.served_model_name,
                endpoint,
            ],
            check=False,
        )

    def start_http_server(self):
        """Start the HTTP server on the configured port."""
        logger.info("Starting HTTP server")
        http_binary = get_http_binary_path()

        self.process = subprocess.Popen(
            [http_binary, "-p", str(self.frontend_config.port)],
            stdout=None,
            stderr=None,
        )

    @async_on_start
    async def register_with_global_scheduler_on_ready(self):
        """Register this pool with the Global Scheduler - REQUIRED for operation."""
        logger.info("Waiting for Processor to be ready before registering with Global Scheduler...")
        
        runtime = dynamo_context["runtime"]
        namespace = dynamo_context["namespace"]
        
        # Get Processor client and wait for it to be responsive
        processor_client = (
            await runtime.namespace(namespace)
            .component("Processor")
            .endpoint("chat/completions")
            .client()
        )
        
        # Wait for Global Scheduler to be ready
        await asyncio.sleep(5)
        
        await register_with_global_scheduler(
            pool_id=self.pool_id,
            slo_level=self.slo_level,
            base_url=self.base_url,
            namespace=self.namespace,
            model_name=self.frontend_config.served_model_name
        )
        logger.info(f"Pool {self.pool_id} registration completed successfully")

    @api()
    def dummy_api(self) -> None:
        """
        Dummy API to enable the HTTP server for the Dynamo operator.
        This API is not used by the model.

        NOTE: this is a temporary solution to expose ingress
        for the LLM examples. Will be fixed and removed in the future.
        The resulting api_endpoints in dynamo.yaml will be incorrect.
        """

    @on_shutdown
    def cleanup(self):
        """Clean up resources before shutdown."""
        if self.pool_id:
            asyncio.run(unregister_with_global_scheduler(self.pool_id))

        namespace = dynamo_context.get("namespace")
        subprocess.run(
            [
                "llmctl",
                "-n", namespace,
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ],
            check=False,
        )
