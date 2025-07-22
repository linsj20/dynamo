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

import argparse
import asyncio
import json
import logging
import random
import time

from pydantic import BaseModel

from components.planner import start_planner  # type: ignore[attr-defined]
from dynamo.planner.defaults import LoadPlannerDefaults
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service
from dynamo.sdk.core.protocol.interface import ComponentType
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "component_type": ComponentType.PLANNER,
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Planner:
    def __init__(self):
        configure_dynamo_logging(service_name="Planner")
        logger.info("Starting planner")
        self.runtime = dynamo_context["runtime"]

        config = ServiceConfig.get_instance()

        # Get namespace directly from dynamo_context as it contains the active namespace
        self.namespace = dynamo_context["namespace"]
        config_instance = config.get("Planner", {})

        # Extract gpu_scope from Common configuration
        common_config = config.get("Common", {})
        gpu_scope = common_config.get("gpu-scope", None)
        if gpu_scope:
            logger.info(f"Using GPU scope from Common config: {gpu_scope}")

        self.args = argparse.Namespace(
            namespace=self.namespace,
            gpu_scope=gpu_scope,
            environment=config_instance.get(
                "environment", LoadPlannerDefaults.environment
            ),
            no_operation=config_instance.get(
                "no-operation", LoadPlannerDefaults.no_operation
            ),
            log_dir=config_instance.get("log-dir", LoadPlannerDefaults.log_dir),
            adjustment_interval=config_instance.get(
                "adjustment-interval", LoadPlannerDefaults.adjustment_interval
            ),
            metric_pulling_interval=config_instance.get(
                "metric-pulling-interval", LoadPlannerDefaults.metric_pulling_interval
            ),
            max_gpu_budget=config_instance.get(
                "max-gpu-budget", LoadPlannerDefaults.max_gpu_budget
            ),
            min_endpoint=config_instance.get(
                "min-endpoint", LoadPlannerDefaults.min_endpoint
            ),
            decode_kv_scale_up_threshold=config_instance.get(
                "decode-kv-scale-up-threshold",
                LoadPlannerDefaults.decode_kv_scale_up_threshold,
            ),
            decode_kv_scale_down_threshold=config_instance.get(
                "decode-kv-scale-down-threshold",
                LoadPlannerDefaults.decode_kv_scale_down_threshold,
            ),
            prefill_queue_scale_up_threshold=config_instance.get(
                "prefill-queue-scale-up-threshold",
                LoadPlannerDefaults.prefill_queue_scale_up_threshold,
            ),
            prefill_queue_scale_down_threshold=config_instance.get(
                "prefill-queue-scale-down-threshold",
                LoadPlannerDefaults.prefill_queue_scale_down_threshold,
            ),
            decode_engine_num_gpu=config_instance.get(
                "decode-engine-num-gpu", LoadPlannerDefaults.decode_engine_num_gpu
            ),
            prefill_engine_num_gpu=config_instance.get(
                "prefill-engine-num-gpu", LoadPlannerDefaults.prefill_engine_num_gpu
            ),
        )
        
        # Communication setup
        self.planner_id = f"{self.namespace}_planner"
        self.communication_task = None
        self.planner_task = None
        
        logger.info(f"PLANNER COMMUNICATION INIT: Planner service initialized with id: {self.planner_id}")

    @async_on_start
    async def async_init(self):
        await asyncio.sleep(5)  # Reduced delay to ensure other services are ready
        logger.info("Calling start_planner")
        
        # Run planner in background task so async_init can complete
        self.planner_task = asyncio.create_task(start_planner(self.runtime, self.args))
        
        logger.info("Planner started in background task")
        
        # Start communication with global scheduler
        logger.info(f"PLANNER COMMUNICATION INIT: Starting communication task for {self.planner_id}")
        self.communication_task = asyncio.create_task(self._communication_loop())

    async def _communication_loop(self):
        """Communicate with global scheduler every 10 seconds"""
        logger.info(f"PLANNER COMMUNICATION: Starting communication loop for {self.planner_id}")
        while True:
            await asyncio.sleep(10)
            logger.info(f"PLANNER COMMUNICATION: Sending metrics for {self.planner_id}")
            await self._send_metrics_to_global_scheduler()
            await asyncio.sleep(5)
            logger.info(f"PLANNER COMMUNICATION: Requesting coordination data for {self.planner_id}")
            await self._request_coordination_data()

    async def _send_metrics_to_global_scheduler(self):
        """Send random metrics to global scheduler"""
        try:
            metrics = {
                "cpu_usage": random.uniform(0.2, 0.8),
                "memory_usage": random.uniform(0.3, 0.7),
                "worker_count": random.randint(1, 5),
                "queue_length": random.randint(0, 20),
                "requests_per_minute": random.randint(10, 100)
            }
            
            scheduler_component = self.runtime.namespace("dynamo").component("GlobalScheduler")
            metrics_endpoint = scheduler_component.endpoint("receive_planner_metrics")
            client = await metrics_endpoint.client()
            logger.info(f"PLANNER: Successfully connected to global scheduler metrics endpoint")
            
            request_data = {"planner_id": self.planner_id, "metrics": metrics}
            response = await client.generate(request_data)
            
            async for response_item in response:
                # Debug: Log the actual response item type and content
                logger.info(f"DEBUG: response_item type: {type(response_item)}, content: {response_item}")
                
                # Handle various Dynamo SDK response formats
                try:
                    if hasattr(response_item, 'data'):
                        data = response_item.data
                    elif hasattr(response_item, '__dict__'):
                        data = response_item.__dict__ 
                    elif isinstance(response_item, dict):
                        data = response_item
                    else:
                        logger.error(f"Unexpected response type: {type(response_item)}")
                        continue
                        
                    if data.get("success", False):
                        logger.info(f"Global scheduler acknowledged metrics: {data}")
                    else:
                        logger.error(f"Failed to send metrics: {data.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error processing response item: {e}, type: {type(response_item)}")
        except Exception as e:
            logger.info(f"Could not send metrics to global scheduler: {e}")

    async def _request_coordination_data(self):
        """Request coordination data from global scheduler"""
        try:
            scheduler_component = self.runtime.namespace("dynamo").component("GlobalScheduler")
            coordination_endpoint = scheduler_component.endpoint("send_coordination_data")
            client = await coordination_endpoint.client()
            logger.info(f"PLANNER: Successfully connected to global scheduler coordination endpoint")
            
            request_data = {"planner_id": self.planner_id}
            response = await client.generate(request_data)
            
            async for response_item in response:
                # Handle Dynamo SDK Annotated response format
                data = response_item.data if hasattr(response_item, 'data') else response_item
                if data.get("success", False):
                    coordination_data = data.get("coordination_data", {})
                    logger.info("=" * 60)
                    logger.info(f"PLANNER {self.planner_id} - RECEIVED COORDINATION DATA:")
                    logger.info(f"Data: {json.dumps(coordination_data, indent=2)}")
                    logger.info("=" * 60)
        except Exception as e:
            logger.info(f"Could not request coordination data from global scheduler: {e}")

    @endpoint()
    async def get_planner_metrics(self, request: dict):
        """
        Provide current metrics when requested by global scheduler.
        
        Args:
            request: Request from global scheduler containing:
                - requester_id: Identifier of the requesting global scheduler
                
        Yields:
            Current planner metrics
        """
        # Handle different request formats
        if isinstance(request, str):
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
        
        requester_id = request_data.get("requester_id", "unknown")
        
        # Generate current metrics
        current_metrics = {
            "cpu_usage": random.uniform(0.2, 0.8),
            "memory_usage": random.uniform(0.3, 0.7),
            "worker_count": random.randint(1, 5),
            "queue_length": random.randint(0, 20),
            "requests_per_minute": random.randint(10, 100),
            "timestamp": time.time()
        }
        
        logger.info("=" * 60)
        logger.info(f"PLANNER {self.planner_id} - PROVIDING METRICS TO: {requester_id}")
        logger.info(f"Metrics: {json.dumps(current_metrics, indent=2)}")
        logger.info("=" * 60)
        
        yield {
            "success": True,
            "planner_id": self.planner_id,
            "metrics": current_metrics
        }

    @endpoint()
    async def receive_coordination_instructions(self, request: dict):
        """
        Receive coordination instructions from global scheduler.
        
        Args:
            request: Instructions from global scheduler containing:
                - sender_id: Identifier of the sending global scheduler
                - instructions: Dictionary of coordination instructions
                
        Yields:
            Acknowledgment of received instructions
        """
        # Handle different request formats
        if isinstance(request, str):
            request_data = json.loads(request)
        elif isinstance(request, dict):
            request_data = request
        else:
            request_data = request
        
        sender_id = request_data.get("sender_id", "unknown")
        instructions = request_data.get("instructions", {})
        
        logger.info("=" * 60)
        logger.info(f"PLANNER {self.planner_id} - RECEIVED INSTRUCTIONS FROM: {sender_id}")
        logger.info(f"Instructions: {json.dumps(instructions, indent=2)}")
        logger.info("=" * 60)
        
        yield {
            "success": True,
            "planner_id": self.planner_id,
            "message": "Instructions received and acknowledged"
        }

    @endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"

    async def cleanup(self):
        """Cleanup background tasks on shutdown"""
        logger.info("Planner cleanup starting...")
        
        # Cancel communication task
        if self.communication_task:
            self.communication_task.cancel()
            try:
                await self.communication_task
            except asyncio.CancelledError:
                pass
        
        # Cancel planner task  
        if self.planner_task:
            self.planner_task.cancel()
            try:
                await self.planner_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Planner cleanup complete")
