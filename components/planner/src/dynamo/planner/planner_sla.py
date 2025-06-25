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
import logging

from pydantic import BaseModel

from dynamo.planner.defaults import SLAPlannerDefaults
from dynamo.planner.utils.planner_core import start_sla_planner
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service
from dynamo.sdk.core.protocol.interface import ComponentType
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)

# start planner 30 seconds after the other components to make sure planner can see them
# TODO: remove this delay
INIT_PLANNER_START_DELAY = 30


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
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

        # Get SLO level first to determine tensor parallelism settings
        slo_level = config_instance.get("slo-level", SLAPlannerDefaults.slo_level)
        
        # Apply SLO-aware tensor parallelism configuration
        tp_config = self._get_slo_tensor_parallelism_config(slo_level)
        
        self.args = argparse.Namespace(
            namespace=self.namespace,
            environment=config_instance.get(
                "environment", SLAPlannerDefaults.environment
            ),
            no_operation=config_instance.get(
                "no-operation", SLAPlannerDefaults.no_operation
            ),
            log_dir=config_instance.get("log-dir", SLAPlannerDefaults.log_dir),
            adjustment_interval=config_instance.get(
                "adjustment-interval", SLAPlannerDefaults.adjustment_interval
            ),
            max_gpu_budget=config_instance.get(
                "max-gpu-budget", SLAPlannerDefaults.max_gpu_budget
            ),
            min_endpoint=config_instance.get(
                "min-endpoint", SLAPlannerDefaults.min_endpoint
            ),
            # Apply SLO-aware tensor parallelism for decode/prefill engines
            decode_engine_num_gpu=config_instance.get(
                "decode-engine-num-gpu", tp_config["decode_tp"]
            ),
            prefill_engine_num_gpu=config_instance.get(
                "prefill-engine-num-gpu", tp_config["prefill_tp"]
            ),
            prometheus_endpoint=config_instance.get(
                "prometheus-endpoint", SLAPlannerDefaults.prometheus_endpoint
            ),
            profile_results_dir=config_instance.get(
                "profile-results-dir", SLAPlannerDefaults.profile_results_dir
            ),
            isl=config_instance.get("isl", SLAPlannerDefaults.isl),
            osl=config_instance.get("osl", SLAPlannerDefaults.osl),
            ttft=config_instance.get("ttft", SLAPlannerDefaults.ttft),
            itl=config_instance.get("itl", SLAPlannerDefaults.itl),
            load_predictor=config_instance.get(
                "load-predictor", SLAPlannerDefaults.load_predictor
            ),
            load_prediction_window_size=config_instance.get(
                "load-prediction-window-size",
                SLAPlannerDefaults.load_prediction_window_size,
            ),
            
            # SLO-specific configuration
            slo_level=slo_level,
            slo_priority=config_instance.get("slo-priority", SLAPlannerDefaults.slo_priority),
            slo_description=config_instance.get("slo-description", SLAPlannerDefaults.slo_description),
            
            # SLO-aware scaling behavior
            enable_proactive_scaling=config_instance.get(
                "enable-proactive-scaling", SLAPlannerDefaults.enable_proactive_scaling
            ),
            enable_burst_protection=config_instance.get(
                "enable-burst-protection", SLAPlannerDefaults.enable_burst_protection
            ),
            slo_violation_threshold=config_instance.get(
                "slo-violation-threshold", SLAPlannerDefaults.slo_violation_threshold
            ),
            scale_up_aggressiveness=config_instance.get(
                "scale-up-aggressiveness", SLAPlannerDefaults.scale_up_aggressiveness
            ),
            
            # Performance monitoring
            performance_buffer_ratio=config_instance.get(
                "performance-buffer-ratio", SLAPlannerDefaults.performance_buffer_ratio
            ),
            latency_percentile=config_instance.get(
                "latency-percentile", SLAPlannerDefaults.latency_percentile
            ),
            cost_optimization_mode=config_instance.get(
                "cost-optimization-mode", SLAPlannerDefaults.cost_optimization_mode
            ),
            
            # Scaling thresholds
            prefill_queue_scale_up_threshold=config_instance.get(
                "prefill-queue-scale-up-threshold", SLAPlannerDefaults.prefill_queue_scale_up_threshold
            ),
            prefill_queue_scale_down_threshold=config_instance.get(
                "prefill-queue-scale-down-threshold", SLAPlannerDefaults.prefill_queue_scale_down_threshold
            ),
            decode_kv_scale_up_threshold=config_instance.get(
                "decode-kv-scale-up-threshold", SLAPlannerDefaults.decode_kv_scale_up_threshold
            ),
            decode_kv_scale_down_threshold=config_instance.get(
                "decode-kv-scale-down-threshold", SLAPlannerDefaults.decode_kv_scale_down_threshold
            ),
            
            # Tensor parallelism configuration
            tensor_parallel_size_decode=tp_config["decode_tp"],
            tensor_parallel_size_prefill=tp_config["prefill_tp"],
            gpu_memory_utilization=tp_config["gpu_memory_util"],
        )
        
        # Log SLO-aware configuration at startup
        logger.info(f"SLA Planner initialized with SLO configuration:")
        logger.info(f"  SLO Level: {self.args.slo_level} (Priority: {self.args.slo_priority})")
        logger.info(f"  Description: {self.args.slo_description}")
        logger.info(f"  SLO Targets: TTFT={self.args.ttft}s, ITL={self.args.itl}s")
        logger.info(f"  Expected workload: ISL={self.args.isl}, OSL={self.args.osl}")
        logger.info(f"  Tensor Parallelism: Decode TP={self.args.decode_engine_num_gpu}, Prefill TP={self.args.prefill_engine_num_gpu}")
        logger.info(f"  GPU Memory Utilization: {self.args.gpu_memory_utilization}")
        logger.info(f"  Scaling behavior: {self.args.scale_up_aggressiveness} aggressiveness")
        logger.info(f"  Proactive scaling: {self.args.enable_proactive_scaling}")
        logger.info(f"  Burst protection: {self.args.enable_burst_protection}")
        logger.info(f"  SLO violation threshold: {self.args.slo_violation_threshold*100}%")
        logger.info(f"  Performance buffer: {self.args.performance_buffer_ratio*100}%")
        logger.info(f"  Cost optimization mode: {self.args.cost_optimization_mode}")
        logger.info(f"  TP Configuration: {tp_config['description']}")

    def _get_slo_tensor_parallelism_config(self, slo_level: str) -> dict:
        """Get tensor parallelism configuration based on SLO level"""
        tp_map = SLAPlannerDefaults.slo_tensor_parallelism_map
        
        if slo_level.upper() in tp_map:
            config = tp_map[slo_level.upper()]
            logger.info(f"Applying SLO-aware tensor parallelism for {slo_level} SLO: {config['description']}")
            return config
        else:
            logger.warning(f"Unknown SLO level '{slo_level}', using MEDIUM defaults")
            return tp_map["MEDIUM"]

    @async_on_start
    async def async_init(self):
        await asyncio.sleep(INIT_PLANNER_START_DELAY)
        logger.info("Calling start_planner")
        await start_sla_planner(self.runtime, self.args)
        logger.info("Planner started")

    @endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
