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


# Source of truth for planner defaults
class BasePlannerDefaults:
    namespace = "dynamo"
    environment = "local"
    no_operation = False
    log_dir = None
    adjustment_interval = 180  # in seconds
    max_gpu_budget = 8
    min_endpoint = 1  # applies to both decode and prefill
    decode_engine_num_gpu = 1
    prefill_engine_num_gpu = 1


class LoadPlannerDefaults(BasePlannerDefaults):
    metric_pulling_interval = 10  # in seconds
    decode_kv_scale_up_threshold = 0.9
    decode_kv_scale_down_threshold = 0.5
    prefill_queue_scale_up_threshold = 5.0
    prefill_queue_scale_down_threshold = 0.2


class SLAPlannerDefaults(BasePlannerDefaults):
    prometheus_endpoint = "http://localhost:9090"
    profile_results_dir = "profiling_results"
    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft = 0.5  # in seconds
    itl = 0.05  # in seconds
    load_predictor = "arima"  # ["constant", "arima", "prophet"]
    load_prediction_window_size = 50  # predict load using how many recent load samples
    
    # SLO-specific configuration defaults
    slo_level = "MEDIUM"  # SLO level: HIGH, MEDIUM, LOW
    slo_priority = 2  # Priority level (1=highest, 3=lowest)
    slo_description = "Standard SLA planner"
    
    # SLO-aware scaling behavior
    enable_proactive_scaling = True  # Scale before hitting limits
    enable_burst_protection = False  # Extra capacity for traffic bursts
    slo_violation_threshold = 0.1  # Allowable SLO violation percentage
    scale_up_aggressiveness = "medium"  # Scaling aggressiveness: high, medium, low
    
    # Performance monitoring
    performance_buffer_ratio = 0.85  # Scale when at X% of SLO target
    latency_percentile = 90  # Monitor Xth percentile latency
    cost_optimization_mode = False  # Enable cost optimization features
    
    # Scaling thresholds (will be adjusted based on SLO level)
    prefill_queue_scale_up_threshold = 2.0
    prefill_queue_scale_down_threshold = 0.5
    decode_kv_scale_up_threshold = 0.7
    decode_kv_scale_down_threshold = 0.3
    
    # Tensor parallelism configuration (SLO-aware)
    # These values get adjusted based on SLO level:
    # HIGH SLO: TP=2 for best performance within 8 GPU budget
    # MEDIUM SLO: TP=1 for balanced performance/cost
    # LOW SLO: TP=1 for maximum cost efficiency
    tensor_parallel_size_decode = 1    # Default TP for decode workers
    tensor_parallel_size_prefill = 1   # Default TP for prefill workers
    gpu_memory_utilization = 0.9       # GPU memory utilization
    
    # SLO-aware tensor parallelism mapping (optimized for 8 GPUs total)
    slo_tensor_parallelism_map = {
        "HIGH": {
            "decode_tp": 2,
            "prefill_tp": 2,
            "gpu_memory_util": 0.95,
            "description": "Medium TP for best performance within 8 GPU budget"
        },
        "MEDIUM": {
            "decode_tp": 1,
            "prefill_tp": 1,
            "gpu_memory_util": 0.9,
            "description": "Low TP for balanced performance/cost"
        },
        "LOW": {
            "decode_tp": 1,
            "prefill_tp": 1,
            "gpu_memory_util": 0.85,
            "description": "Low TP for maximum cost efficiency"
        }
    }
