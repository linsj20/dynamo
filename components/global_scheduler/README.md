# Simple Global Scheduler for Dynamo

A simplified global scheduler implementation that routes requests to different SLO-based pools without complex inter-pool resource conflicts.

## Architecture

Instead of managing pools internally, this global scheduler leverages existing Dynamo components:

- **Global Scheduler**: Routes requests based on SLO requirements 
- **SLO Pools**: Separate namespaces, each containing:
  - **Router**: Handles LLM requests within the pool
  - **SLA Planner**: Auto-scales workers based on SLO targets (now SLO-aware!)
  - **Workers**: Process requests (VllmWorker, PrefillWorker)

## Key Features

### SLO-Aware SLA Planners
Each pool's SLA planner is now explicitly aware of its SLO characteristics at initialization:

- **SLO Level Awareness**: Planners know their priority (HIGH/MEDIUM/LOW) and adjust behavior accordingly
- **Adaptive Scaling**: Different scaling aggressiveness based on SLO requirements
- **Performance Buffers**: SLO-specific performance buffer ratios (High: 80%, Medium: 85%, Low: 95%)
- **Cost Optimization**: Low SLO pools enable cost optimization features

### SLO-Aware Tensor Parallelism
**NEW**: Each pool automatically selects optimal tensor parallelism based on its SLO requirements (optimized for 8 GPUs total):

- **High SLO Pools**: TP=2 for best performance within GPU budget (strict latency requirements)
- **Medium SLO Pools**: TP=1 for balanced performance/cost 
- **Low SLO Pools**: TP=1 for maximum cost efficiency (best effort service)

This allows the system to automatically trade off performance vs cost based on SLO requirements without manual configuration.

### SLO-Specific Behaviors

**High SLO Pool (Premium Service)**:
- **Tensor Parallelism**: TP=2 for decode and prefill workers
- **GPU Budget**: Up to 4 GPUs (2 workers × 2 GPUs each)
- **GPU Memory**: 95% utilization for maximum performance
- Aggressive scaling (50% utilization triggers scale-up)
- Proactive scaling enabled
- Burst protection for traffic spikes
- 95th percentile latency monitoring
- Only 5% SLO violations allowed
- ARIMA load prediction for accuracy

**Medium SLO Pool (Standard Service)**:
- **Tensor Parallelism**: TP=1 for decode and prefill workers
- **GPU Budget**: Up to 3 GPUs (3 workers × 1 GPU each)
- **GPU Memory**: 90% utilization for balanced approach
- Balanced scaling (70% utilization triggers scale-up)
- Moderate proactive scaling
- 90th percentile latency monitoring
- 10% SLO violations allowed
- ARIMA load prediction

**Low SLO Pool (Best Effort)**:
- **Tensor Parallelism**: TP=1 for decode and prefill workers
- **GPU Budget**: Up to 1 GPU (1 worker × 1 GPU)
- **GPU Memory**: 85% utilization for stability and cost optimization
- Conservative scaling (90% utilization triggers scale-up)
- Reactive scaling to minimize costs
- 80th percentile latency monitoring
- 20% SLO violations allowed
- Simple constant load prediction
- Cost optimization mode enabled

## How It Works

### 1. Request Routing
```python
# Client specifies SLO requirement
request = {
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "slo_level": "high"  # Routes to high-slo-pool
}
```

### 2. SLO-Aware Scaling with Tensor Parallelism
Each pool's SLA planner receives SLO-aware configuration at initialization:

```yaml
# High SLO Pool Configuration
Planner:
  # SLO Level Awareness
  slo_level: "HIGH"
  slo_priority: 1
  slo_description: "Premium service with strict latency requirements"
  
  # SLO targets
  ttft: 0.3    # 300ms target
  itl: 0.03    # 30ms target
  
  # Auto-configured tensor parallelism for high SLO (8 GPU setup)
  decode_engine_num_gpu: 2    # TP=2 for decode workers
  prefill_engine_num_gpu: 2   # TP=2 for prefill workers
  
  # Aggressive scaling for high SLO
  prefill_queue_scale_up_threshold: 0.5
  decode_kv_scale_up_threshold: 0.5
  enable_proactive_scaling: true
  enable_burst_protection: true
  performance_buffer_ratio: 0.8
```

### 3. Automatic Tensor Parallelism Selection
The SLA planner automatically chooses TP configuration based on SLO level (optimized for 8 GPUs):

```python
# SLO-aware TP mapping (built into planner, optimized for 8 GPUs)
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
```

### 4. Intelligent Scaling Decisions
The SLA planner uses its SLO awareness to make better scaling decisions:

- **Performance vs Cost Trade-offs**: High SLO prioritizes performance with higher TP, Low SLO optimizes costs with lower TP
- **Resource Allocation**: Different TP configurations automatically adjust GPU budget and worker scaling
- **Proactive vs Reactive**: High SLO scales before hitting limits, Low SLO waits for necessity
- **Buffer Management**: Different performance buffers based on SLO strictness
- **Load Prediction**: More sophisticated prediction for stricter SLOs

## Components

### SimpleGlobalScheduler
- Routes requests to appropriate pools based on SLO level
- Tracks request metrics and routing success
- No complex resource management or inter-pool conflicts

### Pool Namespaces
Each pool is an independent namespace with its own:
- Router (request handling)
- SLA Planner (SLO-aware auto-scaling with TP selection)
- Workers (request processing with SLO-appropriate TP configuration)

## Configuration

See `examples/llm/configs/global_scheduler_demo.yaml` for complete configuration showing:
- SLO-aware SLA planner settings
- Automatic tensor parallelism selection per SLO level (optimized for 8 GPUs)
- Different scaling behaviors per pool
- SLO-specific monitoring and alerting

## Usage

```bash
# Run the global scheduler demo (requires 8 GPUs)
dynamo-run --config examples/llm/configs/global_scheduler_demo.yaml

# Send requests with SLO requirements
python examples/llm/components/simple_client.py
```

## Benefits

1. **Simplified Architecture**: Uses existing components, no custom pool managers
2. **SLO-Aware Scaling**: Each planner optimizes for its specific SLO requirements
3. **Automatic TP Selection**: No manual tensor parallelism configuration needed
4. **Performance/Cost Balance**: High SLO pools prioritize performance, Low SLO optimizes costs
5. **Resource Efficiency**: Different TP configurations optimize GPU usage per SLO level
6. **No Resource Conflicts**: Pools operate independently in separate namespaces
7. **Observability**: Clear separation of metrics per SLO level
8. **Flexible Configuration**: Easy to adjust SLO targets and scaling behavior per pool

## Monitoring

The system provides SLO-specific monitoring:
- SLO compliance rates per pool
- Cost per SLO level
- Scaling efficiency metrics
- Resource utilization by SLO priority
- Tensor parallelism efficiency per pool

Each SLA planner logs its SLO configuration at startup, including:
```
SLA Planner initialized with SLO configuration:
  SLO Level: HIGH (Priority: 1)
  Description: Premium service with strict latency requirements
  SLO Targets: TTFT=0.3s, ITL=0.03s
  Expected workload: ISL=2000, OSL=150
  Tensor Parallelism: Decode TP=2, Prefill TP=2
  GPU Memory Utilization: 0.95
  Scaling behavior: high aggressiveness
  TP Configuration: Medium TP for best performance within 8 GPU budget
```

## Resource Allocation Summary (8 GPUs Total)

| SLO Level | Tensor Parallelism | Max GPUs | GPU Memory | Scaling Strategy | Workers |
|-----------|-------------------|----------|------------|------------------|---------|
| **HIGH**  | TP=2             | 4 GPUs   | 95%        | Aggressive       | 2 max   |
| **MEDIUM**| TP=1             | 3 GPUs   | 90%        | Balanced         | 3 max   |
| **LOW**   | TP=1             | 1 GPU    | 85%        | Conservative     | 1 max   |

**Total GPU Usage**: 4 + 3 + 1 = 8 GPUs maximum (when all pools are fully scaled)

This automatic tensor parallelism selection ensures that each SLO level gets the appropriate performance/cost trade-off while staying within the 8 GPU budget constraint. 