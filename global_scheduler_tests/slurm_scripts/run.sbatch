#!/bin/bash
#SBATCH --job-name=coreai_comparch_sysarch-sj_dynamo-test.dev
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/shengjiel/project/dynamo/global_scheduler_tests/logs/slurm-test-%j-%t.out
#SBATCH --error=/home/shengjiel/project/dynamo/global_scheduler_tests/logs/slurm-test-%j-%t.err
#SBATCH --mail-type=BEGIN
#sSBATCH --mail-user=shengjiel@nvidia.com

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Flexible script for global scheduler tests with PD disaggregated architecture
# This script supports both single-node and multi-node configurations
# For single-node: launches one container and runs python runner.py
# For multi-node: launches containers on all nodes using srun
#
# USAGE:
#   Single-node execution:
#     sbatch --nodes=1 slurm_scripts/run.sbatch
#     MONITOR=true sbatch --nodes=1 slurm_scripts/run.sbatch
#   
#   Multi-node execution (1 high SLO node + 1 low SLO node to test TP):
#     sbatch --nodes=2 slurm_scripts/run.sbatch
#     MONITOR=true sbatch --nodes=2 slurm_scripts/run.sbatch
#   
#   Multi-node execution to test more nodes:
#     sbatch --nodes=3 slurm_scripts/run.sbatch  # 1 high + 2 low SLO nodes
#   
#   Environment variables:
#     TEST_TYPE: "simple" or "scaling" (default: "simple")
#     DEPLOYMENT_MODE: "local", "cluster", or "custom" (default: "cluster")
#     MONITOR: "true" or "false" (default: "false")
#     HIGH_SLO_NODE_COUNT: Number of nodes for high SLO pool (default: "2")
#     LOW_SLO_NODE_COUNT: Number of nodes for low SLO pool (default: "1")
#
# ARCHITECTURE:
#   - Single-node: All pool infrastructure and workers run on one node
#   - Multi-node: Each node runs its own pool (simpler architecture for testing TP)
#                 - First HIGH_SLO_NODE_COUNT nodes: run high SLO pools
#                 - Remaining LOW_SLO_NODE_COUNT nodes: run low SLO pools
#                 - Head node always runs infrastructure services
#
# FOCUS: Testing tensor parallelism (TP) configuration
#   - High SLO pools use TP=2
#   - Low SLO pools use TP=1
#
# COMPATIBILITY:
#   - Single-node: Equivalent to "python runner.py --monitor ..."
#   - Multi-node: Simple per-node pool architecture

set -e

echo "=== SLURM JOB INFORMATION ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Number of nodes: ${SLURM_NNODES}"
echo "Node list: ${SLURM_NODELIST}"

# Container and mount configuration
CONTAINER_IMAGE="/home/shengjiel/project/dynamo-dev.sqsh"
CONTAINER_MOUNTS="/home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage,/project/coreai_comparch_sysarch/shengjiel/dynamo:/workspace,/project/coreai_comparch_sysarch/shengjiel/dynamo/.cache/huggingface:/root/.cache/huggingface"
CONTAINER_WORKDIR="/workspace/global_scheduler_tests"

# Test parameters
TEST_TYPE=${TEST_TYPE:-"simple"}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-"cluster"}
MONITOR=${MONITOR:-false}

# Node count parameters for dynamic pool allocation
HIGH_SLO_NODE_COUNT=${HIGH_SLO_NODE_COUNT:-2}
LOW_SLO_NODE_COUNT=${LOW_SLO_NODE_COUNT:-1}

# Check if this is single-node or multi-node execution
if [ "${SLURM_NNODES}" -eq 1 ]; then
    echo "=== SINGLE-NODE EXECUTION ==="
    echo "Running on single node with PD disaggregated architecture"
    
    # For single-node, we don't need complex multi-node setup
    # Just launch a container and run python runner.py
    srun --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         --container-workdir="${CONTAINER_WORKDIR}" \
         --export=ALL \
         bash -c '
    set -e
    
    # Check container environment
    if [ ! -d "/workspace" ]; then
        echo "ERROR: /workspace directory not found"
        exit 1
    fi
    
    which python
    source /workspace/venv/bin/activate
    which python
    
    echo "✅ Container environment confirmed"
    echo "=== SINGLE-NODE EXECUTION ==="
    echo "Current node: $(hostname)"
    
    # Setup Python path
    export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
    
    cd /workspace/global_scheduler_tests
    
    echo "Starting Global Scheduler with PD disaggregated architecture on single node..."
    python runner.py \
        --test-type '${TEST_TYPE}' \
        --deployment local \
        --config-dir configs \
        $([ "${MONITOR}" = "true" ] && echo "--monitor")
    
    echo "Single-node execution completed: $(hostname)"
    '
    
else
    echo "=== MULTI-NODE EXECUTION ==="
    echo "Running on ${SLURM_NNODES} nodes with PD disaggregated architecture"
    
    # Extract the head node (first node in the list)
    HEAD_NODE=$(scontrol show hostname ${SLURM_NODELIST} | head -1)
    echo "Head node: ${HEAD_NODE}"
    
    # Use hostname directly - services can resolve it internally
    HEAD_NODE_HOST=${HEAD_NODE}
    echo "Using head node hostname: ${HEAD_NODE_HOST}"
    
    # Set total tasks to number of nodes for multi-node
    export SLURM_NTASKS=${SLURM_NNODES}
    
    # Export variables for srun
    export SLURM_JOB_ID TEST_TYPE DEPLOYMENT_MODE MONITOR HEAD_NODE_HOST HIGH_SLO_NODE_COUNT LOW_SLO_NODE_COUNT
    
    # Get all nodes and determine roles based on HIGH_SLO_NODE_COUNT and LOW_SLO_NODE_COUNT
    ALL_NODES=($(scontrol show hostname ${SLURM_NODELIST}))
    TOTAL_NODES=${#ALL_NODES[@]}
    
    echo "Total nodes: ${TOTAL_NODES}"
    echo "HIGH_SLO_NODE_COUNT: ${HIGH_SLO_NODE_COUNT}"
    echo "LOW_SLO_NODE_COUNT: ${LOW_SLO_NODE_COUNT}"
    
    # Validate node count configuration
    EXPECTED_TOTAL=$((HIGH_SLO_NODE_COUNT + LOW_SLO_NODE_COUNT))
    if [ ${TOTAL_NODES} -ne ${EXPECTED_TOTAL} ]; then
        echo "WARNING: Total nodes (${TOTAL_NODES}) doesn't match HIGH_SLO_NODE_COUNT (${HIGH_SLO_NODE_COUNT}) + LOW_SLO_NODE_COUNT (${LOW_SLO_NODE_COUNT}) = ${EXPECTED_TOTAL}"
        echo "Using actual node count and defaulting extra nodes to low SLO"
    fi
    
    # Determine which nodes run high SLO vs low SLO
    HIGH_SLO_NODES=()
    LOW_SLO_NODES=()
    
    for i in $(seq 0 $((TOTAL_NODES - 1))); do
        if [ ${i} -lt ${HIGH_SLO_NODE_COUNT} ]; then
            HIGH_SLO_NODES+=(${ALL_NODES[i]})
        else
            LOW_SLO_NODES+=(${ALL_NODES[i]})
        fi
    done
    
    echo "High SLO nodes: ${HIGH_SLO_NODES[@]}"
    echo "Low SLO nodes: ${LOW_SLO_NODES[@]}"
    
    # Initialize array to store worker PIDs
    WORKER_PIDS=()
    
    # Start high SLO worker nodes (excluding head node)
    echo "Starting high SLO worker nodes in background..."
    for node in "${HIGH_SLO_NODES[@]}"; do
        if [ "${node}" != "${HEAD_NODE}" ]; then
            echo "Starting high SLO worker node: ${node}"
            srun --nodes=1 --nodelist="${node}" \
                 --container-image="${CONTAINER_IMAGE}" \
                 --container-mounts="${CONTAINER_MOUNTS}" \
                 --container-workdir="${CONTAINER_WORKDIR}" \
                 --export=ALL \
                 bash -c "
            set -e
            
            # Check container environment
            if [ ! -d \"/workspace\" ]; then
                echo \"ERROR: /workspace directory not found\"
                exit 1
            fi
            
            which python
            source /workspace/venv/bin/activate
            which python
            
            echo \"✅ Container environment confirmed\"
            echo \"=== HIGH SLO WORKER NODE SETUP ===\"
            echo \"Current node: \$(hostname)\"
            
            # Setup Python path
            export PYTHONPATH=\$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
            
            # Set environment variables for services
            export NATS_SERVER=\"nats://${HEAD_NODE_HOST}:4222\"
            export ETCD_ENDPOINTS=\"${HEAD_NODE_HOST}:2379\"
            export GLOBAL_SCHEDULER_HOST=\"${HEAD_NODE_HOST}\"
            export GLOBAL_SCHEDULER_PORT=\"3999\"
            export HIGH_SLO_NODE_COUNT=\"${HIGH_SLO_NODE_COUNT}\"
            export LOW_SLO_NODE_COUNT=\"${LOW_SLO_NODE_COUNT}\"
            
            echo \"NATS_SERVER: \${NATS_SERVER}\"
            echo \"ETCD_ENDPOINTS: \${ETCD_ENDPOINTS}\"
            echo \"GLOBAL_SCHEDULER_HOST: \${GLOBAL_SCHEDULER_HOST}\"
            echo \"HIGH_SLO_NODE_COUNT: \${HIGH_SLO_NODE_COUNT}\"
            echo \"LOW_SLO_NODE_COUNT: \${LOW_SLO_NODE_COUNT}\"
            
            cd /workspace/global_scheduler_tests
            
            # Wait for head node services to be ready
            echo \"Waiting for head node services to be ready...\"
            sleep 20
            
            # Test connectivity to etcd (required for service registration)
            echo \"Testing connectivity to etcd...\"
            for i in {1..20}; do
                if curl -f \"http://${HEAD_NODE_HOST}:2379/health\" >/dev/null 2>&1; then
                    echo \"✅ etcd connectivity confirmed\"
                    break
                fi
                echo \"Attempt \$i/20: waiting for etcd...\"
                sleep 5
            done
            
            # Start worker node with high SLO pool
            echo \"Starting high SLO pool with PD disaggregated architecture on worker node...\"
            echo \"The pool will register itself with the global scheduler at \${GLOBAL_SCHEDULER_HOST}:\${GLOBAL_SCHEDULER_PORT}\"
            python runner.py \\
                --test-type '${TEST_TYPE}' \\
                --deployment '${DEPLOYMENT_MODE}' \\
                --config-dir configs \\
                --multinode \\
                --head-node-ip '${HEAD_NODE_HOST}' \\
                --head-node-role high_slo \\
                --worker-only
            
            echo \"High SLO worker node completed: \$(hostname)\"
            " &
            
            # Store background job PID
            WORKER_PIDS+=($!)
            echo "High SLO worker node ${node} started with PID ${!}"
        fi
    done
    
    # Start low SLO worker nodes 
    echo "Starting low SLO worker nodes in background..."
    for node in "${LOW_SLO_NODES[@]}"; do
        echo "Starting low SLO worker node: ${node}"
        srun --nodes=1 --nodelist="${node}" \
             --container-image="${CONTAINER_IMAGE}" \
             --container-mounts="${CONTAINER_MOUNTS}" \
             --container-workdir="${CONTAINER_WORKDIR}" \
             --export=ALL \
             bash -c "
        set -e
        
        # Check container environment
        if [ ! -d \"/workspace\" ]; then
            echo \"ERROR: /workspace directory not found\"
            exit 1
        fi
        
        which python
        source /workspace/venv/bin/activate
        which python
        
        echo \"✅ Container environment confirmed\"
        echo \"=== LOW SLO WORKER NODE SETUP ===\"
        echo \"Current node: \$(hostname)\"
        
        # Setup Python path
        export PYTHONPATH=\$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
        
        # Set environment variables for services
        export NATS_SERVER=\"nats://${HEAD_NODE_HOST}:4222\"
        export ETCD_ENDPOINTS=\"${HEAD_NODE_HOST}:2379\"
        export GLOBAL_SCHEDULER_HOST=\"${HEAD_NODE_HOST}\"
        export GLOBAL_SCHEDULER_PORT=\"3999\"
        export HIGH_SLO_NODE_COUNT=\"${HIGH_SLO_NODE_COUNT}\"
        export LOW_SLO_NODE_COUNT=\"${LOW_SLO_NODE_COUNT}\"
        
        echo \"NATS_SERVER: \${NATS_SERVER}\"
        echo \"ETCD_ENDPOINTS: \${ETCD_ENDPOINTS}\"
        echo \"GLOBAL_SCHEDULER_HOST: \${GLOBAL_SCHEDULER_HOST}\"
        echo \"HIGH_SLO_NODE_COUNT: \${HIGH_SLO_NODE_COUNT}\"
        echo \"LOW_SLO_NODE_COUNT: \${LOW_SLO_NODE_COUNT}\"
        
        cd /workspace/global_scheduler_tests
        
        # Wait for head node services to be ready
        echo \"Waiting for head node services to be ready...\"
        sleep 20
        
        # Test connectivity to etcd (required for service registration)
        echo \"Testing connectivity to etcd...\"
        for i in {1..20}; do
            if curl -f \"http://${HEAD_NODE_HOST}:2379/health\" >/dev/null 2>&1; then
                echo \"✅ etcd connectivity confirmed\"
                break
            fi
            echo \"Attempt \$i/20: waiting for etcd...\"
            sleep 5
        done
        
        # Start worker node with low SLO pool
        echo \"Starting low SLO pool with PD disaggregated architecture on worker node...\"
        echo \"The pool will register itself with the global scheduler at \${GLOBAL_SCHEDULER_HOST}:\${GLOBAL_SCHEDULER_PORT}\"
        python runner.py \\
            --test-type '${TEST_TYPE}' \\
            --deployment '${DEPLOYMENT_MODE}' \\
            --config-dir configs \\
            --multinode \\
            --head-node-ip '${HEAD_NODE_HOST}' \\
            --head-node-role low_slo \\
            --worker-only
        
        echo \"Low SLO worker node completed: \$(hostname)\"
        " &
        
        # Store background job PID
        WORKER_PIDS+=($!)
        echo "Low SLO worker node ${node} started with PID ${!}"
    done
    
    # Start head node and wait for completion
    echo "Starting head node: ${HEAD_NODE}"
    srun --nodes=1 --nodelist="${HEAD_NODE}" \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         --container-workdir="${CONTAINER_WORKDIR}" \
         --export=ALL \
         bash -c "
    set -e
    
    # Check container environment
    if [ ! -d \"/workspace\" ]; then
        echo \"ERROR: /workspace directory not found\"
        exit 1
    fi
    
    which python
    source /workspace/venv/bin/activate
    which python
    
    echo \"✅ Container environment confirmed\"
    echo \"=== HEAD NODE SETUP ===\"
    echo \"Current node: \$(hostname)\"
    
    # Setup Python path
    export PYTHONPATH=\$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
    
    # Set environment variables for services
    export NATS_SERVER=\"nats://${HEAD_NODE_HOST}:4222\"
    export ETCD_ENDPOINTS=\"${HEAD_NODE_HOST}:2379\"
    export GLOBAL_SCHEDULER_HOST=\"${HEAD_NODE_HOST}\"
    export GLOBAL_SCHEDULER_PORT=\"3999\"
    export HIGH_SLO_NODE_COUNT=\"${HIGH_SLO_NODE_COUNT}\"
    export LOW_SLO_NODE_COUNT=\"${LOW_SLO_NODE_COUNT}\"
    
    echo \"NATS_SERVER: \${NATS_SERVER}\"
    echo \"ETCD_ENDPOINTS: \${ETCD_ENDPOINTS}\"
    echo \"GLOBAL_SCHEDULER_HOST: \${GLOBAL_SCHEDULER_HOST}\"
    echo \"HIGH_SLO_NODE_COUNT: \${HIGH_SLO_NODE_COUNT}\"
    echo \"LOW_SLO_NODE_COUNT: \${LOW_SLO_NODE_COUNT}\"
    
    cd /workspace/global_scheduler_tests
    
    echo \"Starting global scheduler with PD disaggregated architecture on head node...\"
    
    # Determine head node role based on whether it's in high SLO nodes
    HEAD_NODE_ROLE=\"head_and_high_slo\"
    if [ \${HIGH_SLO_NODE_COUNT} -eq 0 ]; then
        HEAD_NODE_ROLE=\"low_slo\"
    fi
    
    # Start head node with infrastructure services and appropriate pool
    python runner.py \\
        --test-type '${TEST_TYPE}' \\
        --deployment '${DEPLOYMENT_MODE}' \\
        --config-dir configs \\
        \$([ \"${MONITOR}\" = \"true\" ] && echo \"--monitor\") \\
        --multinode \\
        --head-node-ip '${HEAD_NODE_HOST}' \\
        --head-node-role \${HEAD_NODE_ROLE}
    
    echo \"Head node completed: \$(hostname)\"
    "
    
    HEAD_EXIT_CODE=$?
    echo "Head node finished with exit code: ${HEAD_EXIT_CODE}"
    
    # Kill all worker nodes
    echo "Killing worker nodes..."
    for pid in "${WORKER_PIDS[@]}"; do
        echo "Killing worker PID: ${pid}"
        kill ${pid} 2>/dev/null || echo "Worker PID ${pid} already terminated"
        
        # Give process a moment to terminate gracefully
        sleep 2
        
        # Force kill if still running
        if kill -0 ${pid} 2>/dev/null; then
            echo "Force killing worker PID: ${pid}"
            kill -KILL ${pid} 2>/dev/null || echo "Worker PID ${pid} already terminated"
        fi
    done
    
    # Wait for worker nodes to finish
    echo "Waiting for worker nodes to complete..."
    for pid in "${WORKER_PIDS[@]}"; do
        wait ${pid} 2>/dev/null || echo "Worker PID ${pid} finished"
    done
    
    echo "All nodes completed"
    
fi

echo "Test execution completed"