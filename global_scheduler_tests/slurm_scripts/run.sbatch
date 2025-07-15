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
#     sbatch --nodes=1 slurm_scripts/multinode_test_simple.sbatch
#     MONITOR=true sbatch --nodes=1 slurm_scripts/multinode_test_simple.sbatch
#   
#   Multi-node execution:
#     sbatch --nodes=2 slurm_scripts/multinode_test_simple.sbatch
#     MONITOR=true sbatch --nodes=2 slurm_scripts/multinode_test_simple.sbatch
#   
#   Environment variables:
#     TEST_TYPE: "simple" or "scaling" (default: "simple")
#     DEPLOYMENT_MODE: "local", "cluster", or "custom" (default: "cluster")
#     MONITOR: "true" or "false" (default: "false")
#
# COMPATIBILITY:
#   - Single-node: Equivalent to "python runner.py --monitor ..."
#   - Multi-node: Equivalent to "PD_DISAGG=true MONITOR=true sbatch ..." (legacy)

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
    export SLURM_JOB_ID TEST_TYPE DEPLOYMENT_MODE MONITOR HEAD_NODE_HOST
    
    # Get list of worker nodes (all nodes except head)
    WORKER_NODES=$(scontrol show hostname ${SLURM_NODELIST} | tail -n +2 | paste -sd,)
    echo "Worker nodes: ${WORKER_NODES}"
    
    # Initialize array to store worker PIDs
    WORKER_PIDS=()
    
    # Start worker nodes in the background
    echo "Starting worker nodes in background..."
    for worker_node in $(echo ${WORKER_NODES} | tr ',' ' '); do
        echo "Starting worker node: ${worker_node}"
        srun --nodes=1 --nodelist="${worker_node}" \
             --container-image="${CONTAINER_IMAGE}" \
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
        echo "=== WORKER NODE SETUP ==="
        echo "Current node: $(hostname)"
        
        # Setup Python path
        export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
        
        # Set environment variables for services
        export NATS_SERVER="nats://'${HEAD_NODE_HOST}':4222"
        export ETCD_ENDPOINTS="'${HEAD_NODE_HOST}':2379"
        export GLOBAL_SCHEDULER_HOST="'${HEAD_NODE_HOST}'"
        export GLOBAL_SCHEDULER_PORT="3999"
        
        echo "NATS_SERVER: ${NATS_SERVER}"
        echo "ETCD_ENDPOINTS: ${ETCD_ENDPOINTS}"
        echo "GLOBAL_SCHEDULER_HOST: ${GLOBAL_SCHEDULER_HOST}"
        
        cd /workspace/global_scheduler_tests
        
        # Wait for head node services to be ready
        echo "Waiting for head node services to be ready..."
        sleep 20
        
        # Test connectivity to etcd (required for service registration)
        echo "Testing connectivity to etcd..."
        for i in {1..20}; do
            if curl -f "http://'${HEAD_NODE_HOST}':2379/health" >/dev/null 2>&1; then
                echo "✅ etcd connectivity confirmed"
                break
            fi
            echo "Attempt $i/20: waiting for etcd..."
            sleep 5
        done
        
        # Start worker node with low SLO pool
        echo "Starting low SLO pool with PD disaggregated architecture on worker node..."
        echo "The pool will register itself with the global scheduler at ${GLOBAL_SCHEDULER_HOST}:${GLOBAL_SCHEDULER_PORT}"
        python runner.py \
            --test-type '${TEST_TYPE}' \
            --deployment '${DEPLOYMENT_MODE}' \
            --config-dir configs \
            --multinode \
            --head-node-ip '${HEAD_NODE_HOST}' \
            --head-node-role low_slo \
            --worker-only
        
        echo "Worker node completed: $(hostname)"
        ' &
        
        # Store background job PID
        WORKER_PIDS+=($!)
        echo "Worker node ${worker_node} started with PID ${!}"
    done
    
    # Start head node and wait for completion
    echo "Starting head node: ${HEAD_NODE}"
    srun --nodes=1 --nodelist="${HEAD_NODE}" \
         --container-image="${CONTAINER_IMAGE}" \
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
    echo "=== HEAD NODE SETUP ==="
    echo "Current node: $(hostname)"
    
    # Setup Python path
    export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
    
    # Set environment variables for services
    export NATS_SERVER="nats://'${HEAD_NODE_HOST}':4222"
    export ETCD_ENDPOINTS="'${HEAD_NODE_HOST}':2379"
    export GLOBAL_SCHEDULER_HOST="'${HEAD_NODE_HOST}'"
    export GLOBAL_SCHEDULER_PORT="3999"
    
    echo "NATS_SERVER: ${NATS_SERVER}"
    echo "ETCD_ENDPOINTS: ${ETCD_ENDPOINTS}"
    echo "GLOBAL_SCHEDULER_HOST: ${GLOBAL_SCHEDULER_HOST}"
    
    cd /workspace/global_scheduler_tests
    
    echo "Starting global scheduler with PD disaggregated architecture on head node..."
    
    # Start head node with infrastructure services and high SLO pool
    python runner.py \
        --test-type '${TEST_TYPE}' \
        --deployment '${DEPLOYMENT_MODE}' \
        --config-dir configs \
        $([ "'${MONITOR}'" = "true" ] && echo "--monitor") \
        --multinode \
        --head-node-ip '${HEAD_NODE_HOST}' \
        --head-node-role head_and_high_slo
    
    echo "Head node completed: $(hostname)"
    '
    
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