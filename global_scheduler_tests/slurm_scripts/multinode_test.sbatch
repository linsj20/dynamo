#!/bin/bash
#SBATCH --job-name=coreai_comparch_sysarch-sj_dynamo-multinode.dev
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/shengjiel/project/dynamo/global_scheduler_tests/logs/slurm-multinode-%j-%t.out
#SBATCH --error=/home/shengjiel/project/dynamo/global_scheduler_tests/logs/slurm-multinode-%j-%t.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=shengjiel@nvidia.com
#SBATCH --container-image=/home/shengjiel/project/dynamo-dev.sqsh
#SBATCH --container-mounts=/home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage,/project/coreai_comparch_sysarch/shengjiel/dynamo:/workspace,/project/coreai_comparch_sysarch/shengjiel/dynamo/.cache/huggingface:/root/.cache/huggingface
#SBATCH --container-workdir=/workspace/global_scheduler_tests

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-node script for global scheduler tests
# This script runs on BOTH allocated nodes automatically
# SLURM_PROCID=0 runs on first node (head), SLURM_PROCID=1 runs on second node (worker)
#
# Usage:
#   Basic run:
#     sbatch multinode_test.sbatch
#
#   With monitoring:
#     MONITOR=true sbatch multinode_test.sbatch
#
#   With PD disaggregation:
#     PD_DISAGG=true sbatch multinode_test.sbatch
#
#   With both monitoring and PD disaggregation:
#     MONITOR=true PD_DISAGG=true sbatch multinode_test.sbatch
#
#   Custom test type:
#     TEST_TYPE=custom_test sbatch multinode_test.sbatch
#
# Environment variables:
#   TEST_TYPE: Test type to run (default: simple)
#   DEPLOYMENT_MODE: Deployment mode (default: cluster)
#   PD_DISAGG: Enable PD disaggregation (default: false)
#   MONITOR: Enable monitoring (default: false)

set -e

# Check if we're in the container environment
if [ ! -d "/workspace" ]; then
    echo "ERROR: /workspace directory not found. Container environment not properly set up."
    echo "Current working directory: $(pwd)"
    echo "Available directories:"
    ls -la /
    exit 1
fi

echo "✅ Container environment confirmed - /workspace exists"

echo "=== NODE INFORMATION ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Number of nodes: ${SLURM_NNODES}"
echo "Node list: ${SLURM_NODELIST}"
echo "Current node: $(hostname)"

# Print SLURM environment variables for debugging
echo "=== SLURM ENVIRONMENT ==="
echo "SLURM_PROCID: ${SLURM_PROCID:-not set}"
echo "SLURM_LOCALID: ${SLURM_LOCALID:-not set}"
echo "SLURM_NODEID: ${SLURM_NODEID:-not set}"
echo "SLURM_TASK_PID: ${SLURM_TASK_PID:-not set}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-not set}"

# CRITICAL: Determine node role based on SLURM_PROCID
# When using --nodes=2 --ntasks-per-node=1, this script runs on BOTH nodes
# SLURM_PROCID=0 on first node, SLURM_PROCID=1 on second node
if [ "${SLURM_PROCID}" = "0" ]; then
    NODE_ROLE="head"
    echo "✅ Detected as HEAD node (SLURM_PROCID=0)"
elif [ "${SLURM_PROCID}" = "1" ]; then
    NODE_ROLE="worker"
    echo "✅ Detected as WORKER node (SLURM_PROCID=1)"
else
    echo "ERROR: Unexpected SLURM_PROCID value: ${SLURM_PROCID}"
    echo "This script expects exactly 2 tasks (--nodes=2 --ntasks-per-node=1)"
    exit 1
fi

# Paths for container execution
GLOBAL_SCHEDULER_TESTS_DIR="/workspace/global_scheduler_tests"
LOGS_DIR="${GLOBAL_SCHEDULER_TESTS_DIR}/logs"
# Use a shared location accessible from all nodes
COORDINATION_DIR="/project/coreai_comparch_sysarch/shengjiel/dynamo/coordination_${SLURM_JOB_ID}"

# Setup environment for container execution
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src

# Create coordination directory
mkdir -p "$COORDINATION_DIR"

# Get current node info
CURRENT_NODE=$(hostname)
CURRENT_NODE_SHORT=$(hostname -s)
CURRENT_NODE_IP=$(hostname -i)

echo "Current node: ${CURRENT_NODE}"
echo "Current node (short): ${CURRENT_NODE_SHORT}"
echo "Current node IP: ${CURRENT_NODE_IP}"
echo "Node role: ${NODE_ROLE}"

# Set test parameters
TEST_TYPE=${TEST_TYPE:-"simple"}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-"cluster"}
PD_DISAGG=${PD_DISAGG:-false}
MONITOR=${MONITOR:-false}

echo "Test configuration:"
echo "  TEST_TYPE: ${TEST_TYPE}"
echo "  DEPLOYMENT_MODE: ${DEPLOYMENT_MODE}"
echo "  PD_DISAGG: ${PD_DISAGG}"
echo "  MONITOR: ${MONITOR}"
echo "  Logs directory: ${LOGS_DIR}"
echo "  Coordination directory: ${COORDINATION_DIR}"

cd /workspace/global_scheduler_tests

if [ "${NODE_ROLE}" = "head" ]; then
    echo "=== HEAD NODE SETUP ==="
    
    # Get head node IP
    export HEAD_NODE_IP="${CURRENT_NODE_IP}"
    export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
    export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
    export GLOBAL_SCHEDULER_HOST="${HEAD_NODE_IP}"
    export GLOBAL_SCHEDULER_PORT="3999"
    
    echo "Head node IP: ${HEAD_NODE_IP}"
    echo "NATS server: ${NATS_SERVER}"
    echo "ETCD endpoints: ${ETCD_ENDPOINTS}"
    
    # Write coordination files for worker nodes
    echo "Writing coordination files..."
    echo "${HEAD_NODE_IP}" > "${COORDINATION_DIR}/head_node_ip"
    echo "${NATS_SERVER}" > "${COORDINATION_DIR}/nats_server"
    echo "${ETCD_ENDPOINTS}" > "${COORDINATION_DIR}/etcd_endpoints"
    echo "${GLOBAL_SCHEDULER_HOST}" > "${COORDINATION_DIR}/global_scheduler_host"
    echo "${GLOBAL_SCHEDULER_PORT}" > "${COORDINATION_DIR}/global_scheduler_port"
    echo "${CURRENT_NODE_SHORT}" > "${COORDINATION_DIR}/head_node_name"
    
    # Mark as ready
    touch "${COORDINATION_DIR}/ready"
    
    # Verify files were written
    echo "Coordination files written:"
    ls -la "${COORDINATION_DIR}/"
    
    # Run the existing runner with head node parameters
    echo "Starting global scheduler tests on head node..."
    echo "Command: python runner.py --test-type ${TEST_TYPE} --deployment ${DEPLOYMENT_MODE} --config-dir configs $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") $([ "${MONITOR}" = "true" ] && echo "--monitor") --multinode --head-node-ip ${HEAD_NODE_IP} --head-node-role head_and_high_slo"
    
    python runner.py \
        --test-type ${TEST_TYPE} \
        --deployment ${DEPLOYMENT_MODE} \
        --config-dir configs \
        $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") \
        $([ "${MONITOR}" = "true" ] && echo "--monitor") \
        --multinode \
        --head-node-ip ${HEAD_NODE_IP} \
        --head-node-role head_and_high_slo
        
else
    echo "=== WORKER NODE SETUP ==="
    
    # Wait for head node to be ready and read coordination files
    echo "Waiting for head node coordination files..."
    MAX_WAIT=300  # 5 minutes
    WAIT_COUNT=0
    
    while [ ! -f "${COORDINATION_DIR}/ready" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        echo "Waiting for head node to write coordination files... (${WAIT_COUNT}/${MAX_WAIT})"
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    
    if [ ! -f "${COORDINATION_DIR}/ready" ]; then
        echo "ERROR: Coordination files not found after ${MAX_WAIT} seconds"
        echo "Checking coordination directory:"
        ls -la "${COORDINATION_DIR}/" || echo "Coordination directory does not exist"
        exit 1
    fi
    
    # Read coordination information from head node
    export HEAD_NODE_IP=$(cat "${COORDINATION_DIR}/head_node_ip")
    export NATS_SERVER=$(cat "${COORDINATION_DIR}/nats_server")
    export ETCD_ENDPOINTS=$(cat "${COORDINATION_DIR}/etcd_endpoints")
    export GLOBAL_SCHEDULER_HOST=$(cat "${COORDINATION_DIR}/global_scheduler_host")
    export GLOBAL_SCHEDULER_PORT=$(cat "${COORDINATION_DIR}/global_scheduler_port")
    HEAD_NODE_NAME=$(cat "${COORDINATION_DIR}/head_node_name")
    
    echo "Head node name: ${HEAD_NODE_NAME}"
    echo "Head node IP: ${HEAD_NODE_IP}"
    echo "NATS server: ${NATS_SERVER}"
    echo "ETCD endpoints: ${ETCD_ENDPOINTS}"
    
    # Wait a bit more for head node services to be fully ready
    echo "Waiting for head node services to be ready..."
    sleep 30
    
    # Test connectivity to head node services
    echo "Testing connectivity to head node services..."
    for i in {1..10}; do
        if curl -f "http://${HEAD_NODE_IP}:2379/health" >/dev/null 2>&1; then
            echo "etcd connectivity confirmed"
            break
        fi
        echo "Attempt $i: etcd not ready, waiting..."
        sleep 10
    done
    
    # Determine worker role (for now, all workers are low_slo)
    WORKER_ROLE=${WORKER_ROLE:-"low_slo"}
    echo "Worker role: ${WORKER_ROLE}"
    
    # Run the existing runner with worker-specific parameters
    echo "Starting global scheduler tests on worker node..."
    echo "Command: python runner.py --test-type ${TEST_TYPE} --deployment ${DEPLOYMENT_MODE} --config-dir configs $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") --multinode --head-node-ip ${HEAD_NODE_IP} --head-node-role ${WORKER_ROLE} --worker-only"
    
    python runner.py \
        --test-type ${TEST_TYPE} \
        --deployment ${DEPLOYMENT_MODE} \
        --config-dir configs \
        $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") \
        --multinode \
        --head-node-ip ${HEAD_NODE_IP} \
        --head-node-role ${WORKER_ROLE} \
        --worker-only
fi

echo "Node ${CURRENT_NODE} (${NODE_ROLE}) completed" 