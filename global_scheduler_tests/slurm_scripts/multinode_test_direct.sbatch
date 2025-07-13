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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-node script for global scheduler tests
# This script launches containers on BOTH allocated nodes using srun

set -e

echo "=== SLURM JOB INFORMATION ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Number of nodes: ${SLURM_NNODES}"
echo "Node list: ${SLURM_NODELIST}"
echo "Running on hosts: $(echo $(scontrol show hostname))"

# Container and mount configuration
CONTAINER_IMAGE="/home/shengjiel/project/dynamo-dev.sqsh"
CONTAINER_MOUNTS="/home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage,/project/coreai_comparch_sysarch/shengjiel/dynamo:/workspace,/project/coreai_comparch_sysarch/shengjiel/dynamo/.cache/huggingface:/root/.cache/huggingface"
CONTAINER_WORKDIR="/workspace/global_scheduler_tests"

# Test parameters
TEST_TYPE=${TEST_TYPE:-"simple"}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-"cluster"}
PD_DISAGG=${PD_DISAGG:-false}
MONITOR=${MONITOR:-false}

# Export variables for srun
export SLURM_JOB_ID TEST_TYPE DEPLOYMENT_MODE PD_DISAGG MONITOR

# Launch containers on all nodes with direct command execution
echo "Launching containers on all nodes with srun..."

srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir="${CONTAINER_WORKDIR}" \
     --mpi=pmix \
     --export=ALL \
     bash -c '
set -e

# Check container environment
if [ ! -d "/workspace" ]; then
    echo "ERROR: /workspace directory not found"
    exit 1
fi

echo "✅ Container environment confirmed"
echo "=== NODE INFORMATION ==="
echo "Current node: $(hostname)"
echo "SLURM_PROCID: ${SLURM_PROCID}"
echo "SLURM_NODEID: ${SLURM_NODEID}"
echo "SLURM_LOCALID: ${SLURM_LOCALID}"

# Determine node role
if [ "${SLURM_PROCID}" = "0" ]; then
    NODE_ROLE="head"
    echo "✅ HEAD node (SLURM_PROCID=0)"
else
    NODE_ROLE="worker"
    echo "✅ WORKER node (SLURM_PROCID=${SLURM_PROCID})"
fi

# Common setup
COORDINATION_DIR="/project/coreai_comparch_sysarch/shengjiel/dynamo/coordination_${SLURM_JOB_ID}"
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src
mkdir -p "$COORDINATION_DIR"

CURRENT_NODE_IP=$(hostname -i)
echo "Node IP: ${CURRENT_NODE_IP}"
echo "Coordination dir: ${COORDINATION_DIR}"

cd /workspace/global_scheduler_tests

if [ "${NODE_ROLE}" = "head" ]; then
    echo "=== HEAD NODE SETUP ==="
    
    # Set environment
    export HEAD_NODE_IP="${CURRENT_NODE_IP}"
    export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
    export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
    export GLOBAL_SCHEDULER_HOST="${HEAD_NODE_IP}"
    export GLOBAL_SCHEDULER_PORT="3999"
    
    # Write coordination files
    echo "Writing coordination files..."
    echo "${HEAD_NODE_IP}" > "${COORDINATION_DIR}/head_node_ip"
    echo "${NATS_SERVER}" > "${COORDINATION_DIR}/nats_server"
    echo "${ETCD_ENDPOINTS}" > "${COORDINATION_DIR}/etcd_endpoints"
    echo "${GLOBAL_SCHEDULER_HOST}" > "${COORDINATION_DIR}/global_scheduler_host"
    echo "${GLOBAL_SCHEDULER_PORT}" > "${COORDINATION_DIR}/global_scheduler_port"
    touch "${COORDINATION_DIR}/ready"
    
    # Start head node with high SLO pool
    echo "Starting global scheduler on head node..."
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
    
    # Wait for coordination files
    echo "Waiting for head node coordination files..."
    WAIT_COUNT=0
    while [ ! -f "${COORDINATION_DIR}/ready" ] && [ $WAIT_COUNT -lt 300 ]; do
        echo "Waiting... (${WAIT_COUNT}/300)"
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    
    if [ ! -f "${COORDINATION_DIR}/ready" ]; then
        echo "ERROR: Coordination files not found"
        exit 1
    fi
    
    # Read coordination info
    export HEAD_NODE_IP=$(cat "${COORDINATION_DIR}/head_node_ip")
    export NATS_SERVER=$(cat "${COORDINATION_DIR}/nats_server")
    export ETCD_ENDPOINTS=$(cat "${COORDINATION_DIR}/etcd_endpoints")
    export GLOBAL_SCHEDULER_HOST=$(cat "${COORDINATION_DIR}/global_scheduler_host")
    export GLOBAL_SCHEDULER_PORT=$(cat "${COORDINATION_DIR}/global_scheduler_port")
    
    echo "Head node IP: ${HEAD_NODE_IP}"
    echo "NATS: ${NATS_SERVER}"
    
    # Wait for services
    echo "Waiting for head node services..."
    sleep 30
    
    # Test connectivity
    for i in {1..10}; do
        if curl -f "http://${HEAD_NODE_IP}:2379/health" >/dev/null 2>&1; then
            echo "✅ etcd connectivity confirmed"
            break
        fi
        echo "Attempt $i: waiting for etcd..."
        sleep 10
    done
    
    # Start worker node with low SLO pool
    echo "Starting low SLO pool on worker node..."
    python runner.py \
        --test-type ${TEST_TYPE} \
        --deployment ${DEPLOYMENT_MODE} \
        --config-dir configs \
        $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") \
        --multinode \
        --head-node-ip ${HEAD_NODE_IP} \
        --head-node-role low_slo \
        --worker-only
fi

echo "Node completed: $(hostname) (${NODE_ROLE})"
'

echo "Multi-node test completed" 