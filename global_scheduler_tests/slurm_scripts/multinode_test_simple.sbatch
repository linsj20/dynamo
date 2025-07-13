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

# Extract the head node (first node in the list)
# SLURM_NODELIST format: eos[0067,0078] or eos0067,eos0078
HEAD_NODE=$(scontrol show hostname ${SLURM_NODELIST} | head -1)
echo "Head node: ${HEAD_NODE}"

# Use hostname directly - services can resolve it internally
HEAD_NODE_HOST=${HEAD_NODE}
echo "Using head node hostname: ${HEAD_NODE_HOST}"

# Container and mount configuration
CONTAINER_IMAGE="/home/shengjiel/project/dynamo-dev.sqsh"
CONTAINER_MOUNTS="/home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage,/project/coreai_comparch_sysarch/shengjiel/dynamo:/workspace,/project/coreai_comparch_sysarch/shengjiel/dynamo/.cache/huggingface:/root/.cache/huggingface"
CONTAINER_WORKDIR="/workspace/global_scheduler_tests"

# Test parameters
TEST_TYPE=${TEST_TYPE:-"simple"}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-"cluster"}
PD_DISAGG=${PD_DISAGG:-false}
MONITOR=${MONITOR:-false}

# Export variables for srun - including HEAD_NODE_HOST
export SLURM_JOB_ID TEST_TYPE DEPLOYMENT_MODE PD_DISAGG MONITOR HEAD_NODE_HOST

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
echo "Head node hostname (from env): ${HEAD_NODE_HOST}"

# Determine node role based on SLURM_PROCID
if [ "${SLURM_PROCID}" = "0" ]; then
    NODE_ROLE="head"
    echo "✅ HEAD node (SLURM_PROCID=0)"
else
    NODE_ROLE="worker"
    echo "✅ WORKER node (SLURM_PROCID=${SLURM_PROCID})"
fi

# Setup Python path
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/dynamo/sdk/src:/workspace/components/planner/src

# Set environment variables for services
export NATS_SERVER="nats://${HEAD_NODE_HOST}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_HOST}:2379"
export GLOBAL_SCHEDULER_HOST="${HEAD_NODE_HOST}"
export GLOBAL_SCHEDULER_PORT="3999"

echo "NATS_SERVER: ${NATS_SERVER}"
echo "ETCD_ENDPOINTS: ${ETCD_ENDPOINTS}"
echo "GLOBAL_SCHEDULER_HOST: ${GLOBAL_SCHEDULER_HOST}"

cd /workspace/global_scheduler_tests

if [ "${NODE_ROLE}" = "head" ]; then
    echo "=== HEAD NODE SETUP ==="
    
    # Start head node with infrastructure services and high SLO pool
    echo "Starting global scheduler on head node..."
    python runner.py \
        --test-type ${TEST_TYPE} \
        --deployment ${DEPLOYMENT_MODE} \
        --config-dir configs \
        $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") \
        $([ "${MONITOR}" = "true" ] && echo "--monitor") \
        --multinode \
        --head-node-ip ${HEAD_NODE_HOST} \
        --head-node-role head_and_high_slo
        
else
    echo "=== WORKER NODE SETUP ==="
    
    # Wait for head node services to be ready
    echo "Waiting for head node services to be ready..."
    sleep 20
    
    # Test connectivity to etcd (required for service registration)
    echo "Testing connectivity to etcd..."
    for i in {1..20}; do
        if curl -f "http://${HEAD_NODE_HOST}:2379/health" >/dev/null 2>&1; then
            echo "✅ etcd connectivity confirmed"
            break
        fi
        echo "Attempt $i/20: waiting for etcd..."
        sleep 5
    done
    
    # Start worker node with low SLO pool
    # The pool will register itself with the global scheduler and retry if needed
    echo "Starting low SLO pool on worker node..."
    echo "The pool will register itself with the global scheduler at ${GLOBAL_SCHEDULER_HOST}:${GLOBAL_SCHEDULER_PORT}"
    python runner.py \
        --test-type ${TEST_TYPE} \
        --deployment ${DEPLOYMENT_MODE} \
        --config-dir configs \
        $([ "${PD_DISAGG}" = "true" ] && echo "--pd-disagg") \
        --multinode \
        --head-node-ip ${HEAD_NODE_HOST} \
        --head-node-role low_slo \
        --worker-only
fi

echo "Node completed: $(hostname) (${NODE_ROLE})"
'

echo "Multi-node test completed"