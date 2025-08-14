#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

RUN_PREFIX=

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["VLLM"]=1 ["TENSORRTLLM"]=2 ["SGLANG"]=3 ["VLLM_V1"]=4)
DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

HF_CACHE=
DEFAULT_HF_CACHE=${SOURCE_DIR}/.cache/huggingface
MOUNT_WORKSPACE=
REMAINING_ARGS=
WORKDIR=/workspace

# SLURM/srun configuration (default behavior)
SRUN_PARTITION="interactive"
SRUN_ACCOUNT="coreai_comparch_sysarch"
SRUN_JOB_NAME="coreai_comparch_sysarch-sj_dynamo.dev"
SRUN_CONTAINER_IMAGE="/home/shengjiel/project/dynamo-dev.sqsh"
SRUN_CONTAINER_SAVE="/home/shengjiel/project/dynamo-dev.sqsh"
SRUN_CONTAINER_MOUNTS="/home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage"
SRUN_DURATION=""

# Function to convert common time formats to SLURM format
convert_duration() {
    local duration="$1"
    
    # If already in SLURM format (contains colons or dashes), return as-is
    if [[ "$duration" =~ ^[0-9]+(:[0-9]+)*(:[0-9]+)*$|^[0-9]+-[0-9]+(:[0-9]+)*(:[0-9]+)*$ ]]; then
        echo "$duration"
        return
    fi
    
    # Convert common formats to SLURM format
    if [[ "$duration" =~ ^([0-9]+)h$ ]]; then
        # Format: 2h -> 2:00:00
        hours="${BASH_REMATCH[1]}"
        echo "${hours}:00:00"
    elif [[ "$duration" =~ ^([0-9]+)m$ ]]; then
        # Format: 30m -> 30:00
        minutes="${BASH_REMATCH[1]}"
        echo "${minutes}:00"
    elif [[ "$duration" =~ ^([0-9]+)h([0-9]+)m$ ]]; then
        # Format: 1h30m -> 1:30:00
        hours="${BASH_REMATCH[1]}"
        minutes="${BASH_REMATCH[2]}"
        echo "${hours}:${minutes}:00"
    elif [[ "$duration" =~ ^([0-9]+)d$ ]]; then
        # Format: 1d -> 1-00:00:00
        days="${BASH_REMATCH[1]}"
        echo "${days}-00:00:00"
    elif [[ "$duration" =~ ^([0-9]+)d([0-9]+)h$ ]]; then
        # Format: 1d2h -> 1-02:00:00
        days="${BASH_REMATCH[1]}"
        hours="${BASH_REMATCH[2]}"
        printf "%d-%02d:00:00" "$days" "$hours"
    else
        # If format not recognized, return as-is and let SLURM handle it
        echo "$duration"
    fi
}

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
        --framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --hf-cache)
            if [ "$2" ]; then
                HF_CACHE=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --workdir)
            if [ "$2" ]; then
                WORKDIR="$2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --mount-workspace)
            MOUNT_WORKSPACE=TRUE
            ;;
        --srun-partition)
            if [ "$2" ]; then
                SRUN_PARTITION=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --srun-account)
            if [ "$2" ]; then
                SRUN_ACCOUNT=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --srun-job-name)
            if [ "$2" ]; then
                SRUN_JOB_NAME=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --srun-container-image)
            if [ "$2" ]; then
                SRUN_CONTAINER_IMAGE=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --srun-container-save)
            if [ "$2" ]; then
                SRUN_CONTAINER_SAVE=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --srun-container-mounts)
            if [ "$2" ]; then
                SRUN_CONTAINER_MOUNTS=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --duration)
            if [ "$2" ]; then
                SRUN_DURATION=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            ;;
        --)
            shift
            break
            ;;
         -?*)
            error 'ERROR: Unknown option: ' "$1"
            ;;
         ?*)
            error 'ERROR: Unknown option: ' "$1"
            ;;
        *)
            break
            ;;
        esac

        shift
    done

    if [ -z "$FRAMEWORK" ]; then
        FRAMEWORK=$DEFAULT_FRAMEWORK
    fi

    if [ -n "$FRAMEWORK" ]; then
        FRAMEWORK=${FRAMEWORK^^}
        if [[ -z "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
            error 'ERROR: Unknown framework: ' "$FRAMEWORK"
        fi
    fi

    if [ -n "$MOUNT_WORKSPACE" ]; then
        # Add dynamo directory to container mounts for /workspace
        SRUN_CONTAINER_MOUNTS="${SRUN_CONTAINER_MOUNTS},${SOURCE_DIR}:/workspace"

        if [ -z "$HF_CACHE" ]; then
            HF_CACHE=$DEFAULT_HF_CACHE
        fi
    fi

    if [[ ${HF_CACHE^^} == "NONE" ]]; then
        HF_CACHE=
    fi

    if [ -n "$HF_CACHE" ]; then
        mkdir -p "$HF_CACHE"
        # Add HF cache to container mounts
        SRUN_CONTAINER_MOUNTS="${SRUN_CONTAINER_MOUNTS},${HF_CACHE}:/root/.cache/huggingface"
    fi
    REMAINING_ARGS=("$@")
}

show_help() {
    echo "usage: run.sh [options] [command]"
    echo ""
    echo "Dynamo SLURM Container Runner - Uses srun with container images"
    echo ""
    echo "Default Configuration:"
    echo "  --srun-partition: batch"
    echo "  --srun-account: coreai_comparch_sysarch"
    echo "  --srun-job-name: coreai_comparch_sysarch-sj_dynamo.dev"
    echo "  --srun-container-image: /home/shengjiel/project/dynamo-dev.sqsh"
    echo "  --srun-container-save: /home/shengjiel/project/dynamo-dev.sqsh"
    echo "  --srun-container-mounts: /home/shengjiel/project:/home/shengjiel/project,/home/shengjiel/storage:/home/shengjiel/storage"
    echo ""
    echo "Options:"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--dry-run print srun commands without running]"
    echo "  [--hf-cache directory to mount as HF cache]"
    echo "  [--mount-workspace mount dynamo directory to /workspace in container]"
    echo "  [--workdir set the working directory inside the container]"
    echo "  [--duration time limit for the job (e.g., 2h, 30m, 1h30m, 1d, 1d2h, or SLURM format)]"
    echo ""
    echo "SLURM Configuration Override:"
    echo "  [--srun-partition SLURM partition to use]"
    echo "  [--srun-account SLURM account to use]"
    echo "  [--srun-job-name SLURM job name]"
    echo "  [--srun-container-image path to container image (.sqsh file)]"
    echo "  [--srun-container-save path to save container image]"
    echo "  [--srun-container-mounts container mount specification]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh bash                           # Start interactive bash session"
    echo "  ./run.sh --mount-workspace bash        # Mount dynamo to /workspace"
    echo "  ./run.sh --dry-run bash                # Show command without running"
    echo "  ./run.sh --duration 2h bash            # Run for 2 hours maximum"
    echo "  ./run.sh --duration 1h30m ./run_global_scheduler_test_fixed.sh  # Run test for 1.5 hours"
    echo "  ./run.sh dynamo-run --config config.yaml  # Run dynamo with config"
    echo ""
    echo "The script automatically:"
    echo "  - Uses your default SLURM configuration"
    echo "  - Mounts /home/shengjiel/project/dynamo to /workspace when --mount-workspace is used"
    echo "  - Enables interactive terminal (--pty)"
    echo "  - Handles HuggingFace cache mounting"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

# RUN with srun

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

# Build srun command
SRUN_CMD="srun"

if [ -n "$SRUN_PARTITION" ]; then
    SRUN_CMD+=" --partition $SRUN_PARTITION"
fi

if [ -n "$SRUN_ACCOUNT" ]; then
    SRUN_CMD+=" --account $SRUN_ACCOUNT"
fi

if [ -n "$SRUN_JOB_NAME" ]; then
    SRUN_CMD+=" --job-name $SRUN_JOB_NAME"
fi

if [ -n "$SRUN_CONTAINER_IMAGE" ]; then
    SRUN_CMD+=" --container-image=$SRUN_CONTAINER_IMAGE"
fi

if [ -n "$SRUN_CONTAINER_SAVE" ]; then
    SRUN_CMD+=" --container-save=$SRUN_CONTAINER_SAVE"
fi

if [ -n "$SRUN_CONTAINER_MOUNTS" ]; then
    SRUN_CMD+=" --container-mounts=$SRUN_CONTAINER_MOUNTS"
fi

if [ -n "$SRUN_DURATION" ]; then
    CONVERTED_DURATION=$(convert_duration "$SRUN_DURATION")
    SRUN_CMD+=" --time=$CONVERTED_DURATION"
fi

# Add interactive terminal support (always enabled for srun)
SRUN_CMD+=" --pty"

SRUN_CMD+=" --mail-type=BEGIN --mail-user=shengjiel@nvidia.com"

# Execute srun command
${RUN_PREFIX} ${SRUN_CMD} "${REMAINING_ARGS[@]}"

{ set +x; } 2>/dev/null
