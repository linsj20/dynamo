#!/bin/bash

# Sequential SLURM job submission for GenAI-Perf tests
# Usage: ./submit_batch.sh

# Test configuration arrays - modify these lists to run different combinations
GENAI_PERF_REQUEST_RATE_LIST=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
GENAI_PERF_REQUEST_COUNT_LIST=(1800)
GENAI_PERF_TTFT_DISTRIBUTION_LIST=("discrete:2000:0.5:5000:0.5")
GENAI_PERF_ITL_DISTRIBUTION_LIST=(
    "discrete:9:0.5:12:0.5"  
)
#    "discrete:9:0.5:11:0.5"
#    "discrete:9.5:0.6:11:0.4"
TEST_TYPE_LIST=(genai-perf)
HIGH_SLO_NODE_COUNT_LIST=(1)
LOW_SLO_NODE_COUNT_LIST=(1)
GENAI_PERF_DISTRIBUTED_SLOS_LIST=(true)
GENAI_PERF_SLO_STRATEGY_LIST=(round_robin threshold)

# Generate all combinations
CONFIGS=()
for rate in "${GENAI_PERF_REQUEST_RATE_LIST[@]}"; do
    for count in "${GENAI_PERF_REQUEST_COUNT_LIST[@]}"; do
        for ttft in "${GENAI_PERF_TTFT_DISTRIBUTION_LIST[@]}"; do
            for itl in "${GENAI_PERF_ITL_DISTRIBUTION_LIST[@]}"; do
                for test_type in "${TEST_TYPE_LIST[@]}"; do
                    for high_nodes in "${HIGH_SLO_NODE_COUNT_LIST[@]}"; do
                        for low_nodes in "${LOW_SLO_NODE_COUNT_LIST[@]}"; do
                            for dist_slos in "${GENAI_PERF_DISTRIBUTED_SLOS_LIST[@]}"; do
                                for slo_strategy in "${GENAI_PERF_SLO_STRATEGY_LIST[@]}"; do
                                    CONFIGS+=("$rate|$count|$ttft|$itl|$test_type|$high_nodes|$low_nodes|$dist_slos|$slo_strategy")
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Submitting ${#CONFIGS[@]} sequential GenAI-Perf tests..."

PREV_JOB=""

for i in "${!CONFIGS[@]}"; do
    # Parse configuration
    IFS='|' read -r GENAI_PERF_REQUEST_RATE GENAI_PERF_REQUEST_COUNT GENAI_PERF_TTFT_DISTRIBUTION GENAI_PERF_ITL_DISTRIBUTION TEST_TYPE HIGH_SLO_NODE_COUNT LOW_SLO_NODE_COUNT GENAI_PERF_DISTRIBUTED_SLOS GENAI_PERF_SLO_STRATEGY <<< "${CONFIGS[$i]}"
    
    if [ -z "$PREV_JOB" ]; then
        # First job - no dependency
        JOB_ID=$(GENAI_PERF_REQUEST_RATE=$GENAI_PERF_REQUEST_RATE \
                GENAI_PERF_REQUEST_COUNT=$GENAI_PERF_REQUEST_COUNT \
                GENAI_PERF_TTFT_DISTRIBUTION="$GENAI_PERF_TTFT_DISTRIBUTION" \
                GENAI_PERF_ITL_DISTRIBUTION="$GENAI_PERF_ITL_DISTRIBUTION" \
                TEST_TYPE=$TEST_TYPE \
                HIGH_SLO_NODE_COUNT=$HIGH_SLO_NODE_COUNT \
                LOW_SLO_NODE_COUNT=$LOW_SLO_NODE_COUNT \
                GENAI_PERF_DISTRIBUTED_SLOS=$GENAI_PERF_DISTRIBUTED_SLOS \
                GENAI_PERF_SLO_STRATEGY=$GENAI_PERF_SLO_STRATEGY \
                sbatch --nodes=2 slurm_scripts/run.sbatch | awk '{print $4}')
        echo "Job $JOB_ID: Rate=$GENAI_PERF_REQUEST_RATE, ITL=$GENAI_PERF_ITL_DISTRIBUTION, Strategy=$GENAI_PERF_SLO_STRATEGY"
    else
        # Subsequent jobs - depend on previous
        JOB_ID=$(GENAI_PERF_REQUEST_RATE=$GENAI_PERF_REQUEST_RATE \
                GENAI_PERF_REQUEST_COUNT=$GENAI_PERF_REQUEST_COUNT \
                GENAI_PERF_TTFT_DISTRIBUTION="$GENAI_PERF_TTFT_DISTRIBUTION" \
                GENAI_PERF_ITL_DISTRIBUTION="$GENAI_PERF_ITL_DISTRIBUTION" \
                TEST_TYPE=$TEST_TYPE \
                HIGH_SLO_NODE_COUNT=$HIGH_SLO_NODE_COUNT \
                LOW_SLO_NODE_COUNT=$LOW_SLO_NODE_COUNT \
                GENAI_PERF_DISTRIBUTED_SLOS=$GENAI_PERF_DISTRIBUTED_SLOS \
                GENAI_PERF_SLO_STRATEGY=$GENAI_PERF_SLO_STRATEGY \
                sbatch --dependency=afterok:$PREV_JOB --nodes=2 slurm_scripts/run.sbatch | awk '{print $4}')
        echo "Job $JOB_ID: Rate=$GENAI_PERF_REQUEST_RATE, ITL=$GENAI_PERF_ITL_DISTRIBUTION, Strategy=$GENAI_PERF_SLO_STRATEGY (after $PREV_JOB)"
    fi
    
    PREV_JOB=$JOB_ID
done

echo "All jobs submitted. Check with: squeue -u $(whoami)"