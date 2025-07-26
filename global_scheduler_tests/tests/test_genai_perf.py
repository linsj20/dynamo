"""
GenAI-Perf test implementation for Global Scheduler.

This test runs GenAI-Perf performance testing against the Global Scheduler to collect
comprehensive metrics including latency, throughput, and SLO compliance.
"""

import asyncio
import logging
import os
import subprocess
from typing import List

from .test_base import BaseGlobalSchedulerTest, TestRequest

logger = logging.getLogger(__name__)


class GenAIPerfTest(BaseGlobalSchedulerTest):
    """GenAI-Perf performance test for Global Scheduler"""
    
    def __init__(self, config):
        super().__init__(config)
        
    async def run_test_logic(self) -> bool:
        """Execute GenAI-Perf performance testing"""
        logger.info("=" * 60)
        logger.info("STARTING GENAI-PERF PERFORMANCE TEST")
        logger.info("=" * 60)
        
        try:
            # Run GenAI-Perf against the Global Scheduler
            success = await self._run_genai_perf()
            
            if success:
                logger.info("GenAI-Perf test completed successfully!")
                return True
            else:
                logger.error("GenAI-Perf test failed")
                return False
                
        except Exception as e:
            logger.error(f"GenAI-Perf test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def _run_genai_perf(self) -> bool:
        """Run GenAI-Perf performance testing against the Global Scheduler"""
        try:
            # Determine global scheduler URL from config
            gs_url = self.config.get('global_scheduler_url', 'http://localhost:3999')
            logger.info(f"Running GenAI-Perf against Global Scheduler at {gs_url}")
            
            # Get configuration parameters
            request_count = int(os.getenv('GENAI_PERF_REQUEST_COUNT', '50'))
            warmup_requests = int(os.getenv('GENAI_PERF_WARMUP_REQUESTS', '10'))
            input_tokens = int(os.getenv('GENAI_PERF_INPUT_TOKENS', '128'))
            output_tokens = int(os.getenv('GENAI_PERF_OUTPUT_TOKENS', '64'))
            slo_strategy = os.getenv('GENAI_PERF_SLO_STRATEGY', 'round_robin')
            
            # Check if streaming should be enabled
            enable_streaming = self._should_enable_streaming()
            
            # Set up GenAI-Perf command
            genai_perf_cmd = [
                "python", "-m", "genai_perf.main", "profile",
                "-m", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # Use the actual model from pool configs
                "--endpoint-type", "global_scheduler",
                "--url", gs_url,
                "--synthetic-input-tokens-mean", str(input_tokens),
                "--synthetic-input-tokens-stddev", "0", 
                "--output-tokens-mean", str(output_tokens),
                "--output-tokens-stddev", "0",
                "--request-count", str(request_count),
                "--warmup-request-count", str(warmup_requests),
                "--artifact-dir", "artifacts/genai_perf"
            ]
            
            # Add streaming if enabled
            if enable_streaming:
                genai_perf_cmd.append("--streaming")
                logger.info("Streaming enabled for GenAI-Perf test")
                
            # Add SLO strategy through extra inputs
            genai_perf_cmd.extend([
                "--extra-inputs", f"slo_strategy:{slo_strategy}"
            ])
            
            logger.info(f"GenAI-Perf command: {' '.join(genai_perf_cmd)}")
            logger.info(f"Parameters: {request_count} requests, {warmup_requests} warmup, "
                       f"{input_tokens} input tokens, {output_tokens} output tokens")
            
            # Set up environment
            env = os.environ.copy()
            genai_perf_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "perf_analyzer", "genai-perf"
            )
            env['PYTHONPATH'] = f"{genai_perf_dir}:{env.get('PYTHONPATH', '')}"
            
            # Create artifacts directory if it doesn't exist
            artifacts_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "artifacts", "genai_perf"
            )
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Run GenAI-Perf
            logger.info("Starting GenAI-Perf execution...")
            result = subprocess.run(
                genai_perf_cmd,
                cwd=genai_perf_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("GenAI-Perf completed successfully!")
                logger.info("=" * 60)
                logger.info("GENAI-PERF OUTPUT:")
                logger.info("=" * 60)
                logger.info(result.stdout)
                
                # Log artifacts location
                if os.path.exists(artifacts_dir):
                    logger.info(f"GenAI-Perf artifacts saved to: {artifacts_dir}")
                    
                    # List artifact files
                    try:
                        artifact_files = os.listdir(artifacts_dir)
                        if artifact_files:
                            logger.info("Generated artifacts:")
                            for file in sorted(artifact_files):
                                logger.info(f"  - {file}")
                    except Exception as e:
                        logger.warning(f"Could not list artifact files: {e}")
                
                return True
            else:
                logger.error(f"GenAI-Perf failed with return code {result.returncode}")
                logger.error("GenAI-Perf stderr:")
                logger.error(result.stderr)
                logger.error("GenAI-Perf stdout:")
                logger.error(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("GenAI-Perf timed out after 20 minutes")
            return False
        except Exception as e:
            logger.error(f"Failed to run GenAI-Perf: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _should_enable_streaming(self) -> bool:
        """Determine if streaming should be enabled based on configuration"""
        # Enable streaming if explicitly requested via environment variable
        if os.getenv('GENAI_PERF_STREAMING', '').lower() == 'true':
            return True
            
        # Enable streaming if global scheduler supports it (check from config)
        if self.config.get('streaming_enabled', False):
            return True
            
        return False 