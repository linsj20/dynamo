"""
GenAI-Perf test implementation for Global Scheduler.

This test runs GenAI-Perf performance testing against the Global Scheduler to collect
comprehensive metrics including latency, throughput, and SLO compliance.
"""

import asyncio
import logging
import os
import subprocess
import json
import random
import numpy as np
from typing import List, Dict, Tuple

from .test_base import BaseGlobalSchedulerTest, TestRequest

logger = logging.getLogger(__name__)


class DistributedSLOGenerator:
    """Generate per-request SLO requirements from distributions"""
    
    @staticmethod
    def generate_slo_requirements(
        num_requests: int,
        distribution_config: Dict[str, str]
    ) -> List[Dict[str, float]]:
        """
        Generate per-request SLO requirements from distributions.
        
        Args:
            num_requests: Number of requests to generate SLOs for
            distribution_config: Dict mapping metric names to distribution configs
                e.g., {
                    'time_to_first_token': 'normal:100:20',
                    'inter_token_latency': 'normal:15:5',
                    'request_latency': 'uniform:3000:8000'
                }
        
        Returns:
            List of per-request SLO requirement dicts
        """
        slo_requirements = []
        
        for i in range(num_requests):
            request_slos = {}
            for metric_name, dist_config in distribution_config.items():
                value = DistributedSLOGenerator._sample_from_distribution(dist_config, metric_name, i)
                if value is not None:
                    request_slos[metric_name] = value
            slo_requirements.append(request_slos)
        
        return slo_requirements
    
    @staticmethod 
    def _sample_from_distribution(config: str, metric_name: str, request_idx: int) -> float:
        """
        Sample a value from the specified distribution.
        
        Supported formats:
        - 'static:100' -> always return 100
        - 'normal:100:20' -> normal distribution (mean=100, std=20)
        - 'uniform:50:150' -> uniform distribution [50, 150]
        - 'lognormal:4.6:0.2' -> lognormal distribution (ln_mean, ln_std)
        - 'bimodal:50:20:150:30:0.7' -> bimodal (μ1:σ1:μ2:σ2:weight1)
        - 'discrete:5:0.2:7:0.3:10:0.5' -> discrete values (value:prob pairs)
        """
        try:
            parts = config.split(':')
            dist_type = parts[0].lower()
            
            # Set random seed based on request index for reproducibility
            np.random.seed(42 + request_idx)
            
            if dist_type == 'static':
                return float(parts[1])
                
            elif dist_type == 'normal':
                mean, std = float(parts[1]), float(parts[2])
                value = np.random.normal(mean, std)
                return max(1.0, value)  # Ensure positive
                
            elif dist_type == 'uniform':
                min_val, max_val = float(parts[1]), float(parts[2])
                return np.random.uniform(min_val, max_val)
                
            elif dist_type == 'lognormal':
                ln_mean, ln_std = float(parts[1]), float(parts[2])
                return np.random.lognormal(ln_mean, ln_std)
                
            elif dist_type == 'bimodal':
                # Bimodal distribution: two normal distributions
                mean1, std1, mean2, std2, weight1 = map(float, parts[1:6])
                weight2 = 1.0 - weight1
                
                if np.random.random() < weight1:
                    value = np.random.normal(mean1, std1)
                else:
                    value = np.random.normal(mean2, std2)
                return max(8.5, value)
                
            elif dist_type == 'discrete':
                # Discrete distribution: alternating value:probability pairs
                values = [float(parts[i]) for i in range(1, len(parts), 2)]
                probs = [float(parts[i]) for i in range(2, len(parts), 2)]
                return np.random.choice(values, p=probs)
                
            else:
                logger.warning(f"Unknown distribution type '{dist_type}' for {metric_name}")
                return float(parts[1]) if len(parts) > 1 else 100.0
                
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid distribution config '{config}' for {metric_name}: {e}")
            return 100.0  # Default fallback
    
    @staticmethod
    def get_default_distribution_config() -> Dict[str, str]:
        """Get default distribution configurations for different SLO strategies"""
        strategy = os.getenv('GENAI_PERF_SLO_DISTRIBUTION_STRATEGY', 'realistic')
        
        if strategy == 'strict':
            return {
                'time_to_first_token': 'normal:50:10',    # Strict TTFT: mean=50ms, std=10ms
                'inter_token_latency': 'normal:8:2',      # Strict ITL: mean=8ms, std=2ms  
                'request_latency': 'normal:2000:400'      # Strict latency: mean=2s, std=0.4s
            }
        elif strategy == 'realistic':
            return {
                'time_to_first_token': 'lognormal:4.6:0.3',  # Log-normal for realistic latency
                'inter_token_latency': 'normal:20:8',        # Normal ITL: mean=20ms, std=8ms
                'request_latency': 'bimodal:3000:500:8000:1500:0.7'  # Bimodal: fast/slow requests
            }
        elif strategy == 'relaxed':
            return {
                'time_to_first_token': 'uniform:100:500',    # Uniform TTFT: 100-500ms
                'inter_token_latency': 'normal:40:15',       # Relaxed ITL: mean=40ms, std=15ms
                'request_latency': 'normal:10000:3000'       # Relaxed latency: mean=10s, std=3s
            }
        else:  # mixed
            return {
                'time_to_first_token': 'bimodal:80:15:200:50:0.6',   # 60% fast, 40% slow
                'inter_token_latency': 'lognormal:2.5:0.4',          # Log-normal ITL
                'request_latency': 'uniform:2000:12000'              # Wide uniform range
            }
    
    @staticmethod
    def save_slo_requirements(slo_requirements: List[Dict[str, float]], filepath: str):
        """Save per-request SLO requirements to file for analysis"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'total_requests': len(slo_requirements),
                    'slo_requirements': slo_requirements,
                    'statistics': DistributedSLOGenerator._compute_statistics(slo_requirements)
                }, f, indent=2)
            logger.info(f"Saved {len(slo_requirements)} SLO requirements to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save SLO requirements: {e}")
    
    @staticmethod
    def _compute_statistics(slo_requirements: List[Dict[str, float]]) -> Dict:
        """Compute statistics for the generated SLO requirements"""
        stats = {}
        if not slo_requirements:
            return stats
            
        # Get all metric names
        all_metrics = set()
        for req_slos in slo_requirements:
            all_metrics.update(req_slos.keys())
        
        # Compute stats for each metric
        for metric in all_metrics:
            values = [req_slos.get(metric, 0) for req_slos in slo_requirements if metric in req_slos]
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p50': float(np.percentile(values, 50)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99))
                }
        
        return stats


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
            request_count = int(os.getenv('GENAI_PERF_REQUEST_COUNT', '500'))
            warmup_requests = int(os.getenv('GENAI_PERF_WARMUP_REQUESTS', '10'))
            input_tokens = int(os.getenv('GENAI_PERF_INPUT_TOKENS', '512'))
            output_tokens = int(os.getenv('GENAI_PERF_OUTPUT_TOKENS', '256'))
            slo_strategy = os.getenv('GENAI_PERF_SLO_STRATEGY', 'round_robin')
            concurrency = int(os.getenv('GENAI_PERF_CONCURRENCY', '8'))
            request_rate = os.getenv('GENAI_PERF_REQUEST_RATE', None)  # requests per second
            enable_plots = os.getenv('GENAI_PERF_GENERATE_PLOTS', 'false').lower() == 'true'
            
            # Distributed SLO requirements support
            use_distributed_slos = os.getenv('GENAI_PERF_DISTRIBUTED_SLOS', 'false').lower() == 'true'
            
            # Set up GenAI-Perf command with artifact directory
            artifacts_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "artifacts", "genai_perf"
            )
            
            if use_distributed_slos:
                # Generate per-request SLO requirements from distributions
                logger.info("Using distributed per-request SLO requirements")
                
                # Get distribution config from environment or use defaults
                distribution_config = self._get_distribution_config()
                logger.info(f"SLO distribution config: {distribution_config}")
                
                # Generate per-request SLO requirements
                slo_requirements = DistributedSLOGenerator.generate_slo_requirements(
                    request_count, distribution_config
                )
                
                # Save SLO requirements for analysis
                slo_file = os.path.join(artifacts_dir, "distributed_slo_requirements.json")
                os.makedirs(artifacts_dir, exist_ok=True)
                DistributedSLOGenerator.save_slo_requirements(slo_requirements, slo_file)
                
                # Use distributed per-request SLO evaluation with GenAI-Perf
                # NOTE: GenAI-Perf now supports per-request SLO evaluation via --distributed-slo flag
                goodput_constraints = self._compute_average_constraints(slo_requirements)
                logger.info(f"Using distributed SLO evaluation with per-request requirements (averaged constraints for parent class compatibility only): {goodput_constraints}")
                logger.info(f"Per-request SLO requirements will be embedded in request payloads")
                
            else:
                # Use traditional static goodput constraints
                goodput_constraints = []
                if os.getenv('GENAI_PERF_GOODPUT_TTFT'):
                    goodput_constraints.append(f"time_to_first_token:{os.getenv('GENAI_PERF_GOODPUT_TTFT')}")
                if os.getenv('GENAI_PERF_GOODPUT_ITL'):
                    goodput_constraints.append(f"inter_token_latency:{os.getenv('GENAI_PERF_GOODPUT_ITL')}")
                if os.getenv('GENAI_PERF_GOODPUT_LATENCY'):
                    goodput_constraints.append(f"request_latency:{os.getenv('GENAI_PERF_GOODPUT_LATENCY')}")
                if os.getenv('GENAI_PERF_GOODPUT_THROUGHPUT'):
                    goodput_constraints.append(f"output_token_throughput_per_user:{os.getenv('GENAI_PERF_GOODPUT_THROUGHPUT')}")
                
                # Set default goodput constraints if none specified
                if not goodput_constraints:
                    goodput_constraints = [
                        "time_to_first_token:100",     # TTFT < 100ms
                        "inter_token_latency:15",      # ITL < 15ms 
                        "request_latency:10000"        # Total latency < 10s
                    ]
            
            # Force streaming to enable TTFT/TPOT metrics calculation
            enable_streaming = True  # Essential for LLM performance metrics
            
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
                "--artifact-dir", artifacts_dir
            ]
            
            # Add streaming to enable TTFT/TPOT metrics
            if enable_streaming:
                genai_perf_cmd.append("--streaming")
                logger.info("Streaming enabled for TTFT/TPOT metrics calculation")
                
            # Add load configuration: request-rate takes precedence over concurrency
            if request_rate:
                genai_perf_cmd.extend(["--request-rate", request_rate])
                logger.info(f"Using request rate: {request_rate} requests/second")
            else:
                genai_perf_cmd.extend(["--concurrency", str(concurrency)])
                logger.info(f"Using concurrency: {concurrency} parallel requests")
                
            # Add plot generation if requested
            if enable_plots:
                genai_perf_cmd.append("--generate-plots")
                logger.info("Plot generation enabled for TTFT/TPOT visualization")
                
            # Add goodput constraints for SLO compliance measurement
            if goodput_constraints:
                genai_perf_cmd.extend(["--goodput"] + goodput_constraints)
                constraint_type = "per-request (no fallback)" if use_distributed_slos else "static"
                logger.info(f"Goodput SLO constraints ({constraint_type}): {', '.join(goodput_constraints)}")
                if use_distributed_slos:
                    logger.info("Note: These constraints are only used for parent class compatibility - actual evaluation uses per-request SLO requirements")
                
            # Enable distributed SLO evaluation if using per-request requirements
            if use_distributed_slos:
                genai_perf_cmd.append("--distributed-slo")
                logger.info("Enabled distributed per-request SLO evaluation")
                
            # Add SLO strategy through extra inputs
            if use_distributed_slos:
                import json
                # Write requirements to file to avoid command line length limits
                slo_requirements_file = os.path.join(artifacts_dir, "distributed_slo_requirements_for_converter.json")
                with open(slo_requirements_file, 'w') as f:
                    json.dump(slo_requirements, f)
                
                logger.info(f"Distributed SLO requirements file: {slo_requirements_file}")
                
                genai_perf_cmd.extend([
                    "--extra-inputs", f"slo_strategy:{slo_strategy}",
                    "--extra-inputs", f"distributed_slo_requirements_file:{slo_requirements_file}"
                ])
            else:
                genai_perf_cmd.extend([
                    "--extra-inputs", f"slo_strategy:{slo_strategy}"
                ])
            
            logger.info(f"GenAI-Perf command: {' '.join(genai_perf_cmd)}")
            logger.info(f"Parameters: {request_count} requests, {warmup_requests} warmup, "
                       f"{input_tokens} input tokens, {output_tokens} output tokens")
            logger.info("Metrics enabled: TTFT (Time to First Token), TPOT (Time Per Output Token), "
                       "Request Latency, Token Throughput, Goodput (SLO-compliant throughput)")
            
            # Set up environment
            env = os.environ.copy()
            genai_perf_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "perf_analyzer", "genai-perf"
            )
            env['PYTHONPATH'] = f"{genai_perf_dir}:{env.get('PYTHONPATH', '')}"
            
            # Create artifacts directory if it doesn't exist (using the same path as in command)
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Create plots directory if plot generation is enabled
            if enable_plots:
                plots_dir = os.path.join(artifacts_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                logger.info(f"Created plots directory: {plots_dir}")
            
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
                
                # Post-process results with distributed SLO analysis if enabled
                if use_distributed_slos:
                    await self._analyze_distributed_slo_results(artifacts_dir, slo_requirements)
                
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
    
    def _get_distribution_config(self) -> Dict[str, str]:
        """Get distribution configuration from environment variables"""
        config = {}
        
        # Check for explicit distribution configs
        if os.getenv('GENAI_PERF_TTFT_DISTRIBUTION'):
            config['time_to_first_token'] = os.getenv('GENAI_PERF_TTFT_DISTRIBUTION')
        if os.getenv('GENAI_PERF_ITL_DISTRIBUTION'):
            config['inter_token_latency'] = os.getenv('GENAI_PERF_ITL_DISTRIBUTION')
        if os.getenv('GENAI_PERF_LATENCY_DISTRIBUTION'):
            config['request_latency'] = os.getenv('GENAI_PERF_LATENCY_DISTRIBUTION')
        
        # Use defaults if no explicit config provided
        if not config:
            config = DistributedSLOGenerator.get_default_distribution_config()
        
        return config
    
    def _compute_average_constraints(self, slo_requirements: List[Dict[str, float]]) -> List[str]:
        """Compute average constraints from distributed SLO requirements"""
        constraints = []
        
        if not slo_requirements:
            return constraints
        
        # Get all metrics
        all_metrics = set()
        for req_slos in slo_requirements:
            all_metrics.update(req_slos.keys())
        
        # Compute average for each metric
        for metric in all_metrics:
            values = [req_slos[metric] for req_slos in slo_requirements if metric in req_slos]
            if values:
                avg_value = int(np.mean(values))
                constraints.append(f"{metric}:{avg_value}")
        
        return constraints
    
    async def _analyze_distributed_slo_results(self, artifacts_dir: str, slo_requirements: List[Dict[str, float]]):
        """Analyze results against distributed SLO requirements"""
        try:
            # This would be where we implement per-request SLO compliance analysis
            # For now, just log that the analysis would happen here
            logger.info("=" * 60)
            logger.info("DISTRIBUTED SLO ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"Generated {len(slo_requirements)} per-request SLO requirements")
            
            # Compute SLO requirement statistics
            stats = DistributedSLOGenerator._compute_statistics(slo_requirements)
            for metric, metric_stats in stats.items():
                logger.info(f"{metric.upper()} requirements - "
                          f"mean: {metric_stats['mean']:.1f}, "
                          f"p95: {metric_stats['p95']:.1f}, "
                          f"range: [{metric_stats['min']:.1f}, {metric_stats['max']:.1f}]")
            
            # Per-request SLO compliance analysis is now supported via DistributedGoodputCalculator
            # GenAI-Perf will automatically:
            # 1. Extract per-request SLO requirements from request payloads
            # 2. Evaluate each request against its specific SLO requirements
            # 3. Compute distributed goodput metrics and compliance statistics
            logger.info("Per-request SLO compliance analysis will be performed automatically by GenAI-Perf")
            logger.info("Results will include individual request compliance and distributed goodput metrics")
            
        except Exception as e:
            logger.error(f"Failed to analyze distributed SLO results: {e}")
            
    def _should_enable_streaming(self) -> bool:
        """Determine if streaming should be enabled based on configuration"""
        # Enable streaming if explicitly requested via environment variable
        if os.getenv('GENAI_PERF_STREAMING', '').lower() == 'true':
            return True
            
        # Enable streaming if global scheduler supports it (check from config)
        if self.config.get('streaming_enabled', False):
            return True
            
        return False 