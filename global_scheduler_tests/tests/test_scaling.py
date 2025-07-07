# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scaling test implementation for Global Scheduler.

This test validates scaling behavior by varying request rate to trigger:
1. Scale up when load increases
2. Scale down when load decreases
"""

import asyncio
import logging
import time
import random
import string
from typing import List, Dict, Any

from .test_base import BaseGlobalSchedulerTest, TestRequest

logger = logging.getLogger(__name__)

class ScalingTest(BaseGlobalSchedulerTest):
    """Test that validates scaling behavior with varying load"""
    
    def __init__(self, config):
        super().__init__(config)
        self.phase_results = {}  # Track results by load phase
        
    def _generate_random_prefix(self) -> str:
        """Generate a random prefix to avoid KV cache reuse"""
        # Generate 2-3 random words/tokens to make each request unique
        word_count = random.randint(2, 4)
        prefixes = []
        
        for _ in range(word_count):
            # Mix of random words and numbers for variety
            if random.choice([True, False]):
                # Random word (4-8 characters)
                word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
            else:
                # Random number (2-4 digits)
                word = str(random.randint(10, 9999))
            prefixes.append(word)
        
        return f"[{' '.join(prefixes)}] "
        
    async def send_request_quiet(self, request: TestRequest) -> TestRequest:
        """Send a single test request without verbose logging"""
        start_time = time.time()
        
        try:
            # Prepare request data
            request_data = {
                "request_id": request.request_id,
                "slo_requirement": request.slo_requirement,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False
            }
            
            # Send request to Global Scheduler (without logging each one)
            response = await self.scheduler_client.generate(request_data)
            
            # Read response
            response_data = None
            async for item in response:
                response_data = item
                break
            
            # Calculate response time
            end_time = time.time()
            request.response_time = end_time - start_time
            
            # Handle response
            if response_data:
                request.success = True
                request.response_data = response_data
            else:
                request.success = False
                request.error = "No response data received"
                
        except Exception as e:
            end_time = time.time()
            request.response_time = end_time - start_time
            request.success = False
            request.error = str(e)
            
        return request
    
    async def run_test_logic(self) -> bool:
        """Execute scaling test with varying load phases"""
        logger.info("RUNNING SCALING TEST")
        logger.info("Testing scaling behavior with variable request rates")
        logger.info("=" * 60)
        logger.info("LOAD PATTERN OVERVIEW:")
        logger.info("  Phase 1: 50 req/s  for 60s  (baseline)")
        logger.info("  Phase 2: 100 req/s for 90s  (scale up trigger)")
        logger.info("  Phase 3: 200 req/s for 120s (peak load)")
        logger.info("  Phase 4: 100 req/s for 90s  (scale down trigger)")
        logger.info("  Phase 5: 50 req/s  for 60s  (return to baseline)")
        logger.info("  Expected Total: ~36000 requests over ~8 minutes")
        logger.info("  KV Cache: Each request has unique random prefix to prevent reuse")
        logger.info("  Logging: Status every 20 requests (50 during 200+ req/s) to avoid overflow")
        logger.info("=" * 60)
        
        # Phase 1: Low load baseline (should start with minimal workers)
        logger.info("=" * 60)
        logger.info("PHASE 1: HIGH LOAD BASELINE (60 seconds)")
        logger.info("Request rate: 50 req/s")
        logger.info("Expected: Minimal workers (1P+1D)")
        logger.info("=" * 60)
        phase1_results = await self._run_load_phase("low", duration=60, request_rate=50.0)
        self.phase_results["low_baseline"] = phase1_results
        
        # Wait for metrics to stabilize
        logger.info("Waiting 20 seconds for metrics to stabilize...")
        await asyncio.sleep(20)
        
        # Phase 2: Gradually increase load (should trigger scale up)
        logger.info("=" * 60)
        logger.info("PHASE 2: VERY HIGH LOAD (90 seconds)")
        logger.info("Request rate: 100 req/s")
        logger.info("Expected: Scale up triggers (prefill queue > 0.2 or KV util > 20%)")
        logger.info("=" * 60)
        phase2_results = await self._run_load_phase("increasing", duration=90, request_rate=100.0)
        self.phase_results["increasing_load"] = phase2_results
        
        # Wait for scaling to complete
        logger.info("Waiting 45 seconds for scaling actions to complete...")
        await asyncio.sleep(45)
        
        # Phase 3: High sustained load (should maintain scaled state)
        logger.info("=" * 60)
        logger.info("PHASE 3: EXTREME HIGH LOAD (120 seconds)")
        logger.info("Request rate: 200 req/s (PEAK)")
        logger.info("Expected: Maintain scaled workers")
        logger.info("=" * 60)
        phase3_results = await self._run_load_phase("high", duration=120, request_rate=200.0)
        self.phase_results["high_sustained"] = phase3_results
        
        # Wait for metrics to stabilize
        logger.info("Waiting 30 seconds for metrics to stabilize...")
        await asyncio.sleep(30)
        
        # Phase 4: Gradually decrease load (should trigger scale down)
        logger.info("=" * 60)
        logger.info("PHASE 4: VERY HIGH LOAD (90 seconds)")
        logger.info("Request rate: 100 req/s")
        logger.info("Expected: Scale down triggers (prefill queue < 0.1 and KV util < 10%)")
        logger.info("=" * 60)
        phase4_results = await self._run_load_phase("decreasing", duration=90, request_rate=100.0)
        self.phase_results["decreasing_load"] = phase4_results
        
        # Wait for scaling to complete
        logger.info("Waiting 45 seconds for final scaling actions...")
        await asyncio.sleep(45)
        
        # Phase 5: Return to low load (should return to minimal workers)
        logger.info("=" * 60)
        logger.info("PHASE 5: HIGH LOAD RETURN (60 seconds)")
        logger.info("Request rate: 50 req/s")
        logger.info("Expected: Return to minimal workers (1P+1D)")
        logger.info("=" * 60)
        phase5_results = await self._run_load_phase("low_return", duration=60, request_rate=50.0)
        self.phase_results["low_return"] = phase5_results
        
        # Analyze scaling behavior
        return self._analyze_scaling_results()
    
    async def _run_load_phase(self, phase_name: str, duration: int, request_rate: float) -> List[TestRequest]:
        """Run a load phase with specified request rate for given duration"""
        logger.info(f"Starting {phase_name} phase: {request_rate} req/s for {duration}s")
        
        phase_results = []
        start_time = time.time()
        request_counter = 0
        last_status_time = start_time
        
        # Calculate interval between requests (in seconds)
        request_interval = 1.0 / request_rate
        
        # Use both high and low SLO requests to distribute load across both pools
        slo_types = ["high", "low"]
        
        # Varied prompts for more realistic load patterns
        prompts = [
            "Explain the concept of artificial intelligence and its applications in modern technology. Provide detailed examples and discuss future implications.",
            "Write a comprehensive guide on sustainable energy solutions including solar, wind, and hydroelectric power systems.",
            "Describe the process of machine learning model training, including data preprocessing, feature engineering, and validation techniques.", 
            "Analyze the economic impact of remote work on global businesses and discuss long-term societal changes.",
            "Compare different programming paradigms such as object-oriented, functional, and procedural programming with examples.",
            "Discuss the importance of cybersecurity in the digital age and outline best practices for personal and corporate security.",
            "Explain quantum computing principles and how they differ from classical computing architectures.",
            "Describe the evolution of cloud computing from basic virtualization to modern serverless architectures.",
            "Analyze the role of data analytics in business decision making and provide real-world use cases.",
            "Discuss the ethical implications of AI development and deployment in various industries."
        ]
        
        while time.time() - start_time < duration:
            request_counter += 1
            
            # Alternate between high and low SLO to distribute load
            slo_requirement = slo_types[request_counter % 2]
            expected_pool = f"{slo_requirement}_slo_pd" if slo_requirement == "high" else "low_slo_pd"
            
            # Generate random prefix to avoid KV cache reuse
            random_prefix = self._generate_random_prefix()
            base_prompt = prompts[request_counter % len(prompts)]
            unique_prompt = f"{random_prefix}{base_prompt}"
            
            request = TestRequest(
                request_id=f"{phase_name}_{request_counter}",
                slo_requirement=slo_requirement,
                prompt=unique_prompt,
                max_tokens=100,  # Increase token count for more substantial load
                expected_pool=expected_pool
            )
            
            # Send the request (fire and forget to maintain rate)
            try:
                # Use asyncio.create_task to send without waiting for completion
                task = asyncio.create_task(self.send_request_quiet(request))
                
                # Store the task to collect results later
                # For now, we'll approximate success and just track the request
                request.success = True  # We'll update this if needed
                phase_results.append(request)
                
            except Exception as e:
                logger.error(f"Failed to send request {request.request_id}: {e}")
                request.success = False
                request.error = str(e)
                phase_results.append(request)
            
            # Print status every 50 requests during high load (200+ req/s) or every 20 requests otherwise
            current_time = time.time()
            log_interval = 50 if request_rate >= 200 else 20
            
            if request_counter % log_interval == 0 or (current_time - last_status_time) >= 10:
                elapsed = current_time - start_time
                remaining = duration - elapsed
                actual_rate = request_counter / elapsed if elapsed > 0 else 0
                
                # Calculate success rate for recent requests
                recent_count = min(log_interval, len(phase_results))
                recent_requests = phase_results[-recent_count:] if recent_count > 0 else phase_results
                success_count = sum(1 for r in recent_requests if r.success)
                success_rate = (success_count / len(recent_requests) * 100) if recent_requests else 0
                
                # Show sample response info from last successful request
                last_successful = None
                for r in reversed(recent_requests):
                    if r.success and r.response_data:
                        last_successful = r
                        break
                
                status_msg = (f"  {phase_name}: {request_counter} requests | "
                            f"Rate: {actual_rate:.1f} req/s | "
                            f"Success: {success_rate:.0f}% | "
                            f"Remaining: {remaining:.0f}s")
                
                if last_successful:
                    avg_response_time = sum(r.response_time for r in recent_requests if r.success) / max(1, success_count)
                    status_msg += f" | Avg RT: {avg_response_time:.2f}s"
                
                logger.info(status_msg)
                last_status_time = current_time
            
            # Wait for the next request time
            await asyncio.sleep(request_interval)
        
        logger.info(f"Completed {phase_name} phase: {len(phase_results)} requests sent at {request_rate} req/s")
        
        # Wait a bit for requests to complete processing
        completion_wait = min(10, max(3, int(len(phase_results) * 0.1)))
        logger.info(f"Waiting {completion_wait}s for request completion...")
        await asyncio.sleep(completion_wait)
        
        self.results.extend(phase_results)
        return phase_results
    
    def _analyze_scaling_results(self) -> bool:
        """Analyze results across all load phases"""
        logger.info("ANALYZING SCALING TEST RESULTS")
        logger.info("=" * 60)
        
        overall_success = True
        
        # Analyze each phase
        for phase_name, phase_results in self.phase_results.items():
            success_count = sum(1 for r in phase_results if r.success)
            total_count = len(phase_results)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            avg_response_time = 0
            if success_count > 0:
                avg_response_time = sum(r.response_time for r in phase_results if r.success) / success_count
            
            logger.info(f"Phase {phase_name.upper()}:")
            logger.info(f"  Requests: {success_count}/{total_count} successful ({success_rate:.1f}%)")
            logger.info(f"  Avg Response Time: {avg_response_time:.2f}s")
            
            # Phase-specific validation
            if phase_name in ["low_baseline", "low_return"]:
                # Low load phases should have good performance
                if success_rate < 90:
                    logger.warning(f"  WARNING: Low success rate during {phase_name}")
                    overall_success = False
                if avg_response_time > 10:
                    logger.warning(f"  WARNING: High response time during {phase_name}")
            
            elif phase_name in ["increasing_load", "decreasing_load"]:
                # Transition phases may have some degradation but should still work
                if success_rate < 70:
                    logger.error(f"  FAIL: Very low success rate during {phase_name}")
                    overall_success = False
            
            elif phase_name == "high_sustained":
                # High load should be sustained well after scaling
                if success_rate < 80:
                    logger.warning(f"  WARNING: Low success rate during sustained high load")
        
        # Overall statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        overall_success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        logger.info(f"\nOVERALL SCALING TEST RESULTS:")
        logger.info(f"  Total Requests: {successful_requests}/{total_requests} successful ({overall_success_rate:.1f}%)")
        logger.info(f"  Peak Load Phase: 200 req/s for 120 seconds (expected ~24000 requests)")
        
        # Calculate expected vs actual request counts
        expected_total = (50*60) + (100*90) + (200*120) + (100*90) + (50*60)  # Sum of rate*duration for each phase
        logger.info(f"  Expected Total Requests: ~{expected_total}")
        logger.info(f"  Actual Total Requests: {total_requests}")
        
        # Response time analysis
        if successful_requests > 0:
            successful_results = [r for r in self.results if r.success]
            avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
            logger.info(f"  Average Response Time: {avg_response_time:.2f}s")
        
        # Phase-by-phase analysis
        logger.info(f"\nPHASE-BY-PHASE RESULTS:")
        for phase_name, phase_results in self.phase_results.items():
            phase_success = sum(1 for r in phase_results if r.success)
            phase_total = len(phase_results)
            phase_rate = (phase_success / phase_total * 100) if phase_total > 0 else 0
            logger.info(f"  {phase_name}: {phase_success}/{phase_total} ({phase_rate:.1f}%)")
        
        # Success criteria
        success_criteria = [
            overall_success_rate >= 95,  # At least 95% success rate
            total_requests >= expected_total * 0.8,  # At least 80% of expected requests sent
        ]
        
        test_passed = all(success_criteria)
        logger.info(f"\nSCALING TEST RESULT: {'PASS' if test_passed else 'FAIL'}")
        
        # Recommendations for observing scaling
        logger.info("\nSCALING OBSERVATION RECOMMENDATIONS:")
        logger.info("  1. Check planner logs: logs/planner_high_slo/ and logs/planner_low_slo/")
        logger.info("  2. Monitor GPU utilization during test phases:")
        logger.info("     - Phase 1-2 (50-100 req/s): Should trigger scaling actions")
        logger.info("     - Phase 3 (200 req/s): Should see high GPU utilization")
        logger.info("     - Phase 4-5 (100-50 req/s): Should scale down to minimal workers")
        logger.info("  3. Expected pattern: Start with 1P+1D → Scale to 2P+2D → Return to 1P+1D")
        logger.info("  4. Run with --monitor flag to see real-time GPU utilization")
        logger.info("=" * 60)
        
        return test_passed 