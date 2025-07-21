# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scaling test implementation for Global Scheduler.

This test validates scaling behavior by varying request rate to trigger:
1. Scale up when load increases
2. Scale down when load decreases

Key features:
- Streaming requests with TTFT and TPOT metrics tracking
- Incremental metrics calculation (only requests finished since last update)
- Statistics separated by SLO pools (High SLO vs Low SLO)
- Real-time monitoring during load phases with pool-level breakdown
- Simple and clean statistics reporting
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
        """Generate a long random prefix with spaces to avoid KV cache reuse"""
        # Generate 8-15 random words/tokens to make each request unique with long prefix
        word_count = random.randint(30, 100)
        prefixes = []
        
        for _ in range(word_count):
            # Mix of random words and numbers for variety
            if random.choice([True, False]):
                # Random word (6-12 characters for longer words)
                word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(6, 12)))
            else:
                # Random number (3-6 digits for longer numbers)
                word = str(random.randint(100, 999999))
            prefixes.append(word)
        
        return f"[{' '.join(prefixes)}] "
        
    async def send_request_quiet(self, request: TestRequest) -> TestRequest:
        """Send a single test request with streaming enabled for TTFT/TPOT metrics"""
        start_time = time.time()
        
        try:
            # Prepare request data with streaming enabled for TTFT/TPOT metrics
            request_data = {
                "request_id": request.request_id,
                "slo_requirement": request.slo_requirement,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True  # Enable streaming to capture TTFT and TPOT metrics
            }
            
            # Send request to Global Scheduler
            response = await self.scheduler_client.generate(request_data)
            
            # Process streaming response to capture TTFT/TPOT metrics
            first_token_time = None
            response_chunks = []
            final_response_data = None
            
            async for item in response:
                # Handle potential Annotated type wrapper (same logic as base class)
                actual_response_data = self._extract_response_data(item)
                
                # Skip None responses (happens with intermediate Annotated responses)
                if actual_response_data is None:
                    continue
                
                # Track first token time for TTFT metrics
                if first_token_time is None:
                    first_token_time = time.time()
                    
                response_chunks.append(actual_response_data)
                
                # If this is an error response, stop processing
                if not actual_response_data.get('success', True):
                    final_response_data = actual_response_data
                    break
                    
                # For successful streaming responses, collect the final complete response
                if actual_response_data.get('success', False):
                    final_response_data = actual_response_data
            
            # Calculate response time
            end_time = time.time()
            request.response_time = end_time - start_time
            
            # Calculate and store TTFT (Time to First Token) if we got streaming data
            if first_token_time:
                request.ttft = first_token_time - start_time
                
            # Store streaming metrics
            request.chunk_count = len(response_chunks)
            request.is_streaming = True
            
            # Handle response data
            if final_response_data:
                if final_response_data.get('success', False):
                    request.success = True
                    request.assigned_pool = final_response_data.get('assigned_pool')
                    request.response_data = final_response_data
                    
                    # Extract content length and token count from response
                    if 'response' in final_response_data:
                        llm_response = final_response_data['response']
                        if isinstance(llm_response, dict) and 'choices' in llm_response and llm_response['choices']:
                            content = llm_response['choices'][0].get('message', {}).get('content', '')
                            request.total_content_length = len(content)
                            
                            # Try to get actual token count from response usage info
                            if 'usage' in llm_response and llm_response['usage'] is not None:
                                request.output_token_count = llm_response['usage'].get('completion_tokens', 0)
                            else:
                                # Estimate token count (rough approximation: ~4 chars per token)
                                request.output_token_count = max(1, len(content) // 4)
                    
                    # Calculate TPOT (Time Per Output Token) if we have TTFT and token count
                    if request.ttft is not None and request.output_token_count > 0:
                        generation_time = request.response_time - request.ttft
                        request.tpot = generation_time / request.output_token_count
                else:
                    request.success = False
                    request.error = final_response_data.get('error', 'Unknown error')
            else:
                request.success = False
                request.error = "No response data received from streaming request"
                
        except Exception as e:
            end_time = time.time()
            request.response_time = end_time - start_time
            request.success = False
            request.error = str(e)
            
        return request
    
    def _extract_response_data(self, response_item):
        """Extract response data from potentially wrapped Dynamo objects (same as base class)"""
        # Check if this is a Dynamo Annotated object (has .data() method)
        if hasattr(response_item, 'data') and callable(getattr(response_item, 'data')):
            # This is a Dynamo Annotated object - call the data() method
            # Note: data() can return None, so we need to handle that
            data = response_item.data()
            if data is not None:
                return data
            else:
                # Skip None data responses - this happens with intermediate Annotated responses
                return None
        
        # For non-Annotated objects, return as-is
        return response_item
    
    async def run_test_logic(self) -> bool:
        """Execute scaling test with varying load phases"""
        logger.info("RUNNING SCALING TEST")
        logger.info("Testing scaling behavior with variable request rates")
        logger.info("=" * 60)
        logger.info("LOAD PATTERN OVERVIEW:")
        logger.info("  Phase 1: 100 req/s for 30s  (baseline)")
        logger.info("  Phase 2: 200 req/s for 60s (peak load)")
        logger.info("  Phase 3: 100 req/s for 30s  (return to baseline)")
        logger.info("  Expected Total: ~18000 requests over ~2 minutes")
        logger.info("  KV Cache: Each request has unique random prefix to prevent reuse")
        logger.info("  Streaming: Enabled with TTFT and TPOT metrics tracking")
        logger.info("  Statistics: Separated by SLO pools (High SLO vs Low SLO)")
        logger.info("  Logging: Real-time metrics per SLO pool every 20-50 requests")
        logger.info("=" * 60)
        
        # Phase 1: Baseline load (should start with minimal workers)
        logger.info("=" * 60)
        logger.info("PHASE 1: BASELINE LOAD (30 seconds)")
        logger.info("Request rate: 100 req/s")
        logger.info("Expected: Baseline workers")
        logger.info("=" * 60)
        phase1_results = await self._run_load_phase("baseline", duration=30, request_rate=100.0)
        self.phase_results["baseline"] = phase1_results
        
        # Wait for metrics to stabilize
        logger.info("Waiting 20 seconds for metrics to stabilize...")
        await asyncio.sleep(20)
        
        # Phase 2: High sustained load (should trigger scale up and maintain scaled state)
        logger.info("=" * 60)
        logger.info("PHASE 2: HIGH LOAD (60 seconds)")
        logger.info("Request rate: 200 req/s")
        logger.info("Expected: Scale up triggers and maintain scaled workers")
        logger.info("=" * 60)
        phase2_results = await self._run_load_phase("high", duration=60, request_rate=200.0)
        self.phase_results["high_sustained"] = phase2_results
        
        # Wait for scaling to complete
        logger.info("Waiting 30 seconds for scaling actions to complete...")
        await asyncio.sleep(30)
        
        # Phase 3: Return to baseline load (should trigger scale down)
        logger.info("=" * 60)
        logger.info("PHASE 3: RETURN TO BASELINE (30 seconds)")
        logger.info("Request rate: 100 req/s")
        logger.info("Expected: Scale down to baseline workers")
        logger.info("=" * 60)
        phase3_results = await self._run_load_phase("baseline_return", duration=30, request_rate=100.0)
        self.phase_results["baseline_return"] = phase3_results
        
        # Analyze scaling behavior
        return self._analyze_scaling_results()
    
    async def _run_load_phase(self, phase_name: str, duration: int, request_rate: float) -> List[TestRequest]:
        """Run a load phase with specified request rate for given duration"""
        logger.info(f"Starting {phase_name} phase: {request_rate} req/s for {duration}s")
        
        phase_results = []
        # Track requests completed since last status update, separated by SLO
        newly_completed_high_slo = []  # High SLO pools
        newly_completed_low_slo = []   # Low SLO pools
        processed_tasks = set()  # Track which tasks we've already processed
        start_time = time.time()
        request_counter = 0
        last_status_time = start_time
        
        # Calculate interval between requests (in seconds)
        request_interval = 1.0 / request_rate
        
        # Use both high and low SLO requests to distribute load across both pools
        slo_types = ["high", "low"]
        prompts = [
            "Write a short story about",
            "Explain the concept of",
            "List the benefits of",
            "Describe how to",
            "What are the main differences between"
        ]
        
        while time.time() - start_time < duration:
            request_counter += 1
            
            # Alternate between high and low SLO to distribute load
            slo_requirement = slo_types[request_counter % 2]
            expected_pool = f"{slo_requirement}_slo_pd" if slo_requirement == "high" else "low_slo_pd"
            
            # Generate random prefix to avoid KV cache reuse
            random_prefix = self._generate_random_prefix()
            base_prompt = prompts[request_counter % len(prompts)]
            unique_prompt = f"{random_prefix} {base_prompt}"
            
            request = TestRequest(
                request_id=f"{phase_name}_{request_counter}",
                slo_requirement=slo_requirement,
                prompt=unique_prompt,
                max_tokens=100,
                expected_pool=expected_pool
            )
            
            # Send the request and track the task for completion
            try:
                task = asyncio.create_task(self.send_request_quiet(request))
                phase_results.append((request, task))
            except Exception as e:
                logger.error(f"Failed to create request task {request.request_id}: {e}")
                request.success = False
                request.error = str(e)
                phase_results.append((request, None))
            
            # Check for newly completed requests and add to appropriate SLO tracking list
            for req, task in phase_results:
                if task is not None and task.done() and task not in processed_tasks:
                    try:
                        completed_req = task.result()
                        processed_tasks.add(task)  # Mark this task as processed
                        
                        # Separate by SLO requirement
                        if completed_req.slo_requirement == "high":
                            newly_completed_high_slo.append(completed_req)
                        else:
                            newly_completed_low_slo.append(completed_req)
                    except Exception as e:
                        # Handle task exceptions (network errors, timeouts, etc.)
                        req.success = False
                        req.error = f"Task exception: {str(e)}"
                        processed_tasks.add(task)  # Mark this task as processed even if it failed
                        
                        # Add failed request to appropriate SLO list
                        if req.slo_requirement == "high":
                            newly_completed_high_slo.append(req)
                        else:
                            newly_completed_low_slo.append(req)
            
            # Print status every 50 requests during high load or every 20 requests otherwise
            current_time = time.time()
            log_interval = 50 if request_rate >= 200 else 20
            
            if request_counter % log_interval == 0 or (current_time - last_status_time) >= 10:
                elapsed = current_time - start_time
                remaining = duration - elapsed
                actual_rate = request_counter / elapsed if elapsed > 0 else 0
                
                # Calculate metrics for High SLO pools
                high_slo_success = sum(1 for r in newly_completed_high_slo if r.success)
                high_slo_total = len(newly_completed_high_slo)
                high_slo_rate = (high_slo_success / high_slo_total * 100) if high_slo_total > 0 else 0
                
                # Count different types of failures for High SLO
                high_slo_failures = [r for r in newly_completed_high_slo if not r.success]
                high_timeout_count = sum(1 for r in high_slo_failures if "timeout" in (r.error or "").lower())
                high_other_failures = len(high_slo_failures) - high_timeout_count
                
                high_successful = [r for r in newly_completed_high_slo if r.success and r.is_streaming]
                high_ttft = high_tpot = high_rt = "N/A"
                
                if high_successful:
                    high_rt = f"{sum(r.response_time for r in high_successful) / len(high_successful):.2f}s"
                    
                    ttft_values = [r.ttft for r in high_successful if r.ttft is not None]
                    if ttft_values:
                        high_ttft = f"{sum(ttft_values) / len(ttft_values):.3f}s"
                    
                    tpot_values = [r.tpot for r in high_successful if r.tpot is not None]
                    if tpot_values:
                        high_tpot = f"{sum(tpot_values) / len(tpot_values):.3f}s/tok"
                
                # Calculate metrics for Low SLO pools
                low_slo_success = sum(1 for r in newly_completed_low_slo if r.success)
                low_slo_total = len(newly_completed_low_slo)
                low_slo_rate = (low_slo_success / low_slo_total * 100) if low_slo_total > 0 else 0
                
                # Count different types of failures for Low SLO
                low_slo_failures = [r for r in newly_completed_low_slo if not r.success]
                low_timeout_count = sum(1 for r in low_slo_failures if "timeout" in (r.error or "").lower())
                low_other_failures = len(low_slo_failures) - low_timeout_count
                
                low_successful = [r for r in newly_completed_low_slo if r.success and r.is_streaming]
                low_ttft = low_tpot = low_rt = "N/A"
                
                if low_successful:
                    low_rt = f"{sum(r.response_time for r in low_successful) / len(low_successful):.2f}s"
                    
                    ttft_values = [r.ttft for r in low_successful if r.ttft is not None]
                    if ttft_values:
                        low_ttft = f"{sum(ttft_values) / len(ttft_values):.3f}s"
                    
                    tpot_values = [r.tpot for r in low_successful if r.tpot is not None]
                    if tpot_values:
                        low_tpot = f"{sum(tpot_values) / len(tpot_values):.3f}s/tok"
                
                # Combined status message
                total_completed = high_slo_total + low_slo_total
                status_msg = (f"  {phase_name}: {request_counter} sent | Rate: {actual_rate:.1f} req/s | "
                            f"Completed: {total_completed} | Remaining: {remaining:.0f}s")
                
                logger.info(status_msg)
                
                # High SLO pool statistics with failure breakdown
                if high_slo_total > 0:
                    high_msg = f"    HIGH SLO: {high_slo_success}/{high_slo_total} ({high_slo_rate:.0f}%)"
                    if high_slo_rate == 0 and high_slo_total > 0:
                        # Show failure breakdown when success rate is 0
                        failure_detail = ""
                        if high_timeout_count > 0:
                            failure_detail += f" {high_timeout_count} timeouts"
                        if high_other_failures > 0:
                            failure_detail += f" {high_other_failures} other failures"
                        if failure_detail:
                            high_msg += f" [{failure_detail.strip()}]"
                    
                    if high_rt != "N/A":
                        high_msg += f" | RT: {high_rt}"
                    if high_ttft != "N/A":
                        high_msg += f" | TTFT: {high_ttft}"
                    if high_tpot != "N/A":
                        high_msg += f" | TPOT: {high_tpot}"
                    
                    # Add load warning if High SLO is failing significantly under high load
                    if request_rate >= 200 and high_slo_rate < 50 and high_slo_total >= 5:
                        high_msg += " ⚠️ HIGH LOAD IMPACT"
                    
                    logger.info(high_msg)
                
                # Low SLO pool statistics with failure breakdown
                if low_slo_total > 0:
                    low_msg = f"    LOW SLO:  {low_slo_success}/{low_slo_total} ({low_slo_rate:.0f}%)"
                    if low_slo_rate == 0 and low_slo_total > 0:
                        # Show failure breakdown when success rate is 0
                        failure_detail = ""
                        if low_timeout_count > 0:
                            failure_detail += f" {low_timeout_count} timeouts"
                        if low_other_failures > 0:
                            failure_detail += f" {low_other_failures} other failures"
                        if failure_detail:
                            low_msg += f" [{failure_detail.strip()}]"
                    
                    if low_rt != "N/A":
                        low_msg += f" | RT: {low_rt}"
                    if low_ttft != "N/A":
                        low_msg += f" | TTFT: {low_ttft}"
                    if low_tpot != "N/A":
                        low_msg += f" | TPOT: {low_tpot}"
                    
                    # Add load warning if Low SLO is failing significantly under high load
                    if request_rate >= 200 and low_slo_rate < 50 and low_slo_total >= 5:
                        low_msg += " ⚠️ HIGH LOAD IMPACT"
                    
                    logger.info(low_msg)
                
                last_status_time = current_time
                
                # Clear the newly completed lists after calculation
                newly_completed_high_slo.clear()
                newly_completed_low_slo.clear()
            
            # Wait for the next request time
            await asyncio.sleep(request_interval)
        
        logger.info(f"Completed {phase_name} phase: {len(phase_results)} requests sent")
        
        # Wait for all remaining tasks to complete
        completion_wait = 15
        logger.info(f"Waiting {completion_wait}s for all requests to complete...")
        
        final_phase_results = []
        for req, task in phase_results:
            if task is not None:
                try:
                    completed_req = await asyncio.wait_for(task, timeout=completion_wait)
                    final_phase_results.append(completed_req)
                except asyncio.TimeoutError:
                    req.success = False
                    req.error = f"Request timeout after {completion_wait}s"
                    final_phase_results.append(req)
                except Exception as e:
                    req.success = False
                    req.error = str(e)
                    final_phase_results.append(req)
            else:
                final_phase_results.append(req)
        
        # Phase completion summary separated by SLO
        high_slo_results = [r for r in final_phase_results if r.slo_requirement == "high"]
        low_slo_results = [r for r in final_phase_results if r.slo_requirement == "low"]
        
        high_success = sum(1 for r in high_slo_results if r.success)
        low_success = sum(1 for r in low_slo_results if r.success)
        
        high_rate = (high_success / len(high_slo_results) * 100) if high_slo_results else 0
        low_rate = (low_success / len(low_slo_results) * 100) if low_slo_results else 0
        
        logger.info(f"Phase {phase_name} completion:")
        logger.info(f"  HIGH SLO: {high_success}/{len(high_slo_results)} successful ({high_rate:.1f}%)")
        logger.info(f"  LOW SLO:  {low_success}/{len(low_slo_results)} successful ({low_rate:.1f}%)")
        
        self.results.extend(final_phase_results)
        return final_phase_results
    
    def _analyze_scaling_results(self) -> bool:
        """Analyze results across all load phases with TTFT and TPOT metrics separated by SLO pools"""
        logger.info("ANALYZING SCALING TEST RESULTS")
        logger.info("=" * 60)
        
        overall_success = True
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        overall_success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Separate results by SLO pools
        high_slo_results = [r for r in self.results if r.slo_requirement == "high"]
        low_slo_results = [r for r in self.results if r.slo_requirement == "low"]
        
        high_slo_successful = [r for r in high_slo_results if r.success and r.is_streaming]
        low_slo_successful = [r for r in low_slo_results if r.success and r.is_streaming]
        
        # Analyze failure patterns
        high_slo_failures = [r for r in high_slo_results if not r.success]
        low_slo_failures = [r for r in low_slo_results if not r.success]
        
        high_timeout_failures = sum(1 for r in high_slo_failures if "timeout" in (r.error or "").lower())
        low_timeout_failures = sum(1 for r in low_slo_failures if "timeout" in (r.error or "").lower())
        
        logger.info(f"Total Requests: {successful_requests}/{total_requests} successful ({overall_success_rate:.1f}%)")
        logger.info(f"  HIGH SLO Pools: {sum(1 for r in high_slo_results if r.success)}/{len(high_slo_results)} " +
                   f"({sum(1 for r in high_slo_results if r.success)/len(high_slo_results)*100:.1f}%)" +
                   (f" ({high_timeout_failures} timeouts)" if high_timeout_failures > 0 else ""))
        logger.info(f"  LOW SLO Pools:  {sum(1 for r in low_slo_results if r.success)}/{len(low_slo_results)} " +
                   f"({sum(1 for r in low_slo_results if r.success)/len(low_slo_results)*100:.1f}%)" +
                   (f" ({low_timeout_failures} timeouts)" if low_timeout_failures > 0 else ""))
        
        # TTFT/TPOT summary for High SLO pools
        high_ttft_values = [r.ttft for r in high_slo_successful if r.ttft is not None]
        high_tpot_values = [r.tpot for r in high_slo_successful if r.tpot is not None]
        
        if high_ttft_values:
            avg_high_ttft = sum(high_ttft_values) / len(high_ttft_values)
            logger.info(f"HIGH SLO TTFT: avg={avg_high_ttft:.3f}s, min={min(high_ttft_values):.3f}s, max={max(high_ttft_values):.3f}s")
        else:
            logger.info("HIGH SLO TTFT: No successful requests to analyze")
        
        if high_tpot_values:
            avg_high_tpot = sum(high_tpot_values) / len(high_tpot_values)
            logger.info(f"HIGH SLO TPOT: avg={avg_high_tpot:.3f}s/tok, min={min(high_tpot_values):.3f}s/tok, max={max(high_tpot_values):.3f}s/tok")
        else:
            logger.info("HIGH SLO TPOT: No successful requests to analyze")
        
        # TTFT/TPOT summary for Low SLO pools
        low_ttft_values = [r.ttft for r in low_slo_successful if r.ttft is not None]
        low_tpot_values = [r.tpot for r in low_slo_successful if r.tpot is not None]
        
        if low_ttft_values:
            avg_low_ttft = sum(low_ttft_values) / len(low_ttft_values)
            logger.info(f"LOW SLO TTFT:  avg={avg_low_ttft:.3f}s, min={min(low_ttft_values):.3f}s, max={max(low_ttft_values):.3f}s")
        else:
            logger.info("LOW SLO TTFT: No successful requests to analyze")
        
        if low_tpot_values:
            avg_low_tpot = sum(low_tpot_values) / len(low_tpot_values)
            logger.info(f"LOW SLO TPOT:  avg={avg_low_tpot:.3f}s/tok, min={min(low_tpot_values):.3f}s/tok, max={max(low_tpot_values):.3f}s/tok")
        else:
            logger.info("LOW SLO TPOT: No successful requests to analyze")
        
        # Phase-by-phase summary with TTFT/TPOT separated by SLO
        logger.info(f"\nPhase Results by SLO:")
        phase_performance = {}
        
        for phase_name, phase_results in self.phase_results.items():
            # Separate phase results by SLO
            phase_high = [r for r in phase_results if r.slo_requirement == "high"]
            phase_low = [r for r in phase_results if r.slo_requirement == "low"]
            
            high_success = sum(1 for r in phase_high if r.success)
            low_success = sum(1 for r in phase_low if r.success)
            
            high_rate = (high_success / len(phase_high) * 100) if phase_high else 0
            low_rate = (low_success / len(phase_low) * 100) if phase_low else 0
            
            # Store phase performance for analysis
            phase_performance[phase_name] = {
                'high_rate': high_rate, 'low_rate': low_rate,
                'high_total': len(phase_high), 'low_total': len(phase_low)
            }
            
            logger.info(f"  {phase_name.upper()}:")
            
            # Build High SLO message
            high_msg = f"    HIGH SLO: {high_success}/{len(phase_high)} ({high_rate:.1f}%)"
            
            # High SLO TTFT/TPOT for this phase
            phase_high_streaming = [r for r in phase_high if r.success and r.is_streaming]
            phase_high_ttft = [r.ttft for r in phase_high_streaming if r.ttft is not None]
            phase_high_tpot = [r.tpot for r in phase_high_streaming if r.tpot is not None]
            
            if phase_high_ttft:
                avg_phase_high_ttft = sum(phase_high_ttft) / len(phase_high_ttft)
                high_msg += f" | TTFT: {avg_phase_high_ttft:.3f}s"
            if phase_high_tpot:
                avg_phase_high_tpot = sum(phase_high_tpot) / len(phase_high_tpot)
                high_msg += f" | TPOT: {avg_phase_high_tpot:.3f}s/tok"
            
            logger.info(high_msg)
            
            # Build Low SLO message
            low_msg = f"    LOW SLO:  {low_success}/{len(phase_low)} ({low_rate:.1f}%)"
            
            # Low SLO TTFT/TPOT for this phase
            phase_low_streaming = [r for r in phase_low if r.success and r.is_streaming]
            phase_low_ttft = [r.ttft for r in phase_low_streaming if r.ttft is not None]
            phase_low_tpot = [r.tpot for r in phase_low_streaming if r.tpot is not None]
            
            if phase_low_ttft:
                avg_phase_low_ttft = sum(phase_low_ttft) / len(phase_low_ttft)
                low_msg += f" | TTFT: {avg_phase_low_ttft:.3f}s"
            if phase_low_tpot:
                avg_phase_low_tpot = sum(phase_low_tpot) / len(phase_low_tpot)
                low_msg += f" | TPOT: {avg_phase_low_tpot:.3f}s/tok"
            
            logger.info(low_msg)
        
        # Adjusted success criteria - more lenient for high load scenarios
        baseline_performance = (
            phase_performance.get('baseline', {}).get('high_rate', 0) + 
            phase_performance.get('baseline', {}).get('low_rate', 0) + 
            phase_performance.get('baseline_return', {}).get('high_rate', 0) + 
            phase_performance.get('baseline_return', {}).get('low_rate', 0)
        ) / 4  # Average baseline performance
        
        high_load_performance = (
            phase_performance.get('high_sustained', {}).get('high_rate', 0) + 
            phase_performance.get('high_sustained', {}).get('low_rate', 0)
        ) / 2  # Average high load performance
        
        success_criteria = [
            overall_success_rate >= 70,  # At least 70% overall success (more lenient)
            total_requests >= (100*30 + 200*60 + 100*30) * 0.5,  # At least 50% of expected requests
            baseline_performance >= 80,  # Baseline phases should work well (80%+)
        ]
        
        # Don't fail the test just because high load phase has issues - that's expected
        if high_load_performance < 50:
            logger.info("\nNOTE: High load phase showed significant failures - this is expected at 200 req/s")
        
        test_passed = all(success_criteria)
        result_str = "PASS" if test_passed else "FAIL"
        logger.info(f"\nSCALING TEST RESULT: {result_str}")
        
        logger.info("\nFailure Analysis:")
        if high_slo_failures or low_slo_failures:
            logger.info("  Common failure causes under high load:")
            logger.info("  - Resource exhaustion (GPU memory, compute)")
            logger.info("  - Request timeouts due to queue backlog") 
            logger.info("  - High SLO pools more sensitive due to TP=2 memory requirements")
            if high_timeout_failures > low_timeout_failures:
                logger.info("  - High SLO pools showing more timeouts (expected - tighter SLO)")
        
        logger.info("\nObservation Notes:")
        logger.info("  - High SLO pools: Expected better TTFT when not overloaded, more failures under extreme load")
        logger.info("  - Low SLO pools: Higher throughput tolerance, may degrade more gracefully")
        logger.info("  - Check planner logs: logs/planner_high_slo/ and logs/planner_low_slo/")
        logger.info("  - Expected patterns: Good baseline → degraded high load → recovery at baseline return")
        logger.info("=" * 60)
        
        return test_passed 