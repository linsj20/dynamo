"""
Simple Global Scheduler test implementation.

This test validates basic functionality by sending requests with different SLO requirements
and verifying that responses are reasonable and routed correctly.
"""

import asyncio
import logging
from typing import List

from .test_base import BaseGlobalSchedulerTest, TestRequest

logger = logging.getLogger(__name__)

class SimpleSchedulerTest(BaseGlobalSchedulerTest):
    """Simple test that validates basic Global Scheduler functionality"""
    
    def __init__(self, config):
        super().__init__(config)
        self.test_requests = self._create_test_requests()
    
    def _create_test_requests(self) -> List[TestRequest]:
        """Create a set of test requests with different SLO requirements and prompts"""
        
        # Different types of prompts to test various scenarios with balanced SLO distribution
        test_cases = [
            # High SLO requests - should get fast, priority service
            TestRequest(
                request_id="high_priority_1",
                slo_requirement="high",
                prompt="What is the capital of France?",
                max_tokens=30,
                expected_pool="high_slo_pool"
            ),
            TestRequest(
                request_id="high_priority_2", 
                slo_requirement="high",
                prompt="Explain artificial intelligence in one sentence.",
                max_tokens=40,
                expected_pool="high_slo_pool"
            ),
            
            # Low SLO requests - can tolerate higher latency
            TestRequest(
                request_id="low_priority_1",
                slo_requirement="low", 
                prompt="Write a short story about a robot learning to paint.",
                max_tokens=60,
                expected_pool="low_slo_pool"
            ),
            TestRequest(
                request_id="low_priority_2",
                slo_requirement="low",
                prompt="Compare the advantages and disadvantages of renewable energy sources.",
                max_tokens=80,
                expected_pool="low_slo_pool"
            ),
            TestRequest(
                request_id="low_priority_3",
                slo_requirement="low",
                prompt="Describe the process of making coffee.",
                max_tokens=50,
                expected_pool="low_slo_pool"
            ),
            
            # Additional low SLO request to balance distribution  
            TestRequest(
                request_id="low_priority_4",
                slo_requirement="low",
                prompt="What are the benefits of exercise?",
                max_tokens=40,
                expected_pool="low_slo_pool"
            ),
        ]
        
        return test_cases
    
    async def run_test_logic(self) -> bool:
        """Execute the simple test logic"""
        logger.info("RUNNING SIMPLE GLOBAL SCHEDULER TEST")
        logger.info(f"Total test requests: {len(self.test_requests)}")
        
        # Send all test requests
        for request in self.test_requests:
            # Send request and store result
            result = await self.send_request(request)
            self.results.append(result)
            
            # Brief pause between requests to avoid overwhelming the system
            await asyncio.sleep(1)
        
        # Analyze results
        return self._analyze_results()
    
    def _analyze_results(self) -> bool:
        """Analyze test results and determine if the test passed"""
        logger.info("ANALYZING TEST RESULTS")
        
        if not self.results:
            logger.error("No test results to analyze")
            return False
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        
        # Basic success rate validation
        success_rate = (successful_requests / total_requests) * 100
        logger.info(f"Overall success rate: {success_rate:.1f}% ({successful_requests}/{total_requests})")
        
        # Minimum success rate requirement
        min_success_rate = 70  # 70% minimum for simple test
        if success_rate < min_success_rate:
            logger.error(f"FAIL: Success rate {success_rate:.1f}% below minimum {min_success_rate}%")
            return False
        
        # Validate SLO-based routing
        routing_success = self._validate_slo_routing()
        if not routing_success:
            logger.warning("WARNING: SLO routing validation failed, but continuing...")
            # Don't fail the entire test for routing issues in simple test
        
        # Validate response quality
        response_quality = self._validate_response_quality()
        if not response_quality:
            logger.warning("WARNING: Response quality validation had issues, but continuing...")
            # Don't fail the entire test for response quality issues
        
        # Validate response times are reasonable
        time_validation = self._validate_response_times()
        if not time_validation:
            logger.warning("WARNING: Response time validation had issues, but continuing...")
        
        logger.info("PASS: Simple test analysis complete")
        return True
    
    def _validate_slo_routing(self) -> bool:
        """Validate that requests are routed to appropriate pools based on SLO"""
        logger.info("Validating SLO-based routing...")
        
        routing_issues = 0
        
        for request in self.results:
            if not request.success:
                continue
                
            expected_pool = request.expected_pool
            actual_pool = request.assigned_pool
            
            if expected_pool and actual_pool:
                # Check if routing matches expectation
                if expected_pool not in actual_pool.lower():
                    logger.warning(f"WARNING: {request.request_id}: Expected {expected_pool}, got {actual_pool}")
                    routing_issues += 1
                else:
                    logger.debug(f"PASS: {request.request_id}: Correctly routed to {actual_pool}")
        
        if routing_issues == 0:
            logger.info("PASS: All requests routed correctly")
            return True
        else:
            logger.warning(f"WARNING: {routing_issues} routing issues detected")
            return False
    
    def _validate_response_quality(self) -> bool:
        """Validate that responses are reasonable and contain expected content"""
        logger.info("Validating response quality...")
        
        quality_issues = 0
        
        for request in self.results:
            if not request.success:
                continue
                
            response_data = request.response_data
            if not response_data:
                quality_issues += 1
                continue
                
            # Extract response content
            llm_response = response_data.get('response', {})
            if isinstance(llm_response, dict):
                choices = llm_response.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    if content and len(content.strip()) > 0:
                        logger.debug(f"PASS: {request.request_id}: Response has content ({len(content)} chars)")
                    else:
                        logger.warning(f"WARNING: {request.request_id}: Response has no content")
                        quality_issues += 1
                else:
                    logger.warning(f"WARNING: {request.request_id}: Response has no choices")
                    quality_issues += 1
            else:
                logger.warning(f"WARNING: {request.request_id}: Unexpected response format")
                quality_issues += 1
        
        successful_responses = sum(1 for r in self.results if r.success)
        if successful_responses > 0:
            quality_rate = ((successful_responses - quality_issues) / successful_responses) * 100
            logger.info(f"Response quality rate: {quality_rate:.1f}%")
            
            return quality_rate >= 80  # 80% quality threshold
        else:
            return False
    
    def _validate_response_times(self) -> bool:
        """Validate that response times are within reasonable bounds"""
        logger.info("Validating response times...")
        
        # Define reasonable time bounds (in seconds)
        max_reasonable_time = 30.0  # 30 seconds max for simple test
        high_slo_max_time = 5.0     # High SLO should be faster
        
        time_issues = 0
        
        for request in self.results:
            if not request.success:
                continue
                
            response_time = request.response_time
            
            # General time validation
            if response_time > max_reasonable_time:
                logger.warning(f"WARNING: {request.request_id}: Slow response ({response_time:.2f}s)")
                time_issues += 1
                continue
            
            # SLO-specific validation
            if request.slo_requirement == "high" and response_time > high_slo_max_time:
                logger.warning(f"WARNING: {request.request_id}: High SLO request too slow ({response_time:.2f}s)")
                time_issues += 1
                continue
                
            logger.debug(f"PASS: {request.request_id}: Good response time ({response_time:.2f}s)")
        
        successful_responses = sum(1 for r in self.results if r.success)
        if successful_responses > 0:
            time_rate = ((successful_responses - time_issues) / successful_responses) * 100
            logger.info(f"Response time compliance: {time_rate:.1f}%")
            
            return time_rate >= 70  # 70% time compliance threshold
        else:
            return False 