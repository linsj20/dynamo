"""
Streaming-focused Global Scheduler test implementation.

This test validates streaming functionality by sending streaming requests
and verifying proper SSE handling, TTFT metrics, and chunk processing.
"""

import asyncio
import logging
from typing import List

from .test_base import BaseGlobalSchedulerTest, TestRequest

logger = logging.getLogger(__name__)

class StreamingSchedulerTest(BaseGlobalSchedulerTest):
    """Test that validates streaming-specific Global Scheduler functionality"""
    
    def __init__(self, config):
        super().__init__(config)
        self.test_requests = self._create_streaming_test_requests()
    
    def _create_streaming_test_requests(self) -> List[TestRequest]:
        """Create test requests specifically designed to test streaming functionality"""
        
        test_cases = [
            # Short response for quick TTFT validation
            TestRequest(
                request_id="streaming_short_high",
                slo_requirement="high",
                prompt="Say 'Hello, world!' and stop.",
                max_tokens=10,
                expected_pool="high_slo_pool"
            ),
            
            # Medium response for chunk counting
            TestRequest(
                request_id="streaming_medium_high", 
                slo_requirement="high",
                prompt="Count from 1 to 10 with brief explanations.",
                max_tokens=100,
                expected_pool="high_slo_pool"
            ),
            
            # Longer response for comprehensive streaming test
            TestRequest(
                request_id="streaming_long_low",
                slo_requirement="low",
                prompt="Write a detailed explanation of how streaming works in HTTP, including Server-Sent Events.",
                max_tokens=200,
                expected_pool="low_slo_pool"
            ),
            
            # Technical response to test content quality
            TestRequest(
                request_id="streaming_technical_low",
                slo_requirement="low",
                prompt="Explain the differences between REST and GraphQL APIs with examples.",
                max_tokens=150,
                expected_pool="low_slo_pool"
            ),
        ]
        
        return test_cases
    
    async def run_test_logic(self) -> bool:
        """Execute the streaming test logic"""
        logger.info("RUNNING STREAMING GLOBAL SCHEDULER TEST")
        logger.info(f"Total streaming test requests: {len(self.test_requests)}")
        
        # Send all streaming test requests
        for request in self.test_requests:
            logger.info(f"Testing streaming for {request.request_id} (SLO: {request.slo_requirement})")
            
            # Send request and store result
            result = await self.send_request(request)
            self.results.append(result)
            
            # Validate streaming-specific requirements
            if not self._validate_streaming_response(result):
                logger.error(f"FAIL: Streaming validation failed for {request.request_id}")
            
            # Brief pause between requests
            await asyncio.sleep(2)
        
        # Analyze streaming results
        return self._analyze_streaming_results()
    
    def _validate_streaming_response(self, request: TestRequest) -> bool:
        """Validate that the response meets streaming requirements"""
        if not request.success:
            logger.error(f"FAIL: Request {request.request_id} was not successful")
            return False
        
        if not request.is_streaming:
            logger.error(f"FAIL: Request {request.request_id} was not processed as streaming")
            return False
        
        # Validate TTFT is reasonable for SLO level
        if request.ttft is not None:
            if request.slo_requirement == "high" and request.ttft > 1.0:
                logger.warning(f"WARNING: High SLO request {request.request_id} has TTFT {request.ttft:.3f}s > 1.0s")
            elif request.slo_requirement == "low" and request.ttft > 5.0:
                logger.warning(f"WARNING: Low SLO request {request.request_id} has TTFT {request.ttft:.3f}s > 5.0s")
        
        # Validate we got at least one chunk
        if request.chunk_count == 0:
            logger.error(f"FAIL: Request {request.request_id} received no streaming chunks")
            return False
        
        # Validate we got some content
        if request.total_content_length == 0:
            logger.error(f"FAIL: Request {request.request_id} generated no content")
            return False
        
        tpot_str = f", TPOT: {request.tpot:.3f}s/tok" if request.tpot is not None else ""
        logger.info(f"PASS: Streaming validation for {request.request_id} - "
                   f"TTFT: {request.ttft:.3f}s{tpot_str}, Chunks: {request.chunk_count}, "
                   f"Length: {request.total_content_length}, Tokens: {request.output_token_count}")
        
        return True
    
    def _analyze_streaming_results(self) -> bool:
        """Analyze streaming test results and determine if the test passed"""
        logger.info("ANALYZING STREAMING TEST RESULTS")
        
        if not self.results:
            logger.error("FAIL: No results to analyze")
            return False
        
        # Count successful streaming requests
        successful_streaming = [r for r in self.results if r.success and r.is_streaming]
        success_rate = len(successful_streaming) / len(self.results)
        
        logger.info(f"Streaming success rate: {len(successful_streaming)}/{len(self.results)} ({success_rate * 100:.1f}%)")
        
        # Validate SLO-specific streaming performance
        high_slo_requests = [r for r in successful_streaming if r.slo_requirement == "high"]
        low_slo_requests = [r for r in successful_streaming if r.slo_requirement == "low"]
        
        # Check High SLO TTFT requirements
        if high_slo_requests:
            high_ttft_values = [r.ttft for r in high_slo_requests if r.ttft is not None]
            if high_ttft_values:
                avg_high_ttft = sum(high_ttft_values) / len(high_ttft_values)
                max_high_ttft = max(high_ttft_values)
                logger.info(f"High SLO TTFT: avg={avg_high_ttft:.3f}s, max={max_high_ttft:.3f}s")
                
                if max_high_ttft > 2.0:  # Reasonable threshold for testing
                    logger.warning(f"WARNING: High SLO max TTFT {max_high_ttft:.3f}s exceeds 2.0s threshold")
        
        # Check Low SLO streaming completeness
        if low_slo_requests:
            low_content_lengths = [r.total_content_length for r in low_slo_requests]
            if low_content_lengths:
                avg_low_content = sum(low_content_lengths) / len(low_content_lengths)
                logger.info(f"Low SLO avg content length: {avg_low_content:.0f} chars")
        
        # Validate pool assignment accuracy (using prefix match)
        pool_assignment_correct = True
        for request in self.results:
            if request.success and request.expected_pool:
                if not request.assigned_pool or not request.assigned_pool.startswith(request.expected_pool):
                    logger.error(f"FAIL: Request {request.request_id} assigned to {request.assigned_pool}, expected {request.expected_pool} (prefix)")
                    pool_assignment_correct = False
        
        # Overall test success criteria
        test_passed = (
            success_rate >= 0.8 and  # At least 80% success rate
            pool_assignment_correct   # All pools assigned correctly
        )
        
        if test_passed:
            logger.info("PASS: Streaming test completed successfully")
        else:
            logger.error("FAIL: Streaming test failed success criteria")
        
        return test_passed 