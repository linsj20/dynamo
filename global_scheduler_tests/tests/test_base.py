"""
Base test class for Global Scheduler tests.

This module provides common functionality that can be reused across different test types.
"""

import asyncio
import logging
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class TestRequest:
    """Represents a test request with its configuration and results"""
    request_id: str
    slo_requirement: str
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    expected_pool: Optional[str] = None
    
    # Results (filled after execution)
    response_time: float = 0.0
    success: bool = False
    assigned_pool: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Streaming-specific metrics
    ttft: Optional[float] = None  # Time to First Token
    tpot: Optional[float] = None  # Time Per Output Token
    chunk_count: int = 0         # Number of streaming chunks received
    total_content_length: int = 0 # Total length of generated content
    output_token_count: int = 0  # Number of output tokens generated
    is_streaming: bool = True    # Whether this was a streaming request

class BaseGlobalSchedulerTest(ABC):
    """Base class for all Global Scheduler tests"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runtime = None
        self.scheduler_client = None
        self.results: List[TestRequest] = []
        
    async def setup(self) -> bool:
        """Set up the test environment and connections"""
        logger.info("SETTING UP TEST ENVIRONMENT")
        
        # Check prerequisites
        if not await self._check_prerequisites():
            return False
            
        # Connect to Global Scheduler
        if not await self._connect_to_scheduler():
            return False
            
        # Verify system architecture
        if not await self._verify_system_health():
            return False
            
        logger.info("Test environment setup complete")
        return True
    
    async def cleanup(self):
        """Clean up test resources"""
        logger.info("Cleaning up test resources")
        # Close any connections if needed
        if self.runtime:
            # Runtime cleanup would go here if needed
            pass
    
    @abstractmethod
    async def run_test_logic(self) -> bool:
        """Override this method to implement specific test logic"""
        pass
    
    async def run(self) -> bool:
        """Main test execution flow"""
        try:
            # Setup
            if not await self.setup():
                logger.error("Setup failed")
                return False
            
            # Run test logic
            if not await self.run_test_logic():
                logger.error("Test logic failed")
                return False
            
            # Print results summary
            self._print_test_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()
    
    async def _check_prerequisites(self) -> bool:
        """Check that required services are running"""
        logger.info("Checking prerequisites...")
        
        # Check etcd
        try:
            response = requests.get("http://localhost:2379/health", timeout=5)
            if response.status_code != 200:
                raise Exception(f"ETCD health check failed: {response.status_code}")
            logger.info("PASS: etcd is healthy")
        except Exception as e:
            logger.error(f"FAIL: etcd is not accessible: {e}")
            return False
        
        # Check NATS
        try:
            response = requests.get("http://localhost:8222/varz", timeout=5)
            if response.status_code != 200:
                raise Exception(f"NATS health check failed: {response.status_code}")
            logger.info("PASS: NATS is healthy")
        except Exception as e:
            logger.error(f"FAIL: NATS is not accessible: {e}")
            return False
        
        return True
    
    async def _connect_to_scheduler(self) -> bool:
        """Connect to the Global Scheduler"""
        logger.info("Connecting to Global Scheduler...")
        
        try:
            # Import runtime after path setup
            from dynamo.runtime import DistributedRuntime
            
            # Initialize runtime
            loop = asyncio.get_running_loop()
            self.runtime = DistributedRuntime(loop, False)
            
            # Connect to Global Scheduler
            scheduler_component = self.runtime.namespace("dynamo").component("GlobalScheduler")
            generate_endpoint = scheduler_component.endpoint("generate")
            self.scheduler_client = await generate_endpoint.client()
            
            # Verify connection
            instances = self.scheduler_client.instance_ids()
            if not instances:
                raise Exception("No global scheduler instances found")
            
            logger.info(f"PASS: Connected to Global Scheduler ({len(instances)} instances)")
            return True
            
        except Exception as e:
            logger.error(f"FAIL: Failed to connect to Global Scheduler: {e}")
            return False
    
    async def _verify_system_health(self) -> bool:
        """Verify that the system components are healthy"""
        logger.info("Verifying system health...")
        
        try:
            # Check if pools are registered with the Global Scheduler
            status_endpoint = self.runtime.namespace("dynamo").component("GlobalScheduler").endpoint("get_pool_status")
            status_client = await status_endpoint.client()
            
            # Get pool status
            pool_status_response = await status_client.get_pool_status()
            
            # Extract pool status from response
            pool_status = None
            async for status_item in pool_status_response:
                pool_status = status_item
                break
            
            if not pool_status or pool_status.get('total_pools', 0) == 0:
                logger.warning("WARNING: No pools are registered yet - this may be expected during startup")
                return True  # Don't fail - pools might register during test
            
            total_pools = pool_status.get('total_pools', 0)
            connected_pools = pool_status.get('connected_pools', 0)
            
            logger.info(f"PASS: System health: {connected_pools}/{total_pools} pools connected")
            
            return True
            
        except Exception as e:
            logger.warning(f"WARNING: Could not verify pool status: {e} - continuing with tests")
            return True  # Don't fail on this - it's informational
    
    async def send_request(self, request: TestRequest) -> TestRequest:
        """Send a single test request and populate results"""
        logger.info(f"Sending request: {request.request_id} (SLO: {request.slo_requirement})")
        
        start_time = time.time()
        
        try:
            # Prepare request data with streaming enabled
            request_data = {
                "request_id": request.request_id,
                "slo_requirement": request.slo_requirement,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True  # Enable streaming responses
            }
            
            # Send request to Global Scheduler
            response = await self.scheduler_client.generate(request_data)
            
            # Process streaming response
            first_token_time = None
            response_chunks = []
            final_response_data = None
            
            async for item in response:
                # Handle potential Annotated type wrapper
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
            
            # Calculate response metrics
            end_time = time.time()
            request.response_time = end_time - start_time
            
            # Calculate and store TTFT (Time to First Token) if we got streaming data
            if first_token_time:
                request.ttft = first_token_time - start_time
                logger.info(f"TTFT for {request.request_id}: {request.ttft:.3f}s")
                
            # Store streaming metrics
            request.chunk_count = len(response_chunks)
            request.is_streaming = True
                
            # Process final response
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
                        logger.info(f"TPOT for {request.request_id}: {request.tpot:.3f}s/token ({request.output_token_count} tokens)")
                    
                    # Enhanced response summary for streaming
                    self._print_streaming_response_summary(request, final_response_data, len(response_chunks))
                    
                else:
                    request.success = False
                    request.error = final_response_data.get('error', 'Unknown error')
                    logger.error(f"FAIL: Request failed: {request.error}")
            else:
                request.success = False
                request.error = "No response data received from streaming request"
                logger.error(f"FAIL: {request.error}")
            
        except Exception as e:
            request.success = False
            request.error = str(e)
            request.response_time = time.time() - start_time
            logger.error(f"FAIL: Request failed with exception: {e}")
        
        return request
    
    def _extract_response_data(self, response_item):
        """Extract response data from potentially wrapped Dynamo objects"""
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
    
    def _print_streaming_response_summary(self, request: TestRequest, response_data: Dict[str, Any], chunk_count: int):
        """Print a summary of the streaming response in a readable format"""
        metrics_str = f"{request.response_time:.2f}s -> {request.assigned_pool} ({chunk_count} chunks"
        if request.ttft is not None:
            metrics_str += f", TTFT: {request.ttft:.3f}s"
        if request.tpot is not None:
            metrics_str += f", TPOT: {request.tpot:.3f}s/tok"
        metrics_str += ")"
        logger.info(f"PASS: {request.request_id}: {metrics_str}")
        
        # Extract the actual LLM response
        llm_response = response_data.get('response', {})
        if isinstance(llm_response, dict):
            choices = llm_response.get('choices', [])
            if choices:
                message_content = choices[0].get('message', {}).get('content', '')
                if message_content:
                    # Print first 100 characters of the response
                    preview = message_content[:100].replace('\n', ' ')
                    if len(message_content) > 100:
                        preview += "..."
                    logger.info(f"   Response: \"{preview}\"")
                    logger.info(f"   Full length: {len(message_content)} characters")
                else:
                    logger.info(f"   Response: [No content in message]")
            else:
                logger.info(f"   Response: [No choices in response]")
        else:
            logger.info(f"   Response: {str(llm_response)[:100]}...")
    
    def _print_test_summary(self):
        """Print a summary of all test results"""
        if not self.results:
            return
            
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        streaming_requests = sum(1 for r in self.results if r.is_streaming)
        
        # Overall stats
        avg_response_time = sum(r.response_time for r in self.results) / total_requests
        
        logger.info(f"Total Requests: {total_requests}")
        logger.info(f"Successful: {successful_requests}")
        logger.info(f"Failed: {failed_requests}")
        logger.info(f"Streaming: {streaming_requests}")
        logger.info(f"Average Response Time: {avg_response_time:.3f}s")
        
        # Streaming metrics
        successful_streaming = [r for r in self.results if r.success and r.is_streaming]
        if successful_streaming:
            ttft_values = [r.ttft for r in successful_streaming if r.ttft is not None]
            if ttft_values:
                avg_ttft = sum(ttft_values) / len(ttft_values)
                min_ttft = min(ttft_values)
                max_ttft = max(ttft_values)
                logger.info(f"TTFT: avg={avg_ttft:.3f}s, min={min_ttft:.3f}s, max={max_ttft:.3f}s")
            
            tpot_values = [r.tpot for r in successful_streaming if r.tpot is not None]
            if tpot_values:
                avg_tpot = sum(tpot_values) / len(tpot_values)
                min_tpot = min(tpot_values)
                max_tpot = max(tpot_values)
                logger.info(f"TPOT: avg={avg_tpot:.3f}s/tok, min={min_tpot:.3f}s/tok, max={max_tpot:.3f}s/tok")
            
            avg_chunk_count = sum(r.chunk_count for r in successful_streaming) / len(successful_streaming)
            avg_content_length = sum(r.total_content_length for r in successful_streaming) / len(successful_streaming)
            avg_token_count = sum(r.output_token_count for r in successful_streaming) / len(successful_streaming)
            logger.info(f"Avg Chunks per Request: {avg_chunk_count:.1f}")
            logger.info(f"Avg Content Length: {avg_content_length:.0f} chars")
            logger.info(f"Avg Output Tokens: {avg_token_count:.1f} tokens")
        
        # Break down by SLO level
        slo_stats = {}
        for request in self.results:
            slo = request.slo_requirement
            if slo not in slo_stats:
                slo_stats[slo] = {
                    'total': 0, 'success': 0, 'total_time': 0.0, 
                    'ttft_values': [], 'tpot_values': [], 'chunk_counts': [], 'token_counts': []
                }
            
            slo_stats[slo]['total'] += 1
            if request.success:
                slo_stats[slo]['success'] += 1
                if request.ttft is not None:
                    slo_stats[slo]['ttft_values'].append(request.ttft)
                if request.tpot is not None:
                    slo_stats[slo]['tpot_values'].append(request.tpot)
                slo_stats[slo]['chunk_counts'].append(request.chunk_count)
                slo_stats[slo]['token_counts'].append(request.output_token_count)
            slo_stats[slo]['total_time'] += request.response_time
        
        logger.info("\nSLO Level Breakdown:")
        for slo, stats in slo_stats.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_time = stats['total_time'] / stats['total']
            logger.info(f"  {slo.upper()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - Avg: {avg_time:.3f}s")
            
            # TTFT stats per SLO
            if stats['ttft_values']:
                avg_ttft = sum(stats['ttft_values']) / len(stats['ttft_values'])
                logger.info(f"    TTFT: {avg_ttft:.3f}s")
            
            # TPOT stats per SLO
            if stats['tpot_values']:
                avg_tpot = sum(stats['tpot_values']) / len(stats['tpot_values'])
                logger.info(f"    TPOT: {avg_tpot:.3f}s/tok")
            
            # Average chunks and tokens per SLO
            if stats['chunk_counts']:
                avg_chunks = sum(stats['chunk_counts']) / len(stats['chunk_counts'])
                logger.info(f"    Avg Chunks: {avg_chunks:.1f}")
            
            if stats['token_counts']:
                avg_tokens = sum(stats['token_counts']) / len(stats['token_counts'])
                logger.info(f"    Avg Tokens: {avg_tokens:.1f}")
        
        # Pool assignment breakdown
        pool_stats = {}
        for request in self.results:
            if request.success and request.assigned_pool:
                pool = request.assigned_pool
                if pool not in pool_stats:
                    pool_stats[pool] = 0
                pool_stats[pool] += 1
        
        if pool_stats:
            logger.info("\nPool Assignment:")
            for pool, count in pool_stats.items():
                logger.info(f"  {pool}: {count} requests")
        
        logger.info("=" * 60) 