"""
Base test class for Global Scheduler tests.

This module provides common functionality that can be reused across different test types.
"""

import asyncio
import logging
import time
import requests
import aiohttp
import json
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
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestRequest] = []
        
    async def setup(self) -> bool:
        """Set up the test environment and connections"""
        logger.info("SETTING UP TEST ENVIRONMENT")
        
        # Check prerequisites
        if not await self._check_prerequisites():
            return False
            
        # Set up HTTP session for Global Scheduler communication
        if not await self._setup_http_session():
            return False
            
        # Verify system architecture
        if not await self._verify_system_health():
            return False
            
        logger.info("Test environment setup complete")
        return True
    
    async def cleanup(self):
        """Clean up test resources"""
        logger.info("Cleaning up test resources")
        # Close HTTP session if needed
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
    
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
    
    async def _setup_http_session(self) -> bool:
        """Set up HTTP session for Global Scheduler communication"""
        logger.info("Setting up HTTP session for Global Scheduler...")
        
        try:
            # Create HTTP session with appropriate timeout and connection pooling
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'GlobalSchedulerTest/1.0'}
            )
            
            # Test connectivity to Global Scheduler
            scheduler_url = self.config.get('global_scheduler_url', 'http://localhost:3999')
            async with self.http_session.get(f"{scheduler_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info(f"PASS: Successfully connected to Global Scheduler at {scheduler_url}")
                    return True
                else:
                    logger.warning(f"WARNING: Global Scheduler health check returned status {response.status}, but continuing...")
                    return True  # Don't fail if health endpoint doesn't exist
            
        except Exception as e:
            logger.error(f"FAIL: Failed to set up HTTP session for Global Scheduler: {e}")
            return False
    
    async def _verify_system_health(self) -> bool:
        """Verify that the system components are healthy"""
        logger.info("Verifying system health...")
        
        try:
            # Try to get pool status from Global Scheduler via HTTP
            scheduler_url = self.config.get('global_scheduler_url', 'http://localhost:3999')
            
            # Note: This would require the Global Scheduler to expose an HTTP endpoint for pool status
            # For now, we'll just verify basic connectivity which we already did in _setup_http_session
            logger.info("PASS: Basic connectivity to Global Scheduler verified")
            logger.info("Note: Pool status verification would require HTTP endpoint implementation")
            
            return True
            
        except Exception as e:
            logger.warning(f"WARNING: Could not verify system health: {e} - continuing with tests")
            return True  # Don't fail on this - it's informational
    
    async def send_request(self, request: TestRequest) -> TestRequest:
        """Send a single test request and populate results using HTTP v1/chat/completions endpoint"""
        logger.info(f"Sending request: {request.request_id} (SLO: {request.slo_requirement})")
        
        start_time = time.time()
        
        try:
            # Prepare OpenAI-compatible request data
            chat_request = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # Use the actual model from pool configs
                "messages": [
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,  # Enable streaming responses
                "slo_requirement": request.slo_requirement  # Add SLO requirement for Global Scheduler
            }
            
            # Get Global Scheduler URL
            scheduler_url = self.config.get('global_scheduler_url', 'http://localhost:3999')
            url = f"{scheduler_url}/v1/chat/completions"
            
            # Send HTTP request to Global Scheduler
            async with self.http_session.post(url, json=chat_request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ERROR: Global Scheduler returned HTTP {response.status}: {error_text}")
                    request.success = False
                    request.error = f"HTTP {response.status}: {error_text}"
                    request.response_time = time.time() - start_time
                    return request
                
                # Process streaming response
                first_token_time = None
                response_chunks = []
                accumulated_content = ""
                final_response = None
                
                # Process Server-Sent Events stream
                buffer = ""
                async for chunk in response.content.iter_chunked(8192):
                    if not chunk:
                        continue
                        
                    # Decode chunk and add to buffer
                    buffer += chunk.decode('utf-8')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        # Parse SSE data
                        if line.startswith('data: '):
                            data_content = line[6:]
                            if data_content == '[DONE]':
                                break
                                
                            try:
                                chunk_data = json.loads(data_content)
                                response_chunks.append(chunk_data)
                                
                                # Track first token time for TTFT metrics
                                if first_token_time is None:
                                    first_token_time = time.time()
                                
                                # Extract content from streaming chunks
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    choice = chunk_data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        accumulated_content += content
                                    
                                    # Check for completion
                                    if 'finish_reason' in choice and choice['finish_reason'] is not None:
                                        # Create final response in OpenAI format
                                        final_response = {
                                            "id": chunk_data.get("id", f"chatcmpl-{request.request_id}"),
                                            "object": "chat.completion",
                                            "created": int(time.time()),
                                            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                            "choices": [{
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": accumulated_content
                                                },
                                                "finish_reason": choice['finish_reason']
                                            }]
                                        }
                                        
                                        # Add usage information if available
                                        if 'usage' in chunk_data:
                                            final_response['usage'] = chunk_data['usage']
                                        break
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"WARNING: Failed to parse streaming chunk: {e}, data: {data_content}")
                                continue
                
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
                request.total_content_length = len(accumulated_content)
                
                # Process final response
                if final_response:
                    request.success = True
                    request.assigned_pool = None  # HTTP response doesn't include pool info
                    request.response_data = {"success": True, "response": final_response}
                    
                    # Try to get actual token count from response usage info
                    if 'usage' in final_response and final_response['usage'] is not None:
                        request.output_token_count = final_response['usage'].get('completion_tokens', 0)
                    else:
                        # Estimate token count (rough approximation: ~4 chars per token)
                        request.output_token_count = max(1, len(accumulated_content) // 4)
                    
                    # Calculate TPOT (Time Per Output Token) if we have TTFT and token count
                    if request.ttft is not None and request.output_token_count > 0:
                        generation_time = request.response_time - request.ttft
                        request.tpot = generation_time / request.output_token_count
                        logger.info(f"TPOT for {request.request_id}: {request.tpot:.3f}s/token ({request.output_token_count} tokens)")
                    
                    # Enhanced response summary for streaming
                    self._print_streaming_response_summary(request, request.response_data, len(response_chunks))
                    
                elif accumulated_content:
                    # Create final response even if we didn't get explicit finish_reason
                    final_response = {
                        "id": f"chatcmpl-{request.request_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": accumulated_content
                            },
                            "finish_reason": "stop"  # Default finish reason
                        }]
                    }
                    request.success = True
                    request.assigned_pool = None
                    request.response_data = {"success": True, "response": final_response}
                    request.output_token_count = max(1, len(accumulated_content) // 4)
                    
                    # Calculate TPOT if we have TTFT
                    if request.ttft is not None and request.output_token_count > 0:
                        generation_time = request.response_time - request.ttft
                        request.tpot = generation_time / request.output_token_count
                        logger.info(f"TPOT for {request.request_id}: {request.tpot:.3f}s/token ({request.output_token_count} tokens)")
                    
                    self._print_streaming_response_summary(request, request.response_data, len(response_chunks))
                else:
                    logger.warning(f"WARNING: No content received for streaming request {request.request_id}")
                    request.success = False
                    request.error = "Stream ended without content"
            
        except Exception as e:
            request.success = False
            request.error = str(e)
            request.response_time = time.time() - start_time
            logger.error(f"FAIL: Request failed with exception: {e}")
        
        return request
    

    
    def _print_streaming_response_summary(self, request: TestRequest, response_data: Dict[str, Any], chunk_count: int):
        """Print a summary of the streaming response in a readable format"""
        pool_info = request.assigned_pool if request.assigned_pool else "v1/chat/completions"
        metrics_str = f"{request.response_time:.2f}s -> {pool_info} ({chunk_count} chunks"
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