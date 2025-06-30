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
            # Prepare request data
            request_data = {
                "request_id": request.request_id,
                "slo_requirement": request.slo_requirement,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False
            }
            
            # Send request to Global Scheduler
            response = await self.scheduler_client.generate(request_data)
            
            # Read response
            response_data = None
            async for item in response:
                response_data = item
                break
            
            # Calculate response time
            end_time = time.time()
            request.response_time = end_time - start_time
            
            # Handle potential Annotated type wrapper bug in Dynamo runtime
            actual_response_data = response_data
            
            # Check if this is a Dynamo Annotated object (has .data() method)
            if hasattr(response_data, 'data') and callable(getattr(response_data, 'data')):
                # This is a Dynamo Annotated object - call the data() method
                actual_response_data = response_data.data()
                logger.debug(f"Extracted data from Annotated object: {type(actual_response_data)}")
            elif hasattr(response_data, '__origin__') or str(type(response_data)).startswith("<class 'typing."):
                # This is likely a typing.Annotated type wrapper - try to extract the actual data
                if hasattr(response_data, '__args__') and response_data.__args__:
                    actual_response_data = response_data.__args__[0]
                else:
                    logger.error(f"FAIL: Received unexpected type: {type(response_data)} - {response_data}")
                    request.success = False
                    request.error = f"Runtime returned unexpected type: {type(response_data)}"
                    return request
            
            # Check if request was successful
            if actual_response_data and hasattr(actual_response_data, 'get'):
                if actual_response_data.get('success', False):
                    request.success = True
                    request.assigned_pool = actual_response_data.get('assigned_pool')
                    request.response_data = actual_response_data
                    
                    # Extract and print meaningful response content
                    self._print_response_summary(request, actual_response_data)
                    
                else:
                    request.success = False
                    request.error = actual_response_data.get('error', 'Unknown error')
                    logger.error(f"FAIL: Request failed: {request.error}")
            elif isinstance(actual_response_data, dict):
                if actual_response_data.get('success', False):
                    request.success = True
                    request.assigned_pool = actual_response_data.get('assigned_pool')
                    request.response_data = actual_response_data
                    
                    # Extract and print meaningful response content
                    self._print_response_summary(request, actual_response_data)
                    
                else:
                    request.success = False
                    request.error = actual_response_data.get('error', 'Unknown error')
                    logger.error(f"FAIL: Request failed: {request.error}")
            else:
                request.success = False
                request.error = f'Invalid response format: {type(actual_response_data)} - {actual_response_data}'
                logger.error(f"FAIL: Request failed: {request.error}")
            
        except Exception as e:
            request.success = False
            request.error = str(e)
            request.response_time = time.time() - start_time
            logger.error(f"FAIL: Request failed with exception: {e}")
        
        return request
    
    def _print_response_summary(self, request: TestRequest, response_data: Dict[str, Any]):
        """Print a summary of the response in a readable format"""
        logger.info(f"PASS: {request.request_id}: {request.response_time:.2f}s -> {request.assigned_pool}")
        
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
        
        # Overall stats
        avg_response_time = sum(r.response_time for r in self.results) / total_requests
        
        logger.info(f"Total Requests: {total_requests}")
        logger.info(f"Successful: {successful_requests}")
        logger.info(f"Failed: {failed_requests}")
        logger.info(f"Average Response Time: {avg_response_time:.3f}s")
        
        # Break down by SLO level
        slo_stats = {}
        for request in self.results:
            slo = request.slo_requirement
            if slo not in slo_stats:
                slo_stats[slo] = {'total': 0, 'success': 0, 'total_time': 0.0}
            
            slo_stats[slo]['total'] += 1
            if request.success:
                slo_stats[slo]['success'] += 1
            slo_stats[slo]['total_time'] += request.response_time
        
        logger.info("\nSLO Level Breakdown:")
        for slo, stats in slo_stats.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_time = stats['total_time'] / stats['total']
            logger.info(f"  {slo.upper()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - Avg: {avg_time:.3f}s")
        
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