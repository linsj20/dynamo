#!/usr/bin/env python3
"""
Global Scheduler Test Runner

This script provides a complete test framework for the Global Scheduler system.
It starts all required services, runs tests, and cleans up automatically.
"""

import argparse
import asyncio
import os
import sys
import logging
import time
import signal
import subprocess
from typing import Dict, Any, Optional

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib', 'bindings', 'python', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deploy', 'sdk', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components', 'global_scheduler', 'src'))

from tests.test_base import BaseGlobalSchedulerTest
from tests.test_simple import SimpleSchedulerTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

class ServiceManager:
    """Manages the lifecycle of services needed for Global Scheduler testing"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.processes: Dict[str, subprocess.Popen] = {}
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Port counter for dynamic port allocation (multi-node safe)
        self.next_dynamo_port = 4000
        
        # Setup paths
        self._setup_environment()
        
    def _get_next_dynamo_port(self) -> int:
        """Get the next available DYNAMO_PORT and increment counter"""
        port = self.next_dynamo_port
        self.next_dynamo_port += 1
        return port
        
    def _setup_environment(self):
        """Setup environment variables and paths"""
        # Add required paths to PYTHONPATH
        python_paths = [
            os.path.join(self.project_root, 'lib', 'bindings', 'python', 'src'),
            os.path.join(self.project_root, 'deploy', 'sdk', 'src'),
            os.path.join(self.project_root, 'components', 'global_scheduler', 'src'),
            os.path.join(self.project_root, 'examples', 'llm')
        ]
        
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = ':'.join(python_paths + [current_pythonpath] if current_pythonpath else python_paths)
        os.environ['PYTHONPATH'] = new_pythonpath
        
        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def start_all_services(self) -> bool:
        """Start all required services for testing"""
        logger.info("Starting all services for Global Scheduler testing...")
        logger.info(f"Logs will be written to: {self.logs_dir}/")
        
        if not self._start_infrastructure():
            logger.error("Failed to start infrastructure services")
            return False
        if not self._start_global_scheduler():
            logger.error("Failed to start Global Scheduler")
            return False
        if not self._start_pools():
            logger.error("Failed to start SLO pools")
            return False
            
        logger.info("All services started successfully")
        return True
    
    def _start_infrastructure(self) -> bool:
        """Start etcd and NATS services"""
        logger.info("Starting infrastructure services...")
        
        # Start etcd
        try:
            log_file = os.path.join(self.logs_dir, 'etcd.log')
            logger.info(f"  Starting etcd (log: {log_file})")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(['etcd'], stdout=f, stderr=subprocess.STDOUT)
            self.processes['etcd'] = process
            time.sleep(3)
            
            # Check health
            health_check = subprocess.run(['curl', '-f', 'http://localhost:2379/health'], 
                                        capture_output=True, timeout=5)
            if health_check.returncode != 0:
                logger.error(f"FAIL: etcd health check failed - check {log_file}")
                return False
            logger.info("  PASS: etcd is healthy")
        except Exception as e:
            logger.error(f"FAIL: Failed to start etcd: {e} - check {log_file}")
            return False
        
        # Start NATS
        try:
            log_file = os.path.join(self.logs_dir, 'nats.log')
            logger.info(f"  Starting NATS (log: {log_file})")
            with open(log_file, 'w') as f:
                process = subprocess.Popen(['nats-server', '-js', '-m', '8222'], 
                                         stdout=f, stderr=subprocess.STDOUT)
            self.processes['nats'] = process
            time.sleep(3)
            
            # Check health
            health_check = subprocess.run(['curl', '-f', 'http://localhost:8222/varz'], 
                                        capture_output=True, timeout=5)
            if health_check.returncode != 0:
                logger.error(f"FAIL: NATS health check failed - check {log_file}")
                return False
            logger.info("  PASS: NATS is healthy")
        except Exception as e:
            logger.error(f"FAIL: Failed to start NATS: {e} - check {log_file}")
            return False
        
        return True
        
    def _start_global_scheduler(self) -> bool:
        """Start the Global Scheduler service"""
        log_file = os.path.join(self.logs_dir, 'global_scheduler.log')
        logger.info(f"Starting Global Scheduler (log: {log_file})")
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen([
                    'dynamo', 'serve', 
                    'dynamo.global_scheduler.scheduler:GlobalScheduler',
                    '--service-name', 'GlobalScheduler'
                ], stdout=f, stderr=subprocess.STDOUT, env=os.environ)
                
            self.processes['global_scheduler'] = process
            time.sleep(8)  # Give it time to start
            
            if process.poll() is None:
                logger.info("  PASS: Global Scheduler is running")
                return True
            else:
                logger.error(f"  FAIL: Global Scheduler failed to start - check {log_file}")
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start Global Scheduler: {e} - check {log_file}")
            return False
    
    def _start_pools(self) -> bool:
        """Start the high and low SLO pools"""
        logger.info("Starting SLO pools...")
        
        # Start high SLO pool
        if not self._start_pool('high', 8000, '0'):
            return False
            
        # Start low SLO pool  
        if not self._start_pool('low', 8002, '1'):
            return False
            
        return True
    
    def _start_pool(self, slo_level: str, port: int, gpu: str) -> bool:
        """Start a specific SLO pool"""
        log_file = os.path.join(self.logs_dir, f'{slo_level}_slo_pool.log')
        
        try:
            config_file = os.path.join(self.config_dir, f'simple_{slo_level}_slo.yaml')
            if not os.path.exists(config_file):
                logger.error(f"FAIL: Config file not found: {config_file}")
                return False
            
            # Make config file path absolute to work from any directory
            config_file = os.path.abspath(config_file)
                
            # Get next available DYNAMO_PORT using counter
            dynamo_port = self._get_next_dynamo_port()
            
            logger.info(f"  Starting {slo_level.upper()} SLO pool on GPU {gpu}, HTTP port {port}, FastAPI port {dynamo_port} (log: {log_file})")
            
            # Set environment for this pool
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': gpu,
                'DYN_DISABLE_AUTO_GPU_ALLOCATION': '1',
                'DYNAMO_PORT': str(dynamo_port),  # Use dynamic port counter
                'GLOBAL_SCHEDULER_HOST': 'localhost',
                'GLOBAL_SCHEDULER_PORT': '3999',
                'POOL_HOST': 'localhost'
            })
            
            # Change to examples/llm directory
            llm_dir = os.path.join(self.project_root, 'examples', 'llm')
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen([
                    'dynamo', 'serve', 'graphs.agg_router:Frontend',
                    '-f', config_file,
                    f'--Frontend.port={port}',
                    f'--Frontend.ServiceArgs.dynamo.namespace={slo_level}_slo',
                    f'--Processor.ServiceArgs.dynamo.namespace={slo_level}_slo',
                    f'--Router.ServiceArgs.dynamo.namespace={slo_level}_slo',
                    f'--VllmWorker.ServiceArgs.dynamo.namespace={slo_level}_slo',
                    f'--Planner.ServiceArgs.dynamo.namespace={slo_level}_slo'
                ], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=llm_dir)
                
            self.processes[f'{slo_level}_pool'] = process
            
            # Wait for startup
            time.sleep(10)
            
            if process.poll() is None:
                logger.info(f"    PASS: {slo_level.upper()} SLO pool is running (PID: {process.pid})")
                return True
            else:
                logger.error(f"    FAIL: {slo_level.upper()} SLO pool failed to start - check {log_file}")
                # Show the last few lines of the log for quick debugging
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            logger.error(f"    Last few lines from {log_file}:")
                            for line in lines[-5:]:
                                logger.error(f"      {line.strip()}")
                except:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start {slo_level} SLO pool: {e} - check {log_file}")
            return False
    
    def stop_all(self):
        """Stop all running services"""
        if not self.processes:
            return
            
        logger.info("Stopping all services...")
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Still running
                    logger.info(f"  Stopping {name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"  Force killing {name}...")
                        process.kill()
                        process.wait()
                        
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("All services stopped")

class TestRunner:
    """Main test runner that orchestrates the complete test lifecycle"""
    
    def __init__(self, test_type: str = "simple", deployment: str = "local", 
                 config_dir: str = "configs", start_services: bool = True):
        self.test_type = test_type
        self.deployment = deployment
        self.start_services = start_services
        self.config = self._get_deployment_config()
        self.service_manager = ServiceManager(config_dir) if start_services else None
        
    def _get_deployment_config(self) -> Dict[str, Any]:
        """Get configuration based on deployment mode"""
        if self.deployment == "local":
            return {
                "global_scheduler_url": "http://localhost:3999",
                "high_slo_pool_url": "http://localhost:8000", 
                "low_slo_pool_url": "http://localhost:8002",
                "pool_host": "localhost"
            }
        elif self.deployment == "cluster":
            return {
                "global_scheduler_url": f"http://{os.getenv('GLOBAL_SCHEDULER_HOST', 'dynamo-global-scheduler')}:{os.getenv('GLOBAL_SCHEDULER_PORT', '3999')}",
                "high_slo_pool_url": f"http://{os.getenv('POOL_HOST', 'localhost')}:8000",
                "low_slo_pool_url": f"http://{os.getenv('POOL_HOST', 'localhost')}:8002",
                "pool_host": os.getenv('POOL_HOST', 'localhost')
            }
        else:  # custom
            return {
                "global_scheduler_url": os.getenv('GLOBAL_SCHEDULER_URL', 'http://localhost:3999'),
                "high_slo_pool_url": os.getenv('HIGH_SLO_POOL_URL', 'http://localhost:8000'),
                "low_slo_pool_url": os.getenv('LOW_SLO_POOL_URL', 'http://localhost:8002'),
                "pool_host": os.getenv('POOL_HOST', 'localhost')
            }
    
    async def run_complete_test(self) -> bool:
        """Run the complete test including service management"""
        logger.info("=" * 60)
        logger.info(f"STARTING GLOBAL SCHEDULER COMPLETE TEST")
        logger.info(f"Test Type: {self.test_type}")
        logger.info(f"Deployment: {self.deployment}")
        logger.info(f"Start Services: {self.start_services}")
        logger.info("=" * 60)
        
        try:
            # Start services if requested
            if self.start_services:
                if not self.service_manager.start_all_services():
                    logger.error("Failed to start services")
                    logger.error("Check the following log files for details:")
                    logger.error(f"  - {self.service_manager.logs_dir}/etcd.log")
                    logger.error(f"  - {self.service_manager.logs_dir}/nats.log")
                    logger.error(f"  - {self.service_manager.logs_dir}/global_scheduler.log")
                    logger.error(f"  - {self.service_manager.logs_dir}/high_slo_pool.log")
                    logger.error(f"  - {self.service_manager.logs_dir}/low_slo_pool.log")
                    return False
                
                # Wait for services to be fully ready
                logger.info("Waiting for services to initialize (60 seconds)...")
                time.sleep(60)
            
            # Run the actual tests
            if self.test_type == "simple":
                test_instance = SimpleSchedulerTest(self.config)
                result = await test_instance.run()
            else:
                logger.error(f"Unknown test type: {self.test_type}")
                result = False
                
            return result
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Always cleanup services if we started them
            if self.start_services and self.service_manager:
                self.service_manager.stop_all()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Global Scheduler Complete Test Runner")
    parser.add_argument("--test-type", choices=["simple"], default="simple",
                       help="Type of test to run (default: simple)")
    parser.add_argument("--deployment", choices=["local", "cluster", "custom"], default="local",
                       help="Deployment mode (default: local)")
    parser.add_argument("--config-dir", default="configs",
                       help="Directory containing configuration files (default: configs)")
    parser.add_argument("--no-start-services", action="store_true",
                       help="Skip starting services (assume they're already running)")
    
    args = parser.parse_args()
    
    # Setup signal handler for cleanup
    def signal_handler(signum, frame):
        logger.info("Interrupt received, shutting down...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the test runner
    runner = TestRunner(
        test_type=args.test_type, 
        deployment=args.deployment,
        config_dir=args.config_dir,
        start_services=not args.no_start_services
    )
    
    success = asyncio.run(runner.run_complete_test())
    
    if success:
        logger.info("ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
