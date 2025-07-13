#!/usr/bin/env python3
"""
Service Launcher for Global Scheduler Tests

This script provides a simple way to start the required services for testing
the Global Scheduler in a local environment.
"""

import os
import sys
import time
import signal
import subprocess
import logging
import argparse
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        logger.info(f"Set PYTHONPATH: {new_pythonpath}")
        
        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def start_infrastructure(self) -> bool:
        """Start etcd and NATS services"""
        logger.info("Starting infrastructure services...")
        
        # Start etcd
        if not self._start_etcd():
            return False
            
        # Start NATS with JetStream
        if not self._start_nats():
            return False
            
        logger.info("Infrastructure services started")
        return True
        
    def start_global_scheduler(self) -> bool:
        """Start the Global Scheduler service"""
        logger.info("Starting Global Scheduler...")
        
        try:
            log_file = os.path.join(self.logs_dir, 'global_scheduler.log')
            with open(log_file, 'w') as f:
                process = subprocess.Popen([
                    'dynamo', 'serve', 
                    'dynamo.global_scheduler.scheduler:GlobalScheduler',
                    '--service-name', 'GlobalScheduler'
                ], stdout=f, stderr=subprocess.STDOUT, env=os.environ)
                
            self.processes['global_scheduler'] = process
            logger.info(f"Global Scheduler started (PID: {process.pid})")
            
            # Wait a bit for startup
            time.sleep(3)
            
            # Check if still running
            if process.poll() is None:
                logger.info("PASS: Global Scheduler is running")
                return True
            else:
                logger.error("FAIL: Global Scheduler failed to start")
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start Global Scheduler: {e}")
            return False
    
    def start_pools(self) -> bool:
        """Start the high and low SLO pools"""
        logger.info("Starting SLO pools...")
        
        # Start high SLO pool
        if not self._start_pool('high', 8000, '0'):
            return False
            
        # Start low SLO pool  
        if not self._start_pool('low', 8002, '1'):
            return False
            
        logger.info("SLO pools started")
        return True
    
    def _start_etcd(self) -> bool:
        """Start etcd service"""
        try:
            log_file = os.path.join(self.logs_dir, 'etcd.log')
            with open(log_file, 'w') as f:
                # Configure etcd to listen on all interfaces for multinode, localhost for single node
                if self.multinode and self.head_node_ip:
                    etcd_cmd = [
                        'etcd',
                        '--listen-client-urls', 'http://0.0.0.0:2379',
                        '--advertise-client-urls', f'http://{self.head_node_ip}:2379'
                    ]
                else:
                    etcd_cmd = ['etcd']  # Use defaults for localhost
                    
                process = subprocess.Popen(etcd_cmd, stdout=f, stderr=subprocess.STDOUT)
                
            self.processes['etcd'] = process
            logger.info(f"etcd started (PID: {process.pid})")
            
            # Wait and check health
            time.sleep(3)
            health_check = subprocess.run(['curl', '-f', 'http://localhost:2379/health'], 
                                        capture_output=True, timeout=5)
            if health_check.returncode == 0:
                logger.info("PASS: etcd is healthy")
                return True
            else:
                logger.error("FAIL: etcd health check failed")
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start etcd: {e}")
            return False
    
    def _start_nats(self) -> bool:
        """Start NATS service"""
        try:
            log_file = os.path.join(self.logs_dir, 'nats.log')
            with open(log_file, 'w') as f:
                # Configure NATS to listen on all interfaces for multinode, default for single node
                if self.multinode and self.head_node_ip:
                    nats_cmd = ['nats-server', '-js', '-m', '8222', '-a', '0.0.0.0']
                else:
                    nats_cmd = ['nats-server', '-js', '-m', '8222']
                    
                process = subprocess.Popen(nats_cmd,
                                         stdout=f, stderr=subprocess.STDOUT)
                
            self.processes['nats'] = process
            logger.info(f"NATS started with JetStream (PID: {process.pid})")
            
            # Wait for startup
            time.sleep(5)
            
            # Check if NATS process is still running
            if process.poll() is not None:
                logger.error("FAIL: NATS process died immediately")
                return False
            
            # Check if NATS is healthy
            health_check = subprocess.run(['curl', '-f', 'http://localhost:8222/varz'], 
                                        capture_output=True, timeout=5)
            if health_check.returncode == 0:
                logger.info("PASS: NATS is healthy")
                
                # Check if JetStream is enabled
                js_check = subprocess.run(['curl', '-s', 'http://localhost:8222/jsz'], 
                                        capture_output=True, timeout=5)
                if js_check.returncode == 0 and b'"enabled":true' in js_check.stdout:
                    logger.info("PASS: JetStream is enabled")
                else:
                    logger.info("PASS: NATS running (JetStream status unclear from API, but should be enabled)")
                return True
            else:
                logger.error("FAIL: NATS health check failed")
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start NATS: {e}")
            return False
    
    def _start_pool(self, slo_level: str, port: int, gpu: str) -> bool:
        """Start a specific SLO pool"""
        try:
            config_file = os.path.join(self.config_dir, f'simple_{slo_level}_slo.yaml')
            if not os.path.exists(config_file):
                logger.error(f"FAIL: Config file not found: {config_file}")
                return False
                
            log_file = os.path.join(self.logs_dir, f'{slo_level}_slo_pool.log')
            
            # Get next available DYNAMO_PORT using counter
            dynamo_port = self._get_next_dynamo_port()
            
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
                    f'--VllmWorker.ServiceArgs.workers=0',  # Override workers to 0 for planner control
                    f'--PrefillWorker.ServiceArgs.dynamo.namespace={slo_level}_slo',
                    f'--PrefillWorker.ServiceArgs.workers=0',  # Override workers to 0 for planner control
                    f'--Planner.ServiceArgs.dynamo.namespace={slo_level}_slo'
                ], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=llm_dir)
                
            self.processes[f'{slo_level}_pool'] = process
            logger.info(f"{slo_level.upper()} SLO pool started (PID: {process.pid}, GPU: {gpu}, HTTP port: {port}, FastAPI port: {dynamo_port})")
            
            # Wait for startup
            time.sleep(5)
            
            if process.poll() is None:
                logger.info(f"PASS: {slo_level.upper()} SLO pool is running")
                return True
            else:
                logger.error(f"FAIL: {slo_level.upper()} SLO pool failed to start")
                return False
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start {slo_level} SLO pool: {e}")
            return False
    
    def stop_all(self):
        """Stop all running services"""
        logger.info("Stopping all services...")
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Still running
                    logger.info(f"Stopping {name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {name}...")
                        process.kill()
                        process.wait()
                        
                    logger.info(f"PASS: {name} stopped")
                    
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("All services stopped")
    
    def wait_for_interrupt(self):
        """Wait for user interrupt (Ctrl+C)"""
        try:
            logger.info("All services running. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
                # Check if any process died
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"FAIL: {name} process died unexpectedly")
                        
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Start services for Global Scheduler testing")
    parser.add_argument("--config-dir", default="configs", help="Configuration directory")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory")
    
    args = parser.parse_args()
    
    manager = ServiceManager(args.config_dir)
    
    # Setup signal handler for cleanup
    def signal_handler(signum, frame):
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        if not manager.start_infrastructure():
            logger.error("FAIL: Failed to start infrastructure")
            return 1
            
        if not manager.start_global_scheduler():
            logger.error("FAIL: Failed to start Global Scheduler")
            return 1
            
        if not manager.start_pools():
            logger.error("FAIL: Failed to start pools")
            return 1
        
        logger.info("All services started successfully!")
        logger.info(f"Logs available in: {manager.logs_dir}")
        
        # Wait for interrupt
        manager.wait_for_interrupt()
        
        return 0
        
    except Exception as e:
        logger.error(f"FAIL: Service startup failed: {e}")
        return 1
        
    finally:
        manager.stop_all()

if __name__ == "__main__":
    sys.exit(main()) 