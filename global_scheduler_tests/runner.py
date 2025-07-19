#!/usr/bin/env python3
"""
Global Scheduler Test Runner

This script provides a complete test framework for the Global Scheduler system with PD disaggregated architecture.
It starts all required services, runs tests, and cleans up automatically.

USAGE:
  Single-node execution:
    python runner.py --monitor
    python runner.py --test-type simple --monitor
    python runner.py --test-type scaling --monitor
    python runner.py --test-type streaming --monitor
  
  Multi-node execution:
    python runner.py --multinode --head-node-ip <IP> --head-node-role head_and_high_slo
python runner.py --multinode --head-node-ip <IP> --head-node-role high_slo --worker-only
python runner.py --multinode --head-node-ip <IP> --head-node-role low_slo --worker-only
  
  SLURM execution (recommended):
    sbatch --nodes=1 slurm_scripts/multinode_test_simple.sbatch  # Single-node
    sbatch --nodes=2 slurm_scripts/multinode_test_simple.sbatch  # Multi-node

NOTE: In multinode deployments, log files are named with hostname suffixes to avoid conflicts:
  - Single-node: high_slo_pool.log, low_slo_pool.log
  - Multi-node: high_slo_pool_<hostname>.log, low_slo_pool_<hostname>.log
"""

import argparse
import asyncio
import os
import sys
import logging
import time
import signal
import subprocess
from typing import Dict, Any, Optional, List

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib', 'bindings', 'python', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deploy', 'sdk', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components', 'global_scheduler', 'src'))

from tests.test_base import BaseGlobalSchedulerTest
from tests.test_simple import SimpleSchedulerTest
from tests.test_scaling import ScalingTest
from tests.test_streaming import StreamingSchedulerTest
from monitor import SystemMonitor

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
    """Manages the lifecycle of services needed for Global Scheduler testing with PD disaggregated architecture"""
    
    def __init__(self, config_dir: str = "configs", multinode: bool = False, 
                 head_node_ip: str = None, head_node_role: str = None, worker_only: bool = False,
                 high_slo_nodes: int = None, low_slo_nodes: int = None):
        self.config_dir = config_dir
        self.multinode = multinode
        self.head_node_ip = head_node_ip
        self.head_node_role = head_node_role
        self.worker_only = worker_only
        self.processes: Dict[str, subprocess.Popen] = {}
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Port counter for dynamic port allocation (multi-node safe)
        self.next_dynamo_port = 4000
        
        # Handle node count configuration with environment variable fallback
        self.high_slo_nodes = high_slo_nodes or int(os.getenv('HIGH_SLO_NODE_COUNT', '2'))
        self.low_slo_nodes = low_slo_nodes or int(os.getenv('LOW_SLO_NODE_COUNT', '1'))
        
        # GPU allocation for PD disaggregated architecture
        # Single-node: High SLO (0-3), Low SLO (4-7) 
        # Multinode with dynamic node counts: High SLO (high_slo_nodes x 8 GPUs), Low SLO (low_slo_nodes x 8 GPUs)
        
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
        
        # Set ETCD_ENDPOINTS based on deployment mode
        if self.multinode and self.head_node_ip:
            os.environ['ETCD_ENDPOINTS'] = f"http://{self.head_node_ip}:2379"
        else:
            os.environ['ETCD_ENDPOINTS'] = "http://localhost:2379"
        
        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def start_all_services(self) -> bool:
        """Start all required services for testing"""
        if self.multinode:
            logger.info(f"Starting multinode services for Global Scheduler testing with PD disaggregated architecture...")
            logger.info(f"Node role: {self.head_node_role}")
            logger.info(f"Head node IP: {self.head_node_ip}")
            logger.info(f"Worker only: {self.worker_only}")
        else:
            logger.info("Starting all services for Global Scheduler testing with PD disaggregated architecture...")
        if self.multinode:
            logger.info(f"GPU Allocation: Both pools have access to all GPUs (0-7) on their respective nodes")
            logger.info(f"Each pool starts with 2 GPUs (1P+1D), can scale to 8 GPUs")
        else:
            logger.info(f"GPU Allocation: High SLO (0-3), Low SLO (4-7)")
            logger.info(f"Each pool starts with 2 GPUs (1P+1D), can scale to 4 GPUs")
        logger.info(f"Logs will be written to: {self.logs_dir}/")
        
        if self.multinode and self.worker_only:
            # Worker node - only start pools, no infrastructure or global scheduler
            logger.info("Worker node - starting only pools...")
            if not self._start_pools():
                logger.error("Failed to start SLO pools")
                return False
        else:
            # Head node - start infrastructure, global scheduler, and pools
            if not self._start_infrastructure():
                logger.error("Failed to start infrastructure services")
                return False
            if not self._start_global_scheduler():
                logger.error("Failed to start Global Scheduler")
                return False
            if not self._start_pools():
                logger.error("Failed to start SLO pools")
                return False
            
        # Additional stabilization time for multinode setups
        if self.multinode:
            logger.info("Waiting 20 seconds for all services to stabilize and NIXL connections to establish...")
            time.sleep(20)
        
        logger.info("PASS: All services started successfully")
        return True
    
    def _start_infrastructure(self) -> bool:
        """Start etcd and NATS services"""
        logger.info("Starting infrastructure services...")
        
        # Start etcd
        try:
            log_file = os.path.join(self.logs_dir, 'etcd.log')
            logger.info(f"  Starting etcd (log: {log_file})")
            with open(log_file, 'w') as f:
                # Configure etcd to listen on all interfaces for multinode, localhost for single node
                if self.multinode:
                    etcd_cmd = [
                        'etcd',
                        '--listen-client-urls', 'http://0.0.0.0:2379',
                        '--advertise-client-urls', f'http://{self.head_node_ip}:2379'
                    ]
                else:
                    etcd_cmd = ['etcd']  # Use defaults for localhost
                    
                process = subprocess.Popen(etcd_cmd, stdout=f, stderr=subprocess.STDOUT)
            self.processes['etcd'] = process
            time.sleep(3)
            
            # Check health
            health_check = subprocess.run(['curl', '-f', 'http://localhost:2379/health'], 
                                        capture_output=True, timeout=5)
            if health_check.returncode != 0:
                logger.error(f"FAIL: etcd health check failed - check {log_file}")
                return False
            logger.info("  PASS: etcd is healthy")
            
            # Clear stale model registrations from previous test runs
            logger.info("  Cleaning up stale model registrations from etcd...")
            cleanup_result = subprocess.run(
                ['etcdctl', 'del', '--prefix', 'models/'], 
                capture_output=True, timeout=10
            )
            if cleanup_result.returncode == 0:
                logger.info("  PASS: Cleared stale model registrations")
            else:
                logger.warning(f"  WARNING: Could not clear model registrations: {cleanup_result.stderr.decode()}")
                
            # Also clear model deployment cards
            mdc_cleanup_result = subprocess.run(
                ['etcdctl', 'del', '--prefix', 'mdc/'], 
                capture_output=True, timeout=10
            )
            if mdc_cleanup_result.returncode == 0:
                logger.info("  PASS: Cleared stale model deployment cards")
            else:
                logger.warning(f"  WARNING: Could not clear model deployment cards: {mdc_cleanup_result.stderr.decode()}")
                
            # Clear disaggregation router configurations
            disagg_cleanup_result = subprocess.run(
                ['etcdctl', 'del', '--prefix', 'public/components/disagg_router/'], 
                capture_output=True, timeout=10
            )
            if disagg_cleanup_result.returncode == 0:
                logger.info("  PASS: Cleared stale disaggregation router configurations")
            else:
                logger.warning(f"  WARNING: Could not clear disaggregation router configurations: {disagg_cleanup_result.stderr.decode()}")
                
        except Exception as e:
            logger.error(f"FAIL: Failed to start etcd: {e} - check {log_file}")
            return False
        
        # Start NATS
        try:
            log_file = os.path.join(self.logs_dir, 'nats.log')
            logger.info(f"  Starting NATS (log: {log_file})")
            with open(log_file, 'w') as f:
                # Configure NATS to listen on all interfaces for multinode, default for single node
                if self.multinode:
                    nats_cmd = ['nats-server', '-js', '-m', '8222', '-a', '0.0.0.0']
                else:
                    nats_cmd = ['nats-server', '-js', '-m', '8222']
                    
                process = subprocess.Popen(nats_cmd, stdout=f, stderr=subprocess.STDOUT)
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
        """Start SLO-based pools with PD disaggregated architecture"""
        if self.multinode:
            return self._start_multinode_pools()
        else:
            return self._start_local_pools()
    
    def _start_local_pools(self) -> bool:
        """Start the high and low SLO pools for single node deployment with PD disaggregated architecture"""
        logger.info("Starting SLO pools with PD disaggregated architecture...")
        
        # Start high SLO pool with PD disaggregated architecture
        if not self._start_pool('high', 8000):
            return False
            
        # Start low SLO pool with PD disaggregated architecture  
        if not self._start_pool('low', 8002):
            return False
            
        return True
    
    def _start_multinode_pools(self) -> bool:
        """Start the high and low SLO pools for multinode deployment with PD disaggregated architecture"""
        logger.info("Starting SLO pools with PD disaggregated architecture...")
        
        # Determine which pools to start based on node role
        if os.getenv('DYNAMO_CONTAINER_MODE', 'false').lower() == 'true' or self.head_node_role == "all":
            logger.info("Container/testing mode detected - starting both SLO pools")
            # Start both pools
            if not self._start_pool('high', 8000, multinode=True):
                return False
            if not self._start_pool('low', 8002, multinode=True):
                return False
        elif self.head_node_role == "head_and_high_slo":
            # Head node runs high SLO pool only
            if not self._start_pool('high', 8000, multinode=True):
                return False
        elif self.head_node_role == "high_slo":
            # Worker node runs high SLO pool only
            # Add extra delay in multinode to ensure other node's services are ready
            logger.info("Waiting 10 seconds for other node's services to initialize...")
            time.sleep(10)
            
            if not self._start_pool('high', 8000, multinode=True):
                return False
        elif self.head_node_role == "low_slo":
            # Worker node runs low SLO pool only
            # Add extra delay in multinode to ensure other node's services are ready
            logger.info("Waiting 10 seconds for other node's services to initialize...")
            time.sleep(10)
            
            if not self._start_pool('low', 8002, multinode=True):
                return False
        else:
            logger.error(f"Unknown node role: {self.head_node_role}")
            return False
        
        return True
    
    def _get_pool_log_files(self) -> List[str]:
        """Get list of all pool log files for debugging"""
        import glob
        log_pattern = os.path.join(self.logs_dir, '*_slo_pool*.log')
        return glob.glob(log_pattern)
    
    def _start_pool(self, slo_level: str, port: int, multinode: bool = False) -> bool:
        """Start a specific SLO pool with PD disaggregated architecture"""
        if multinode:
            # Include hostname in log file name to avoid conflicts between nodes
            import socket
            hostname = socket.gethostname()
            log_file = os.path.join(self.logs_dir, f'{slo_level}_slo_pool_{hostname}.log')
        else:
            log_file = os.path.join(self.logs_dir, f'{slo_level}_slo_pool.log')
        
        try:
            # Choose configuration file based on deployment mode
            if multinode:
                # Use new tensor parallelism configs for multinode deployment
                config_file = os.path.join(self.config_dir, f'{slo_level}_slo.yaml')
            else:
                # Use original single-node config
                config_file = os.path.join(self.config_dir, f'pd_disagg_{slo_level}_slo.yaml')
                
            if not os.path.exists(config_file):
                logger.error(f"FAIL: Config file not found: {config_file}")
                return False
            
            # Make config file path absolute to work from any directory
            config_file = os.path.abspath(config_file)
            
            # Get next available DYNAMO_PORT using counter
            dynamo_port = self._get_next_dynamo_port()
            
            # Calculate GPU allocation info for this pool
            if multinode:
                if slo_level == 'high':
                    node_count = self.high_slo_nodes
                    total_gpus = node_count * 8
                    tp_size = 2
                else:
                    node_count = self.low_slo_nodes
                    total_gpus = node_count * 8
                    tp_size = 1
            else:
                node_count = 1
                total_gpus = 4  # Single node uses 4 GPUs per pool
                tp_size = 1
            
            logger.info(f"  Starting {slo_level.upper()} SLO pool:")
            logger.info(f"    - Config file: {os.path.basename(config_file)}")
            logger.info(f"    - Node count: {node_count}, Total GPUs: {total_gpus}")
            logger.info(f"    - Tensor parallel size: {tp_size}")
            logger.info(f"    - HTTP port {port}, FastAPI port {dynamo_port}")
            logger.info(f"    - Starts with 1P+1D workers, can scale via planner")
            logger.info(f"    - Log: {log_file}")
            
            # Set environment for this pool
            env = os.environ.copy()
            
            # For multinode, use actual hostname so global scheduler can reach the pool
            import socket
            pool_host = socket.gethostname() if multinode else 'localhost'
            
            # Ensure ETCD_ENDPOINTS is preserved from parent environment
            etcd_endpoints = os.environ.get('ETCD_ENDPOINTS')
            if etcd_endpoints:
                env['ETCD_ENDPOINTS'] = etcd_endpoints
            
            env.update({
                'DYN_DISABLE_AUTO_GPU_ALLOCATION': '0',  # Enable auto allocation within scope
                'DYNAMO_PORT': str(dynamo_port),
                'GLOBAL_SCHEDULER_HOST': self.head_node_ip if multinode else 'localhost',
                'GLOBAL_SCHEDULER_PORT': '3999',
                'POOL_HOST': pool_host
            })
            
            # Log ETCD_ENDPOINTS for debugging
            if multinode:
                logger.info(f"    ETCD_ENDPOINTS from environment: {env.get('ETCD_ENDPOINTS', 'NOT SET')}")
                
                # Add UCX/NIXL optimization for multinode
                env.update({
                    'UCX_NET_DEVICES': 'all',  # Use all available network devices
                    'UCX_TLS': 'tcp,cuda_copy,cuda_ipc',  # Transport layers
                    'UCX_LOG_LEVEL': 'info',  # Enable UCX logging for debugging
                    'UCX_SOCKADDR_TLS_PRIORITY': 'tcp',  # Ensure TCP is prioritized for cross-node
                    'UCX_TCP_CM_REUSEADDR': 'y',  # Allow socket reuse for multi-node
                    'UCX_TCP_NODELAY': 'y',  # Disable Nagle's algorithm for low latency
                    'UCX_RC_TIMEOUT': '10s',  # Increase RC timeout
                    'UCX_UD_TIMEOUT': '10s',  # Increase UD timeout
                    'UCX_MAX_RNDV_RAILS': '1',  # Limit rendezvous rails for stability
                    'UCX_IB_PKEY': 'default',  # Use default partition key for InfiniBand
                    'UCX_MM_FIFO_SIZE': '1024',  # Increase FIFO size for better throughput
                })
            
            logger.info(f"    Pool will register with base URL: http://{pool_host}:{port}")
            
            # Change to examples/llm directory
            llm_dir = os.path.join(self.project_root, 'examples', 'llm')
            
            # Build command arguments
            cmd_args = [
                'dynamo', 'serve', 'graphs.disagg_router:Frontend',
                '-f', config_file,
                f'--Frontend.port={port}'
            ]
            
            # Override namespace for multinode deployment
            if multinode:
                namespace_suffix = "_multinode"
                namespace = f"{slo_level}_slo_pd{namespace_suffix}"
                cmd_args.extend([
                    f'--Frontend.ServiceArgs.dynamo.namespace={namespace}',
                    f'--Processor.ServiceArgs.dynamo.namespace={namespace}',
                    f'--Router.ServiceArgs.dynamo.namespace={namespace}',
                    f'--VllmWorker.ServiceArgs.dynamo.namespace={namespace}',
                    f'--PrefillWorker.ServiceArgs.dynamo.namespace={namespace}',
                    f'--Planner.ServiceArgs.dynamo.namespace={namespace}'
                ])
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(cmd_args, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=llm_dir)
                
            self.processes[f'{slo_level}_pool'] = process
            
            # Wait for startup (longer for disaggregated setup)
            startup_wait = 15
            if multinode:
                # Extra time for NIXL agent discovery in multinode setups
                startup_wait = 25
                logger.info(f"    Waiting {startup_wait}s for NIXL and services to initialize in multinode mode...")
            else:
                logger.info(f"    Waiting {startup_wait}s for services to initialize...")
            
            time.sleep(startup_wait)
            
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
                 config_dir: str = "configs", start_services: bool = True, monitor: bool = False,
                 multinode: bool = False, head_node_ip: str = None, head_node_role: str = None, worker_only: bool = False,
                 high_slo_nodes: int = None, low_slo_nodes: int = None):
        self.test_type = test_type
        self.deployment = deployment
        self.start_services = start_services
        self.monitor = monitor
        self.multinode = multinode
        self.head_node_ip = head_node_ip
        self.head_node_role = head_node_role
        self.worker_only = worker_only
        self.high_slo_nodes = high_slo_nodes
        self.low_slo_nodes = low_slo_nodes
        self.config = self._get_deployment_config()
        self.service_manager = ServiceManager(config_dir, multinode, head_node_ip, head_node_role, worker_only, high_slo_nodes, low_slo_nodes) if start_services else None
        self.system_monitor = None
        
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
        logger.info(f"Monitor: {self.monitor}")
        logger.info(f"Architecture: PD Disaggregated (always enabled)")
        if self.multinode:
            logger.info(f"GPU Allocation: Both pools have access to all GPUs (0-7) on their respective nodes")
            logger.info(f"Each pool starts with 2 GPUs (1P+1D), can scale to 8 GPUs")
        else:
            logger.info(f"GPU Allocation: High SLO (0-3), Low SLO (4-7)")
            logger.info(f"Each pool starts with 2 GPUs (1P+1D), can scale to 4 GPUs")
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
                    logger.error(f"  - {self.service_manager.logs_dir}/*_slo_pool*.log (pool logs)")
                    return False
                
                # Wait for services to be fully ready
                logger.info("Waiting for services to initialize (240 seconds)...")
                time.sleep(240)
            
            # Start monitoring if requested
            if self.monitor:
                pool_urls = {
                    'high_slo': self.config['high_slo_pool_url'],
                    'low_slo': self.config['low_slo_pool_url']
                }
                logs_dir = self.service_manager.logs_dir if self.service_manager else os.path.join(os.path.dirname(__file__), 'logs')
                
                # Define GPU mapping for pool-specific monitoring
                if self.multinode:
                    # In multinode mode, each pool uses all GPUs (0-7) on its dedicated nodes
                    pool_gpu_mapping = {
                        'high_slo': list(range(8)),  # GPUs 0-7 on high SLO nodes
                        'low_slo': list(range(8))    # GPUs 0-7 on low SLO nodes
                    }
                else:
                    # In single-node mode, split GPUs between pools
                    pool_gpu_mapping = {
                        'high_slo': list(range(4)),  # GPUs 0-3
                        'low_slo': list(range(4, 8))  # GPUs 4-7
                    }
                
                self.system_monitor = SystemMonitor(pool_urls, logs_dir, pool_gpu_mapping=pool_gpu_mapping)
                await self.system_monitor.start()
            
            # Only run tests on the head node (not on worker-only nodes)
            if self.worker_only:
                logger.info("Worker node - skipping test execution, keeping services running")
                # Keep the worker node running to serve requests until killed by head node
                try:
                    while True:
                        await asyncio.sleep(60)
                        # Could add health checks here if needed
                except KeyboardInterrupt:
                    logger.info("Worker node interrupted, shutting down...")
                
                logger.info("Worker node shutting down...")
                return True  # Successful shutdown
            else:
                # Run the actual tests only on the head node
                logger.info("Head node - executing test logic")
                
                # In multinode mode, check network connectivity to worker nodes
                #if self.multinode:
                #    await self._check_multinode_connectivity()
                
                if self.test_type == "simple":
                    test_instance = SimpleSchedulerTest(self.config)
                    result = await test_instance.run()
                elif self.test_type == "scaling":
                    test_instance = ScalingTest(self.config)
                    result = await test_instance.run()
                elif self.test_type == "streaming":
                    test_instance = StreamingSchedulerTest(self.config)
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
            # Stop monitoring if it was started
            if self.system_monitor:
                await self.system_monitor.stop()
            
            # Always cleanup services if we started them
            if self.start_services and self.service_manager:
                self.service_manager.stop_all()

    async def _check_multinode_connectivity(self):
        """Check network connectivity to all registered pools"""
        logger.info("Checking multinode network connectivity...")
        
        # Import runtime
        from dynamo.runtime import DistributedRuntime
        
        # Initialize runtime
        loop = asyncio.get_running_loop()
        runtime = DistributedRuntime(loop, False)
        
        # Get pool status from Global Scheduler
        scheduler_component = runtime.namespace("dynamo").component("GlobalScheduler")
        status_endpoint = scheduler_component.endpoint("get_pool_status")
        status_client = await status_endpoint.client()
        
        pool_status_response = await status_client.get_pool_status()
        
        # Extract pool status
        pool_status = None
        async for status_item in pool_status_response:
            pool_status = status_item
            break
        
        if pool_status and pool_status.get('pools'):
            logger.info(f"Found {len(pool_status['pools'])} registered pools")
            
            # Check connectivity to each pool
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for pool_id, pool_info in pool_status['pools'].items():
                    base_url = pool_info['pool_config']['base_url']
                    logger.info(f"Checking connectivity to {pool_id} at {base_url}...")
                    
                    # Try a simple health check
                    response = await session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5))
                    if response.status == 200:
                        logger.info(f"  ✓ Successfully connected to {pool_id}")
                    else:
                        logger.warning(f"  ✗ {pool_id} returned status {response.status}")
                        
        else:
            logger.warning("No pools registered yet")
        
        logger.info("Network connectivity check complete")

        
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Global Scheduler Test Runner with PD Disaggregated Architecture")
    parser.add_argument("--test-type", choices=["simple", "scaling", "streaming"], default="simple",
                       help="Type of test to run (default: simple)")
    parser.add_argument("--deployment", choices=["local", "cluster", "custom"], default="local",
                       help="Deployment mode (default: local)")
    parser.add_argument("--config-dir", default="configs",
                       help="Directory containing configuration files (default: configs)")
    parser.add_argument("--no-start-services", action="store_true",
                       help="Skip starting services (assume they're already running)")
    parser.add_argument("--monitor", action="store_true", default=False,
                       help="Enable lightweight online monitoring of GPU utilization and pool load (default: false)")
    
    # Multinode arguments
    parser.add_argument("--multinode", action="store_true", default=False,
                       help="Enable multinode deployment mode")
    parser.add_argument("--head-node-ip", type=str, default=None,
                       help="IP address of the head node (required for multinode)")
    parser.add_argument("--head-node-role", type=str, default=None,
                       choices=["head_and_high_slo", "high_slo", "low_slo", "all"],
                       help="Role of the current node: head_and_high_slo (head node with high SLO pool), high_slo (worker node with high SLO pool), low_slo (worker node with low SLO pool), all (both pools for testing)")
    parser.add_argument("--worker-only", action="store_true", default=False,
                       help="Only start worker pools, not infrastructure services")
    
    # Pool node count arguments
    parser.add_argument("--high-slo-nodes", type=int, default=None,
                       help="Number of nodes for high SLO pool (env: HIGH_SLO_NODE_COUNT)")
    parser.add_argument("--low-slo-nodes", type=int, default=None,
                       help="Number of nodes for low SLO pool (env: LOW_SLO_NODE_COUNT)")
    
    args = parser.parse_args()
    
    # Handle environment variable fallback for node counts
    high_slo_nodes = args.high_slo_nodes or (int(os.getenv('HIGH_SLO_NODE_COUNT')) if os.getenv('HIGH_SLO_NODE_COUNT') else None)
    low_slo_nodes = args.low_slo_nodes or (int(os.getenv('LOW_SLO_NODE_COUNT')) if os.getenv('LOW_SLO_NODE_COUNT') else None)
    
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
        start_services=not args.no_start_services,
        monitor=args.monitor,
        multinode=args.multinode,
        head_node_ip=args.head_node_ip,
        head_node_role=args.head_node_role,
        worker_only=args.worker_only,
        high_slo_nodes=high_slo_nodes,
        low_slo_nodes=low_slo_nodes
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
