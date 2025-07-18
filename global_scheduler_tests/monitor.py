#!/usr/bin/env python3
"""
Lightweight Online Monitor for Global Scheduler Tests

This module provides real-time monitoring of:
- GPU utilization across different devices
- Pool load conditions and performance metrics
- System resource usage during test execution
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import threading

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU utilization using nvidia-smi"""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.gpu_data: List[Dict[str, Any]] = []
        self.running = False
        
    def get_gpu_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Get current GPU statistics"""
        try:
            # Query nvidia-smi for JSON output
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
                
            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': float(parts[2]) if parts[2] != '[Not Supported]' else 0.0,
                        'memory_util': float(parts[3]) if parts[3] != '[Not Supported]' else 0.0,
                        'memory_used_mb': float(parts[4]) if parts[4] != '[Not Supported]' else 0.0,
                        'memory_total_mb': float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                        'temperature': float(parts[6]) if parts[6] != '[Not Supported]' else 0.0,
                        'power_draw': float(parts[7]) if parts[7] != '[Not Supported]' else 0.0,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return gpu_stats
            
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return None
    
    async def monitor_loop(self):
        """Main monitoring loop for GPU stats"""
        while self.running:
            stats = self.get_gpu_stats()
            if stats:
                self.gpu_data.extend(stats)
                # Keep only last 100 readings per GPU to prevent memory growth
                if len(self.gpu_data) > 1000:
                    self.gpu_data = self.gpu_data[-1000:]
            
            await asyncio.sleep(self.update_interval)

class PoolMonitor:
    """Monitor pool load conditions and performance"""
    
    def __init__(self, pool_urls: Dict[str, str], update_interval: float = 5.0):
        self.pool_urls = pool_urls
        self.update_interval = update_interval
        self.pool_data: List[Dict[str, Any]] = []
        self.running = False
        
    async def get_pool_stats(self, session: aiohttp.ClientSession, pool_name: str, url: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific pool"""
        try:
            # Try to get basic health/stats from the pool
            async with session.get(f"{url}/health", timeout=3) as response:
                if response.status == 200:
                    health_data = await response.text()
                    return {
                        'pool_name': pool_name,
                        'url': url,
                        'status': 'healthy',
                        'response_time_ms': 0,  # Could measure this
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.debug(f"Pool {pool_name} health check failed: {e}")
            
        # If health check fails, try a simple connection test
        try:
            start_time = time.time()
            async with session.get(url, timeout=3) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    'pool_name': pool_name,
                    'url': url,
                    'status': 'reachable' if response.status < 500 else 'degraded',
                    'status_code': response.status,
                    'response_time_ms': response_time,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'pool_name': pool_name,
                'url': url,
                'status': 'unreachable',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def monitor_loop(self):
        """Main monitoring loop for pool stats"""
        async with aiohttp.ClientSession() as session:
            while self.running:
                timestamp = datetime.now()
                
                # Collect stats from all pools
                tasks = []
                for pool_name, url in self.pool_urls.items():
                    tasks.append(self.get_pool_stats(session, pool_name, url))
                
                pool_stats = await asyncio.gather(*tasks, return_exceptions=True)
                
                for stat in pool_stats:
                    if isinstance(stat, dict):
                        self.pool_data.append(stat)
                
                # Keep only last 200 readings to prevent memory growth
                if len(self.pool_data) > 200:
                    self.pool_data = self.pool_data[-200:]
                
                await asyncio.sleep(self.update_interval)

class SystemMonitor:
    """Combined system monitor that orchestrates GPU and Pool monitoring"""
    
    def __init__(self, pool_urls: Dict[str, str], logs_dir: str, 
                 gpu_interval: float = 2.0, pool_interval: float = 5.0,
                 pool_gpu_mapping: Dict[str, List[int]] = None):
        self.gpu_monitor = GPUMonitor(gpu_interval)
        self.pool_monitor = PoolMonitor(pool_urls, pool_interval)
        self.logs_dir = logs_dir
        self.running = False
        self.monitor_tasks: List[asyncio.Task] = []
        
        # Pool-specific GPU mapping for enhanced monitoring
        self.pool_gpu_mapping = pool_gpu_mapping or {}
        
        # Create monitoring log file
        self.monitor_log_file = os.path.join(logs_dir, 'monitor.log')
        
    async def start(self):
        """Start all monitoring tasks"""
        if self.running:
            return
            
        logger.info("Starting system monitoring...")
        logger.info(f"Monitor logs will be written to: {self.monitor_log_file}")
        
        self.running = True
        self.gpu_monitor.running = True
        self.pool_monitor.running = True
        
        # Start monitoring tasks
        self.monitor_tasks = [
            asyncio.create_task(self.gpu_monitor.monitor_loop()),
            asyncio.create_task(self.pool_monitor.monitor_loop()),
            asyncio.create_task(self.periodic_report())
        ]
        
        logger.info("System monitoring started")
    
    async def stop(self):
        """Stop all monitoring tasks"""
        if not self.running:
            return
            
        logger.info("Stopping system monitoring...")
        
        self.running = False
        self.gpu_monitor.running = False
        self.pool_monitor.running = False
        
        # Cancel all tasks
        for task in self.monitor_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitor_tasks, return_exceptions=True)
        self.monitor_tasks.clear()
        
        # Generate final report
        await self.generate_final_report()
        
        logger.info("System monitoring stopped")
    
    async def periodic_report(self):
        """Generate periodic monitoring reports"""
        report_interval = 30.0  # Report every 30 seconds
        
        while self.running:
            await asyncio.sleep(report_interval)
            
            # Generate current status report
            await self.generate_status_report()
    
    async def generate_status_report(self):
        """Generate a current status report"""
        try:
            # Get latest GPU data
            gpu_summary = self.get_gpu_summary()
            pool_summary = self.get_pool_summary()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'gpu_summary': gpu_summary,
                'pool_summary': pool_summary
            }
            
            # Log the report
            with open(self.monitor_log_file, 'a') as f:
                f.write(f"{json.dumps(report)}\n")
            
            # Also log key metrics to console
            if gpu_summary:
                if self.pool_gpu_mapping:
                    # Show GPU utilization per pool
                    for pool_name, gpu_indices in self.pool_gpu_mapping.items():
                        pool_gpu_utils = []
                        for gpu_data in gpu_summary:
                            if gpu_data['index'] in gpu_indices:
                                pool_gpu_utils.append(gpu_data['gpu_util'])
                        if pool_gpu_utils:
                            avg_util = sum(pool_gpu_utils) / len(pool_gpu_utils)
                            logger.info(f"{pool_name.upper()} SLO Pool GPU Utilization (GPUs {gpu_indices}): {pool_gpu_utils} % (avg: {avg_util:.1f}%)")
                else:
                    # Fallback to overall GPU utilization
                    gpu_utils = [g['gpu_util'] for g in gpu_summary if 'gpu_util' in g]
                    if gpu_utils:
                        logger.info(f"GPU Utilization: {gpu_utils} %")
            
            if pool_summary:
                pool_statuses = [(p['pool_name'], p['status']) for p in pool_summary]
                logger.info(f"Pool Status: {dict(pool_statuses)}")
                
        except Exception as e:
            logger.warning(f"Failed to generate status report: {e}")
    
    def get_gpu_summary(self) -> List[Dict[str, Any]]:
        """Get summary of latest GPU statistics"""
        if not self.gpu_monitor.gpu_data:
            return []
        
        # Get the latest reading for each GPU
        latest_by_gpu = {}
        for data in reversed(self.gpu_monitor.gpu_data):
            gpu_idx = data['index']
            if gpu_idx not in latest_by_gpu:
                latest_by_gpu[gpu_idx] = data
        
        return list(latest_by_gpu.values())
    
    def get_pool_summary(self) -> List[Dict[str, Any]]:
        """Get summary of latest pool statistics"""
        if not self.pool_monitor.pool_data:
            return []
        
        # Get the latest reading for each pool
        latest_by_pool = {}
        for data in reversed(self.pool_monitor.pool_data):
            pool_name = data['pool_name']
            if pool_name not in latest_by_pool:
                latest_by_pool[pool_name] = data
        
        return list(latest_by_pool.values())
    
    async def generate_final_report(self):
        """Generate a comprehensive final report"""
        try:
            final_report = {
                'test_completion_time': datetime.now().isoformat(),
                'gpu_data_points': len(self.gpu_monitor.gpu_data),
                'pool_data_points': len(self.pool_monitor.pool_data),
                'gpu_summary': self.get_gpu_summary(),
                'pool_summary': self.get_pool_summary()
            }
            
            # Calculate GPU utilization statistics
            if self.gpu_monitor.gpu_data:
                gpu_stats = {}
                for data in self.gpu_monitor.gpu_data:
                    gpu_idx = data['index']
                    if gpu_idx not in gpu_stats:
                        gpu_stats[gpu_idx] = []
                    gpu_stats[gpu_idx].append(data['gpu_util'])
                
                # Add pool-specific GPU statistics if mapping is available
                if self.pool_gpu_mapping:
                    pool_gpu_stats = {}
                    for pool_name, gpu_indices in self.pool_gpu_mapping.items():
                        pool_utils = []
                        for gpu_idx in gpu_indices:
                            if gpu_idx in gpu_stats:
                                pool_utils.extend(gpu_stats[gpu_idx])
                        if pool_utils:
                            pool_gpu_stats[pool_name] = {
                                'avg_utilization': sum(pool_utils) / len(pool_utils),
                                'max_utilization': max(pool_utils),
                                'min_utilization': min(pool_utils),
                                'gpu_count': len(gpu_indices),
                                'data_points': len(pool_utils)
                            }
                    final_report['pool_gpu_stats'] = pool_gpu_stats
                
                for gpu_idx, utils in gpu_stats.items():
                    final_report[f'gpu_{gpu_idx}_avg_util'] = sum(utils) / len(utils)
                    final_report[f'gpu_{gpu_idx}_max_util'] = max(utils)
                    final_report[f'gpu_{gpu_idx}_min_util'] = min(utils)
            
            # Write final report
            final_report_file = os.path.join(self.logs_dir, 'monitor_final_report.json')
            with open(final_report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            logger.info(f"Final monitoring report written to: {final_report_file}")
            
            # Print summary to console
            if self.gpu_monitor.gpu_data:
                logger.info("=== GPU UTILIZATION SUMMARY ===")
                for gpu_idx in sorted(gpu_stats.keys()):
                    avg_util = final_report[f'gpu_{gpu_idx}_avg_util']
                    max_util = final_report[f'gpu_{gpu_idx}_max_util']
                    logger.info(f"GPU {gpu_idx}: Avg={avg_util:.1f}%, Max={max_util:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}") 