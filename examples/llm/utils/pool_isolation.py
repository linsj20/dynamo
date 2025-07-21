# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pool Isolation Utilities

Provides unique identifiers to ensure complete separation between pools
running on different nodes, specifically for prefill queues and NIXL metadata stores.
ETCD and NATS remain shared to maintain the distributed environment.
"""

import os
import socket
import logging

logger = logging.getLogger(__name__)


def get_pool_unique_id() -> str:
    """Get a unique identifier for this pool/node instance.
    
    Returns:
        A unique string that identifies this specific pool instance
    """
    # Try environment variables first (allows explicit override)
    pool_id = os.environ.get("POOL_UNIQUE_ID")
    if pool_id:
        pool_id_clean = pool_id.replace(".", "_").replace("-", "_")
        logger.info(f"POOL_ISOLATION: Using POOL_UNIQUE_ID env var: {pool_id} -> {pool_id_clean}")
        return pool_id_clean
    
    # Try pool host from environment
    pool_host = os.environ.get("POOL_HOST")
    if pool_host:
        pool_host_clean = pool_host.replace(".", "_").replace("-", "_")
        logger.info(f"POOL_ISOLATION: Using POOL_HOST env var: {pool_host} -> {pool_host_clean}")
        return pool_host_clean
    
    # Try NODE_IP from environment (often set in k8s/slurm environments)
    node_ip = os.environ.get("NODE_IP")
    if node_ip:
        node_ip_clean = node_ip.replace(".", "_").replace("-", "_")
        logger.info(f"POOL_ISOLATION: Using NODE_IP env var: {node_ip} -> {node_ip_clean}")
        return node_ip_clean
    
    # Try HOSTNAME from environment
    hostname_env = os.environ.get("HOSTNAME")
    if hostname_env:
        hostname_clean = hostname_env.replace(".", "_").replace("-", "_")
        logger.info(f"POOL_ISOLATION: Using HOSTNAME env var: {hostname_env} -> {hostname_clean}")
        return hostname_clean
    
    # Fall back to system hostname
    hostname = socket.gethostname()
    hostname_clean = hostname.replace(".", "_").replace("-", "_")
    logger.info(f"POOL_ISOLATION: Using socket.gethostname(): {hostname} -> {hostname_clean}")
    return hostname_clean


def get_unique_prefill_queue_name(namespace: str) -> str:
    """Get a unique prefill queue name for this pool.
    
    Args:
        namespace: The base namespace
        
    Returns:
        A unique prefill queue name that isolates this pool from others
    """
    pool_id = get_pool_unique_id()
    unique_name = f"{namespace}_{pool_id}_prefill_queue"
    logger.info(f"Pool prefill queue: {unique_name}")
    return unique_name


def get_unique_nixl_namespace(namespace: str) -> str:
    """Get a unique namespace for NIXL metadata store for this pool.
    
    Args:
        namespace: The base namespace
        
    Returns:
        A unique namespace that isolates this pool's NIXL metadata from others
    """
    pool_id = get_pool_unique_id()
    unique_namespace = f"{namespace}_{pool_id}"
    logger.info(f"Pool NIXL namespace: {unique_namespace}")
    return unique_namespace


def log_pool_isolation_info():
    """Log comprehensive information about pool isolation configuration."""
    pool_id = get_pool_unique_id()
    
    logger.info(f"Pool isolation: ID={pool_id}, Strategy=separate_prefill_queues_and_nixl_namespaces")


def validate_pool_isolation(namespace: str):
    """Validate that pool isolation is properly configured."""
    try:
        pool_id = get_pool_unique_id()
        prefill_queue_name = get_unique_prefill_queue_name(namespace)
        nixl_namespace = get_unique_nixl_namespace(namespace)
        
        # Check for potential issues
        if pool_id == "localhost":
            logger.warning("POOL_ISOLATION: Using 'localhost' as pool ID - may cause conflicts in multi-node setup")
        
        return True
        
    except Exception as e:
        logger.error(f"POOL_ISOLATION: Validation failed: {e}")
        return False 