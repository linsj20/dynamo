#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import logging
import os
from typing import Any

from dynamo.sdk.core.protocol.interface import ServiceInterface

# Import our own resource module
from dynamo.sdk.lib.resource import (
    NVIDIA_GPU,
    GPUManager,
    ResourceError,
    system_resources,
)

logger = logging.getLogger(__name__)

# Constants
DYN_DISABLE_AUTO_GPU_ALLOCATION = "DYN_DISABLE_AUTO_GPU_ALLOCATION"
DYN_DEPLOYMENT_ENV = "DYN_DEPLOYMENT_ENV"

logger = logging.getLogger(__name__)


def format_memory_gb(memory_bytes: float) -> str:
    """Convert memory from bytes to formatted GB string.
    Args:
        memory_bytes: Memory size in bytes
    Returns:
        Formatted string with memory size in GB with 1 decimal place
    """
    return f"{memory_bytes/1024/1024/1024:.1f}GB"


class ResourceAllocator:
    def __init__(self) -> None:
        """Initialize the resource allocator."""
        self.system_resources = system_resources()
        self.gpu_manager = GPUManager()
        self.available_gpu_indices = list(range(len(self.system_resources[NVIDIA_GPU])))
        self.remaining_gpus = len(self.system_resources[NVIDIA_GPU])
        self.gpu_scope = None
        self._available_gpus: list[tuple[float, float]] = [
            (1.0, 1.0)  # each item is (remaining, unit)
            for _ in range(self.remaining_gpus)
        ]
        self._service_gpu_allocations: dict[str, list[int]] = {}
        logger.debug(
            f"ResourceAllocator initialized with {self.remaining_gpus} GPUs available"
        )
    
    def _get_gpu_scope(self, service_config: dict = None) -> list[int]:
        """Get GPU scope from service configuration and parse it into GPU indices."""
        if not service_config or "gpu-scope" not in service_config:
            return []
            
        gpu_scope_config = service_config["gpu-scope"]
        logger.info(f"Using GPU scope from service config: {gpu_scope_config}")
        
        if not gpu_scope_config:
            return []
        
        gpu_indices = []
        for part in gpu_scope_config.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                gpu_indices.extend(range(start, end + 1))
            else:
                gpu_indices.append(int(part))
        return gpu_indices

    def assign_gpus(self, count: float, service_name: str = "") -> list[int]:
        """
        Assign GPUs for use.

        Args:
            count: Number of GPUs to assign (can be fractional)
            service_name: Name of the service for tracking

        Returns:
            List of GPU indices that were assigned (logical indices within the scope)
        """
        if count > self.remaining_gpus:
            logger.warning(
                f"Requested {count} GPUs, but only {self.remaining_gpus} are remaining "
                f"from scope {self.available_gpu_indices}. "
                f"Serving may fail due to inadequate GPUs. Set {DYN_DISABLE_AUTO_GPU_ALLOCATION}=1 "
                "to disable automatic allocation and allocate GPUs manually."
            )
        self.remaining_gpus = int(max(0, self.remaining_gpus - count))

        assigned = []  # Will store assigned GPU indices

        if count < 1:  # a fractional GPU
            try:
                # try to find the GPU used with the same fragment
                gpu = next(
                    i
                    for i, v in enumerate(self._available_gpus)
                    if v[0] > 0 and v[1] == count
                )
            except StopIteration:
                try:
                    gpu = next(
                        i for i, v in enumerate(self._available_gpus) if v[0] == 1.0
                    )
                except StopIteration:
                    gpu = len(self._available_gpus)
                    self._available_gpus.append((1.0, count))
            remaining, _ = self._available_gpus[gpu]
            if (remaining := remaining - count) < count:
                # can't assign to the next one, mark it as zero.
                self._available_gpus[gpu] = (0.0, count)
            else:
                self._available_gpus[gpu] = (remaining, count)
            assigned = [gpu]
        else:  # allocate n GPUs, n is a positive integer
            if int(count) != count:
                raise ResourceError("Float GPUs larger than 1 is not supported")
            count = int(count)
            unassigned = [
                gpu
                for gpu, value in enumerate(self._available_gpus)
                if value[0] > 0 and value[1] == 1.0
            ]
            if len(unassigned) < count:
                logger.warning(f"Not enough GPUs to be assigned, {count} is requested")
                for _ in range(count - len(unassigned)):
                    unassigned.append(len(self._available_gpus))
                    self._available_gpus.append((1.0, 1.0))
            for gpu in unassigned[:count]:
                self._available_gpus[gpu] = (0.0, 1.0)
            assigned = unassigned[:count]

        # Store the allocation if service_name is provided
        if service_name and assigned:
            if service_name in self._service_gpu_allocations:
                self._service_gpu_allocations[service_name].extend(assigned)
                logger.debug(
                    f"Additional GPUs {assigned} allocated to service '{service_name}', "
                    f"total GPUs: {self._service_gpu_allocations[service_name]}"
                )
            else:
                self._service_gpu_allocations[service_name] = assigned
                logger.debug(f"GPUs {assigned} allocated to service '{service_name}'")
        elif assigned:
            logger.debug(f"GPUs {assigned} allocated without service name tracking")

        # Log the GPU allocation
        if assigned:
            if self.gpu_scope:
                physical_gpus = [self.available_gpu_indices[i] for i in assigned]
                logger.debug(f"Allocated logical GPUs {assigned} (physical GPUs {physical_gpus}) from scope {self.available_gpu_indices}")
            else:
                logger.debug(f"Allocated GPUs {assigned}")
        
        return assigned

    def get_gpu_stats(self) -> list[dict[str, Any]]:
        """Get detailed statistics for all GPUs."""
        return self.gpu_manager.get_gpu_stats()

    def get_resource_envs(
        self,
        service: ServiceInterface[Any],
    ) -> tuple[int, list[dict[str, str]]]:
        """
        Get resource environment variables for a service.

        Args:
            service: The service to get resource environment variables for

        Returns:
            Tuple of (number of workers, list of environment variables dictionaries)
        """
        logger.info(f"Getting resource envs for service {service.name}")
        services = service.get_service_configs()
        if service.name not in services:
            logger.warning(f"No service configs found for {service.name}")
            return 1, []  # Default to 1 worker, no special resources

        config = services[service.name]
        
        # Check for GPU scope in Common configuration
        common_config = services.get("Common", {})
        gpu_scope = self._get_gpu_scope(common_config)
        
        # Update GPU scope for this service if specified
        if gpu_scope:
            self.gpu_scope = gpu_scope
            self.available_gpu_indices = gpu_scope
            # Reset allocations for new scope
            self.remaining_gpus = len(gpu_scope)
            self._available_gpus = [(1.0, 1.0) for _ in range(self.remaining_gpus)]
            logger.info(f"ResourceAllocator using GPU scope: {gpu_scope}")
        else:
            # Use all available GPUs if no scope specified
            self.gpu_scope = None
            self.available_gpu_indices = list(range(len(self.system_resources[NVIDIA_GPU])))
            self.remaining_gpus = len(self.system_resources[NVIDIA_GPU])
            self._available_gpus = [(1.0, 1.0) for _ in range(self.remaining_gpus)]
            logger.info(f"ResourceAllocator using all GPUs: {self.available_gpu_indices}")
            
        logger.debug(f"Using config for {service.name}: {config}")

        num_gpus = 0
        num_workers = 1
        resource_envs: list[dict[str, str]] = []

        # Check for GPU requirements from config
        if "gpu" in (config.get("resources") or {}):
            try:
                num_gpus = int(config["resources"]["gpu"])
                logger.info(f"GPU requirement from config: {num_gpus}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid GPU value in config: {config['resources']['gpu']}")

        # Check if we have enough GPUs
        if num_gpus > 0:
            if self.gpu_scope:
                available_gpus = self.available_gpu_indices
                if num_gpus > len(available_gpus):
                    logger.warning(
                        f"Requested {num_gpus} GPUs, but only {len(available_gpus)} are available "
                        f"in scope {available_gpus}. Service may fail due to inadequate GPU resources."
                    )
            else:
                available_gpus = self.gpu_manager.get_available_gpus()
                if num_gpus > len(available_gpus):
                    logger.warning(
                        f"Requested {num_gpus} GPUs, but only {len(available_gpus)} are available. "
                        f"Service may fail due to inadequate GPU resources."
                    )

        # Determine number of workers
        if config.get("workers"):
            num_workers = config["workers"]
            logger.info(f"Using configured worker count: {num_workers}")

        # Handle GPU allocation
        if num_gpus and DYN_DISABLE_AUTO_GPU_ALLOCATION not in os.environ:
            logger.info("GPU allocation enabled")

            if os.environ.get(DYN_DEPLOYMENT_ENV):
                logger.info("K8s deployment detected")
                # K8s replicas: Assumes DYNAMO_DEPLOYMENT_ENV is set
                # each pod in replicaset will have separate GPU with same CUDA_VISIBLE_DEVICES
                assigned = self.assign_gpus(num_gpus, service.name)
                logger.info(f"Assigned GPUs for K8s: {assigned}")

                # Generate environment variables for each worker
                for _ in range(num_workers):
                    env_vars = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                    resource_envs.append(env_vars)
            else:
                logger.info(
                    f"Local deployment detected. Allocating GPUs for {num_workers} workers of '{service.name}'"
                )
                # Local deployment where we split all available GPUs across workers
                for worker_id in range(num_workers):
                    assigned = self.assign_gpus(num_gpus, service.name)
                    logger.debug(
                        f"Worker {worker_id} of '{service.name}' assigned GPUs: {assigned}"
                    )

                    # Generate environment variables for this worker
                    env_vars = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}

                    # If we have comprehensive GPU stats, log them
                    try:
                        gpu_stats = [
                            stat
                            for stat in self.get_gpu_stats()
                            if stat["index"] in assigned
                        ]
                        for stat in gpu_stats:
                            logger.info(
                                f"GPU {stat['index']} ({stat['name']}): "
                                f"Memory: {format_memory_gb(stat['free_memory'])} free / "
                                f"{format_memory_gb(stat['total_memory'])} total, "
                                f"Utilization: {stat['gpu_utilization']}% "
                            )
                    except Exception as e:
                        logger.debug(f"Failed to get GPU stats: {e}")

                    resource_envs.append(env_vars)

        logger.info(
            f"Final resource allocation - workers: {num_workers}, envs: {resource_envs}"
        )
        return num_workers, resource_envs

    def reset_allocations(self):
        """Reset all GPU allocations."""
        self.gpu_manager.reset_allocations()
        # Reset legacy tracking
        self._available_gpus = [(1.0, 1.0) for _ in range(self.remaining_gpus)]
