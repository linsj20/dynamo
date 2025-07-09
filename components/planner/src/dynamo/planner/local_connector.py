# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import filelock

from dynamo.planner.circusd import CircusController
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class LocalConnector(PlannerConnector):
    def __init__(self, namespace: str, runtime: DistributedRuntime, gpu_scope: str = None):
        """
        Initialize LocalConnector and connect to CircusController.

        Args:
            namespace: The Dynamo namespace
            runtime: Optional DistributedRuntime instance
            gpu_scope: GPU scope string to limit available GPUs (e.g., "0,1,2,3" or "0-3")
        """
        self.namespace = namespace
        self.runtime = runtime
        self.gpu_scope = gpu_scope
        self.state_file = Path.home() / ".dynamo" / "state" / f"{namespace}.json"
        self.circus = CircusController.from_state_file(namespace)
        self._lockfile = self.state_file.with_suffix(".lock")
        self._file_lock = filelock.FileLock(self._lockfile)
        self.worker_client: Any | None = None
        self.prefill_client: Any | None = None
        self.etcd_client: Any | None = None

    async def _load_state(self) -> Dict[str, Any]:
        """Load state from state file.

        Returns:
            State dictionary
        """
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")

        with self._file_lock:
            with open(self.state_file, "r") as f:
                return json.load(f)

    async def _save_state(self, state: Dict[str, Any]) -> bool:
        """Save state to state file.

        Args:
            state: State dictionary to save

        Returns:
            True if successful
        """
        try:
            with self._file_lock:
                with open(self.state_file, "w") as f:
                    json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def _parse_gpu_scope(self, gpu_scope: str) -> List[str]:
        """Parse gpu_scope string to get list of GPU IDs.
        
        Args:
            gpu_scope: GPU scope string (e.g., "0,1,2,3" or "0-3")
            
        Returns:
            List of GPU IDs
        """
        if not gpu_scope:
            return []
        
        gpu_ids = []
        for part in gpu_scope.split(','):
            part = part.strip()
            if '-' in part:
                # Handle ranges like "0-3"
                start, end = part.split('-')
                gpu_ids.extend(str(i) for i in range(int(start), int(end) + 1))
            else:
                # Handle individual IDs
                gpu_ids.append(part)
        
        return gpu_ids

    async def _get_available_gpus(self) -> List[str]:
        """Get list of unallocated GPU IDs.

        Returns:
            List of available GPU IDs
        """
        state = await self._load_state()
        
        # If gpu_scope is set, use it as the available GPU pool
        if self.gpu_scope:
            all_gpus = set(self._parse_gpu_scope(self.gpu_scope))
        else:
            # Fallback to system resources
            system_resources = state.get("environment", {}).get("SYSTEM_RESOURCES", {})
            all_gpus = set(str(gpu) for gpu in system_resources.get("gpu_info", []))

        allocated_gpus: set[str] = set()
        for component_info in state.get("components", {}).values():
            resources = component_info.get("resources", {})
            gpu_list = resources.get("allocated_gpus", [])
            allocated_gpus.update(str(gpu) for gpu in gpu_list)

        logger.info(f"GPU scope: {self.gpu_scope}")
        logger.info(f"All GPUs: {all_gpus}")
        logger.info(f"Allocated GPUs: {allocated_gpus}")
        available = sorted(list(all_gpus - allocated_gpus))
        logger.info(f"Available GPUs: {available}")
        return available

    async def add_component(self, component_name: str, blocking: bool = True) -> bool:
        """
        Add a component. The steps are as follows:

        1. Load state
        2. Find max suffix to create unique watcher name
        3. Built environment and command for watcher
        4. Block until component is running

        Args:
            component_name: Name of the component

        Returns:
            True if successful
        """
        state = await self._load_state()
        # Find max suffix
        max_suffix = 0
        for watcher_name in state["components"].keys():
            if watcher_name.startswith(f"{self.namespace}_{component_name}_"):
                suffix = int(
                    watcher_name.replace(f"{self.namespace}_{component_name}_", "")
                )
                max_suffix = max(max_suffix, suffix)

        watcher_name = f"{self.namespace}_{component_name}_{max_suffix + 1}"

        available_components = [c.replace(f"{self.namespace}_", "") for c in state["components"]]
        logger.info(f"State components: {list(state['components'].keys())}, Available after namespace strip: {available_components}")
        
        if component_name not in available_components:
            raise ValueError(
                f"Component {component_name} not found in state configuration"
            )

        # Get base command and config
        component_info = state["components"][f"{self.namespace}_{component_name}"]
        base_cmd = component_info["cmd"].split("--worker-env")[0].strip()
        service_config = state["environment"].get("DYNAMO_SERVICE_CONFIG")

        # Build environment
        watcher_env = os.environ.copy()
        gpu_id = None
        if component_name in ["VllmWorker", "PrefillWorker"]:
            # Always use the GPU allocation logic to get individual GPUs
            # Reload state to ensure we have the latest GPU allocations
            state = await self._load_state()
            available_gpus = await self._get_available_gpus()
            if not available_gpus:
                raise ValueError("No GPUs available for allocation")
            gpu_id = available_gpus[0]
            watcher_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            logger.info(f"Setting CUDA_VISIBLE_DEVICES to individual GPU: {gpu_id}")
            
            # IMMEDIATELY mark GPU as allocated to prevent race conditions
            # Add GPU to allocated_gpus list (should be empty for new workers)
            existing_resources = state["components"].get(watcher_name, {}).get("resources", {})
            allocated_gpus = existing_resources.get("allocated_gpus", [])
            assert gpu_id not in allocated_gpus, f"GPU {gpu_id} already allocated"
            allocated_gpus.append(gpu_id)
            
            resources = existing_resources.copy()
            resources["allocated_gpus"] = allocated_gpus
            
            state["components"][watcher_name] = {
                "watcher_name": watcher_name,
                "cmd": "pending",  # Will be updated after successful start
                "resources": resources,
            }
            await self._save_state(state)
            logger.info(f"Pre-allocated GPU {gpu_id} for {watcher_name} (allocated_gpus: {allocated_gpus})")

        watcher_env["DYNAMO_SERVICE_CONFIG"] = service_config

        # Build worker env list and command
        worker_env_list = [watcher_env]
        worker_env_arg = json.dumps(worker_env_list)
        # We add a custom component name to ensure that the lease is attatched to this specific watcher
        full_cmd = f"{base_cmd} --worker-env '{worker_env_arg}' --custom-component-name '{watcher_name}'"

        pre_add_endpoint_ids = await self._count_instance_ids(component_name)
        logger.info(f"Pre-add endpoint IDs: {pre_add_endpoint_ids}")

        logger.info(f"Adding watcher {watcher_name}")
        success = await self.circus.add_watcher(
            name=watcher_name, cmd=full_cmd, env=watcher_env, singleton=True
        )

        if success:
            # Update the command in the state now that worker started successfully
            if watcher_name in state["components"]:
                state["components"][watcher_name]["cmd"] = full_cmd
            else:
                # Non-GPU component
                resources = {}
                state["components"][watcher_name] = {
                    "watcher_name": watcher_name,
                    "cmd": full_cmd,
                    "resources": resources,
                }
            await self._save_state(state)
            logger.info(
                f"Succesfully created {watcher_name}. Waiting for worker to start..."
            )
        else:
            # Worker failed to start, clean up GPU allocation
            if component_name in ["VllmWorker", "PrefillWorker"] and watcher_name in state["components"]:
                logger.error(f"Worker {watcher_name} failed to start, releasing GPU {gpu_id}")
                del state["components"][watcher_name]
                await self._save_state(state)
            raise RuntimeError(f"Failed to start watcher {watcher_name}")

        if blocking:
            required_endpoint_ids = pre_add_endpoint_ids + 1
            max_retries = 30  # Wait up to 150 seconds (30 * 5 seconds)
            retry_count = 0
            
            while retry_count < max_retries:
                current_endpoint_ids = await self._count_instance_ids(component_name)
                if current_endpoint_ids == required_endpoint_ids:
                    break
                logger.info(
                    f"Waiting for {component_name} to start. Current endpoint IDs: {current_endpoint_ids}, Required endpoint IDs: {required_endpoint_ids}"
                )
                await asyncio.sleep(5)
                retry_count += 1
                
            if retry_count >= max_retries:
                # Worker failed to start within timeout, clean up GPU allocation
                logger.error(f"Worker {watcher_name} failed to start within timeout, cleaning up GPU allocation")
                if component_name in ["VllmWorker", "PrefillWorker"] and watcher_name in state["components"]:
                    del state["components"][watcher_name]
                    await self._save_state(state)
                    
                # Also try to remove the circus watcher
                await self.circus.remove_watcher(name=watcher_name, blocking=False)
                return False

        return success

    async def remove_component(
        self, component_name: str, blocking: bool = True
    ) -> bool:
        """
        Remove a component. The initial components are not numbered so we simply remove their resources
        and lease but keep the entry in order to use the cmd. This allows us to re-add the component
        without having to re-specify the cmd. For components that have been added, we remove their entry
        entry

        Args:
            component_name: Name of the component

        Returns:
            True if successful
        """
        logger.info(f"Attempting to remove component {component_name}")
        state = await self._load_state()
        matching_components = {}

        base_name = f"{self.namespace}_{component_name}"
        base_name_with_underscore = f"{base_name}_"

        for watcher_name in state["components"].keys():
            if watcher_name == base_name:
                matching_components[0] = watcher_name
            elif watcher_name.startswith(base_name_with_underscore):
                suffix = int(watcher_name.replace(base_name_with_underscore, ""))
                matching_components[suffix] = watcher_name

        if not matching_components:
            logger.error(f"No matching components found for {component_name}")
            return False

        highest_suffix = max(matching_components.keys())
        target_watcher = matching_components[highest_suffix]
        logger.info(f"Removing watcher {target_watcher}")

        success = await self.circus.remove_watcher(
            name=target_watcher, blocking=blocking
        )
        if not blocking:
            logger.info(
                f"Circus remove_watcher for {target_watcher} {'succeeded' if success else 'failed'}"
            )

        if success:
            if highest_suffix > 0:  # Numbered watcher - remove entire entry
                if target_watcher in state["components"]:
                    del state["components"][target_watcher]
            else:  # Base watcher - just clear resources and lease
                if target_watcher in state["components"]:
                    state["components"][target_watcher]["resources"] = {}
                    state["components"][target_watcher]["lease"] = None
            await self._save_state(state)

        return success

    async def _count_instance_ids(self, component_name: str) -> int:
        """
        Count the instance IDs for the 'generate' endpoint of given component.

        Args:
            component_name: Name of the component

        Returns:
            Number of endpoint IDs for a component
        """
        if component_name == "VllmWorker":
            if self.worker_client is None:
                self.worker_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(component_name)
                    .endpoint("generate")
                    .client()
                )
            worker_ids = self.worker_client.instance_ids()
            return len(worker_ids)
        elif component_name == "PrefillWorker":
            if self.prefill_client is None:
                self.prefill_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(component_name)
                    .endpoint("mock")
                    .client()
                )
            prefill_ids = self.prefill_client.instance_ids()
            return len(prefill_ids)
        else:
            raise ValueError(f"Component {component_name} not supported")

    async def _revoke_lease(self, lease_id: int) -> bool:
        """
        Wrapper function around the etcd client to revoke a lease

        Args:
            lease_id: Lease ID to revoke

        Returns:
            True if successful
        """
        if self.etcd_client is None:
            self.etcd_client = self.runtime.etcd_client()  # type: ignore
        try:
            await self.etcd_client.revoke_lease(lease_id)
            logger.info(f"Revoked lease {lease_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke lease {lease_id}: {e}")
            return False

    def __del__(self):
        """Cleanup circus controller connection on deletion."""
        if hasattr(self, "circus"):
            self.circus.close()
