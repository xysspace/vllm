# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVCacheManager extensions for Optane tier integration.

This module extends the KVCacheManager to support Optane persistent memory
tier management, including automatic block promotion/demotion and tier-aware
allocation strategies.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging

import torch

from vllm.logger import init_logger
from vllm.config.optane import OptaneConfig
from vllm.v1.core.optane_manager import OptaneManager
from vllm.v1.core.block_pool_optane_ext import TierAwareBlockPool

logger = init_logger(__name__)


class OptaneIntegrationContext:
    """Context for Optane integration within KVCacheManager.

    Manages the interaction between KVCacheManager and OptaneManager,
    including promotion triggers and tier-aware allocation.
    """

    def __init__(
        self,
        optane_config: Optional[OptaneConfig],
        base_kv_cache_manager: Any,
        enable_optane: bool = False,
    ):
        """Initialize Optane integration context.

        Args:
            optane_config: Optane configuration (can be None if disabled)
            base_kv_cache_manager: The base KVCacheManager instance
            enable_optane: Whether Optane tier is enabled
        """
        self.optane_config = optane_config
        self.base_manager = base_kv_cache_manager
        self.enable_optane = enable_optane
        self.optane_manager: Optional[OptaneManager] = None
        self.tier_aware_pool: Optional[TierAwareBlockPool] = None

        if enable_optane and optane_config:
            self._initialize_optane()
        else:
            logger.info("Optane tier is disabled")

    def _initialize_optane(self) -> None:
        """Initialize Optane manager and tier-aware block pool."""
        try:
            # Calculate max Optane blocks (conservative: half of GPU blocks)
            num_gpu_blocks = getattr(self.base_manager, 'block_pool', None)
            if num_gpu_blocks is None:
                logger.warning("Cannot determine GPU blocks, disabling Optane")
                self.enable_optane = False
                return

            max_optane_blocks = max(100, num_gpu_blocks.num_gpu_blocks // 2)

            # Initialize OptaneManager
            self.optane_manager = OptaneManager(
                optane_config=self.optane_config,
                block_size=self.base_manager.block_pool.hash_block_size,
                max_blocks=max_optane_blocks,
                enable_events=getattr(self.optane_config, 'optane_enable_events', False),
                enable_metrics=True,
            )

            # Wrap block pool with tier awareness
            self.tier_aware_pool = TierAwareBlockPool(self.base_manager.block_pool)

            logger.info(
                f"Initialized Optane integration: "
                f"{max_optane_blocks} blocks, "
                f"{self.optane_config.get_optane_bytes() / (1024**3):.2f} GiB, "
                f"policy={self.optane_config.optane_eviction_policy}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Optane: {e}")
            self.enable_optane = False
            self.optane_manager = None
            self.tier_aware_pool = None

    def should_trigger_promotion(
        self,
        gpu_usage: float,
        optane_usage: float,
    ) -> bool:
        """Determine if blocks should be promoted to Optane.

        Args:
            gpu_usage: Current GPU memory usage fraction (0.0-1.0)
            optane_usage: Current Optane usage fraction (0.0-1.0)

        Returns:
            True if promotion should be triggered, False otherwise
        """
        if not self.enable_optane or self.optane_manager is None:
            return False

        return self.optane_manager.should_promote_to_optane(
            gpu_usage_fraction=gpu_usage,
            optane_usage_fraction=optane_usage,
        )

    def try_promote_cold_blocks_to_optane(
        self,
        num_candidates: int = 5,
        block_data_map: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[int, int]:
        """Try to promote cold blocks from GPU to Optane.

        Args:
            num_candidates: Number of blocks to consider for promotion
            block_data_map: Optional map of block_id -> KV tensor data

        Returns:
            Tuple of (num_promoted, num_attempted)
        """
        if not self.enable_optane or self.tier_aware_pool is None:
            return 0, 0

        try:
            # Get promotion candidates from tier-aware pool
            candidates = self.tier_aware_pool.get_promotion_recommendation(
                gpu_memory_usage=self.base_manager.usage,
                optane_usage=self.optane_manager.get_usage(),
                num_candidates=num_candidates,
            )

            if not candidates:
                return 0, 0

            logger.debug(f"Attempting to promote {len(candidates)} blocks to Optane")

            # If no block data provided, skip promotion
            if block_data_map is None:
                return 0, len(candidates)

            # Promote blocks
            promoted, num_promoted = self.tier_aware_pool.promote_to_optane(
                candidates,
                self.optane_manager,
                block_data_map,
            )

            if num_promoted > 0:
                logger.info(
                    f"Promoted {num_promoted}/{len(candidates)} blocks to Optane. "
                    f"GPU usage: {self.base_manager.usage:.2%}, "
                    f"Optane usage: {self.optane_manager.get_usage():.2%}"
                )

            return num_promoted, len(candidates)

        except Exception as e:
            logger.error(f"Error during block promotion: {e}")
            return 0, num_candidates

    def handle_optane_full_eviction(
        self,
        num_to_evict: int = 3,
    ) -> int:
        """Handle eviction from full Optane tier to SSD/Remote.

        Args:
            num_to_evict: Number of blocks to evict from Optane

        Returns:
            Number of blocks successfully evicted
        """
        if not self.enable_optane or self.optane_manager is None:
            return 0

        try:
            candidates = self.optane_manager.get_eviction_candidates(num_to_evict)
            if not candidates:
                return 0

            evicted = self.optane_manager.evict_blocks_from_optane(candidates)

            if evicted > 0:
                logger.info(
                    f"Evicted {evicted} blocks from Optane to SSD. "
                    f"Optane usage: {self.optane_manager.get_usage():.2%}"
                )

            return evicted

        except Exception as e:
            logger.error(f"Error during Optane eviction: {e}")
            return 0

    def get_block_from_optane_if_available(
        self,
        block_id: int,
        device: torch.device = torch.device("cuda"),
    ) -> Optional[torch.Tensor]:
        """Try to retrieve a block from Optane.

        Args:
            block_id: The block ID
            device: Target device for the tensor

        Returns:
            KV tensor if block is in Optane, None otherwise
        """
        if not self.enable_optane or self.optane_manager is None:
            return None

        # Check if block is in Optane
        if block_id not in self.optane_manager.get_promoted_block_ids():
            return None

        # Retrieve from Optane
        kv_data = self.optane_manager.demote_block_from_optane(block_id, device)
        if kv_data is not None:
            self.tier_aware_pool.extensions.mark_demoted_from_optane(
                block_id, 'gpu' if device.type == 'cuda' else 'cpu'
            )

        return kv_data

    def get_optane_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Optane metrics.

        Returns:
            Dictionary containing Optane statistics and usage
        """
        if not self.enable_optane or self.optane_manager is None:
            return {}

        metrics = {
            'enabled': True,
            'optane': self.optane_manager.get_metrics(),
            'tier_stats': self.tier_aware_pool.get_tier_stats() if self.tier_aware_pool else {},
        }

        return metrics

    def get_optane_events(self) -> List[Any]:
        """Get and clear all Optane KV-events.

        Returns:
            List of KV-events
        """
        if not self.enable_optane or self.optane_manager is None:
            return []

        return self.optane_manager.get_events()

    def reset(self) -> None:
        """Reset all Optane state."""
        if self.optane_manager:
            self.optane_manager.reset()
        if self.tier_aware_pool:
            self.tier_aware_pool.reset()

    def shutdown(self) -> None:
        """Cleanly shutdown Optane integration."""
        if self.optane_manager:
            self.optane_manager.shutdown()
        if self.tier_aware_pool:
            self.tier_aware_pool.reset()


class KVCacheManagerOptaneExtension:
    """Mixin-style extension for KVCacheManager Optane integration.

    Provides methods to be added to KVCacheManager for Optane support.
    """

    @staticmethod
    def create_optane_context(
        kv_cache_manager: Any,
        optane_config: Optional[OptaneConfig] = None,
        enable_optane: bool = False,
    ) -> Optional[OptaneIntegrationContext]:
        """Create an OptaneIntegrationContext for a KVCacheManager.

        Args:
            kv_cache_manager: The KVCacheManager instance
            optane_config: Optane configuration
            enable_optane: Whether to enable Optane

        Returns:
            OptaneIntegrationContext or None if disabled
        """
        if not enable_optane or optane_config is None:
            return None

        return OptaneIntegrationContext(
            optane_config=optane_config,
            base_kv_cache_manager=kv_cache_manager,
            enable_optane=True,
        )

    @staticmethod
    def integrate_with_allocate_slots(
        original_allocate_slots_func,
        optane_context: Optional[OptaneIntegrationContext],
    ):
        """Wrap allocate_slots to integrate Optane promotion logic.

        This function wraps the original allocate_slots method to add
        Optane promotion triggers when GPU memory is under pressure.

        Args:
            original_allocate_slots_func: The original allocate_slots function
            optane_context: The Optane integration context (or None if disabled)

        Returns:
            Wrapped allocate_slots function
        """

        def wrapped_allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=None,
            num_lookahead_tokens=0,
            num_external_computed_tokens=0,
            delay_cache_blocks=False,
            num_encoder_tokens=0,
            full_sequence_must_fit=False,
            reserved_blocks=0,
            has_scheduled_reqs=True,
        ):
            """Wrapped allocate_slots with Optane integration."""

            # Call original allocation
            blocks = original_allocate_slots_func(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=full_sequence_must_fit,
                reserved_blocks=reserved_blocks,
                has_scheduled_reqs=has_scheduled_reqs,
            )

            # NEW: Try Optane promotion if enabled and GPU is under pressure
            if optane_context and optane_context.enable_optane:
                gpu_usage = self.usage
                optane_usage = optane_context.optane_manager.get_usage()

                # Trigger promotion if GPU > 85% and Optane has space
                if optane_context.should_trigger_promotion(gpu_usage, optane_usage):
                    num_promoted, num_attempted = (
                        optane_context.try_promote_cold_blocks_to_optane(
                            num_candidates=3
                        )
                    )
                    logger.debug(
                        f"Optane promotion: {num_promoted}/{num_attempted} blocks "
                        f"(GPU: {gpu_usage:.2%}, Optane: {optane_usage:.2%})"
                    )

            return blocks

        return wrapped_allocate_slots

    @staticmethod
    def integrate_with_free(
        original_free_func,
        optane_context: Optional[OptaneIntegrationContext],
    ):
        """Wrap free to handle Optane cleanup on block eviction.

        Args:
            original_free_func: The original free function
            optane_context: The Optane integration context

        Returns:
            Wrapped free function
        """

        def wrapped_free(self, request):
            """Wrapped free with Optane cleanup."""
            # Call original free
            original_free_func(self, request)

            # NEW: Clean up Optane tracking if enabled
            if optane_context and optane_context.enable_optane:
                # Optane blocks will be handled by block pool
                pass

        return wrapped_free

    @staticmethod
    def add_optane_metrics_method(
        kv_cache_manager_class,
        optane_context: Optional[OptaneIntegrationContext],
    ):
        """Add get_optane_metrics method to KVCacheManager.

        Args:
            kv_cache_manager_class: The KVCacheManager class
            optane_context: The Optane integration context
        """

        def get_optane_metrics(self) -> Dict[str, Any]:
            """Get Optane tier metrics.

            Returns:
                Dictionary with Optane statistics and usage
            """
            if optane_context is None:
                return {}
            return optane_context.get_optane_metrics()

        kv_cache_manager_class.get_optane_metrics = get_optane_metrics

    @staticmethod
    def add_optane_events_method(
        kv_cache_manager_class,
        optane_context: Optional[OptaneIntegrationContext],
    ):
        """Add get_optane_events method to KVCacheManager.

        Args:
            kv_cache_manager_class: The KVCacheManager class
            optane_context: The Optane integration context
        """

        def get_optane_events(self) -> List[Any]:
            """Get and clear Optane KV-events.

            Returns:
                List of KV-events
            """
            if optane_context is None:
                return []
            return optane_context.get_optane_events()

        kv_cache_manager_class.get_optane_events = get_optane_events
