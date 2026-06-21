# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OptaneManager: Core bridge for persistent memory tier management."""

from typing import Dict, Optional, Tuple
import time
import logging

from vllm.config.optane import OptaneConfig
from vllm.logger import init_logger
from vllm.v1.core.optane_types import (
    EvictionTier,
    OptaneBlockInfo,
    BlockState,
)

logger = init_logger(__name__)


class OptaneManager:
    """Manages Optane persistent memory tier for KV cache blocks.
    
    Responsibilities:
    - Track Optane block allocation and usage
    - Handle block promotion (GPU → Optane) and demotion (Optane → GPU)
    - Coordinate tier transitions
    - Report metrics and statistics
    """

    def __init__(
        self,
        config: OptaneConfig,
        block_size_bytes: int,
        max_blocks: int,
        enable_events: bool = False,
    ) -> None:
        """Initialize OptaneManager.
        
        Args:
            config: OptaneConfig with Optane parameters
            block_size_bytes: Size of each KV cache block
            max_blocks: Maximum blocks to store in Optane
            enable_events: Whether to emit KV events
        """
        self.config = config
        self.block_size_bytes = block_size_bytes
        self.max_blocks = max_blocks
        self.enable_events = enable_events

        # Block tracking
        self.blocks: Dict[int, OptaneBlockInfo] = {}
        self.free_offsets: list[int] = []
        self.next_offset = 0

        # Capacity
        self.capacity_bytes = int(config.get_optane_bytes())
        self.used_bytes = 0

        # Statistics
        self.total_promoted = 0
        self.total_demoted = 0
        self.total_evicted = 0

        logger.info(
            f"OptaneManager initialized: capacity={self.capacity_bytes / 1e9:.1f}GB, "
            f"max_blocks={max_blocks}, block_size={block_size_bytes} bytes"
        )

    def allocate_block(
        self,
        block_id: int,
        size_bytes: int,
        compressed: bool = False,
    ) -> Optional[OptaneBlockInfo]:
        """Allocate space in Optane for a block.
        
        Args:
            block_id: Unique block identifier
            size_bytes: Size to allocate
            compressed: Whether block is compressed
            
        Returns:
            OptaneBlockInfo if successful, None if out of space
        """
        required_space = size_bytes if not compressed else int(size_bytes * 0.5)

        # Check if we have space
        if self.used_bytes + required_space > self.capacity_bytes:
            logger.debug(
                f"Optane full: {self.used_bytes} + {required_space} "
                f"> {self.capacity_bytes}"
            )
            return None

        # Allocate offset
        if self.free_offsets:
            offset = self.free_offsets.pop(0)
        else:
            offset = self.next_offset
            self.next_offset += required_space

        # Create block info
        block_info = OptaneBlockInfo(
            block_id=block_id,
            physical_offset=offset,
            size_bytes=size_bytes,
            compressed_size_bytes=required_space,
            current_tier=EvictionTier.OPTANE,
            compressed=compressed,
            compression_ratio=1.0 if not compressed else float(size_bytes) / required_space,
            access_time=time.time(),
            access_count=0,
            ref_cnt=1,
        )

        self.blocks[block_id] = block_info
        self.used_bytes += required_space

        logger.debug(
            f"Allocated block {block_id} in Optane: "
            f"{required_space} bytes at offset {offset}"
        )
        return block_info

    def free_block(self, block_id: int) -> bool:
        """Free an allocated Optane block.
        
        Args:
            block_id: Block to free
            
        Returns:
            True if successful, False if block not found
        """
        if block_id not in self.blocks:
            return False

        block_info = self.blocks[block_id]
        self.used_bytes -= block_info.compressed_size_bytes
        self.free_offsets.append(block_info.physical_offset)
        del self.blocks[block_id]

        logger.debug(
            f"Freed block {block_id} from Optane: "
            f"{block_info.compressed_size_bytes} bytes"
        )
        return True

    def promote_block_to_optane(
        self,
        block_id: int,
        size_bytes: int,
        compressed: bool = False,
    ) -> bool:
        """Promote a block from GPU/CPU to Optane.
        
        Args:
            block_id: Block to promote
            size_bytes: Block size
            compressed: Whether data is compressed
            
        Returns:
            True if successful, False if out of space
        """
        block_info = self.allocate_block(block_id, size_bytes, compressed)
        if block_info is None:
            return False

        self.total_promoted += 1
        logger.debug(f"Promoted block {block_id} to Optane (total: {self.total_promoted})")
        return True

    def demote_block_from_optane(self, block_id: int) -> bool:
        """Demote a block from Optane back to GPU/CPU.
        
        Args:
            block_id: Block to demote
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            return False

        self.total_demoted += 1
        logger.debug(f"Demoted block {block_id} from Optane (total: {self.total_demoted})")
        return self.free_block(block_id)

    def evict_block_from_optane(self, block_id: int) -> bool:
        """Evict a block from Optane to SSD/remote.
        
        Args:
            block_id: Block to evict
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            return False

        self.total_evicted += 1
        logger.debug(f"Evicted block {block_id} from Optane (total: {self.total_evicted})")
        return self.free_block(block_id)

    def get_available_space(self) -> int:
        """Get currently available Optane space in bytes.
        
        Returns:
            Number of available bytes
        """
        return self.capacity_bytes - self.used_bytes

    def get_utilization(self) -> float:
        """Get Optane utilization as fraction (0.0-1.0).
        
        Returns:
            Utilization fraction
        """
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

    def record_access(self, block_id: int) -> None:
        """Record that a block was accessed.
        
        Args:
            block_id: Block that was accessed
        """
        if block_id in self.blocks:
            block_info = self.blocks[block_id]
            block_info.access_time = time.time()
            block_info.access_count += 1

    def update_ref_count(self, block_id: int, delta: int) -> int:
        """Update reference count for a block.
        
        Args:
            block_id: Block to update
            delta: Change in ref count (+1 or -1)
            
        Returns:
            New ref count, or -1 if block not found
        """
        if block_id not in self.blocks:
            return -1

        block_info = self.blocks[block_id]
        block_info.ref_cnt += delta
        return block_info.ref_cnt

    def get_block_info(self, block_id: int) -> Optional[OptaneBlockInfo]:
        """Get metadata for a block.
        
        Args:
            block_id: Block to query
            
        Returns:
            OptaneBlockInfo if found, None otherwise
        """
        return self.blocks.get(block_id)

    def get_blocks_by_tier(self, tier: EvictionTier) -> list[int]:
        """Get all blocks currently in a tier.
        
        Args:
            tier: Tier to query
            
        Returns:
            List of block IDs in that tier
        """
        return [
            block_id
            for block_id, info in self.blocks.items()
            if info.current_tier == tier
        ]

    def get_statistics(self) -> dict:
        """Get manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_blocks": len(self.blocks),
            "used_bytes": self.used_bytes,
            "available_bytes": self.get_available_space(),
            "utilization": self.get_utilization(),
            "total_promoted": self.total_promoted,
            "total_demoted": self.total_demoted,
            "total_evicted": self.total_evicted,
        }
