# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Dict, List, Optional, Set, Tuple
import mmap
import os
import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.config.optane import OptaneConfig, OptaneEvictionPolicy

logger = init_logger(__name__)


class OptaneBlockMetadata:
    """Metadata for a KV cache block stored in Optane.
    
    Tracks block lifecycle, access patterns, and tier information.
    """

    def __init__(
        self,
        block_id: int,
        block_size: int,
        timestamp: float,
        access_count: int = 0,
    ):
        self.block_id = block_id
        self.block_size = block_size
        self.timestamp = timestamp  # When block was stored
        self.access_count = access_count  # For LFU eviction
        self.last_access_time = timestamp
        self.is_compressed = False
        self.compressed_size = block_size
        self.parent_block_hash: Optional[str] = None


class OptaneBlockPool:
    """Manages KV cache blocks stored in Optane persistent memory.
    
    Provides block allocation, storage, retrieval, and eviction for Optane tier.
    Integrates with vLLM's tiering infrastructure (GPU → CPU → Optane → SSD).
    
    Args:
        config: OptaneConfig object with Optane parameters
        block_size: Size of each KV cache block in tokens
        num_blocks: Total number of Optane blocks to allocate
    """

    def __init__(
        self,
        config: OptaneConfig,
        block_size: int,
        num_blocks: int,
    ):
        self.config = config
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.optane_bytes = config.get_optane_bytes()

        # Block metadata tracking
        self.block_metadata: Dict[int, OptaneBlockMetadata] = {}
        self.allocated_blocks: Set[int] = set()
        self.free_blocks: Set[int] = set(range(num_blocks))

        # Storage backend
        self.backend = config.optane_backend
        self.mount_path = config.optane_mount_path
        self.mmap_file: Optional[mmap.mmap] = None
        self.storage_buffer: Optional[np.ndarray] = None

        # Eviction policy
        self.eviction_policy = config.optane_eviction_policy
        self.eviction_queue: List[int] = []  # For LRU/LFU

        # Metrics
        self.enable_metrics = config.optane_enable_metrics
        self.blocks_stored = 0
        self.blocks_retrieved = 0
        self.blocks_evicted = 0
        self.total_store_time = 0.0
        self.total_retrieve_time = 0.0

        self._initialize_storage()
        logger.info(
            f"Initialized OptaneBlockPool: {num_blocks} blocks, "
            f"{self.optane_bytes / (1024**3):.2f} GiB, backend={self.backend}"
        )

    def _initialize_storage(self) -> None:
        """Initialize storage backend (mmap or PMDK)."""
        if self.backend == "native":
            self._init_native_backend()
        elif self.backend == "pmdk":
            self._init_pmdk_backend()
        else:
            raise ValueError(f"Unknown Optane backend: {self.backend}")

    def _init_native_backend(self) -> None:
        """Initialize native mmap-based backend."""
        try:
            # Create a temporary file for mmap
            temp_file = f"{self.mount_path}/vllm_optane_kv_cache.bin"
            os.makedirs(self.mount_path, exist_ok=True)

            # Pre-allocate file
            with open(temp_file, "wb") as f:
                f.write(b"\0" * self.optane_bytes)

            # mmap the file
            with open(temp_file, "r+b") as f:
                self.mmap_file = mmap.mmap(f.fileno(), self.optane_bytes)

            logger.info(
                f"Initialized native Optane backend at {temp_file} "
                f"({self.optane_bytes / (1024**3):.2f} GiB)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize native Optane backend: {e}")
            raise

    def _init_pmdk_backend(self) -> None:
        """Initialize PMDK-based backend (requires libpmem)."""
        try:
            import pmemobj

            pool_path = self.config.optane_pmdk_pool_path or f"{self.mount_path}/vllm_optane.pool"
            pool_size = self.config.optane_pmdk_pool_size * 1024 * 1024

            # Create or open PMDK pool
            self.pmdk_pool = pmemobj.create(
                pool_path, pool_size, 0o666, "vllm_optane"
            )
            logger.info(
                f"Initialized PMDK Optane backend at {pool_path} "
                f"({pool_size / (1024**3):.2f} GiB)"
            )
        except ImportError:
            logger.error(
                "PMDK backend requires 'pmemobj' package. "
                "Install with: pip install pmemobj"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PMDK Optane backend: {e}")
            raise

    def allocate_block(self, block_id: int) -> bool:
        """Allocate a block in Optane storage.
        
        Args:
            block_id: ID of the block to allocate
            
        Returns:
            True if allocation succeeded, False otherwise
        """
        if block_id not in self.free_blocks:
            logger.warning(f"Block {block_id} not available for allocation")
            return False

        self.free_blocks.remove(block_id)
        self.allocated_blocks.add(block_id)

        import time

        self.block_metadata[block_id] = OptaneBlockMetadata(
            block_id=block_id,
            block_size=self.block_size,
            timestamp=time.time(),
        )
        return True

    def store_kv_block(
        self,
        block_id: int,
        kv_data: torch.Tensor,
        parent_block_hash: Optional[str] = None,
    ) -> bool:
        """Store a KV cache block to Optane.
        
        Args:
            block_id: ID of the block to store
            kv_data: KV tensor data (typically shape [2, num_tokens, num_heads, head_dim])
            parent_block_hash: Optional hash of parent block for prefix caching
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if block_id not in self.allocated_blocks:
            logger.warning(f"Block {block_id} not allocated")
            return False

        import time

        start_time = time.time()

        try:
            # Convert tensor to numpy for storage
            kv_numpy = kv_data.cpu().numpy()
            block_data = kv_numpy.tobytes()

            # Apply compression if enabled
            if self.config.optane_enable_compression:
                import zlib

                compressed_data = zlib.compress(
                    block_data, self.config.optane_compression_level
                )
                self.block_metadata[block_id].is_compressed = True
                self.block_metadata[block_id].compressed_size = len(compressed_data)
                block_data = compressed_data

            # Store to Optane
            offset = block_id * self.optane_bytes // self.num_blocks
            if self.mmap_file:
                self.mmap_file.seek(offset)
                self.mmap_file.write(block_data)
                self.mmap_file.flush()

            # Update metadata
            if parent_block_hash:
                self.block_metadata[block_id].parent_block_hash = parent_block_hash
            self.block_metadata[block_id].last_access_time = time.time()

            if self.enable_metrics:
                elapsed = time.time() - start_time
                self.total_store_time += elapsed
                self.blocks_stored += 1

            return True
        except Exception as e:
            logger.error(f"Failed to store block {block_id} to Optane: {e}")
            return False

    def retrieve_kv_block(
        self,
        block_id: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[torch.Tensor]:
        """Retrieve a KV cache block from Optane.
        
        Args:
            block_id: ID of the block to retrieve
            device: Target device for the tensor
            
        Returns:
            Retrieved KV tensor or None if retrieval failed
        """
        if block_id not in self.allocated_blocks:
            logger.warning(f"Block {block_id} not allocated")
            return None

        import time

        start_time = time.time()

        try:
            # Retrieve from Optane
            offset = block_id * self.optane_bytes // self.num_blocks
            if self.mmap_file:
                self.mmap_file.seek(offset)
                block_size = self.block_metadata[block_id].compressed_size
                block_data = self.mmap_file.read(block_size)

            # Decompress if needed
            if self.block_metadata[block_id].is_compressed:
                import zlib

                block_data = zlib.decompress(block_data)

            # Convert back to tensor
            kv_numpy = np.frombuffer(block_data, dtype=np.float16)
            kv_tensor = torch.from_numpy(kv_numpy).to(device)

            # Update access tracking
            self.block_metadata[block_id].access_count += 1
            self.block_metadata[block_id].last_access_time = time.time()

            if self.enable_metrics:
                elapsed = time.time() - start_time
                self.total_retrieve_time += elapsed
                self.blocks_retrieved += 1

            return kv_tensor
        except Exception as e:
            logger.error(f"Failed to retrieve block {block_id} from Optane: {e}")
            return None

    def evict_block(self, block_id: int) -> bool:
        """Evict a block from Optane storage.
        
        Args:
            block_id: ID of the block to evict
            
        Returns:
            True if eviction succeeded, False otherwise
        """
        if block_id not in self.allocated_blocks:
            return False

        self.allocated_blocks.remove(block_id)
        self.free_blocks.add(block_id)

        if block_id in self.block_metadata:
            del self.block_metadata[block_id]

        if self.enable_metrics:
            self.blocks_evicted += 1

        return True

    def get_eviction_candidates(self, num_candidates: int) -> List[int]:
        """Get candidate blocks for eviction based on policy.
        
        Args:
            num_candidates: Number of eviction candidates to return
            
        Returns:
            List of block IDs to evict
        """
        if not self.allocated_blocks:
            return []

        candidates = []

        if self.eviction_policy == "lru":
            # Sort by last access time (oldest first)
            sorted_blocks = sorted(
                self.allocated_blocks,
                key=lambda b: self.block_metadata[b].last_access_time,
            )
            candidates = sorted_blocks[:num_candidates]

        elif self.eviction_policy == "lfu":
            # Sort by access count (least frequently used first)
            sorted_blocks = sorted(
                self.allocated_blocks,
                key=lambda b: self.block_metadata[b].access_count,
            )
            candidates = sorted_blocks[:num_candidates]

        elif self.eviction_policy in ("cascade", "parallel"):
            # For cascade/parallel, return oldest blocks
            sorted_blocks = sorted(
                self.allocated_blocks,
                key=lambda b: self.block_metadata[b].timestamp,
            )
            candidates = sorted_blocks[:num_candidates]

        return candidates

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self.free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Get number of allocated blocks."""
        return len(self.allocated_blocks)

    def get_usage(self) -> float:
        """Get Optane usage as fraction (0.0 to 1.0)."""
        if self.num_blocks == 0:
            return 0.0
        return len(self.allocated_blocks) / self.num_blocks

    def get_metrics(self) -> Dict[str, any]:
        """Get performance metrics."""
        return {
            "blocks_stored": self.blocks_stored,
            "blocks_retrieved": self.blocks_retrieved,
            "blocks_evicted": self.blocks_evicted,
            "total_store_time": self.total_store_time,
            "total_retrieve_time": self.total_retrieve_time,
            "avg_store_time": (
                self.total_store_time / self.blocks_stored
                if self.blocks_stored > 0
                else 0.0
            ),
            "avg_retrieve_time": (
                self.total_retrieve_time / self.blocks_retrieved
                if self.blocks_retrieved > 0
                else 0.0
            ),
            "usage_fraction": self.get_usage(),
            "allocated_blocks": self.get_num_allocated_blocks(),
            "free_blocks": self.get_num_free_blocks(),
        }

    def reset(self) -> None:
        """Reset all blocks and metrics."""
        self.allocated_blocks.clear()
        self.free_blocks = set(range(self.num_blocks))
        self.block_metadata.clear()
        self.blocks_stored = 0
        self.blocks_retrieved = 0
        self.blocks_evicted = 0
        self.total_store_time = 0.0
        self.total_retrieve_time = 0.0
        logger.info("OptaneBlockPool reset")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self.mmap_file:
            self.mmap_file.close()
