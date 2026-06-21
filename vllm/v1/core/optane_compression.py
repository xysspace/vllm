# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compression manager for KV cache blocks in Optane tier."""

import time
from typing import Optional, Tuple
import zstandard as zstd

from vllm.logger import init_logger
from vllm.v1.core.optane_types import CompressedBlock, CompressionStats

logger = init_logger(__name__)


class CompressionManager:
    """Manages compression/decompression of KV cache blocks.
    
    Supports zstd compression with configurable levels and thresholds.
    """

    def __init__(
        self,
        enable_compression: bool = True,
        compression_level: int = 3,
        min_compress_size: int = 1024,  # 1KB minimum
    ):
        """Initialize compression manager.
        
        Args:
            enable_compression: Whether to enable compression
            compression_level: zstd level 1-22 (default 3 = balanced)
            min_compress_size: Minimum block size to compress
        """
        self.enable_compression = enable_compression
        self.compression_level = max(1, min(compression_level, 22))
        self.min_compress_size = min_compress_size

        # Statistics
        self.stats = CompressionStats()

        # Compression context
        self.cctx = zstd.ZstdCompressor(level=self.compression_level)
        self.dctx = zstd.ZstdDecompressor()

        logger.info(
            f"CompressionManager initialized: "
            f"enabled={enable_compression}, level={compression_level}, "
            f"min_size={min_compress_size} bytes"
        )

    def should_compress(self, block_size: int) -> bool:
        """Determine if a block should be compressed.
        
        Args:
            block_size: Size of block in bytes
            
        Returns:
            True if should compress
        """
        if not self.enable_compression:
            return False

        # Don't compress small blocks (overhead not worth it)
        if block_size < self.min_compress_size:
            return False

        # Always compress large blocks (likely to compress well)
        if block_size > 1024 * 1024:  # >1MB
            return True

        # For medium blocks, compress if past threshold
        return True

    def compress(self, data: bytes, level: Optional[int] = None) -> CompressedBlock:
        """Compress block data with zstd.
        
        Args:
            data: Uncompressed block data
            level: Compression level (uses default if not specified)
            
        Returns:
            CompressedBlock with compressed data and metadata
        """
        if not self.enable_compression or len(data) < self.min_compress_size:
            # Return uncompressed
            return CompressedBlock(
                original_size=len(data),
                compressed_size=len(data),
                compression_time_us=0.0,
                decompression_time_us=0.0,
                data=data,
                compression_algorithm="none",
            )

        # Compress with timing
        start_time = time.perf_counter()
        if level is None:
            level = self.compression_level
        cctx = zstd.ZstdCompressor(level=min(max(level, 1), 22))
        compressed_data = cctx.compress(data)
        compression_time_us = (time.perf_counter() - start_time) * 1e6

        # Create result
        result = CompressedBlock(
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression_time_us=compression_time_us,
            decompression_time_us=0.0,
            data=compressed_data,
            compression_algorithm="zstd",
        )

        # Update statistics
        self.stats.total_blocks_compressed += 1
        self.stats.total_bytes_original += len(data)
        self.stats.total_bytes_compressed += len(compressed_data)
        self.stats.total_compression_time_us += compression_time_us
        self.stats.max_compression_ratio = max(
            self.stats.max_compression_ratio,
            result.compression_ratio,
        )
        self.stats.min_compression_ratio = min(
            self.stats.min_compression_ratio,
            result.compression_ratio,
        )

        logger.debug(
            f"Compressed block: {len(data)} -> {len(compressed_data)} bytes "
            f"({result.compression_ratio:.2f}x) in {compression_time_us:.2f}us"
        )

        return result

    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress block data.
        
        Args:
            compressed_data: Compressed block data
            
        Returns:
            Original uncompressed data
        """
        start_time = time.perf_counter()
        uncompressed_data = self.dctx.decompress(compressed_data)
        decompression_time_us = (time.perf_counter() - start_time) * 1e6

        logger.debug(
            f"Decompressed block: {len(compressed_data)} -> {len(uncompressed_data)} bytes "
            f"in {decompression_time_us:.2f}us"
        )

        return uncompressed_data

    def get_compression_stats(self) -> CompressionStats:
        """Get aggregate compression statistics.
        
        Returns:
            CompressionStats with current aggregates
        """
        return self.stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = CompressionStats()

    def estimate_compressed_size(self, block_size: int) -> int:
        """Estimate compressed size for a block.
        
        Uses average compression ratio if available.
        
        Args:
            block_size: Original block size
            
        Returns:
            Estimated compressed size
        """
        if not self.enable_compression or block_size < self.min_compress_size:
            return block_size

        # Use average compression ratio
        avg_ratio = self.stats.avg_compression_ratio
        if avg_ratio <= 1.0:
            avg_ratio = 2.0  # Default estimate if no data yet

        return int(block_size / avg_ratio)
