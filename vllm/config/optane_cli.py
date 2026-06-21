# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI argument registration for Optane configuration.

This module provides the `add_cli_args` classmethod for OptaneConfig
to enable command-line argument parsing in vLLM's serve and engine interfaces.
"""

from __future__ import annotations

from vllm.config.optane import OptaneConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _add_optane_cli_args(parser: FlexibleArgumentParser) -> None:
    """Add Optane configuration arguments to the CLI parser.
    
    Args:
        parser: The argument parser to add arguments to.
    """
    group = parser.add_argument_group("Optane KV Cache Configuration")

    group.add_argument(
        "--enable-optane",
        action="store_true",
        default=False,
        help=(
            "Enable Intel Optane persistent memory tier for KV cache offloading. "
            "Requires Optane hardware and proper mount configuration."
        ),
    )

    group.add_argument(
        "--optane-cache-size",
        type=float,
        default=None,
        help=(
            "Size of Optane cache in GiB per rank. If not specified, Optane tier "
            "is disabled. Recommended: 50-500 GiB depending on hardware."
        ),
    )

    group.add_argument(
        "--optane-mount-path",
        type=str,
        default="/mnt/optane",
        help=(
            "Mount path for DAX-enabled Optane persistent memory device. "
            "Must be a valid path with read/write permissions. "
            "Typical: /mnt/optane, /dev/dax0.0, etc."
        ),
    )

    group.add_argument(
        "--optane-backend",
        type=str,
        choices=["native", "pmdk"],
        default="native",
        help=(
            "Backend for Optane allocation. 'native' uses direct mmap-based "
            "allocation (simpler, lower overhead). 'pmdk' uses Intel PMDK library "
            "(transactional, crash-safe, higher overhead)."
        ),
    )

    group.add_argument(
        "--optane-eviction-policy",
        type=str,
        choices=["lru", "lfu", "cascade", "parallel"],
        default="cascade",
        help=(
            "Eviction policy for Optane tier. "
            "'lru': Least Recently Used. "
            "'lfu': Least Frequently Used. "
            "'cascade': Cascade to next tier (SSD). "
            "'parallel': Parallel eviction across tiers."
        ),
    )

    group.add_argument(
        "--optane-sliding-window",
        type=int,
        default=None,
        help=(
            "Sliding window size for Optane KV cache in tokens. "
            "If specified, only recent tokens are kept in Optane. "
            "Default: None (keep all tokens)."
        ),
    )

    group.add_argument(
        "--optane-retention-interval",
        type=int,
        default=None,
        help=(
            "Retention interval for cached blocks in Optane in seconds. "
            "Blocks older than this interval can be evicted. "
            "Default: None (no time-based eviction)."
        ),
    )

    group.add_argument(
        "--optane-max-pinned-blocks",
        type=int,
        default=None,
        help=(
            "Maximum number of blocks that can be pinned in Optane memory. "
            "Pinned blocks are never evicted. "
            "Default: None (no limit)."
        ),
    )

    group.add_argument(
        "--optane-enable-compression",
        action="store_true",
        default=False,
        help=(
            "Enable KV cache compression when offloading to Optane. "
            "Reduces memory usage but increases CPU overhead during compression."
        ),
    )

    group.add_argument(
        "--optane-compression-level",
        type=int,
        default=4,
        help=(
            "Compression level for Optane KV cache (1-9, if compression enabled). "
            "Higher values = better compression, more CPU overhead. Default: 4."
        ),
    )

    group.add_argument(
        "--optane-prefetch-size",
        type=int,
        default=None,
        help=(
            "Number of tokens to prefetch from Optane to GPU in advance. "
            "Reduces latency at cost of additional memory transfers. "
            "Default: None (no prefetching)."
        ),
    )

    group.add_argument(
        "--optane-enable-metrics",
        action="store_true",
        default=False,
        help="Enable detailed metrics collection for Optane KV cache operations.",
    )


# Patch the OptaneConfig class to add the CLI args method
if not hasattr(OptaneConfig, "add_cli_args"):
    OptaneConfig.add_cli_args = classmethod(_add_optane_cli_args)
