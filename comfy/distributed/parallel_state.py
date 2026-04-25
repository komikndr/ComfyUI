"""
ComfyUI-owned distributed parallel state management.

Provides a ComfyUI-owned interface for distributed parallel execution with xFuser.
This module stores parallel configuration and provides access to sequence parallel
rank/world size information without depending on Raylight.
"""

_parallel_config = None
_initialized = False
_ray_runtime_available = False


class ParallelConfig:
    """Parallel execution configuration for xFuser."""

    def __init__(
        self,
        ulysses_degree=1,
        ring_degree=1,
        cfg_degree=1,
        attention_backend="TORCH_FLASH",
        sync_ulysses=False,
    ):
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.cfg_degree = cfg_degree
        self.attention_backend = attention_backend
        self.sync_ulysses = sync_ulysses

    @property
    def sequence_parallel_degree(self):
        return self.ulysses_degree * self.ring_degree

    @property
    def dit_parallel_size(self):
        return self.sequence_parallel_degree * self.cfg_degree

    @property
    def model_parallel_size(self):
        return self.sequence_parallel_degree * self.cfg_degree


def set_ray_runtime_available(available):
    """Set whether the Ray runtime has been initialized."""
    global _ray_runtime_available
    _ray_runtime_available = available


def is_ray_enabled() -> bool:
    """Check if distributed parallel execution is fully enabled.

    Returns True only when:
      - Ray runtime has been initialized (_ray_runtime_available)
      - Parallel config has been set (_parallel_config)
      - xFuser SP world size > 1
    """
    if not _ray_runtime_available:
        return False
    if _parallel_config is None:
        return False
    try:
        from xfuser.core.distributed import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size() > 1
    except Exception:
        return False


def is_ray_configured() -> bool:
    """Check if parallel config has been set (regardless of SP world size)."""
    return _parallel_config is not None


def configure_ray_parallel(
    ulysses_degree=1,
    ring_degree=1,
    cfg_degree=1,
    attention_backend="TORCH_FLASH",
    sync_ulysses=False,
):
    """Configure parallel execution parameters.

    Called at startup when --ray is enabled. The actual xFuser distributed
    environment initialization is handled by the Ray worker runtime.
    """
    global _parallel_config, _initialized
    _parallel_config = ParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        cfg_degree=cfg_degree,
        attention_backend=attention_backend,
        sync_ulysses=sync_ulysses,
    )
    _initialized = True


def get_parallel_config():
    """Get the current parallel configuration."""
    return _parallel_config


def get_sp_world_size() -> int:
    """Get sequence parallel world size from xFuser."""
    try:
        from xfuser.core.distributed import get_sequence_parallel_world_size
        return get_sequence_parallel_world_size()
    except Exception:
        return 1


def get_sp_rank() -> int:
    """Get sequence parallel rank from xFuser."""
    try:
        from xfuser.core.distributed import get_sequence_parallel_rank
        return get_sequence_parallel_rank()
    except Exception:
        return 0


def get_sp_group():
    """Get sequence parallel group from xFuser."""
    try:
        from xfuser.core.distributed import get_sp_group
        return get_sp_group()
    except Exception:
        return None
