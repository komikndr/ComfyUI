"""
ComfyUI-owned Ray model executor for distributed inference.

Provides a high-level API to load models on Ray workers and dispatch
inference tasks with USP (Ulysses Sequence Parallelism) support.
"""

from __future__ import annotations

import torch
import ray

from comfy.distributed.ray_runtime import (
    get_ray_actors,
    get_parallel_dict,
    _ray_runtime,
)
from comfy.distributed.parallel_state import (
    get_sp_group,
    get_sp_rank,
    get_sp_world_size,
    is_ray_enabled,
)


class RayModelExecutor:
    """Manages model loading and inference across Ray workers.

    Attributes:
        unet_path: Path to the UNet model file.
        model_options: ComfyUI model options dict passed to the loader.
        _model_loaded: Whether the model has been loaded on all workers.
    """

    def __init__(self):
        self.unet_path: str | None = None
        self.model_options: dict = {}
        self._model_loaded = False


    def load_model(self, unet_path: str, model_options: dict | None = None) -> bool:
        """Load the model on all Ray workers.

        Args:
            unet_path: Path to the UNet model file.
            model_options: ComfyUI model options dict.

        Returns:
            True if the model was loaded successfully on all workers.
        """
        if self._model_loaded and self.unet_path == unet_path:
            return True

        actors = get_ray_actors()
        if not actors:
            raise RuntimeError("No Ray actors available. Initialize the runtime first.")

        self.unet_path = unet_path
        self.model_options = model_options or {}

        futures = [
            actor.load_unet.remote(unet_path, self.model_options)
            for actor in actors
        ]
        try:
            ray.get(futures)
        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Failed to load model on Ray workers: {e}") from e

        self._model_loaded = True
        return True

    def unload_model(self):
        """Unload the model from all Ray workers."""
        if not self._model_loaded:
            return
        actors = get_ray_actors()
        if not actors:
            return
        try:
            ray.get([actor.unload_model.remote() for actor in actors])
        except Exception:
            pass
        self._model_loaded = False
        self.unet_path = None


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        transformer_options: dict | None = None,
        control = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch a forward pass to ALL Ray workers simultaneously for USP.

        For USP (Ulysses Sequence Parallelism), the WAN model's
        ``forward_orig()`` handles input splitting and output gathering
        internally.  All workers must participate in the forward pass
        because attention requires all-to-all communication and the
        final output is all-gathered.

        Args:
            x: Latent input tensor of shape [B, C_in, F, H, W].
            t: Diffusion timestep tensor of shape [B].
            context: Text embeddings of shape [B, L, C].
            clip_fea: Optional CLIP image features for I2V.
            freqs: RoPE frequencies (ignored; model computes internally).
            transformer_options: ComfyUI transformer options dict.
            control: ControlNet control signals (optional).
            **kwargs: Additional model-specific arguments.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        actors = get_ray_actors()
        if not actors:
            raise RuntimeError("No Ray actors available.")

        # Dispatch to ALL workers simultaneously for USP.
        # The WAN model's forward_orig() splits input by SP rank,
        # processes through blocks with all-to-all in attention,
        # and all-gathers the final output. All workers produce
        # the same output after all-gather.
        try:
            futures = [
                actor.forward.remote(
                    x, t, context, clip_fea, freqs,
                    transformer_options or {}, control, **kwargs
                )
                for actor in actors
            ]
            results = ray.get(futures)
            return results[0]
        except Exception as e:
            raise RuntimeError(f"Ray forward failed: {e}") from e


def _split_for_sp(tensor: torch.Tensor, dim: int = 1):
    """Pad tensor to world size and split by SP rank.

    Returns:
        Tuple of (split_tensor, original_size).
    """
    if tensor is None:
        return None, 0

    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return tensor, tensor.size(dim)

    orig_size = tensor.size(dim)
    pad = (sp_world_size - orig_size % sp_world_size) % sp_world_size
    if pad > 0:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=dim,
        )

    tensor = torch.chunk(tensor, sp_world_size, dim=dim)[get_sp_rank()]
    return tensor, orig_size


_executor: RayModelExecutor | None = None


def get_model_executor() -> RayModelExecutor:
    """Return the singleton RayModelExecutor, creating it if needed."""
    global _executor
    if _executor is None:
        _executor = RayModelExecutor()
    return _executor


def reset_model_executor():
    """Reset the singleton executor (useful for testing / model switching)."""
    global _executor
    if _executor is not None:
        _executor.unload_model()
    _executor = None
