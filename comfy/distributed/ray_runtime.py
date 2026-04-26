"""
ComfyUI-owned Ray distributed runtime.

Manages Ray process startup, actor lifecycle, and torch distributed + xFuser
initialization on workers.  Provides the entry point that ComfyUI calls at
startup when ``--ray`` is enabled.
"""

from __future__ import annotations

import os

import ray
import torch
import torch.distributed as dist

from comfy.distributed.parallel_state import (
    configure_ray_parallel,
    set_ray_runtime_available,
    get_parallel_config,
)

# GLobal runtime state

_ray_runtime = None


class RayRuntime:
    """Holder for Ray runtime objects and configuration."""

    def __init__(self, ray_ctx, ray_actors, gpu_actors, parallel_dict):
        self.ray_ctx = ray_ctx          # ray.init() context (if local)
        self.ray_actors = ray_actors    # dict with "workers" key
        self.gpu_actors = gpu_actors    # list of actor handles
        self.parallel_dict = parallel_dict


def _get_runtime() -> RayRuntime | None:
    """Return the current RayRuntime, or None if not initialized."""
    return _ray_runtime


def _setup_sage_fp8_cuda_kernel():
    """Ensure SageAttention FP8 CUDA kernel is available for xFuser."""
    try:
        __import__("sageattention")
    except ImportError as e:
        raise RuntimeError("SAGE_FP8_CUDA requires the sageattention package") from e


def _setup_sage_fp8_sm90_kernel():
    """Ensure SageAttention FP8 SM90 kernel is available for xFuser."""
    try:
        __import__("sageattention")
    except ImportError as e:
        raise RuntimeError("SAGE_FP8_SM90 requires the sageattention package") from e


def _normalized_degree(value: int | None) -> int:
    if value is None:
        return 1
    value = int(value)
    return 1 if value <= 0 else value


def init_ray_runtime(
    ray_gpus: int | None = None,
    ray_ulysses_degree: int | None = None,
    ray_ring_degree: int | None = None,
    ray_cfg_degree: int | None = None,
    ray_cluster_address: str | None = None,
    ray_attention: str | None = None,
    ray_sync_ulysses: bool | None = None,
    ray_skip_comm_test: bool | None = None,
    args: object | None = None,
) -> RayRuntime | None:
    """Initialize the Ray distributed runtime.

    Starts Ray (locally or connects to an existing cluster), creates worker
    actors, initializes torch distributed + xFuser groups on each worker, and
    stores the runtime state for later use.

    Accepts either keyword arguments or an ``args`` namespace object
    (e.g. from ``comfy.cli_args.parser.parse_args()``).  When ``args`` is
    provided it takes precedence over individually passed kwargs.

    Returns the RayRuntime instance on success, or None if Ray is already
    initialized.
    """
    global _ray_runtime

    # Resolve from args namespace if provided
    if args is not None:
        ray_gpus = getattr(args, "ray_gpus", ray_gpus) if ray_gpus is None else ray_gpus
        ray_ulysses_degree = getattr(args, "ray_ulysses_degree", ray_ulysses_degree) if ray_ulysses_degree is None else ray_ulysses_degree
        ray_ring_degree = getattr(args, "ray_ring_degree", ray_ring_degree) if ray_ring_degree is None else ray_ring_degree
        ray_cfg_degree = getattr(args, "ray_cfg_degree", ray_cfg_degree) if ray_cfg_degree is None else ray_cfg_degree
        ray_cluster_address = getattr(args, "ray_cluster_address", ray_cluster_address) if ray_cluster_address is None else ray_cluster_address
        ray_attention = getattr(args, "ray_attention", ray_attention) if ray_attention is None else ray_attention
        ray_sync_ulysses = getattr(args, "ray_sync_ulysses", ray_sync_ulysses) if ray_sync_ulysses is None else ray_sync_ulysses
        ray_skip_comm_test = getattr(args, "ray_skip_comm_test", ray_skip_comm_test) if ray_skip_comm_test is None else ray_skip_comm_test

    if _ray_runtime is not None:
        return _ray_runtime

    detected_gpus = torch.cuda.device_count()
    if ray_gpus is None:
        ray_gpus = detected_gpus
    elif detected_gpus > 0 and ray_gpus > detected_gpus:
        print(
            f"[Ray] WARNING: Requested {ray_gpus} GPUs but only {detected_gpus} detected. "
            f"Using {detected_gpus} GPU(s)."
        )
        ray_gpus = detected_gpus

    # Validate: world_size must be divisible by model parallel size
    model_parallel_size = ray_ulysses_degree * ray_ring_degree * ray_cfg_degree
    if ray_gpus % model_parallel_size != 0:
        raise ValueError(
            f"GPU count ({ray_gpus}) must be divisible by "
            f"ulysses({ray_ulysses_degree}) × ring({ray_ring_degree}) × cfg({ray_cfg_degree}) = {model_parallel_size}"
        )

    # Build parallel dict for workers
    parallel_dict = {
        "ulysses_degree": ray_ulysses_degree,
        "ring_degree": ray_ring_degree,
        "cfg_degree": ray_cfg_degree,
        "is_xdit": True,
        "attention": ray_attention,
        "sync_ulysses": ray_sync_ulysses,
    }

    # Start / connect Ray
    ray_ctx = None
    if ray_cluster_address == "local":
        ray_ctx = ray.init(
            num_gpus=ray_gpus,
            include_dashboard=False,
        )
    else:
        ray.init(address=ray_cluster_address)

    # create worker actors
    world_size = ray_gpus
    # Add world_size to parallel_dict so workers know the total
    parallel_dict["world_size"] = world_size
    gpu_actor = ray.remote(_RayWorker)
    gpu_actors = []

    for local_rank in range(world_size):
        gpu_actors.append(
            gpu_actor.options(num_gpus=1).remote(
                local_rank=local_rank,
                device_id=local_rank,
                parallel_dict=parallel_dict,
            )
        )

    # Wait for actors to be ready
    for actor in gpu_actors:
        ray.get(actor.__ray_ready__.remote())

    # Configure parallel state on the driver side
    configure_ray_parallel(
        ulysses_degree=ray_ulysses_degree,
        ring_degree=ray_ring_degree,
        cfg_degree=ray_cfg_degree,
        attention_backend=ray_attention,
        sync_ulysses=ray_sync_ulysses,
    )
    set_ray_runtime_available(True)

    # Run communication test if requested
    if not ray_skip_comm_test:
        _run_comm_test(gpu_actors)

    _ray_runtime = RayRuntime(
        ray_ctx=ray_ctx,
        ray_actors={"workers": gpu_actors},
        gpu_actors=gpu_actors,
        parallel_dict=parallel_dict,
    )

    return _ray_runtime


def _run_comm_test(gpu_actors):
    """Run an NCCL all-reduce communication test on existing workers."""
    try:
        ray.get([actor.comm_test.remote() for actor in gpu_actors])
    except Exception as e:
        print(f"[Ray] Communication test failed: {e}")


def destroy_ray_runtime():
    """Shut down the Ray runtime and destroy all actors."""
    global _ray_runtime
    if _ray_runtime is None:
        return

    for actor in _ray_runtime.gpu_actors:
        try:
            ray.get(actor.kill.remote())
        except Exception:
            pass

    if _ray_runtime.ray_ctx is not None:
        ray.shutdown()

    _ray_runtime = None
    set_ray_runtime_available(False)


def get_ray_actors():
    """Return the list of Ray worker actor handles."""
    if _ray_runtime is None:
        return []
    return _ray_runtime.gpu_actors


def get_parallel_dict():
    """Return the parallel configuration dict."""
    if _ray_runtime is None:
        return {}
    return _ray_runtime.parallel_dict


class _RayWorker:
    """Lightweight worker that initializes torch distributed + xFuser."""

    def __init__(self, local_rank: int, device_id: int, parallel_dict: dict):
        from datetime import timedelta

        import os

        self.local_rank = local_rank
        self.device_id = device_id
        self.parallel_dict = parallel_dict

        # World size comes from the driver (number of Ray actors)
        self.global_world_size = self.parallel_dict.get("world_size", 1)

        # Set CUDA_VISIBLE_DEVICES BEFORE importing torch
        # (PyTorch reads this env var during CUDA initialization)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        # Initialize xFuser (which internally calls dist.init_process_group)
        self._init_xfuser()

    def _init_xfuser(self):
        """Initialize xFuser distributed environment."""
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        parallel_dict = dict(self.parallel_dict)
        ulysses_degree = _normalized_degree(parallel_dict.get("ulysses_degree"))
        ring_degree = _normalized_degree(parallel_dict.get("ring_degree"))
        cfg_degree = _normalized_degree(parallel_dict.get("cfg_degree"))
        pp_degree = _normalized_degree(parallel_dict.get("pp_degree"))

        model_parallel_size = ulysses_degree * ring_degree * cfg_degree * pp_degree
        if self.global_world_size % model_parallel_size != 0:
            raise ValueError(
                "Ray worker count must be divisible by "
                "pp_degree * ulysses_degree * ring_degree * cfg_degree: "
                f"{self.global_world_size} is not divisible by {pp_degree} * {ulysses_degree} * {ring_degree} * {cfg_degree}"
            )

        init_distributed_environment(rank=self.local_rank, world_size=self.global_world_size)
        initialize_model_parallel(
            data_parallel_degree=self.global_world_size // model_parallel_size,
            sequence_parallel_degree=ulysses_degree * ring_degree,
            classifier_free_guidance_degree=cfg_degree,
            ring_degree=ring_degree,
            ulysses_degree=ulysses_degree,
            pipeline_parallel_degree=pp_degree,
        )

        print(
            f"[Rank {self.local_rank}] Parallel Degree: "
            f"Ulysses={ulysses_degree}, "
            f"Ring={ring_degree}, "
            f"CFG={cfg_degree}"
        )

    def get_local_rank(self):
        return self.local_rank

    def get_parallel_dict(self):
        return self.parallel_dict

    def comm_test(self):
        """Run a tiny all-reduce on the worker's initialized process group."""
        import torch
        import torch.distributed as dist

        tensor = torch.ones(1, device="cuda")
        dist.all_reduce(tensor)
        expected = float(self.global_world_size)
        actual = float(tensor.item())
        if actual != expected:
            raise RuntimeError(f"NCCL test failed on rank {self.local_rank}: got {actual}, expected {expected}")
        return True

    def load_unet(self, unet_path: str, model_options: dict) -> bool:
        """Load the UNet model on this worker.

        Args:
            unet_path: Path to the UNet model file.
            model_options: ComfyUI model options dict.

        Returns:
            True if the model was loaded successfully.
        """
        import comfy.sd as comfy_sd
        import gc
        import torch.cuda

        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.free_fsdp_vram()
            except Exception:
                pass
            try:
                self._model.detach()
            except Exception:
                pass

        self._model = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            self._model = comfy_sd.load_diffusion_model(unet_path, model_options=model_options)
            self._model_loaded = True
            return True
        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Failed to load model on rank {self.local_rank}: {e}") from e

    def unload_model(self):
        """Unload the model from this worker."""
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.free_fsdp_vram()
            except Exception:
                pass
            try:
                self._model.detach()
            except Exception:
                pass
            self._model = None
        self._model_loaded = False

    def forward(
        self,
        x,
        t,
        context,
        clip_fea,
        freqs,
        transformer_options,
        control=None,
        **kwargs,
    ):
        """Run a forward pass on this worker.

        For sequence-parallel (USP) execution, the model's forward_orig()
        handles the splitting/gathering internally.  This method passes
        the (possibly pre-split) inputs through the model.

        Args:
            x: Latent input tensor (full or pre-split for USP).
            t: Timestep tensor.
            context: Text embeddings.
            clip_fea: CLIP image features (optional).
            freqs: RoPE frequencies (ignored; model computes internally).
            transformer_options: ComfyUI transformer options.
            control: ControlNet control signals (optional).
            **kwargs: Additional arguments (reference_latent, etc.).

        Returns:
            Output tensor.
        """
        if not hasattr(self, "_model") or self._model is None:
            raise RuntimeError("Model not loaded. Call load_unet() first.")

        model = self._model
        base_model = model.model if hasattr(model, "model") else model

        if hasattr(base_model, "forward"):
            output = base_model.forward(
                x,
                timestep=t,
                context=context,
                clip_fea=clip_fea,
                control=control,
                transformer_options=transformer_options,
                **kwargs,
            )
        else:
            raise RuntimeError(f"Model has no forward method: {type(base_model)}")

        return output

    def kill(self):
        dist.destroy_process_group()
        ray.actor.exit_actor()


class _RayCOMMTester:
    """Worker for NCCL communication testing."""

    def __init__(self, local_rank: int, world_size: int, device_id: int):
        from datetime import timedelta

        import torch
        import torch.distributed as dist
        import os

        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
            timeout=timedelta(minutes=1),
        )

    def test(self):
        """Run an all-reduce sum test."""
        import torch
        import torch.distributed as dist

        x = torch.ones(1, device=self.device) * (self.local_rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        result = x.item()
        expected = self.world_size * (self.world_size + 1) // 2

        if abs(result - expected) > 1e-3:
            raise RuntimeError(
                f"[Rank {self.local_rank}] COMM test failed: got {result}, "
                f"expected {expected}. world_size may be mismatched!"
            )
        else:
            print(f"[Rank {self.local_rank}] COMM test passed (result={result})")

    def kill(self):
        import torch.distributed as dist
        dist.destroy_process_group()
        ray.actor.exit_actor()
