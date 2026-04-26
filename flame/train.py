# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Iterable

import fla  # noqa
import torch
import torch.distributed.checkpoint as dcp
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss # commented out for now as it is not used so far; also not available in flash-linear-attention 0.4.0
from fla.ops.utils import prepare_position_ids
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.tools.utils import get_nparams_and_flops, get_parameter_counts

import custom_models
# torch.utils.checkpoint.set_checkpoint_debug_enabled(True)

def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)


register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


@contextmanager
def _allow_partial_checkpoint_load():
    """Allow loading checkpoints that do not cover newly added model weights."""

    original_load = dcp.load

    def _log_partial_checkpoint_load(state_dict, checkpoint_id) -> None:
        if state_dict is None or checkpoint_id is None:
            return

        try:
            metadata = dcp.FileSystemReader(checkpoint_id).read_metadata()
            # print (f"{metadata=}")
            planner = DefaultLoadPlanner(allow_partial_load=True)
            planner.set_up_planner(state_dict, metadata, is_coordinator=False)
            current_keys = set(planner.state_dict.keys())
            # print (f"{list(current_keys)[:20]=}")
            checkpoint_keys = set(metadata.state_dict_metadata.keys())
            # print (f"{list(checkpoint_keys)[:20]=}")
            missing_keys = sorted(current_keys - checkpoint_keys)
            if missing_keys:
                max_logged_keys = 16
                preview = "\n".join(missing_keys[:max_logged_keys])
                if len(missing_keys) > max_logged_keys:
                    preview = f"{preview}, ..."
                logger.warning(
                    f"{utils.Color.red}Partial checkpoint load enabled: %d model keys are missing in %s "
                    f"and will keep their initialized values. Missing keys: %s{utils.Color.reset}",
                    len(missing_keys),
                    checkpoint_id,
                    preview,
                )
        except Exception as exc:
            logger.warning(
                "Unable to inspect checkpoint metadata for partial-load logging at %s: %s",
                checkpoint_id,
                exc,
            )

    def _load_with_partial_planner(*args, **kwargs):
        state_dict = kwargs.get("state_dict")
        if state_dict is None and args:
            state_dict = args[0]
        checkpoint_id = kwargs.get("checkpoint_id")
        # print(f"{state_dict.keys()}")
        # if 'model' in state_dict and len(state_dict.keys()) == 1:
        #     state_dict = state_dict['model']
        _log_partial_checkpoint_load(state_dict, checkpoint_id)
        kwargs.setdefault("planner", DefaultLoadPlanner(allow_partial_load=True))
        return original_load(*args, **kwargs)

    dcp.load = _load_with_partial_planner
    try:
        yield
    finally:
        dcp.load = original_load


def _log_initial_weight_value_stats(
    model_parts: Iterable[torch.nn.Module], max_tensors: int | None = None
) -> None:
    """Log value statistics for initialized parameters before training starts."""

    def _tensor_stats(tensor: torch.Tensor, chunk_size: int = 16_777_216) -> dict[str, float | int]:
        with torch.no_grad():
            values = tensor.detach()
            if hasattr(values, "to_local"):
                values = values.to_local()
            numel = values.numel()
            if values.device.type == "meta":
                return {
                    "numel": numel,
                    "finite": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                }
            flat_values = values.reshape(-1)
            finite = 0
            total_sum = 0.0
            total_sumsq = 0.0
            total_min = float("inf")
            total_max = float("-inf")
            total_absmax = 0.0
            for start in range(0, numel, chunk_size):
                chunk = flat_values[start : start + chunk_size].float()
                finite_mask = torch.isfinite(chunk)
                finite_count = finite_mask.sum().item()
                if finite_count == 0:
                    continue
                finite_values = chunk[finite_mask]
                finite += finite_count
                total_sum += finite_values.sum().item()
                total_sumsq += finite_values.square().sum().item()
                total_min = min(total_min, finite_values.min().item())
                total_max = max(total_max, finite_values.max().item())
                total_absmax = max(total_absmax, finite_values.abs().max().item())
            if finite == 0:
                return {
                    "numel": numel,
                    "finite": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                }
            return {
                "numel": numel,
                "finite": finite,
                "sum": total_sum,
                "sumsq": total_sumsq,
                "min": total_min,
                "max": total_max,
                "absmax": total_absmax,
            }

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    model_parts = list(model_parts)
    logger.info(
        "[INIT-WEIGHT-STATS] Local value statistics after model.post_init(), before checkpoint load."
    )

    total_numel = 0
    total_finite = 0
    total_sum = 0.0
    total_sumsq = 0.0
    total_min = float("inf")
    total_max = float("-inf")
    total_absmax = 0.0
    logged_tensors = 0
    skipped_tensors = 0

    for part_idx, model_part in enumerate(model_parts):
        prefix = f"part{part_idx}." if len(model_parts) > 1 else ""
        for name, param in model_part.named_parameters():
            stats = _tensor_stats(param)
            numel = int(stats["numel"])
            finite = int(stats["finite"])
            total_numel += numel
            total_finite += finite
            total_sum += float(stats["sum"])
            total_sumsq += float(stats["sumsq"])
            if finite > 0:
                total_min = min(total_min, float(stats["min"]))
                total_max = max(total_max, float(stats["max"]))
                total_absmax = max(total_absmax, float(stats["absmax"]))

            if max_tensors is None or logged_tensors < max_tensors:
                mean = float(stats["sum"]) / finite if finite else float("nan")
                variance = max(float(stats["sumsq"]) / finite - mean * mean, 0.0) if finite else float("nan")
                rms = (float(stats["sumsq"]) / finite) ** 0.5 if finite else float("nan")
                finite_ratio = finite / numel if numel else float("nan")
                print(
                    "\n"
                    f"[INIT-WEIGHT-STATS][rank={rank}] {prefix}{name}\n"
                    f"  shape:        {tuple(param.shape)}\n"
                    f"  dtype:        {param.dtype}\n"
                    f"  numel:        {numel:,}\n"
                    f"  finite:       {finite_ratio:.6f}\n"
                    f"  mean:         {mean:.6e}\n"
                    f"  std:          {variance**0.5:.6e}\n"
                    f"  rms:          {rms:.6e}\n"
                    f"  min:          {float(stats['min']):.6e}\n"
                    f"  max:          {float(stats['max']):.6e}\n"
                    f"  absmax:       {float(stats['absmax']):.6e}"
                )
                logged_tensors += 1
            else:
                skipped_tensors += 1

    if total_finite > 0:
        total_mean = total_sum / total_finite
        total_variance = max(total_sumsq / total_finite - total_mean * total_mean, 0.0)
        total_rms = (total_sumsq / total_finite) ** 0.5
        total_finite_ratio = total_finite / total_numel if total_numel else float("nan")
    else:
        total_mean = total_variance = total_rms = total_finite_ratio = float("nan")
        total_min = total_max = total_absmax = float("nan")

    if skipped_tensors:
        print(
            "\n"
            f"[INIT-WEIGHT-STATS][rank={rank}] skipped logging "
            f"{skipped_tensors:,} tensors after max_tensors={max_tensors}"
        )
    print(
        "\n"
        f"[INIT-WEIGHT-STATS][rank={rank}][summary]\n"
        f"  tensors_logged: {logged_tensors:,}\n"
        f"  total_numel:    {total_numel:,}\n"
        f"  finite:         {total_finite_ratio:.6f}\n"
        f"  mean:           {total_mean:.6e}\n"
        f"  std:            {total_variance**0.5:.6e}\n"
        f"  rms:            {total_rms:.6e}\n"
        f"  min:            {total_min:.6e}\n"
        f"  max:            {total_max:.6e}\n"
        f"  absmax:         {total_absmax:.6e}"
    )


def _log_gradient_value_stats(
    model_parts: Iterable[torch.nn.Module],
    step: int,
    top_k: int | None = None,
    first_n_steps: int | None = None,
    local_norm_threshold: float | None = None,
) -> None:
    """Log the largest local gradient contributors before gradient clipping."""

    top_k = int(os.environ.get("FLAME_GRAD_DEBUG_TOPK", top_k or 6))
    first_n_steps = int(os.environ.get("FLAME_GRAD_DEBUG_FIRST_N_STEPS", first_n_steps or 4))
    local_norm_threshold = float(
        os.environ.get("FLAME_GRAD_DEBUG_LOCAL_NORM_THRESHOLD", local_norm_threshold or 100.0)
    )

    def _to_local(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        return tensor

    def _scan_tensor(tensor: torch.Tensor, chunk_size: int = 16_777_216) -> dict[str, float | int]:
        with torch.no_grad():
            values = _to_local(tensor)
            numel = values.numel()
            if values.device.type == "meta":
                return {
                    "numel": numel,
                    "finite": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                }
            flat_values = values.reshape(-1)
            finite = 0
            total_sum = 0.0
            total_sumsq = 0.0
            total_min = float("inf")
            total_max = float("-inf")
            total_absmax = 0.0
            for start in range(0, numel, chunk_size):
                chunk = flat_values[start : start + chunk_size].float()
                finite_mask = torch.isfinite(chunk)
                finite_count = finite_mask.sum().item()
                if finite_count == 0:
                    continue
                finite_values = chunk[finite_mask]
                finite += finite_count
                total_sum += finite_values.sum().item()
                total_sumsq += finite_values.square().sum().item()
                total_min = min(total_min, finite_values.min().item())
                total_max = max(total_max, finite_values.max().item())
                total_absmax = max(total_absmax, finite_values.abs().max().item())
            if finite == 0:
                return {
                    "numel": numel,
                    "finite": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                }
            return {
                "numel": numel,
                "finite": finite,
                "sum": total_sum,
                "sumsq": total_sumsq,
                "min": total_min,
                "max": total_max,
                "absmax": total_absmax,
            }

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    model_parts = list(model_parts)
    rows = []
    total_sumsq = 0.0
    total_numel = 0
    total_finite = 0

    for part_idx, model_part in enumerate(model_parts):
        prefix = f"part{part_idx}." if len(model_parts) > 1 else ""
        for name, param in model_part.named_parameters():
            if param.grad is None:
                continue
            grad_stats = _scan_tensor(param.grad)
            grad_sumsq = float(grad_stats["sumsq"])
            grad_finite = int(grad_stats["finite"])
            grad_numel = int(grad_stats["numel"])
            total_sumsq += grad_sumsq
            total_numel += grad_numel
            total_finite += grad_finite
            if grad_finite == 0:
                grad_mean = grad_std = grad_rms = float("nan")
            else:
                grad_mean = float(grad_stats["sum"]) / grad_finite
                grad_variance = max(grad_sumsq / grad_finite - grad_mean * grad_mean, 0.0)
                grad_std = grad_variance**0.5
                grad_rms = (grad_sumsq / grad_finite) ** 0.5
            param_stats = _scan_tensor(param)
            param_norm = float(param_stats["sumsq"]) ** 0.5
            grad_norm = grad_sumsq**0.5
            rows.append(
                {
                    "name": f"{prefix}{name}",
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype),
                    "grad_numel": grad_numel,
                    "grad_finite_ratio": grad_finite / grad_numel if grad_numel else float("nan"),
                    "grad_norm": grad_norm,
                    "grad_mean": grad_mean,
                    "grad_std": grad_std,
                    "grad_rms": grad_rms,
                    "grad_min": float(grad_stats["min"]),
                    "grad_max": float(grad_stats["max"]),
                    "grad_absmax": float(grad_stats["absmax"]),
                    "param_norm": param_norm,
                    "grad_param_ratio": grad_norm / param_norm if param_norm > 0 else float("inf"),
                }
            )

    local_grad_norm = total_sumsq**0.5
    should_log = step <= first_n_steps or local_grad_norm >= local_norm_threshold
    if not should_log:
        return

    rows.sort(key=lambda row: row["grad_norm"], reverse=True)
    total_finite_ratio = total_finite / total_numel if total_numel else float("nan")
    print(
        "\n"
        f"[GRAD-STATS][rank={rank}][step={step}][pre-clip]\n"
        f"  local_grad_norm: {local_grad_norm:.6e}\n"
        f"  tensors_with_grad: {len(rows):,}\n"
        f"  finite:          {total_finite_ratio:.6f}\n"
        f"  top_k:           {min(top_k, len(rows)):,}"
    )

    for idx, row in enumerate(rows[:top_k], start=1):
        print(
            "\n"
            f"[GRAD-STATS][rank={rank}][step={step}][#{idx}] {row['name']}\n"
            f"  shape:            {row['shape']}\n"
            # f"  dtype:            {row['dtype']}\n"
            # f"  grad_numel:       {row['grad_numel']:,}\n"
            # f"  grad_finite:      {row['grad_finite_ratio']:.6f}\n"
            f"  grad_norm:        {row['grad_norm']:.6e}\n"
            f"  grad_mean:        {row['grad_mean']:.6e}\n"
            f"  grad_std:         {row['grad_std']:.6e}\n"
            # f"  grad_rms:         {row['grad_rms']:.6e}\n"
            # f"  grad_min:         {row['grad_min']:.6e}\n"
            # f"  grad_max:         {row['grad_max']:.6e}\n"
            f"  grad_absmax:      {row['grad_absmax']:.6e}\n"
            f"  param_norm:       {row['param_norm']:.6e}\n"
            # f"  grad/param_norm:  {row['grad_param_ratio']:.6e}"
        )


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):

    # torch.cuda.memory._record_memory_history(max_entries=100000)

    # # get model_type attr from the json file at job_config.model.config, and import
    # import json
    # with open(job_config.model.config, "r") as f:
    #     model_json = json.load(f)
    #     model_type = model_json.get("model_type")
    #     layer_types = model_json.get("layer_types")
    # from custom_models import MODEL_TYPE_TO_PARENT_DIR
    # parent_dir = MODEL_TYPE_TO_PARENT_DIR[model_type]
    # import importlib
    # importlib.import_module(f"custom_models.{parent_dir}")
    model_type=None
    layer_types=None


    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0


    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    tokenizer = None
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    tokenizer.eos_token_id = tokenizer.pad_token_id # for qwen3 base
    logger.info(f"{tokenizer}")

    is_tokenized = False if job_config.training.tokenized_dataset_dir is None else True

    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset is not None
        else f"Loading pre-tokenized dataset {job_config.training.tokenized_dataset_dir}"
    )
    # breakpoint()
    dataset = build_dataset(
        dataset=job_config.training.dataset if job_config.training.tokenized_dataset_dir is None else None,
        tokenized_dataset_dir=job_config.training.tokenized_dataset_dir,
        data_format=job_config.training.data_format,
        seq_len=job_config.training.seq_len,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        data_mix_stopping_strategy=job_config.training.data_mix_stopping_strategy,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        is_tokenized=is_tokenized,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
        sample_trunc_seq=job_config.training.sample_trunc_seq,
        add_eos_token=job_config.training.add_eos_token_to_sample,
    )
    # breakpoint()
    # data_iterator = iter(dataloader)

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_linear_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )

    # E2E-TTT performs inner-loop autograd.grad inside forward, then uses outer loss.backward.
    # With torch.compile + donated buffers, AOT backward requires retain_graph=False/create_graph=False
    # across all backward calls, which conflicts with this pattern. Disable donated buffers for stability.
    if job_config.training.compile and bool(getattr(model_config, "use_e2e_ttt", False)):
        try:
            job_config.training.compile = False
            if getattr(torch._functorch.config, "donated_buffer", False):
                torch._functorch.config.donated_buffer = False

                logger.info(
                    "Disabled torch._functorch.config.donated_buffer for E2E-TTT inner-loop autograd compatibility."
                )
        except Exception as e:
            logger.warning(
                f"Failed to set torch._functorch.config.donated_buffer=False ({e}). "
                "If you hit donated buffer backward errors, disable it manually."
            )

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # # Source - https://stackoverflow.com/a/79627453
    # # Posted by Om Rastogi
    # # Retrieved 2026-03-07, License - CC BY-SA 4.0

    # for i, (name, tensor) in enumerate(model.state_dict().items()):
    #     print(f"{i}: {name} → shape={tensor.shape}, dtype={tensor.dtype}")


    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, nparams_embedding, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )
    (
        model_param_count_og,
        model_trainable_param_count,
        model_non_trainable_param_count,
    ) = get_parameter_counts(model)
    logger.info(
        f"{color.green}Model parameters at init (before parallelism): "
        f"total={model_param_count_og:,}, "
        f"trainable={model_trainable_param_count:,}, "
        f"non-trainable={model_non_trainable_param_count:,}{color.reset}"
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        ignored_params = None
        if model_type == 'qwen3_gdn' and layer_types is not None:
            # FSDP fully_shard expects ignored_params to be a set of nn.Parameter
            ignored_params = {
                model.model.layers[idx].self_attn.A_log
                for idx, layer_type in enumerate(layer_types)
                if layer_type == 'linear_attention'
            }
        # train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config, ignored_params=ignored_params)
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()

        model_parts = [model]

    # _log_initial_weight_value_stats(model_parts)

    ###
    from collections import defaultdict

    # dtypes = defaultdict(list)

    # for name, p in model.named_parameters():
    #     dtypes[p.dtype].append(name)

    # for k, v in dtypes.items():
    #     print(k, len(v))
    for name, p in model.named_parameters():
        if p.dtype == torch.float32:
            print(name, "is in", p.dtype)
    ###

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)


    # # Sanity check: ensure lr_proj / momentum_proj params are covered by the optimizer
    # name_to_param = dict(model.named_parameters())
    # target_names = [n for n in name_to_param if "attn.lr_proj" in n or "attn.momentum_proj" in n]
    # # Collect all params that the (first) underlying optimizer actually optimizes
    # from itertools import chain
    # # TorchTitan's build_optimizers returns a container with a list of torch.optim optimizers
    # all_opt_params = set(
    #     chain.from_iterable(
    #         group["params"] for opt in optimizers.optimizers for group in opt.param_groups
    #     )
    # )
    # missing = [n for n in target_names if name_to_param[n] not in all_opt_params]
    # if missing:
    #     raise RuntimeError(f"The following attn extra params are NOT in the optimizer: {missing}")


    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    with _allow_partial_checkpoint_load():
        checkpoint.load(step=job_config.checkpoint.load_step)
    # # NOTE[flame]:
    # # When loading checkpoints created with an older model/optimizer configuration,
    # # TorchTitan's CheckpointManager may raise:
    # #   "checkpoint's metadata does not match its state_dict"
    # # especially for newly introduced optimizer states such as
    # #   optimizer.state.model.layers.*.attn.lr_proj.*
    # #   optimizer.state.model.layers.*.attn.momentum_proj.*.
    # #
    # # PyTorch's distributed checkpoint utilities currently require an exact match
    # # between optimizer state metadata and the in-memory optimizer state, and do
    # # not support partially missing per-parameter optimizer entries.
    # #
    # # To allow resuming from such checkpoints while the model architecture has
    # # changed, we fall back to *ignoring optimizer (and scheduler) state* on
    # # load, and only restore model weights and training state.
    # try:
    #     checkpoint.load(step=job_config.checkpoint.load_step)
    # except Exception as e:
    #     if "metadata does not match its state_dict" in str(e):
    #         logger.warning(
    #             "Checkpoint load failed due to optimizer metadata/state mismatch. "
    #             "Falling back to loading model weights and train_state only; "
    #             "optimizer and lr_schedulers will be reinitialized."
    #         )
    #         # Ensure optimizer / scheduler are excluded from the next load attempt.
    #         exclude = list(getattr(job_config.checkpoint, "exclude_from_loading", []))
    #         for key in ["optimizer", "lr_scheduler"]:
    #             if key not in exclude:
    #                 exclude.append(key)
    #         job_config.checkpoint.exclude_from_loading = exclude

    #         # Recreate CheckpointManager so it observes the updated config.
    #         checkpoint = CheckpointManager(
    #             dataloader=dataloader,
    #             model_parts=model_parts,
    #             optimizers=optimizers,
    #             lr_schedulers=lr_schedulers,
    #             states={"train_state": train_state},
    #             job_config=job_config,
    #             ft_manager=ft_manager,
    #         )
    #         checkpoint.load(step=job_config.checkpoint.load_step)
    #     else:
    #         raise




    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )
    logger.info(
        f"{color.green}  Number of embedding parameters = {nparams_embedding:,} {color.reset}"
    )
    if model_config.tie_word_embeddings:
        logger.info("Model ties input and output word embeddings, so")
        logger.info(f"Number of distinct parameters = {model_param_count - nparams_embedding:,}")
    else:
        logger.info(f"Model does not tie input and output word embeddings, so the number of unique model parameters is the above.")
    

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):

        while train_state.step < job_config.training.steps:
            train_state.step += 1

            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()

                batch = next(data_iterator) 

                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )
                
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage                logger.info(f"{ parallel_dims=}")
                # logger.info(f"{ parallel_dims=}")
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        with maybe_enable_amp:
                            output = model(
                                input_ids=input_ids,
                                labels=labels,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                                use_cache=False
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
                # logger.info(1)
            loss = sum(losses)

            _log_gradient_value_stats(model_parts, train_state.step)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )
            # print (f"clipped {grad_norm=}")

            # optimizer step
            # logger.info(1)
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/grad_norm": grad_norm.item(),
                        "optimizer/skipped_step": train_state.skipped_step,
                    },
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            # breakpoint()
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    # Configure logger format to include filename and line number
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
