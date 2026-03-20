# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import time
from datetime import timedelta

import fla  # noqa
import torch
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss # commented out for now as it is not used so far; also not available in flash-linear-attention 0.4.0
from fla.ops.utils import prepare_position_ids
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


            if train_state.step == 25:
                torch.cuda.cudart().cudaProfilerStart()
                if os.environ.get("PROFILE_MODE") == "record_function":
                    _rf_profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        with_stack=True,
                    )
                    _rf_profiler.__enter__()
            if train_state.step == 30:
                torch.cuda.cudart().cudaProfilerStop()
                if os.environ.get("PROFILE_MODE") == "record_function" and '_rf_profiler' in dir():
                    _rf_profiler.__exit__(None, None, None)
                    if torch.distributed.get_rank() == 0:
                        logger.info("\n" + _rf_profiler.key_averages(group_by_stack_n=3).table(sort_by="self_device_time_total", row_limit=200))


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
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
                # logger.info(1)
            loss = sum(losses)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

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