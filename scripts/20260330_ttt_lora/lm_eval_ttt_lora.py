#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import logging
from typing import Any

import custom_models  # noqa: F401
import torch
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    handle_stop_sequences,
    normalize_gen_kwargs,
    postprocess_generated_text,
)
from lm_eval.models.utils_hf import get_dtype
from packaging.version import parse as vparse
from tqdm import tqdm

from run_ttt_lora import (
    BETAS,
    CHUNK_SIZE,
    GRAD_CLIP,
    LR,
    LOCAL_TRAIN_WINDOW,
    LOSS_MODE,
    LOSS_TOPK_FRACTION,
    LORA_ALPHA,
    LORA_R,
    STEPS_PER_CHUNK,
    UPDATE_MODE,
    WEIGHT_DECAY,
    TOP_LAYER_FRACTION,
    attach_ttt_lora,
    clone_trainable_state,
    compute_delta_norm,
    generate_from_adapted_state,
    get_trainable_lora_parameters,
    normalize_update_mode,
    rebuild_cache_for_prefix,
    reset_trainable_state,
    run_chunk_update_step,
    should_prefer_flash_attention_2,
)


eval_logger = logging.getLogger(__name__)


@register_model("hf-ttt-lora", "hf_ttt_lora")
class HFTTTLoRALM(HFLM):
    def _create_model(
        self,
        pretrained: str,
        revision: str = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config=None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        model_kwargs = kwargs or {}
        explicit_torch_dtype = model_kwargs.pop("torch_dtype", None)
        effective_dtype = explicit_torch_dtype if explicit_torch_dtype is not None else dtype
        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        if not autogptq and not gptqmodel:
            if model_kwargs.get("load_in_4bit"):
                assert vparse(transformers.__version__) >= vparse("4.30.0"), (
                    "load_in_4bit requires transformers >= 4.30.0"
                )
                if compute_dtype := model_kwargs.get("bnb_4bit_compute_dtype"):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(compute_dtype)
            pretrained_kwargs = dict(
                revision=revision,
                torch_dtype=get_dtype(effective_dtype),
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **model_kwargs,
            )
            if should_prefer_flash_attention_2(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=get_dtype(effective_dtype),
            ):
                try:
                    self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                        pretrained,
                        attn_implementation="flash_attention_2",
                        **pretrained_kwargs,
                    )
                except Exception:
                    self._model = self.AUTO_MODEL_CLASS.from_pretrained(pretrained, **pretrained_kwargs)
            else:
                self._model = self.AUTO_MODEL_CLASS.from_pretrained(pretrained, **pretrained_kwargs)
        else:
            super()._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **kwargs,
            )

    def __init__(
        self,
        *args,
        ttt_enable: bool = True,
        ttt_chunk_size: int = CHUNK_SIZE,
        ttt_steps_per_chunk: int = STEPS_PER_CHUNK,
        ttt_update_mode: str = UPDATE_MODE,
        ttt_local_train_window: int = LOCAL_TRAIN_WINDOW,
        ttt_lora_r: int = LORA_R,
        ttt_lora_alpha: int = LORA_ALPHA,
        ttt_loss_mode: str = LOSS_MODE,
        ttt_loss_topk_fraction: float = LOSS_TOPK_FRACTION,
        ttt_lr: float = LR,
        ttt_beta1: float = BETAS[0],
        ttt_beta2: float = BETAS[1],
        ttt_weight_decay: float = WEIGHT_DECAY,
        ttt_grad_clip: float = GRAD_CLIP,
        ttt_raw_chars_per_token: float = 4.0,
        ttt_raw_trunc_safety_margin: float = 1.15,
        ttt_log_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ttt_enable = bool(ttt_enable)
        if self.ttt_enable:
            self._model = attach_ttt_lora(
                self.model,
                top_layer_fraction=TOP_LAYER_FRACTION,
                lora_r=int(ttt_lora_r),
                lora_alpha=int(ttt_lora_alpha),
            )
        self._model.eval()

        self.ttt_chunk_size = int(ttt_chunk_size)
        self.ttt_steps_per_chunk = int(ttt_steps_per_chunk)
        self.ttt_update_mode = normalize_update_mode(str(ttt_update_mode))
        self.ttt_local_train_window = int(ttt_local_train_window)
        self.ttt_lora_r = int(ttt_lora_r)
        self.ttt_lora_alpha = int(ttt_lora_alpha)
        self.ttt_loss_mode = str(ttt_loss_mode)
        self.ttt_loss_topk_fraction = float(ttt_loss_topk_fraction)
        self.ttt_lr = float(ttt_lr)
        self.ttt_betas = (float(ttt_beta1), float(ttt_beta2))
        self.ttt_weight_decay = float(ttt_weight_decay)
        self.ttt_grad_clip = float(ttt_grad_clip)
        self.ttt_raw_chars_per_token = float(ttt_raw_chars_per_token)
        self.ttt_raw_trunc_safety_margin = float(ttt_raw_trunc_safety_margin)
        self.ttt_log_path = ttt_log_path
        self.trainable_lora_params = get_trainable_lora_parameters(self.model)
        self.initial_lora_state = clone_trainable_state(self.trainable_lora_params) if self.ttt_enable else {}
        self._ttt_log_handle = open(ttt_log_path, "a", encoding="utf-8") if ttt_log_path else None

    def _write_ttt_log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        if self._ttt_log_handle is not None:
            self._ttt_log_handle.write(line + "\n")
            self._ttt_log_handle.flush()
        eval_logger.info(line)

    def _memory_stats(self) -> dict[str, float | None]:
        if not torch.cuda.is_available():
            return {
                "cuda_allocated_gb": None,
                "cuda_reserved_gb": None,
                "cuda_peak_allocated_gb": None,
                "cuda_peak_reserved_gb": None,
            }
        return {
            "cuda_allocated_gb": torch.cuda.memory_allocated(self.device) / (1024 ** 3),
            "cuda_reserved_gb": torch.cuda.memory_reserved(self.device) / (1024 ** 3),
            "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
            "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
        }

    def _adapt_prefix(self, input_ids):
        if not self.ttt_enable:
            return rebuild_cache_for_prefix(
                self.model,
                input_ids,
                chunk_size=self.ttt_chunk_size,
            ), []

        reset_trainable_state(self.trainable_lora_params, self.initial_lora_state)
        optimizer = torch.optim.AdamW(
            [param for _, param in self.trainable_lora_params],
            lr=self.ttt_lr,
            betas=self.ttt_betas,
            weight_decay=self.ttt_weight_decay,
        )
        last_rebuild = None
        logs = []
        seq_len = input_ids.shape[1]

        for chunk_idx, start in enumerate(range(0, seq_len, self.ttt_chunk_size)):
            end = min(start + self.ttt_chunk_size, seq_len)
            step_logs = []
            prefix_cache_for_chunk = None if start == 0 or last_rebuild is None else last_rebuild.past_key_values
            for step_idx in range(self.ttt_steps_per_chunk):
                loss, grad_norm = run_chunk_update_step(
                    model=self.model,
                    optimizer=optimizer,
                    trainable_lora_params=self.trainable_lora_params,
                    full_input_ids=input_ids,
                    chunk_start=start,
                    chunk_end=end,
                    chunk_size=self.ttt_chunk_size,
                    update_mode=self.ttt_update_mode,
                    local_train_window=self.ttt_local_train_window,
                    loss_mode=self.ttt_loss_mode,
                    loss_topk_fraction=self.ttt_loss_topk_fraction,
                    base_prefix_cache=prefix_cache_for_chunk,
                )

                step_logs.append(
                    {
                        "chunk_idx": chunk_idx,
                        "step_idx": step_idx,
                        "steps_per_chunk": self.ttt_steps_per_chunk,
                        "update_mode": self.ttt_update_mode,
                        "local_train_window": self.ttt_local_train_window if self.ttt_update_mode == "local_window" else None,
                        "lora_r": self.ttt_lora_r,
                        "lora_alpha": self.ttt_lora_alpha,
                        "loss_mode": self.ttt_loss_mode,
                        "loss_topk_fraction": self.ttt_loss_topk_fraction if self.ttt_loss_mode == "topk_fraction" else None,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_tokens": end - start,
                        "loss": None if loss is None else float(loss.detach().cpu().item()),
                        "grad_norm": None
                        if grad_norm is None
                        else float(grad_norm.detach().cpu().item())
                        if torch.is_tensor(grad_norm)
                        else float(grad_norm),
                        "lora_norm": float(compute_delta_norm(self.trainable_lora_params, self.initial_lora_state)),
                    }
                )

            last_rebuild = rebuild_cache_for_prefix(
                self.model,
                input_ids[:, :end],
                chunk_size=self.ttt_chunk_size,
            )
            logs.extend(step_logs)

        if last_rebuild is None:
            last_rebuild = rebuild_cache_for_prefix(
                self.model,
                input_ids,
                chunk_size=self.ttt_chunk_size,
            )
        return last_rebuild, logs

    def _clip_grad_norm(self) -> float:
        import torch

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for _, param in self.trainable_lora_params],
            max_norm=self.ttt_grad_clip,
        )
        return float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)

    def _raw_pretruncate_context(self, context: str, max_ctx_len: int) -> tuple[str, int]:
        estimated_char_budget = int(
            max_ctx_len * self.ttt_raw_chars_per_token * self.ttt_raw_trunc_safety_margin
        )
        if estimated_char_budget <= 0 or len(context) <= estimated_char_budget:
            return context, estimated_char_budget
        return context[-estimated_char_budget:], estimated_char_budget

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        if self.batch_size_per_gpu != 1:
            raise ValueError("hf-ttt-lora currently supports batch_size=1 only.")

        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with TTT-LoRA",
        )
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        for req_idx, req in enumerate(requests):
            context, gen_kwargs = req.args
            kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = kwargs.pop("max_gen_toks")

            if self.backend != "causal":
                raise NotImplementedError("hf-ttt-lora only supports causal models.")

            max_ctx_len = self.max_length - max_gen_toks
            raw_context, raw_char_budget = self._raw_pretruncate_context(context, max_ctx_len)
            context_enc, _ = self.tok_batch_encode(
                [raw_context],
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            original_token_len = len(self.tok_encode(context))
            truncated_token_len = int(context_enc.shape[1])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            do_sample = bool(kwargs.pop("do_sample", False))
            temperature = float(kwargs.pop("temperature", 1.0))
            top_p = float(kwargs.pop("top_p", 1.0))
            if kwargs:
                eval_logger.warning(f"Unused generation kwargs for hf-ttt-lora: {sorted(kwargs.keys())}")

            self._write_ttt_log(
                {
                    "event": "request_start",
                    "request_idx": req_idx,
                    "task_name": getattr(req, "task_name", None),
                    "doc_id": getattr(req, "doc_id", None),
                    "instance_idx": getattr(req, "idx", None),
                    "original_char_len": len(context),
                    "raw_pretruncated_char_len": len(raw_context),
                    "raw_char_budget": raw_char_budget,
                    "original_token_len": original_token_len,
                    "truncated_token_len": truncated_token_len,
                    "max_ctx_len": max_ctx_len,
                    "max_gen_toks": max_gen_toks,
                    "rank": self.rank,
                    **self._memory_stats(),
                }
            )

            try:
                rebuild, logs = self._adapt_prefix(context_enc)
                for chunk_log in logs:
                    self._write_ttt_log(
                        {
                            "event": "chunk_update",
                            "request_idx": req_idx,
                            "rank": self.rank,
                            **chunk_log,
                            **self._memory_stats(),
                        }
                    )

                cont_toks = generate_from_adapted_state(
                    model=self.model,
                    prefix_ids=context_enc,
                    past_key_values=rebuild.past_key_values,
                    last_logits=rebuild.last_logits,
                    max_new_tokens=max_gen_toks,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.eot_token_id,
                )
                continuation = self.tok_decode(cont_toks[0].tolist())
                continuation = postprocess_generated_text(
                    generation=continuation,
                    stop=until,
                    think_end_token=self.think_end_token if isinstance(self.think_end_token, str) else None,
                )
                res.append(continuation)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), continuation)
                self._write_ttt_log(
                    {
                        "event": "request_end",
                        "request_idx": req_idx,
                        "task_name": getattr(req, "task_name", None),
                        "doc_id": getattr(req, "doc_id", None),
                        "instance_idx": getattr(req, "idx", None),
                        "rank": self.rank,
                        "generated_text_len": len(continuation),
                        **self._memory_stats(),
                    }
                )
            except torch.OutOfMemoryError as exc:
                self._write_ttt_log(
                    {
                        "event": "request_oom",
                        "request_idx": req_idx,
                        "task_name": getattr(req, "task_name", None),
                        "doc_id": getattr(req, "doc_id", None),
                        "instance_idx": getattr(req, "idx", None),
                        "rank": self.rank,
                        "error": str(exc),
                        **self._memory_stats(),
                    }
                )
                raise
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self._write_ttt_log(
                        {
                            "event": "request_oom",
                            "request_idx": req_idx,
                            "task_name": getattr(req, "task_name", None),
                            "doc_id": getattr(req, "doc_id", None),
                            "instance_idx": getattr(req, "idx", None),
                            "rank": self.rank,
                            "error": str(exc),
                            **self._memory_stats(),
                        }
                    )
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.update(1)

        pbar.close()
        return res

    def __del__(self) -> None:
        handle = getattr(self, "_ttt_log_handle", None)
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    cli_evaluate()
