#!/usr/bin/env python3
from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Optional

import custom_models  # noqa: F401
import torch
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text
from lm_eval.models.utils_hf import get_dtype
from packaging.version import parse as vparse
from tqdm import tqdm

from run_ttt_lora import (
    BETAS,
    CHUNK_SIZE,
    GRAD_CLIP,
    LR,
    LORA_ALPHA,
    LORA_R,
    NUM_QA_CANDIDATES,
    NUM_JUDGE_CANDIDATES,
    NUM_SELECTED_QA,
    QA_GENERATION_MAX_NEW_TOKENS,
    QA_JUDGE_MAX_NEW_TOKENS,
    TOP_LAYER_FRACTION,
    UPDATE_MODE,
    WEIGHT_DECAY,
    SampleSpec,
    adapt_and_generate_for_sample,
    attach_ttt_lora,
    clone_trainable_state,
    get_task_family,
    get_trainable_lora_parameters,
    should_prefer_flash_attention_2,
)


DEFAULT_SAMPLES_BASE = Path(
    "/work/yufei/projects/flame/results/"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4/"
    "lb/__storage__backup__yufei__ttt__flame__exp__"
    "20260322_hybrid_qwen3_lact_0p6B_swa_2k_chunk_1k_rerun12_prolong_prolong_from_run12_step9535_v4"
)


eval_logger = logging.getLogger(__name__)


def _load_legacy_ttt_module():
    legacy_path = Path("/work/yufei/projects/flame/scripts/20260330_ttt_lora/run_ttt_lora.py")
    spec = importlib.util.spec_from_file_location("legacy_ttt_lora_run", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy TTT-LoRA module from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_TTT = _load_legacy_ttt_module()


def _normalize_task_question(doc: dict[str, Any], task_name: Optional[str]) -> str:
    question = str(doc.get("question") or "").strip()
    if question:
        return question
    dataset = str(doc.get("dataset") or task_name or "")
    if dataset in {"gov_report", "qmsum", "multi_news", "vcsum", "samsum"}:
        return "Summarize the full document faithfully."
    if dataset in {"lcc", "repobench-p"}:
        return "Complete the code consistently with the repository context."
    return "Answer the final benchmark task using information from the document."


def _render_truncated_text(tokenizer, text: str, max_ctx_len: int, device: torch.device) -> tuple[str, torch.LongTensor]:
    encoded = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][:, -max_ctx_len:].to(device)
    rendered = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return rendered, input_ids


@register_model("hf-test-time-distill", "hf_test_time_distill")
class HFTestTimeDistillLM(HFLM):
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
        ttt_update_mode: str = UPDATE_MODE,
        ttt_lora_r: int = LORA_R,
        ttt_lora_alpha: int = LORA_ALPHA,
        ttt_lr: float = LR,
        ttt_beta1: float = BETAS[0],
        ttt_beta2: float = BETAS[1],
        ttt_weight_decay: float = WEIGHT_DECAY,
        ttt_grad_clip: float = GRAD_CLIP,
        ttt_num_qa_candidates: int = NUM_QA_CANDIDATES,
        ttt_num_judge_candidates: int = NUM_JUDGE_CANDIDATES,
        ttt_num_selected_qa: int = NUM_SELECTED_QA,
        ttt_qa_generation_max_new_tokens: int = QA_GENERATION_MAX_NEW_TOKENS,
        ttt_qa_judge_max_new_tokens: int = QA_JUDGE_MAX_NEW_TOKENS,
        ttt_log_path: str | None = None,
        ttt_longbench_samples_base: str | None = None,
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
        self.ttt_update_mode = str(ttt_update_mode)
        self.ttt_lora_r = int(ttt_lora_r)
        self.ttt_lora_alpha = int(ttt_lora_alpha)
        self.ttt_lr = float(ttt_lr)
        self.ttt_betas = (float(ttt_beta1), float(ttt_beta2))
        self.ttt_weight_decay = float(ttt_weight_decay)
        self.ttt_grad_clip = float(ttt_grad_clip)
        self.ttt_num_qa_candidates = int(ttt_num_qa_candidates)
        self.ttt_num_judge_candidates = int(ttt_num_judge_candidates)
        self.ttt_num_selected_qa = int(ttt_num_selected_qa)
        self.ttt_qa_generation_max_new_tokens = int(ttt_qa_generation_max_new_tokens)
        self.ttt_qa_judge_max_new_tokens = int(ttt_qa_judge_max_new_tokens)
        self.ttt_log_path = ttt_log_path
        self.ttt_longbench_samples_base = Path(ttt_longbench_samples_base) if ttt_longbench_samples_base else DEFAULT_SAMPLES_BASE
        self.trainable_lora_params = get_trainable_lora_parameters(self.model)
        self.initial_lora_state = clone_trainable_state(self.trainable_lora_params) if self.ttt_enable else {}
        self._ttt_log_handle = open(ttt_log_path, "a", encoding="utf-8") if ttt_log_path else None
        self._sample_cache: dict[tuple[str, int], dict[str, Any]] = {}

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

    def _load_longbench_sample(self, task_name: str, doc_id: int) -> Optional[dict[str, Any]]:
        key = (task_name, int(doc_id))
        if key in self._sample_cache:
            return self._sample_cache[key]

        candidates = sorted(self.ttt_longbench_samples_base.glob(f"samples_{task_name}_*.jsonl"))
        if not candidates:
            return None
        sample_path = candidates[-1]
        with sample_path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                if int(obj["doc_id"]) == int(doc_id):
                    self._sample_cache[key] = obj
                    return obj
        return None

    def _legacy_ntp_adapt(
        self,
        input_ids: torch.LongTensor,
    ):
        reset_trainable_state = LEGACY_TTT.reset_trainable_state
        rebuild_cache_for_prefix = LEGACY_TTT.rebuild_cache_for_prefix
        run_chunk_update_step = LEGACY_TTT.run_chunk_update_step
        compute_delta_norm = LEGACY_TTT.compute_delta_norm

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
            prefix_cache_for_chunk = None if start == 0 or last_rebuild is None else last_rebuild.past_key_values
            loss, grad_norm = run_chunk_update_step(
                model=self.model,
                optimizer=optimizer,
                trainable_lora_params=self.trainable_lora_params,
                full_input_ids=input_ids,
                chunk_start=start,
                chunk_end=end,
                chunk_size=self.ttt_chunk_size,
                update_mode="full_prefix_approx",
                local_train_window=2048,
                loss_mode="topk_fraction",
                loss_topk_fraction=0.2,
                base_prefix_cache=prefix_cache_for_chunk,
            )
            last_rebuild = rebuild_cache_for_prefix(
                self.model,
                input_ids[:, :end],
                chunk_size=self.ttt_chunk_size,
            )
            logs.append(
                {
                    "event": "chunk_update",
                    "adaptation_method": "legacy_ntp_ttt_lora",
                    "chunk_idx": chunk_idx,
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

        if last_rebuild is None:
            last_rebuild = rebuild_cache_for_prefix(
                self.model,
                input_ids,
                chunk_size=self.ttt_chunk_size,
            )
        return last_rebuild, logs

    def _build_sample_spec(
        self,
        rendered_prompt: str,
        req,
        max_ctx_len: int,
    ) -> SampleSpec:
        task_name = getattr(req, "task_name", None)
        doc_id = getattr(req, "doc_id", None)
        sample_obj = None
        if task_name is not None and doc_id is not None:
            sample_obj = self._load_longbench_sample(str(task_name), int(doc_id))

        if sample_obj is not None:
            doc = sample_obj["doc"]
            final_prompt, _ = _render_truncated_text(self.tokenizer, rendered_prompt, max_ctx_len=max_ctx_len, device=self.device)
            adaptation_text, _ = _render_truncated_text(
                self.tokenizer,
                str(doc.get("context") or rendered_prompt),
                max_ctx_len=max_ctx_len,
                device=self.device,
            )
            return SampleSpec(
                final_prompt=final_prompt,
                adaptation_text=adaptation_text,
                task_question=_normalize_task_question(doc, task_name=str(task_name)),
                metadata={
                    "task_name": doc.get("task") or doc.get("dataset") or task_name,
                    "doc_id": doc_id,
                    "sample_source": "longbench_doc",
                },
            )

        final_prompt, _ = _render_truncated_text(self.tokenizer, rendered_prompt, max_ctx_len=max_ctx_len, device=self.device)
        return SampleSpec(
            final_prompt=final_prompt,
            adaptation_text=final_prompt,
            task_question="Answer the final benchmark task using information from the prompt.",
            metadata={
                "task_name": task_name,
                "doc_id": doc_id,
                "sample_source": "rendered_prompt_fallback",
            },
        )

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        if self.batch_size_per_gpu != 1:
            raise ValueError("hf-test-time-distill currently supports batch_size=1 only.")

        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with test-time distillation",
        )
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        for req_idx, req in enumerate(requests):
            context, gen_kwargs = req.args
            kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = kwargs.pop("max_gen_toks")

            if self.backend != "causal":
                raise NotImplementedError("hf-test-time-distill only supports causal models.")

            max_ctx_len = self.max_length - max_gen_toks
            sample = self._build_sample_spec(context, req=req, max_ctx_len=max_ctx_len)
            task_family = get_task_family(str(sample.metadata.get("task_name") or ""))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            do_sample = bool(kwargs.pop("do_sample", False))
            temperature = float(kwargs.pop("temperature", 1.0))
            top_p = float(kwargs.pop("top_p", 1.0))
            if kwargs:
                eval_logger.warning(f"Unused generation kwargs for hf-test-time-distill: {sorted(kwargs.keys())}")

            self._write_ttt_log(
                {
                    "event": "request_start",
                    "request_idx": req_idx,
                    "task_name": getattr(req, "task_name", None),
                    "doc_id": getattr(req, "doc_id", None),
                    "instance_idx": getattr(req, "idx", None),
                    "sample_source": sample.metadata.get("sample_source"),
                    "task_family": task_family,
                    "final_prompt_char_len": len(sample.final_prompt),
                    "adaptation_char_len": len(sample.adaptation_text),
                    "task_question": sample.task_question,
                    "max_ctx_len": max_ctx_len,
                    "max_gen_toks": max_gen_toks,
                    "rank": self.rank,
                    **self._memory_stats(),
                }
            )

            try:
                if self.ttt_enable:
                    if task_family == "qa":
                        result = adapt_and_generate_for_sample(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            sample=sample,
                            sample_index=req_idx,
                            chunk_size=self.ttt_chunk_size,
                            update_mode=self.ttt_update_mode,
                            lr=self.ttt_lr,
                            num_qa_candidates=self.ttt_num_qa_candidates,
                            num_judge_candidates=self.ttt_num_judge_candidates,
                            num_selected_qa=self.ttt_num_selected_qa,
                            qa_generation_max_new_tokens=self.ttt_qa_generation_max_new_tokens,
                            qa_judge_max_new_tokens=self.ttt_qa_judge_max_new_tokens,
                            max_new_tokens=max_gen_toks,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            initial_lora_state=self.initial_lora_state,
                            trainable_lora_params=self.trainable_lora_params,
                            log_file=self._ttt_log_handle,
                        )
                        continuation = result["generated_text"]
                    else:
                        prompt_ids, _ = self.tok_batch_encode(
                            [sample.final_prompt],
                            left_truncate_len=max_ctx_len,
                            truncation=self.truncation,
                        )
                        prompt_ids = prompt_ids.to(self.device)
                        rebuild, logs = self._legacy_ntp_adapt(prompt_ids)
                        for chunk_log in logs:
                            self._write_ttt_log(
                                {
                                    "request_idx": req_idx,
                                    "rank": self.rank,
                                    "task_name": getattr(req, "task_name", None),
                                    "doc_id": getattr(req, "doc_id", None),
                                    **chunk_log,
                                    **self._memory_stats(),
                                }
                            )
                        cont_toks = LEGACY_TTT.generate_from_adapted_state(
                            model=self.model,
                            prefix_ids=prompt_ids,
                            past_key_values=rebuild.past_key_values,
                            last_logits=rebuild.last_logits,
                            max_new_tokens=max_gen_toks,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            eos_token_id=self.eot_token_id,
                        )
                        continuation = self.tok_decode(cont_toks[0].tolist())
                else:
                    context_enc, _ = self.tok_batch_encode(
                        [sample.final_prompt],
                        left_truncate_len=max_ctx_len,
                        truncation=self.truncation,
                    )
                    context_enc = context_enc.to(self.device)
                    from run_ttt_lora import rebuild_cache_for_prefix, generate_from_adapted_state

                    rebuild = rebuild_cache_for_prefix(self.model, context_enc, chunk_size=self.ttt_chunk_size)
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
