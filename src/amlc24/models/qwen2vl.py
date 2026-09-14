"""Load Qwen2-VL-7B-Instruct as a quantised LoRA model sized for a 16 GB T4.

Constraints this module encodes, all of them Turing-specific:

* **fp16, never bf16.** T4 (SM 7.5) has no bf16 tensor cores. Requesting bf16
  either errors or silently falls back to a slow path.
* **``attn_implementation="sdpa"``.** flash-attention-2 requires Ampere or
  newer. SDPA gives memory-efficient attention on Turing.
* **``max_pixels`` is the real memory knob.** Qwen2-VL's dynamic resolution
  produces one visual token per 28x28 patch *after* its own resize, so a default
  configuration can emit thousands of tokens per image and OOM instantly. At
  ``256 * 28 * 28`` each image costs ~256 visual tokens, which is what makes a
  7B model fit here at all.
* **Vision tower frozen, LoRA on the LM's attention projections only.** The
  vision encoder already reads text in images well; what the model lacks is the
  output *format*, which lives in the language model.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
PATCH_AREA = 28 * 28  # one visual token per 28x28 patch


def _torch():
    import torch
    return torch


def pin_single_gpu(device: int = 0) -> bool:
    """Restrict the session to one GPU. Call **before** torch initialises CUDA.

    Kaggle's default accelerator is **T4 x2**, so ``torch.cuda.device_count()``
    is 2. HF ``Trainer`` reacts to that by wrapping the model in
    ``nn.DataParallel``, which replicates it across both devices every step.
    That is wrong here for two reasons:

    * the model is loaded with ``device_map={"": 0}``, i.e. already pinned to
      GPU 0, and DataParallel replication of an 8-bit bitsandbytes model either
      raises or silently degrades;
    * a 7B model at 8-bit with ``max_pixels=256*28*28`` fits in one 16 GB T4, so
      there is nothing to gain from splitting it.

    Returns True if the variable was set, False if CUDA was already initialised
    (in which case the caller must restart the kernel for it to take effect).
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info("CUDA_VISIBLE_DEVICES already set to %r; leaving it alone",
                    os.environ["CUDA_VISIBLE_DEVICES"])
        return False

    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        logger.warning(
            "CUDA is already initialised, so pinning to one GPU has no effect. "
            "Restart the kernel and call pin_single_gpu() before importing torch "
            "if Trainer reports n_gpu > 1."
        )
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    logger.info("Pinned session to GPU %d (avoids Trainer's DataParallel path)", device)
    return True


def gpu_report() -> dict[str, Any]:
    """Device name, VRAM, capability, and whether this GPU is the expected T4."""
    torch = _torch()
    if not torch.cuda.is_available():
        return {"cuda": False, "device_name": None, "is_t4": False, "supports_bf16": False}

    props = torch.cuda.get_device_properties(0)
    name = props.name
    major, minor = props.major, props.minor
    report = {
        "cuda": True,
        "device_name": name,
        "device_count": torch.cuda.device_count(),
        "total_vram_gb": round(props.total_memory / 1024 ** 3, 2),
        "capability": f"{major}.{minor}",
        "is_t4": "T4" in name,
        "is_p100": "P100" in name,
        "supports_bf16": major >= 8,
        "supports_flash_attn2": major >= 8,
    }
    logger.info(
        "GPU: %s x%d, %.1f GB, SM %s (bf16=%s, flash-attn2=%s)",
        name, report["device_count"], report["total_vram_gb"], report["capability"],
        report["supports_bf16"], report["supports_flash_attn2"],
    )
    if report["is_p100"]:
        logger.warning(
            "P100 detected. It is slower than T4 for fp16 (no tensor cores for "
            "this workload) and the run may exceed the 12-hour session limit. "
            "Prefer a T4 x2 accelerator."
        )
    if report["device_count"] > 1:
        logger.warning(
            "%d GPUs visible. HF Trainer will wrap the model in DataParallel, "
            "which breaks an 8-bit model already pinned to GPU 0. Call "
            "pin_single_gpu() before importing torch (and restart the kernel).",
            report["device_count"],
        )
    return report


def build_quantization_config(bits: int = 8) -> Any:
    """``BitsAndBytesConfig`` for 8-bit or 4-bit loading.

    8-bit is the default: it keeps more fidelity and fits within the estimated
    13-14 GB budget. 4-bit (NF4, double-quantised) is the OOM escape hatch and
    saves roughly 3.5-4 GB of weight memory.
    """
    from transformers import BitsAndBytesConfig

    torch = _torch()
    if bits == 8:
        logger.info("Quantisation: 8-bit (LLM.int8)")
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            # The vision tower stays in fp16: int8 on the ViT costs accuracy for
            # very little memory, since it is a small fraction of the weights.
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    if bits == 4:
        logger.info("Quantisation: 4-bit (NF4, double quant, fp16 compute)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(f"quantization.bits must be 8 or 4, got {bits!r}")


def load_processor(
    model_id: str = DEFAULT_MODEL_ID,
    min_pixels_tokens: int = 64,
    max_pixels_tokens: int = 256,
) -> Any:
    """Load ``AutoProcessor`` with an explicit visual-token budget.

    Token counts are converted to pixel areas here so the config can speak in
    the unit that actually matters (visual tokens per image).
    """
    from transformers import AutoProcessor

    min_pixels = int(min_pixels_tokens) * PATCH_AREA
    max_pixels = int(max_pixels_tokens) * PATCH_AREA

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token was unset; using eos_token")

    logger.info(
        "Processor: min_pixels=%d (%d tokens), max_pixels=%d (%d tokens)",
        min_pixels, min_pixels_tokens, max_pixels, max_pixels_tokens,
    )
    return processor


def prepare_quantized_for_training(model: Any, use_gradient_checkpointing: bool = True) -> Any:
    """Freeze the base model and enable checkpointing, WITHOUT upcasting to fp32.

    Replaces peft's ``prepare_model_for_kbit_training``, which casts every
    non-int8 parameter to fp32. Here that means the fp16 vision tower, the
    embeddings and ``lm_head`` (~2.2B params) double in size -- about 4 GB extra,
    which OOM'd a 14.56 GiB T4. Those weights are frozen, so fp32 buys nothing;
    fp16 autocast covers the forward pass.
    """
    for param in model.parameters():
        param.requires_grad = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # With a frozen embedding layer, checkpointed segments would otherwise
        # have no input that requires grad, and LoRA would receive no gradient.
        model.enable_input_require_grads()
    return model


def count_parameters(model: Any) -> dict[str, int | float]:
    """Trainable vs total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stats = {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }
    logger.info(
        "Parameters: %s trainable / %s total (%.4f%%)",
        f"{trainable:,}", f"{total:,}", stats["trainable_pct"],
    )
    return stats


def freeze_vision_tower(model: Any) -> int:
    """Freeze the ViT. Returns how many tensors were frozen.

    Qwen2-VL exposes the encoder as ``model.visual``; we match by name so the
    merger/projector inside it is frozen too.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if "visual" in name.split(".") or name.startswith("visual."):
            if param.requires_grad:
                param.requires_grad = False
            frozen += 1
    logger.info("Froze %d vision-tower parameter tensor(s)", frozen)
    return frozen


def build_lora_config(lora_cfg: Any) -> Any:
    """PEFT ``LoraConfig`` from the ``lora:`` config block."""
    from peft import LoraConfig

    target_modules = list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        # Without this the adapter would also wrap the vision tower's attention
        # projections, which share these names and which we want frozen.
        modules_to_save=None,
    )
    logger.info(
        "LoRA: r=%d alpha=%d dropout=%.3f targets=%s",
        config.r, config.lora_alpha, config.lora_dropout, target_modules,
    )
    return config


def _exclude_vision_targets(model: Any, target_modules: list[str]) -> list[str]:
    """Expand target names to explicit module paths outside the vision tower.

    ``q_proj`` matches in both the ViT and the LM. PEFT matches on the suffix,
    so we hand it fully-qualified names from the language model only -- that is
    what "LoRA on language-model attention projections only" requires.
    """
    names = [
        name for name, module in model.named_modules()
        if any(name.endswith(f".{t}") or name == t for t in target_modules)
        and "visual" not in name.split(".")
    ]
    logger.info(
        "Resolved %d LoRA target module(s) in the language model (vision tower excluded)",
        len(names),
    )
    return names


def load_model(
    cfg: Any,
    processor: Any | None = None,
    for_training: bool = True,
) -> tuple[Any, Any]:
    """Load the quantised base model and attach the LoRA adapter.

    Returns ``(model, processor)``.
    """
    from transformers import Qwen2VLForConditionalGeneration
    from peft import get_peft_model

    torch = _torch()
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id", DEFAULT_MODEL_ID)
    attn = model_cfg.get("attn_implementation", "sdpa")
    bits = int(cfg.get("quantization", {}).get("bits", 8))

    report = gpu_report()
    if attn == "flash_attention_2" and not report.get("supports_flash_attn2", False):
        logger.warning("flash_attention_2 unsupported on this GPU; falling back to sdpa")
        attn = "sdpa"

    proc_cfg = cfg.get("processor", {})
    if processor is None:
        processor = load_processor(
            model_id,
            int(proc_cfg.get("min_pixels_tokens", 64)),
            int(proc_cfg.get("max_pixels_tokens", 256)),
        )

    logger.info("Loading %s (%d-bit, attn=%s, fp16)", model_id, bits, attn)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(bits),
        torch_dtype=torch.float16,
        attn_implementation=attn,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not for_training

    if not for_training:
        count_parameters(model)
        return model, processor

    grad_ckpt = bool(cfg.get("train", {}).get("gradient_checkpointing", True))
    model = prepare_quantized_for_training(model, use_gradient_checkpointing=grad_ckpt)

    lora_cfg = cfg.get("lora", {})
    if bool(lora_cfg.get("freeze_vision_tower", True)):
        freeze_vision_tower(model)

    peft_config = build_lora_config(lora_cfg)
    peft_config.target_modules = _exclude_vision_targets(
        model, list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    )
    model = get_peft_model(model, peft_config)

    if bool(lora_cfg.get("freeze_vision_tower", True)):
        freeze_vision_tower(model)  # re-assert after PEFT wrapping

    # LoRA weights must be fp32: fp16 trainable params make the fp16 GradScaler
    # raise "Attempting to unscale FP16 gradients". They are tiny (~20M params).
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()

    count_parameters(model)
    if torch.cuda.is_available():
        logger.info("VRAM after model load: %.2f GB allocated",
                    torch.cuda.memory_allocated() / 1024 ** 3)
    return model, processor


def load_for_inference(cfg: Any, adapter_path: str | None = None) -> tuple[Any, Any]:
    """Load the base model and, when given, merge in a trained adapter."""
    from peft import PeftModel

    model, processor = load_model(cfg, for_training=False)
    if adapter_path:
        logger.info("Attaching LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model, processor


__all__ = [
    "load_model", "load_processor", "load_for_inference", "gpu_report",
    "build_quantization_config", "build_lora_config", "count_parameters",
    "freeze_vision_tower", "pin_single_gpu", "prepare_quantized_for_training",
    "DEFAULT_MODEL_ID", "PATCH_AREA",
]
