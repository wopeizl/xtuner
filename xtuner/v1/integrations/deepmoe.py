"""Lazy DeepMoE integration boundary.

In ``dispatcher="deepmoe"`` mode XTuner owns only the transformer shell
(attention, normalization, residual connections, and the language-model
head). DeepMoE owns the complete MoE block: routing, shared and routed
experts, token movement, auxiliary losses, and parallel ownership.

The older ``moonep`` and ``ultraep`` values remain specialized model-owned
runtime adapters for compatibility.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _deepmoe_api(backend: str) -> tuple[type[Any], Any]:
    try:
        if backend == "moonep":
            from deepmoe.integrations.xtuner import (
                MoonEPRuntimeGroup,
                validate_moonep_model_config,
            )

            return MoonEPRuntimeGroup, validate_moonep_model_config
        if backend == "ultraep":
            from deepmoe.integrations.xtuner_ultraep_bank import (
                UltraEPXTunerRuntime,
                validate_ultraep_model_config,
            )

            return UltraEPXTunerRuntime, validate_ultraep_model_config
        raise ValueError(f"unsupported DeepMoE backend {backend!r}")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"dispatcher={backend!r} requires the optional DeepMoE package; "
            "install XTuner with the 'deepmoe' extra or install DeepMoE from "
            "https://github.com/wopeizl/deepmoe"
        ) from exc


def validate_deepmoe_model_config(
    config: Any,
    *,
    fsdp_config: Any | None = None,
    intra_layer_micro_batch: int = 1,
    checkpoint_restore: bool = False,
) -> None:
    """Validate the selected DeepMoE phase-1 envelope before allocation."""
    backend = getattr(config, "dispatcher", None)
    if backend == "deepmoe":
        failures: list[str] = []
        if getattr(config, "float8_cfg", None) is not None:
            failures.append("float8/fp8")
        if getattr(config, "gate_bias", False):
            failures.append("router bias")
        if getattr(config, "moe_bias", False):
            failures.append("expert bias")
        if getattr(config, "with_shared_expert_gate", False):
            failures.append("shared-expert gate")
        if getattr(config, "mtp_config", None) is not None:
            failures.append("MTP")
        if getattr(config, "generate_config", None) is not None:
            failures.append("generation/decoding")
        if getattr(config, "router_async_offload", False):
            failures.append("router async offload")
        if getattr(config, "compile_cfg", False) is not False:
            failures.append("torch.compile")
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
            failures.append("XTuner activation offload")
        if intra_layer_micro_batch != 1:
            failures.append("async intra-layer microbatch overlap")
        if checkpoint_restore:
            failures.append("checkpoint restore")

        ep_size = int(getattr(config, "ep_size", 1))
        if int(getattr(config, "expert_tp_size", 1)) != 1:
            failures.append("XTuner expert_tp_size must be 1; use FSDPConfig.tp_size for dense TP")
        num_experts = int(getattr(config, "n_routed_experts", 0))
        if ep_size <= 0 or num_experts <= 0 or num_experts % max(ep_size, 1):
            failures.append("n_routed_experts must be positive and divisible by ep_size")
        if int(getattr(config, "moe_intermediate_size", 0)) <= 0:
            failures.append("moe_intermediate_size must be positive")
        if failures:
            raise RuntimeError("DeepMoE full-layer unsupported configuration: " + "; ".join(failures))
        return
    if backend not in {"moonep", "ultraep"}:
        return
    _, validate = _deepmoe_api(backend)
    validate(
        config,
        fsdp_config=fsdp_config,
        intra_layer_micro_batch=intra_layer_micro_batch,
        checkpoint_restore=checkpoint_restore,
    )


def build_deepmoe_runtime(*, backend: str, **kwargs: Any) -> Any:
    """Build the selected model-owned DeepMoE runtime."""
    runtime_type, _ = _deepmoe_api(backend)
    return runtime_type(**kwargs)


def build_deepmoe_layer(config: Any, *, layer_idx: int) -> torch.nn.Module:
    """Build a complete DeepMoE layer from an XTuner MoE configuration.

    Args:
        config (Any): XTuner MoE model configuration.
        layer_idx (int): Transformer layer index.

    Returns:
        torch.nn.Module: DeepMoE-owned router, experts, transport, and losses.
    """
    validate_deepmoe_model_config(config)
    if getattr(config, "dispatcher", None) != "deepmoe":
        raise ValueError("build_deepmoe_layer() requires dispatcher='deepmoe'")

    try:
        from deepmoe import MoEConfig as DeepMoEConfig
        from deepmoe import MoELayer
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "dispatcher='deepmoe' requires the optional DeepMoE package; "
            "install XTuner with the 'deepmoe' extra"
        ) from exc

    hidden_size = int(config.hidden_size)
    shared_width = int(config.n_shared_experts) * int(config.moe_intermediate_size)
    shared_ratio = shared_width / (4 * hidden_size) if shared_width else 0.0
    balancing_cfg = getattr(config, "balancing_loss_cfg", None)
    z_loss_cfg = getattr(config, "z_loss_cfg", None)
    deepmoe_config = DeepMoEConfig(
        model_dim=hidden_size,
        num_experts=int(config.n_routed_experts),
        top_k=int(config.num_experts_per_tok),
        expert_hidden_dim=int(config.moe_intermediate_size),
        shared_expert=shared_width > 0,
        shared_expert_width_ratio=shared_ratio,
        ep_size=int(config.ep_size),
        tp_size=1,
        aux_loss_coef=(
            float(balancing_cfg.balancing_loss_alpha) if balancing_cfg is not None else 0.0
        ),
        z_loss_coef=float(z_loss_cfg.z_loss_alpha) if z_loss_cfg is not None else 0.0,
        compute_dtype=torch.bfloat16,
    )
    return MoELayer(deepmoe_config, layer_idx=layer_idx)


def is_deepmoe_runtime(runtime: object | None) -> bool:
    """Return whether an object implements the selected DeepMoE runtime."""
    return runtime is not None and getattr(runtime, "backend", None) in {"moonep", "ultraep"} and callable(runtime)
