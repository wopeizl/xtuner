"""Four-GPU DeepMoE MetaMoE decoder correctness smoke."""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from xtuner.v1.model import Qwen3MetaMoE10BA1BConfig, deepmoe_model_session
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
from xtuner.v1.module.dispatcher import build_dispatcher


WORLD_SIZE = 4
SEQUENCE_LENGTH = 8
NUM_EXPERTS = int(os.environ.get("DEEPMOE_TEST_NUM_EXPERTS", "2048"))
TOP_K = 8
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE,
    reason="DeepMoE MoonEP integration requires torchrun with four ranks",
)


def _config() -> Qwen3MetaMoE10BA1BConfig:
    return Qwen3MetaMoE10BA1BConfig(
        vocab_size=128,
        num_hidden_layers=8,
        max_window_layers=8,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        moe_intermediate_size=INTERMEDIATE_SIZE,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moonep_sequence_length=SEQUENCE_LENGTH,
        attention=MHAConfig(
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=HIDDEN_SIZE // 2,
            dropout=0.0,
            qkv_bias=False,
            qk_norm=True,
            o_bias=False,
            sliding_window=None,
        ),
    )


def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def _oracle(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden, dtype=torch.float32)
    hidden_float = hidden.float()
    for slot in range(TOP_K):
        for expert in torch.unique(topk_ids[:, slot]).tolist():
            selected = topk_ids[:, slot] == expert
            x = hidden_float[selected]
            projected = (torch.nn.functional.silu(x @ gate[expert]) * (x @ up[expert])) @ down[expert]
            output[selected] += projected * topk_weights[selected, slot, None]
    return output


def _xtuner_all2all_reference(
    model,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    bank,
) -> torch.Tensor:
    with torch.device(hidden.device):
        experts = MoEBlock(
            hidden_size=HIDDEN_SIZE,
            moe_intermediate_size=INTERMEDIATE_SIZE,
            n_routed_experts=NUM_EXPERTS,
            ep_mesh=model.ep_mesh,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
    experts.to(dtype=hidden.dtype)
    fused_w1w3 = experts.fused_w1w3.weight.to_local()
    fused_w2 = experts.fused_w2.weight.to_local()
    with torch.no_grad():
        gate_up = torch.cat(
            (bank.gate_weight.transpose(1, 2), bank.up_weight.transpose(1, 2)),
            dim=1,
        )
        fused_w1w3.copy_(gate_up.reshape_as(fused_w1w3).to(fused_w1w3.dtype))
        fused_w2.copy_(bank.down_weight.transpose(1, 2).reshape_as(fused_w2).to(fused_w2.dtype))

    dispatcher = build_dispatcher(
        dispatcher="all2all",
        n_routed_experts=NUM_EXPERTS,
        ep_group=model.ep_mesh.get_group(),
    )
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids)
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre_dispatched,
        topk_weights=topk_weights,
        decoding=False,
    )
    post_dispatched = dispatcher.dispatch_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
    )
    expert_output = experts(
        post_dispatched["hidden_states"],
        post_dispatched["tokens_per_expert"],
        decoding=False,
    )
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=expert_output,
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        decoding=False,
    )
    combined = dispatcher.combine(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        decoding=False,
    )
    return dispatcher.combine_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        combined=combined,
    )["hidden_states"]


def test_deepmoe_decoder_matches_replicated_swiglu_oracle_and_closes():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    runtime = None
    completed = False

    try:
        with deepmoe_model_session(_config(), device=device) as model:
            runtime = model.deepmoe_runtime
            assert runtime is not None
            assert [model.layers[str(i)]._deepmoe_bank_index for i in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
            master_names = [name for name, _ in model.named_parameters() if name.startswith("deepmoe_runtime.banks.")]
            assert len(master_names) == 12
            assert all("layers." not in name for name in master_names)
            assert {parameter.dtype for name, parameter in model.named_parameters() if name in master_names} == {
                torch.float32
            }

            layer = model.layers["0"]
            token = torch.arange(SEQUENCE_LENGTH, device=device)
            expert_stride = NUM_EXPERTS // TOP_K
            topk_ids = token[:, None] + torch.arange(TOP_K, device=device)[None, :] * expert_stride
            topk_ids = topk_ids.remainder(NUM_EXPERTS).to(torch.int64)
            topk_weights = torch.arange(TOP_K, 0, -1, device=device, dtype=torch.float32)
            topk_weights = (topk_weights / topk_weights.sum()).expand(SEQUENCE_LENGTH, -1).clone()
            topk_weights.requires_grad_(True)
            tokens_per_expert = torch.bincount(topk_ids.flatten(), minlength=NUM_EXPERTS)
            router_results = {
                "logits": torch.zeros(SEQUENCE_LENGTH, NUM_EXPERTS, device=device),
                "router_weights": torch.zeros(SEQUENCE_LENGTH, NUM_EXPERTS, device=device),
                "topk_ids": topk_ids,
                "topk_weights": topk_weights,
                "topkens_per_expert": tokens_per_expert,
            }

            def pre_moe(self, *, hidden_states, **_kwargs):
                return torch.zeros_like(hidden_states), hidden_states, router_results

            def post_moe(self, *, combined_hidden_states, **_kwargs):
                return combined_hidden_states

            layer._pre_moe_forward = MethodType(pre_moe, layer)
            layer._post_moe_forward = MethodType(post_moe, layer)
            hidden = torch.randn(
                1,
                SEQUENCE_LENGTH,
                HIDDEN_SIZE,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output, returned_logits, returned_router_weights, returned_ids = layer._forward(
                hidden_states=hidden,
                seq_ctx=SimpleNamespace(),
                position_embeddings=(torch.empty(0, device=device), torch.empty(0, device=device)),
            )

            bank = runtime.banks[0]
            expected = _oracle(
                hidden.detach().view(SEQUENCE_LENGTH, HIDDEN_SIZE),
                topk_ids,
                topk_weights.detach(),
                _all_gather(bank.gate_weight.detach()),
                _all_gather(bank.up_weight.detach()),
                _all_gather(bank.down_weight.detach()),
            )
            torch.testing.assert_close(output.view_as(expected).float(), expected, atol=5e-2, rtol=5e-2)
            original_xtuner = _xtuner_all2all_reference(
                model,
                hidden.detach().view(SEQUENCE_LENGTH, HIDDEN_SIZE),
                topk_ids,
                topk_weights.detach(),
                bank,
            )
            torch.testing.assert_close(
                output.view_as(original_xtuner).float(),
                original_xtuner.float(),
                atol=5e-2,
                rtol=5e-2,
            )
            assert returned_logits is router_results["logits"]
            assert returned_router_weights is router_results["router_weights"]
            assert returned_ids is topk_ids

            output.float().square().mean().backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
            assert topk_weights.grad is not None and torch.isfinite(topk_weights.grad).all()
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in bank.parameters()
            )
            assert bank.last_forward_plan_id == bank.last_backward_plan_id
            assert runtime.trace[:11] == [
                "dispatch(new plan)",
                "prefetch",
                "expert",
                "combine",
                "dispatch(saved plan)",
                "prefetch(saved plan)",
                "expert backward",
                "combine activation/route gradients",
                "reduce_grad",
                "synchronize",
                "expose owned fp32 grads",
            ]
            completed = True
    finally:
        if runtime is not None:
            assert runtime.closed
            if completed:
                assert runtime.trace.count("buffer.destroy") == 4
            else:
                assert runtime.trace.count("buffer.abandon") == 4
            assert runtime.trace.count("storage.close") == 12
        dist.destroy_process_group()
