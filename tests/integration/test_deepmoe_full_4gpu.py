"""Four-GPU smoke tests for XTuner's complete DeepMoE mode."""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.integrations.deepmoe import build_deepmoe_layer
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig


WORLD_SIZE = 4

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE,
    reason="complete DeepMoE integration smoke requires torchrun with four ranks",
)


@pytest.fixture(scope="module", autouse=True)
def distributed() -> Iterator[None]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def release_cuda_cache() -> Iterator[None]:
    yield
    torch.cuda.empty_cache()


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        dispatcher="deepmoe",
        mesh_prefix="deepmoe_test",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        ep_size=2,
        deepmoe_transport=os.environ.get("DEEPMOE_TEST_TRANSPORT", "native"),
        deepmoe_max_tokens_per_rank=6,
        balancing_loss_cfg=SimpleNamespace(balancing_loss_alpha=0.01),
        z_loss_cfg=SimpleNamespace(z_loss_alpha=0.001),
    )


def test_complete_deepmoe_layer_forward_backward_and_gradient_ownership() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    layer = build_deepmoe_layer(_config(), layer_idx=0).to(device)
    hidden = torch.randn(2, 3, 8, device=device, dtype=torch.bfloat16, requires_grad=True)

    output, aux = layer(hidden, collect_router_stats=True)
    loss = output.float().square().mean()
    loss = loss + layer.config.aux_loss_coef * aux.aux_loss + layer.config.z_loss_coef * aux.z_loss
    loss.backward()
    layer.reduce_gradients()

    assert output.shape == hidden.shape
    assert aux.router_logits is not None and aux.router_logits.shape == (6, 4)
    assert aux.topk_expert_ids is not None and aux.topk_expert_ids.shape == (6, 2)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
    layer.close()


def test_torchtitan_dense_and_sparse_meshes_share_one_world_mesh() -> None:
    model = MoE.__new__(MoE)
    torch.nn.Module.__init__(model)
    model.config = _config()
    model.fsdp_config = FSDPConfig(ep_size=2, tp_size=2, hsdp_sharding_size=1)

    model._init_deepmoe_device_mesh(device="cuda", world_size=WORLD_SIZE)

    assert model.tp_mesh.size() == 2
    assert model.ep_mesh.size() == 2
    assert model.fsdp_mesh.size() == 1
    assert model.efsdp_mesh.size() == 1
    assert model.hsdp_mesh.size() == 2
    assert model.expert_hsdp_mesh.size() == 2


def test_complete_deepmoe_model_supports_nested_expert_hsdp() -> None:
    config = Qwen3MoEConfig(
        vocab_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=2, head_dim=4),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        ep_size=2,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        dispatcher="deepmoe",
        compile_cfg=False,
    )
    with torch.device("meta"):
        model = config.build()

    model.fully_shard(FSDPConfig(ep_size=2, tp_size=1, hsdp_sharding_size=2))
    model.init_weights()

    deepmoe_layer = model.layers["0"].deepmoe_layer
    assert deepmoe_layer is not None
    assert model.hsdp_mesh.size() == 4
    assert model.expert_hsdp_mesh.size() == 2
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    model.close()
