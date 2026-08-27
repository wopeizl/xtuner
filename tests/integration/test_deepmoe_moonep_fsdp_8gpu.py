"""Eight-GPU MoonEP EP8 + FSDP lifecycle and autograd smoke."""

from __future__ import annotations

import os
import typing
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist


if not hasattr(typing, "Self"):
    from typing_extensions import Self

    typing.Self = Self

from xtuner.v1.config.fsdp import FSDPConfig  # noqa: E402
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig  # noqa: E402
from xtuner.v1.module.attention import MHAConfig  # noqa: E402
from xtuner.v1.module.router import GreedyRouterConfig  # noqa: E402


WORLD_SIZE = 8

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE,
    reason="MoonEP EP8/FSDP smoke requires torchrun with eight ranks",
)


@pytest.fixture(scope="module", autouse=True)
def distributed() -> Iterator[None]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


def _config() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=2,
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=8, num_key_value_heads=8, head_dim=64),
        tie_word_embeddings=False,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
        ep_size=8,
        router=GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        ),
        dispatcher="moonep",
        moonep_sequence_length=8,
        moonep_planning_backend="auto",
        compile_cfg=False,
    )


def _run_moonep_ep8_materializes_after_fsdp_and_runs_backward() -> None:
    with torch.device("meta"):
        model = _config().build()

    runtime = model.deepmoe_runtime
    assert runtime is not None
    assert runtime.runtime_buffers == ()
    assert all(parameter.device.type == "meta" for parameter in runtime.parameters())

    model.fully_shard(FSDPConfig(ep_size=8, tp_size=1, recompute_ratio=1.0))
    model.init_weights()

    assert len(runtime.runtime_buffers) == 1
    assert len(runtime.storages) == 3
    assert len({id(bank._buffer) for bank in runtime.banks}) == 1
    assert not hasattr(model.layers["0"], "_checkpoint_wrapped_module")
    assert model.layers["0"]._checkpoint_attention is True
    assert model.layers["1"]._checkpoint_attention is False

    hidden = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    base = torch.arange(8, device="cuda", dtype=torch.int64).unsqueeze(1)
    topk_ids = torch.cat((base, base + 1), dim=1).remainder(16)
    topk_weights = torch.full((8, 2), 0.5, device="cuda", dtype=torch.float32)
    tokens_per_expert = torch.bincount(topk_ids.reshape(-1), minlength=16)

    output = runtime(0, hidden, topk_ids, topk_weights, tokens_per_expert)
    output.float().square().mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None for parameter in runtime.banks[0].parameters())
    model.close()


def test_moonep_ep8_materializes_after_fsdp_and_runs_backward() -> None:
    _run_moonep_ep8_materializes_after_fsdp_and_runs_backward()


def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise SystemExit("run with torchrun --standalone --nproc-per-node=8")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        _run_moonep_ep8_materializes_after_fsdp_and_runs_backward()
        if dist.get_rank() == 0:
            print("XTuner MoonEP EP8 + FSDP2 lifecycle/backward: PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
