from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.integrations.deepmoe import build_deepmoe_layer, validate_deepmoe_model_config
from xtuner.v1.model.moe.moe import MoE, _checkpoint_moe_decoder_layer
from xtuner.v1.model.moe.qwen3 import Qwen3MetaMoE10BA1BConfig, Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.utils import ForwardState


class _FakeRuntime:
    backend = "moonep"

    def __init__(self):
        self.calls = []

    def __call__(self, bank_index, hidden, topk_ids, topk_weights, tokens_per_expert):
        self.calls.append((bank_index, hidden, topk_ids, topk_weights, tokens_per_expert))
        return hidden + 3


class _Generation:
    def __init__(self, calls):
        self.calls = calls

    def close(self):
        self.calls.append("close")


class _FakeUltraRuntime:
    backend = "ultraep"

    def __init__(self):
        self.calls = []

    def new_generation(self, *, layer_idx):
        self.calls.append(("generation", layer_idx))
        return _Generation(self.calls)

    def wrap_transformer_input(self, hidden, *, layer_idx, generation):
        self.calls.append(("boundary", layer_idx, generation))
        return hidden

    def __call__(
        self,
        bank_index,
        hidden,
        topk_ids,
        topk_weights,
        tokens_per_expert,
        *,
        layer_idx,
        generation,
    ):
        self.calls.append(("runtime", bank_index, layer_idx, generation))
        return hidden + 4


def _bare_deepmoe_layer(runtime):
    layer = MoEDecoderLayer.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = 5
    layer.n_shared_experts = 0
    layer.hidden_factor = 1.0
    layer.deepmoe_layer = None
    layer.shared_experts = None
    object.__setattr__(layer, "_deepmoe_runtime", runtime)
    layer._deepmoe_bank_index = 1

    router_results = {
        "logits": torch.randn(4, 8),
        "router_weights": torch.randn(4, 8),
        "topk_ids": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
        "topk_weights": torch.full((4, 2), 0.5),
        "topkens_per_expert": torch.tensor([2, 2, 2, 2]),
    }

    def pre_moe(self, *, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states), hidden_states, router_results

    def post_moe(self, *, combined_hidden_states, **_kwargs):
        return combined_hidden_states

    layer._pre_moe_forward = MethodType(pre_moe, layer)
    layer._post_moe_forward = MethodType(post_moe, layer)
    return layer, router_results


class _FakeCompleteDeepMoE(nn.Module):
    def __init__(self, *, differentiable_aux=False):
        super().__init__()
        self.router = nn.Linear(6, 8, bias=False)
        self.config = SimpleNamespace(aux_loss_coef=0.1, z_loss_coef=0.2)
        self.calls = []
        self.differentiable_aux = differentiable_aux

    def forward(self, hidden, *, token_mask, collect_router_stats):
        self.calls.append((hidden, token_mask, collect_router_stats))
        tokens = hidden.numel() // hidden.shape[-1]
        logits = torch.randn(tokens, 8)
        topk_ids = torch.tensor([[0, 1]]).expand(tokens, -1)
        topk_weights = torch.full((tokens, 2), 0.5)
        aux_loss = hidden.square().mean() if self.differentiable_aux else torch.tensor(2.0)
        z_loss = hidden.mean() if self.differentiable_aux else torch.tensor(3.0)
        aux = SimpleNamespace(
            aux_loss=aux_loss,
            z_loss=z_loss,
            router_logits=logits,
            topk_expert_ids=topk_ids,
            topk_weights=topk_weights,
        )
        return hidden + 5, aux


def test_decoder_delegates_the_complete_moe_block_to_deepmoe():
    deepmoe_layer = _FakeCompleteDeepMoE()
    layer = MoEDecoderLayer.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = 2
    layer.hidden_factor = 1.0
    layer.n_shared_experts = 1
    layer.shared_experts = None
    layer.deepmoe_layer = deepmoe_layer
    object.__setattr__(layer, "_deepmoe_runtime", None)

    def pre_moe_input(self, *, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states), hidden_states

    layer._pre_moe_input_forward = MethodType(pre_moe_input, layer)
    hidden = torch.randn(1, 4, 6)
    mask = torch.ones(1, 4, dtype=torch.bool)
    output, logits, weights, expert_ids, aux_loss, z_loss = layer._forward(
        hidden_states=hidden,
        seq_ctx=SimpleNamespace(mask=mask),
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )

    torch.testing.assert_close(output, hidden + 5)
    assert len(deepmoe_layer.calls) == 1
    called_hidden, called_mask, collect_router_stats = deepmoe_layer.calls[0]
    assert called_hidden is hidden and called_mask is mask and collect_router_stats is True
    assert logits.shape == (4, 8)
    assert weights.shape == expert_ids.shape == (4, 2)
    assert aux_loss.item() == 2.0
    assert z_loss.item() == 3.0


def test_complete_deepmoe_aux_losses_survive_reentrant_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deepmoe_layer = _FakeCompleteDeepMoE(differentiable_aux=True)
    layer = MoEDecoderLayer.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = 2
    layer.hidden_factor = 1.0
    layer.n_shared_experts = 0
    layer.shared_experts = None
    layer.deepmoe_layer = deepmoe_layer
    object.__setattr__(layer, "_deepmoe_runtime", None)

    def pre_moe_input(self, *, hidden_states, **_kwargs):
        return torch.zeros_like(hidden_states), hidden_states

    layer._pre_moe_input_forward = MethodType(pre_moe_input, layer)
    hidden = torch.randn(1, 4, 6, device=device, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)

    outputs = checkpoint(
        lambda value: layer._forward(
            hidden_states=value,
            seq_ctx=SimpleNamespace(mask=mask),
            position_embeddings=(torch.empty(0), torch.empty(0)),
        ),
        hidden,
        use_reentrant=True,
    )
    aux_loss, z_loss = outputs[-2:]
    (aux_loss + z_loss).backward()

    expected = 2 * hidden.detach() / hidden.numel() + torch.full_like(hidden, 1 / hidden.numel())
    torch.testing.assert_close(hidden.grad, expected)
    assert len(deepmoe_layer.calls) == 2


class _CountingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        position_embeddings=None,
        seq_ctx=None,
    ) -> dict[str, torch.Tensor]:
        del position_embeddings, seq_ctx
        value = hidden if hidden is not None else hidden_states
        assert value is not None
        self.calls += 1
        return {"projected_output": value.square()}


class _StatefulRuntimeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _CountingAttention()
        self._checkpoint_attention = False
        self.runtime_calls = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._checkpoint_attention:
            hidden = checkpoint(
                lambda value: self.self_attn(value)["projected_output"],
                hidden,
                use_reentrant=True,
            )
        else:
            hidden = self.self_attn(hidden)["projected_output"]
        self.runtime_calls += 1
        return hidden * 3


@pytest.mark.parametrize("backend", ["moonep", "ultraep"])
def test_specialized_deepmoe_recompute_does_not_replay_stateful_runtime(backend):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = _StatefulRuntimeLayer()
    wrapped = _checkpoint_moe_decoder_layer(layer, backend)
    hidden = torch.randn(4, device=device, requires_grad=True)

    wrapped(hidden).sum().backward()

    assert wrapped is layer
    assert layer._checkpoint_attention is True
    assert layer.self_attn.calls == 2
    assert layer.runtime_calls == 1


def test_decoder_attention_checkpoint_replays_tensor_output_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = MoEDecoderLayer.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.self_attn = _CountingAttention()
    layer._checkpoint_attention = True
    hidden = torch.randn(1, 4, 6, device=device, requires_grad=True)
    position_embeddings = (
        torch.randn(1, 4, 6, device=device),
        torch.randn(1, 4, 6, device=device),
    )

    _, moe_input = layer._pre_moe_input_forward(
        hidden_states=hidden,
        seq_ctx=SimpleNamespace(),
        position_embeddings=position_embeddings,
        state=ForwardState.TRAINING,
    )
    moe_input.sum().backward()

    torch.testing.assert_close(hidden.grad, 1 + 2 * hidden.detach())
    assert layer.self_attn.calls == 2


def test_complete_deepmoe_builder_owns_router_routed_and_shared_experts():
    pytest.importorskip("deepmoe")
    config = SimpleNamespace(
        dispatcher="deepmoe",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        ep_size=1,
        balancing_loss_cfg=SimpleNamespace(balancing_loss_alpha=0.01),
        z_loss_cfg=SimpleNamespace(z_loss_alpha=0.02),
    )

    layer = build_deepmoe_layer(config, layer_idx=3)

    assert layer.layer_idx == 3
    assert layer.router.out_features == 4
    assert layer.expert_w1.shape == (4, 8, 16)
    assert layer.shared_expert.c_fc.out_features == 16
    assert layer.config.aux_loss_coef == 0.01
    assert layer.config.z_loss_coef == 0.02
    assert layer.config.transport_provider == "native"


def test_complete_deepmoe_rejects_accelerated_transport_without_ep():
    config = SimpleNamespace(
        dispatcher="deepmoe",
        deepmoe_transport="nccl_ep",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        ep_size=1,
    )

    with pytest.raises(RuntimeError, match="nccl_ep requires ep_size > 1"):
        validate_deepmoe_model_config(config)


def test_complete_deepmoe_model_uses_deepmoe_initialization():
    pytest.importorskip("deepmoe")
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
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        dispatcher="deepmoe",
        compile_cfg=False,
    )

    model = config.build()
    model.init_weights()

    deepmoe_layer = model.layers["0"].deepmoe_layer
    assert deepmoe_layer is not None
    assert torch.isfinite(deepmoe_layer.expert_w1).all()
    assert torch.count_nonzero(deepmoe_layer.expert_w1) > 0
    assert torch.count_nonzero(deepmoe_layer.expert_w2) == 0
    model.close()


def test_complete_deepmoe_rejects_xtuner_owned_shared_expert_gate():
    config = SimpleNamespace(
        dispatcher="deepmoe",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        ep_size=1,
        with_shared_expert_gate=True,
    )

    with pytest.raises(RuntimeError, match="shared-expert gate"):
        validate_deepmoe_model_config(config)


def test_complete_deepmoe_rejects_xtuner_compile_path():
    config = SimpleNamespace(
        dispatcher="deepmoe",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        ep_size=1,
        compile_cfg=None,
    )

    with pytest.raises(RuntimeError, match="torch.compile"):
        validate_deepmoe_model_config(config)


def test_fsdp_config_accepts_torchtitan_hsdp_ep_tp_factorization():
    config = FSDPConfig(ep_size=8, tp_size=2, hsdp_sharding_size=8)

    assert (config.ep_size, config.tp_size, config.hsdp_sharding_size) == (8, 2, 8)


def test_decoder_delegates_only_routed_expert_execution_to_deepmoe():
    runtime = _FakeRuntime()
    layer, router_results = _bare_deepmoe_layer(runtime)
    hidden = torch.randn(1, 4, 6)

    output, logits, router_weights, topk_ids = layer._forward(
        hidden_states=hidden,
        seq_ctx=SimpleNamespace(),
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )

    torch.testing.assert_close(output, hidden + 3)
    assert logits is router_results["logits"]
    assert router_weights is router_results["router_weights"]
    assert topk_ids is router_results["topk_ids"]
    assert len(runtime.calls) == 1
    bank_index, routed_hidden, routed_ids, routed_weights, routed_counts = runtime.calls[0]
    assert bank_index == 1
    torch.testing.assert_close(routed_hidden, hidden.view(-1, hidden.shape[-1]))
    assert routed_ids is router_results["topk_ids"]
    assert routed_weights is router_results["topk_weights"]
    assert routed_counts is router_results["topkens_per_expert"]


def test_decoder_places_ultraep_join_before_attention_and_reuses_generation():
    runtime = _FakeUltraRuntime()
    layer, _ = _bare_deepmoe_layer(runtime)
    hidden = torch.randn(1, 4, 6)

    output, *_ = layer._forward(
        hidden_states=hidden,
        seq_ctx=SimpleNamespace(),
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )

    torch.testing.assert_close(output, hidden + 4)
    generation = runtime.calls[0]
    assert generation == ("generation", 5)
    boundary_generation = runtime.calls[1][2]
    runtime_generation = runtime.calls[2][3]
    assert boundary_generation is runtime_generation
    assert runtime.calls[-1] == "close"


def test_decoder_rejects_deepmoe_intra_layer_microbatch_before_work():
    layer, _ = _bare_deepmoe_layer(_FakeRuntime())

    with pytest.raises(RuntimeError, match="does not support async intra-layer microbatch"):
        layer._micro_batch_forward(
            hidden_states_list=[torch.empty(1, 2, 3), torch.empty(1, 2, 3)],
            seq_ctx_list=[SimpleNamespace(), SimpleNamespace()],
            position_embeddings_list=[
                (torch.empty(0), torch.empty(0)),
                (torch.empty(0), torch.empty(0)),
            ],
        )


def test_qwen3_metamoe_keeps_mlp_and_projection_topologies_distinct():
    config = Qwen3MetaMoE10BA1BConfig()

    assert config.dispatcher == "moonep"
    assert config.ep_size == 4
    assert (config.n_routed_experts, config.num_experts_per_tok) == (2048, 8)
    assert (
        config.moe_proj_n_routed_experts,
        config.moe_proj_num_experts_per_tok,
    ) == (32, 2)


def test_qwen3_metamoe_preserves_xtuner_router_and_aux_loss_configuration():
    config = Qwen3MetaMoE10BA1BConfig()

    assert config.router.scoring_func == "softmax"
    assert config.router.norm_topk_prob is True
    assert config.balancing_loss_cfg is not None
    assert config.balancing_loss_cfg.balancing_loss_alpha == 0.001
    assert config.z_loss_cfg is None


class _CleanupRuntime:
    def __init__(self):
        self.closed = False
        self.calls = []

    def close(self, *, collective=True):
        self.calls.append(collective)
        self.closed = True


def _bare_model(runtime):
    model = MoE.__new__(MoE)
    nn.Module.__init__(model)
    model._deepmoe_closed = False
    model.deepmoe_runtime = runtime
    return model


def test_model_close_releases_deepmoe_runtime_once():
    runtime = _CleanupRuntime()
    model = _bare_model(runtime)

    model.close()
    model.close()

    assert runtime.calls == [True]
    assert model._deepmoe_closed is True


def test_ultraep_scale_and_reduce_grad_normalizes_experts_and_reduces_dense(monkeypatch):
    dense = nn.Parameter(torch.ones(2))
    expert = nn.Parameter(torch.ones(2))
    dense.grad = torch.full_like(dense, 8)
    expert.grad = torch.full_like(expert, 12)

    class _Runtime:
        backend = "ultraep"

        def scale_expert_grads(self):
            expert.grad.div_(4)

    group = object()
    model = MoE.__new__(MoE)
    nn.Module.__init__(model)
    object.__setattr__(model, "deepmoe_runtime", _Runtime())
    model.ep_mesh = SimpleNamespace(size=lambda: 4, get_group=lambda: group)
    model.trainable_parameters = MethodType(
        lambda _self: [
            ("layers.0.attention.weight", dense),
            ("deepmoe_runtime.banks.0.gate_weight", expert),
        ],
        model,
    )
    reduced = []
    monkeypatch.setattr(torch.distributed, "_coalescing_manager", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda grad, *_args, **_kwargs: reduced.append(grad),
    )

    model.scale_and_reduce_grad()

    torch.testing.assert_close(expert.grad, torch.full_like(expert, 3))
    torch.testing.assert_close(dense.grad, torch.full_like(dense, 2))
    assert len(reduced) == 1 and reduced[0] is dense.grad


def test_complete_deepmoe_scale_and_reduce_uses_native_ownership(monkeypatch):
    class _DeepMoEReducer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.calls = 0

        def reduce_gradients(self):
            self.calls += 1

    reducer = _DeepMoEReducer()
    reducer.weight.grad = torch.full_like(reducer.weight, 3)
    decoder = nn.Module()
    decoder.deepmoe_layer = reducer
    dense = nn.Parameter(torch.ones(2))
    dense.grad = torch.full_like(dense, 8)

    model = MoE.__new__(MoE)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(dispatcher="deepmoe")
    model.fsdp_config = None
    model.layers = nn.ModuleDict({"0": decoder})
    model.dense = dense
    reduced = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "_coalescing_manager", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda grad, **_kwargs: reduced.append(grad))

    model.scale_and_reduce_grad()

    assert reducer.calls == 1
    torch.testing.assert_close(dense.grad, torch.full_like(dense, 2))
    assert len(reduced) == 1 and reduced[0] is dense.grad


@pytest.mark.parametrize(("raises", "expected_collective"), [(False, True), (True, False)])
def test_model_context_selects_safe_cleanup_mode(raises, expected_collective):
    runtime = _CleanupRuntime()
    model = _bare_model(runtime)

    if raises:
        with pytest.raises(RuntimeError, match="boom"):
            with model:
                raise RuntimeError("boom")
    else:
        with model:
            pass

    assert runtime.calls == [expected_collective]
