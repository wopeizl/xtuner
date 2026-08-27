# DeepMoE integration

`dispatcher="deepmoe"` is the complete integration mode. XTuner owns the
transformer shell—attention, normalization, residual order, embeddings and the
language-model head—while DeepMoE owns the whole MoE block:

```text
XTuner attention + post-attention norm
    -> DeepMoE router + top-k + dispatch + routed experts
       + shared expert + combine + aux/z loss
    -> XTuner residual
```

No XTuner `MoEGate`, `MoEBlock`, shared-expert MLP, or dispatcher is created in
this mode. DeepMoE parameter ownership is preserved: routed experts are local
EP shards, while router/shared parameters are replicated over EP. Under FSDP,
XTuner builds Torchtitan-style dense and sparse views over one world mesh:
`(dp_replicate, fsdp, tp)` for attention/dense parameters and
`(dp_replicate, efsdp, ep)` for the complete DeepMoE child.

The complete mode currently follows DeepMoE's native semantics (ReLU² routed
and shared experts, softmax top-k, and DeepMoE aux/z loss formulas). XTuner
router variants, SwiGLU expert weights, router bias, shared-expert gating, MTP,
generation, checkpoint restore, torch.compile, float8 and async intra-layer
microbatches are rejected rather than silently mixed into the DeepMoE path.
When FSDP activation recompute is enabled, the complete decoder layer is
checkpointed. DeepMoE aux/z losses are explicit decoder outputs, so reentrant
checkpoint replay preserves their router-gradient graph instead of publishing
no-grad tensors through module state.

`dispatcher="moonep"` and `dispatcher="ultraep"` remain compatibility adapters
for the earlier MetaMoE experiments. They delegate only routed-expert execution
and are not the complete mode described above.

The specialized MoonEP path is the fixed-shape Qwen3 MetaMoE training envelope
on EP2/EP4/EP8. Four model-owned expert banks are shared by decoder layers using
`layer_idx % 4`. DeepMoE owns MoonEP buffers, VMM training storage, fp32 expert
masters, forward/backward communication plans, and collective cleanup. The
UltraEP path uses the same model-owned bank boundary with official placement,
replica synchronization, reroute, and gradient reduction. Decoder layers keep
non-owning runtime references so parameters have one canonical state-dict path.

Both specialized paths support FSDP2/HSDP and `FSDPConfig.recompute_ratio`.
Their expert custom-autograd functions already recompute expert math during
backward, so XTuner checkpoints only attention for selected decoder layers. It
must not checkpoint the whole layer: doing so would replay stateful MoonEP
dispatch plans or UltraEP generation/placement bookkeeping. Thus attention is
executed twice, while dispatch, placement, and generation lifecycle operations
are executed once per training forward.

Install the optional package with `pip install 'xtuner[deepmoe]'`, or install a
local DeepMoE checkout before installing XTuner. Use `build_deepmoe_model()` or
`deepmoe_model_session()` after initializing the required NCCL process group.
Checkpoint restore without an explicit HF-to-MetaMoE bank converter,
generation, MTP, torch.compile, fp8, activation/router offload, FSDP CPU offload,
and intra-layer microbatch overlap are rejected before runtime allocation.

The `moe_proj_*` fields describe a separate Q/K/V/O projection topology. They
are not consumed by the decoder-MLP runtime and do not imply that routed
attention projections are implemented.
# DeepEP 2.x 与 NVIDIA NCCL EP

`dispatcher="deepmoe"` 下由 `deepmoe_transport` 选择 token transport：

- `native`：PyTorch/NCCL `all_to_all_single`，默认值；
- `deepep`：DeepEP 2.x `ElasticBuffer`，兼容旧 `Buffer`；
- `nccl_ep`：NVIDIA NCCL EP v0.1，使用 HT + FLAT 训练路径。

公共参数包括 `deepmoe_max_tokens_per_rank`、`deepmoe_num_sms`、
`deepmoe_num_channels`、`deepmoe_buffer_mib` 和
`deepmoe_transport_fallback`。可选后端在第一次 CUDA forward 时延迟构造，
因此不会破坏 XTuner 的 meta 初始化和 FSDP2 materialization 顺序。

NCCL EP 当前官方 Python 数据面只支持 CUDA 13。CUDA 12.x 环境会在所有 EP
rank 上一致失败，不会静默退回，除非显式设置
`deepmoe_transport_fallback="native"`。
