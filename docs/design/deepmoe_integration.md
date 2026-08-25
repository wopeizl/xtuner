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

`dispatcher="moonep"` and `dispatcher="ultraep"` remain compatibility adapters
for the earlier MetaMoE experiments. They delegate only routed-expert execution
and are not the complete mode described above.

The specialized MoonEP phase-1 path is the fixed-shape, checkpoint-free Qwen3 MetaMoE
training envelope on EP4. Four model-owned router/expert banks are shared by
decoder layers using `layer_idx % 4`. DeepMoE owns MoonEP buffers, VMM training
storage, fp32 expert masters, forward/backward communication plans, and
collective cleanup. Decoder layers keep non-owning references so parameters have
one canonical state-dict path.

Install the optional package with `pip install 'xtuner[deepmoe]'`, or install a
local DeepMoE checkout before installing XTuner. Use `build_deepmoe_model()` or
`deepmoe_model_session()` after initializing a four-rank NCCL process group.
The regular `TrainEngine` meta/FSDP path fails before allocation because this
MoonEP phase does not support FSDP, checkpoint restore, generation, MTP,
torch.compile, fp8, activation/router offload, or intra-layer microbatch overlap.

The `moe_proj_*` fields describe a separate Q/K/V/O projection topology. They
are not consumed by the decoder-MLP runtime and do not imply that routed
attention projections are implemented.
