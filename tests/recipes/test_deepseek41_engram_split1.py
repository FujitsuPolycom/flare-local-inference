# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from tests.recipes.test_deepseek41_engram_attention_owner import construct
from tests.recipes.test_deepseek41_outgoing_owner import declaration


@pytest.mark.parametrize("owner", [False, True])
def test_engram_mhc_declaration_uses_the_reference_config_only_for_owned_rows(monkeypatch, declaration, owner):
    from ignition.components import mhc as component
    from ignition.components.mhc import MHCWeights, MHCScratch

    kwargs = {} if owner else dict(row_shard=None, output_rows=None, engram_input=None)
    layer = construct(monkeypatch, declaration, **kwargs)
    rows, hidden = (2048 if owner else 8192), 5120
    weights = MHCWeights(torch.empty(24, 4*hidden), torch.empty(3), torch.empty(24),
                         torch.empty(hidden, dtype=torch.bfloat16))
    # Collect real b12x requests without executing a CUDA prepare callback.
    monkeypatch.setattr(component, "declare", lambda session, requests: None)
    def operand(shape, dtype):
        return torch.empty((), dtype=dtype).expand(shape)
    residual = operand((rows,4,hidden), torch.bfloat16)
    mixed = operand((rows,4), torch.float32)
    block = operand((rows,hidden), torch.bfloat16)
    sub = operand((rows,4,4), torch.float32)
    for leaf, inputs, operation in [(layer.attention_hc,(residual,),"pre"), (layer.ffn_hc,(block,residual,mixed,sub),"post_pre")]:
        config = leaf.options.get("tuning_config")
        assert (config is not None) == owner
        if owner:
            expected = (128,32,128,2,8) if operation=="pre" else (192,32,64,3,12)
            assert (config.projection_tile_m,config.projection_tile_n,config.projection_tile_k,
                    config.projection_num_stages,config.projection_num_m_warps)==expected
            assert config.backend=="tf32_tma" and config.projection_k_splits==1
            assert config.projection_num_n_warps==1 and not config.lagged_prepare and config.partials_per_cta==4
        actual = component.PreparedMHC(inputs,weights,MHCScratch.allocate(rows,hidden,"meta"),
            session=None,name="component",workspace=None,rms_eps=1e-20,norm_eps=1e-20,
            hc_eps=1e-6,sinkhorn_iters=20,pre_mix=mixed,pre_out=mixed,tuning_config=config)
        assert actual.plan.query.max_tokens==rows and actual.plan.query.operation==operation
        assert actual.plan.override==config
    assert not torch.cuda.is_initialized()
