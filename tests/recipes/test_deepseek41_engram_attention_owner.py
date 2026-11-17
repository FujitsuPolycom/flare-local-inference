# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Normal-constructor checks for owner mHC around full-row attention.

GPU leaves record declarations. These checks cover row ownership, selection
wiring and fallback; they do not establish CUDA arithmetic or replay behavior.
"""
from dataclasses import dataclass,replace
from types import SimpleNamespace

import pytest
import torch
from tests.recipes.test_deepseek41_outgoing_owner import declaration,leaves,layer_weights,text,Leaf


def construct(monkeypatch,declaration,**changes):
    from ignition.recipes.deepseek41 import decoder
    leaves(monkeypatch,decoder)
    owner=declaration.owner
    options=dict(shard=SimpleNamespace(world_size=4),mode="prefill",runtime=declaration.runtime,
        coordinator=None,session=None,name="layer1",workspace=SimpleNamespace(overlap=False),
        carry=False,defer_post=False,row_shard=owner,attention_shard=False,
        engram_input=(owner,None),output_rows=owner,reseed=False)
    options.update(changes)
    return decoder.PreparedLayer(1,layer_weights(),None,declaration.scratch,declaration.metadata,
        SimpleNamespace(config=SimpleNamespace(text=text())),None,**options)


@pytest.mark.parametrize("rank",range(4))
def test_normal_constructor_keeps_mhc_owned_and_attention_full(monkeypatch,declaration,rank):
    from ignition.recipes.deepseek41 import decoder
    owner=replace(declaration.owner,rank=rank,start=rank*2048);declaration.owner=owner
    move=decoder.SelectionMove(14,"full_to_shard")
    layer=construct(monkeypatch,declaration,selection_move=move)
    full=declaration.scratch
    assert layer.row_shard is owner and not layer.attention_shard and not layer.reseed
    assert layer.engram.args[3] is owner.residual and layer.engram.args[4].shape==(2048,4,5120)
    assert layer.engram.options["owner_rows"] is owner.rows_index
    assert layer.engram.options["output_gather"] is None
    assert layer.attention_hc.args[0][0].shape==(2048,4,5120)
    assert layer.attention_hc.args[2] is owner.attention_hc
    assert layer.attention_hc.options["pre_mix"] is owner.pre_mix
    assert layer.attention_hc.options["pre_out"] is owner.attn_mix
    assert layer.attention_gather.args==(owner.attention_hc.normalized,full.attention_hc.normalized)
    assert layer.attention.args[2] is full.attention_hc.normalized
    assert layer.attention.args[3] is full.attention and layer.attention.options["row_shard"] is None
    assert layer.attention.options["world"]==4
    assert layer.attention_reduce.kind=="PreparedReduceScatter"
    assert layer.attention_reduce.args[0] is full.attention.output
    assert layer.attention_reduce.args[1] is owner.attention_output
    assert layer.ffn_hc.args[2] is owner.ffn_hc
    assert layer.ffn_hc.options["pre_mix"] is owner.attn_mix
    assert layer.ffn_hc.options["pre_out"] is owner.pre_mix
    assert layer.ffn_gather.args==(owner.ffn_hc.normalized,full.ffn_hc.normalized)
    assert layer.router_projection.args[0] is full.ffn_hc.normalized
    assert layer.experts.args[0] is full.ffn_hc.normalized and layer.shared.args[0] is full.ffn_hc.normalized
    assert layer.route_gather is None
    assert layer.mlp_reduce.kind=="PreparedReduceScatter" and layer.mlp_reduce.args[1] is owner.mlp_output
    assert layer.post.args[:5]==(owner.mlp_output,owner.ffn_hc.residual,owner.ffn_hc.post,owner.ffn_hc.comb,owner.residual)
    assert layer.output_row_indices is None and layer.residual_gather is None and layer.mix_gather is None
    assert layer.selection_move.args[0] is move and layer.selection_move.options["full"] is full.attention
    assert layer.selection_move.options["shard"] is owner.attention


@pytest.mark.parametrize("change",("missing_input","full_output_gather","no_owner_output","attention_shard","carry","reseed","different_output_owner"))
def test_normal_constructor_rejects_incoherent_owner_layout(monkeypatch,declaration,change):
    owner=declaration.owner;options={}
    if change=="missing_input":options["engram_input"]=None
    elif change=="full_output_gather":options["engram_input"]=(owner,object())
    elif change=="no_owner_output":options["output_rows"]=None
    elif change=="attention_shard":options["attention_shard"]=True
    elif change=="carry":options["carry"]=True
    elif change=="reseed":options["reseed"]=True
    elif change=="different_output_owner":options["output_rows"]=replace(owner,rank=1,start=2048)
    with pytest.raises(ValueError,match="owner FFN output requires|owner Engram mHC requires"):
        construct(monkeypatch,declaration,**options)


class MixPlan:
    def __init__(self,caps):self.caps=caps
    def request(self,**kwargs):return SimpleNamespace(**kwargs)


@pytest.mark.parametrize("gather",(False,True))
def test_actual_engram_constructor_declares_full_lookup_projection_and_owner_mix(monkeypatch,declaration,gather):
    from ignition.recipes.deepseek41 import engram
    for name in ("PreparedEngramLookup","QuantizedProjection","PreparedColumnReduceScatter"):
        monkeypatch.setattr(engram,name,lambda *args,_kind=name,**kwargs:Leaf(_kind,*args,**kwargs))
    monkeypatch.setattr(engram.hyperconnection,"Caps",lambda **kwargs:SimpleNamespace(**kwargs))
    monkeypatch.setattr(engram.hyperconnection,"plan",lambda caps,**kwargs:MixPlan(caps))
    monkeypatch.setattr(engram,"declare",lambda *args:None)
    owner=declaration.owner;full=declaration.scratch
    tables=SimpleNamespace(shard=SimpleNamespace(world_size=4),inputs={1:object()},readiness_runtime=None,
        token_map=(0,),geometry=object(),columns=torch.empty(1536,dtype=torch.int64,device="meta"))
    weights={"projection":object(),"weight":torch.empty((4,5120),dtype=torch.float32,device="meta")}
    output=full.engram_residual if gather else full.engram_residual[:owner.rows]
    result=engram.PreparedEngram(1,weights,tables,owner.residual,output,declaration.metadata,
        SimpleNamespace(config=SimpleNamespace(text=text())),scratch=full.engram,session=None,
        name="layer1.engram",workspace=None,runtime=declaration.runtime,owner_rows=owner.rows_index,
        output_gather=object() if gather else None)
    assert result.lookup.args[1].shape==(8192,6144)
    assert result.projection.args[0].shape==(8192,1536)
    assert result.projection.args[2].shape==(8192,25600)
    assert result.reduce.args[0].shape==(8192,25600) and result.reduce.args[1].shape==(2048,25600)
    assert result.mix_plan.caps.max_tokens==2048
    assert (result.output_gather is not None) is gather


def test_owner_without_gather_rejects_a_full_height_declared_output(declaration):
    from ignition.recipes.deepseek41.engram import PreparedEngram
    with pytest.raises(ValueError,match="owner-row Engram requires"):
        PreparedEngram(1,None,None,declaration.owner.residual,declaration.scratch.engram_residual,
            declaration.metadata,None,scratch=None,session=None,name="invalid",workspace=None,
            owner_rows=declaration.owner.rows_index)


@dataclass
class DecoderLayerWeights:
    engram:object
    attention:dict
    experts:object


def decoder_model(monkeypatch,declaration,case,attention_shard):
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.deepseek41 import decoder
    leaves(monkeypatch,decoder)
    monkeypatch.setattr(decoder,"Rotary",lambda *args:object())
    monkeypatch.setattr(decoder,"AttentionIndices",lambda *args,**kwargs:Leaf("indices"))
    monkeypatch.setattr(decoder.PreparedDecoder,"_prepare_head",lambda *args,**kwargs:None)
    cfg=SimpleNamespace(text=text(),layers=40,decoder_start=20,sources=(),draft=SimpleNamespace(target_layers=(33,34,35)))
    cfg.prefill_stop=lambda scored:39 if scored else 19
    cfg.shardable_attention_layers=lambda stop,aux:tuple(i for i in range(stop) if i not in cfg.text["engram_layer_ids"] and i not in aux)
    cfg.index_source=lambda index:None if index<2 else 2 if index<14 else 14
    layers=[]
    for i in range(40):
        w=layer_weights(i in (1,14));layers.append(DecoderLayerWeights(w.engram,w.attention,w.experts))
    weights=SimpleNamespace(config=cfg,layers=layers,draft=object(),head=None,
        embedding=torch.empty((129280,5120),dtype=torch.bfloat16,device="meta"),shard=SimpleNamespace(world_size=4))
    scratch=declaration.scratch;metadata=declaration.metadata
    scoring=SimpleNamespace(prepare=lambda *args,**kwargs:object()) if case=="scored" else None
    if case=="4096":scratch=scratch.rows(4096,1);metadata=Metadata.allocate(4096,1,96,"meta")
    if case=="no_collectives":declaration.runtime.should_reduce_scatter=lambda value:False
    model=decoder.PreparedDecoder(weights,SimpleNamespace(shard=SimpleNamespace(world_size=4)),scratch,metadata,
        SimpleNamespace(config=cfg),mode="prefill",runtime=declaration.runtime,vocabulary_runtime=declaration.runtime,
        coordinator=None,session=None,name="decoder",workspace=SimpleNamespace(overlap=False),
        row_shard_attention=attention_shard,scoring=scoring)
    stop=cfg.prefill_stop(model.packing is None)
    original_schedule=decoder.shard_schedule(cfg,stop,model.row_shard,model.aux_layers)
    original_moves=decoder.selection_moves(cfg,original_schedule,attention_shard=model.attention_shard)
    return model,scratch,original_moves


@pytest.mark.parametrize("attention_shard",(False,True))
@pytest.mark.parametrize("case",("eligible","4096","scored","no_collectives"))
def test_normal_decoder_keeps_attention_selection_and_only_eligible_owners(monkeypatch,declaration,case,attention_shard):
    model,scratch,moves=decoder_model(monkeypatch,declaration,case,attention_shard)
    for index,layer in enumerate(model.layers):
        assert (None if layer.selection_move is None else layer.selection_move.args[0])==moves[index]
    for index in (1,14):
        layer=model.layers[index]
        assert not layer.attention_shard
        assert layer.attention.options["row_shard"] is None
        if case=="eligible":
            assert layer.row_shard is model.row_shard
            assert layer.attention_hc.args[2] is model.row_shard.attention_hc
            assert layer.attention.args[2] is scratch.attention_hc.normalized
            assert layer.attention_reduce.kind=="PreparedReduceScatter"
            assert layer.ffn_hc.args[2] is model.row_shard.ffn_hc
            assert layer.mlp_reduce.kind=="PreparedReduceScatter"
            assert layer.post is not None and layer.residual_gather is layer.mix_gather is None
            assert layer.output_row_indices is None
            assert model.layers[index-1].residual_gather is model.layers[index-1].mix_gather is None
            assert not model.layers[index+1].reseed
        else:
            assert layer.row_shard is None
            assert layer.attention_hc.args[2] is scratch.attention_hc
            assert layer.attention_reduce.kind=="PreparedAllReduce"
            assert layer.mlp_reduce.kind=="PreparedAllReduce"
    if attention_shard and case=="eligible":
        assert model.layers[15].selection_move.args[0].selected=="full_to_shard"
