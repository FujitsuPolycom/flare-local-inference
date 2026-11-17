# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU declaration tests for the full-FFN to owner-post transition.

The real recipe constructors and eligibility predicate run on meta tensors.
GPU leaf operators are declaration recorders; these tests establish wiring and
scope, not CUDA arithmetic, replay or native-state qualification.
"""
from dataclasses import dataclass,replace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def declaration(monkeypatch):
    from b12x.comm.roce import _proxy
    from ignition.components import collectives
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.deepseek41 import decoder

    monkeypatch.setattr(_proxy,"ABI_VERSION",14)
    monkeypatch.setattr(collectives,"_roce",lambda runtime:True)
    runtime=SimpleNamespace(world_size=4,pipe_bytes=80<<20,
        should_reduce_scatter=lambda source:True,should_gather_shards=lambda source:True)
    scratch=decoder.Scratch.allocate(8192,1,4,"meta",row_shard=SimpleNamespace(rank=0),attention_shard=True)
    schedule=[(True,True,False)]*40
    for index in (1,14):schedule[index]=(False,False,False);schedule[index+1]=(True,True,True)
    return SimpleNamespace(owner=scratch.row_shard,scratch=scratch,runtime=runtime,schedule=schedule,
        metadata=Metadata.allocate(8192,1,96,"meta"),mode="prefill",scoring=None,packing=object(),
        engram_layers=(1,14),aux_layers=(33,34,35),decoder_start=20)


def ready(value,index=1):
    from ignition.recipes.deepseek41.decoder import outgoing_owner_ready
    return outgoing_owner_ready(index,value.schedule,mode=value.mode,scoring=value.scoring,
        packing=value.packing,engram_layers=value.engram_layers,aux_layers=value.aux_layers,
        decoder_start=value.decoder_start,metadata=value.metadata,scratch=value.scratch,
        owner=value.owner,runtime=value.runtime)


@pytest.mark.parametrize("index",(1,14))
@pytest.mark.parametrize("rank",range(4))
def test_declared_engram_edges_accept_every_owner(declaration,index,rank):
    declaration.owner=replace(declaration.owner,rank=rank,start=rank*2048)
    assert ready(declaration,index)


def test_eligibility_uses_declaration_and_capability_instead_of_layer_identity(declaration):
    declaration.engram_layers=(5,)
    declaration.schedule[5]=(False,False,False);declaration.schedule[6]=(True,True,True)
    assert ready(declaration,5)
    assert not ready(declaration,1)


@pytest.mark.parametrize("change",("scored","decode","no_packing","no_owner","no_runtime","not_declared",
    "current_sharded","deferred","next_full","no_reseed","current_aux","next_aux","suffix",
    "wrong_world","wrong_pipe","wrong_start","short_indices","partial_dtype","post_shape","comb_stride","smaller_capacity"))
def test_ineligible_edges_keep_full_post(declaration,change):
    from ignition.components.paged_batch import Metadata
    value=declaration
    if change=="scored":value.scoring=object()
    elif change=="decode":value.mode="decode"
    elif change=="no_packing":value.packing=None
    elif change=="no_owner":value.owner=None
    elif change=="no_runtime":value.runtime=None
    elif change=="not_declared":value.engram_layers=()
    elif change=="current_sharded":value.schedule[1]=(True,False,False)
    elif change=="deferred":value.schedule[1]=(False,True,False)
    elif change=="next_full":value.schedule[2]=(False,False,False)
    elif change=="no_reseed":value.schedule[2]=(True,False,False)
    elif change=="current_aux":value.aux_layers=(1,33)
    elif change=="next_aux":value.aux_layers=(2,33)
    elif change=="suffix":value.decoder_start=2
    elif change=="wrong_world":value.runtime.world_size=2
    elif change=="wrong_pipe":value.runtime.pipe_bytes=64<<20
    elif change=="wrong_start":value.owner=replace(value.owner,start=1)
    elif change=="short_indices":value.owner=replace(value.owner,rows_index=value.owner.rows_index[:2047])
    elif change=="partial_dtype":value.scratch=replace(value.scratch,mlp_partial=value.scratch.mlp_partial.float())
    elif change=="post_shape":value.scratch=replace(value.scratch,ffn_hc=replace(value.scratch.ffn_hc,post=value.scratch.ffn_hc.post[:4096]))
    elif change=="comb_stride":
        comb=torch.empty((8192,4,5),dtype=torch.float32,device="meta")[:,:,:4]
        value.scratch=replace(value.scratch,ffn_hc=replace(value.scratch.ffn_hc,comb=comb))
    elif change=="smaller_capacity":value.metadata=Metadata.allocate(4096,1,96,"meta")
    assert not ready(value)


class Leaf:
    def __init__(self,kind,*args,**kwargs):
        self.kind,self.args,self.options=kind,args,kwargs
        self.plan=object()
    def functional(self,*args,**kwargs):return self


def leaves(monkeypatch,decoder):
    for name in ("PreparedMHC","PreparedAttention","PreparedAllReduce","PreparedReduceScatter",
                 "PreparedChunkedGatherShards","PreparedGatherShards","UnquantizedProjection",
                 "PreparedRouter","PreparedRoutedExperts","PreparedMLP","PreparedMHCPost","PreparedEngram",
                 "PreparedSelectionRefill","PreparedTableRequests","PreparedEmbedding","PreparedDecoderBoundary"):
        original=getattr(decoder,name)
        factory=lambda *args,_kind=name,**kwargs:Leaf(_kind,*args,**kwargs)
        if hasattr(original,"fits"):factory.fits=original.fits
        monkeypatch.setattr(decoder,name,factory)
    monkeypatch.setattr(decoder,"_mhc_weights",lambda *args:None)
    monkeypatch.setattr(decoder,"register_finalizer",lambda *args,**kwargs:None)


def layer_weights(engram=True):
    return SimpleNamespace(engram={} if engram else None,attention={},experts=SimpleNamespace(
        vision_correction=None,router=None,correction=None,experts=None,shared=None))


def text():
    return dict(rms_norm_eps=1e-20,hc_eps=1e-6,hc_sinkhorn_iters=20,scoring_func="sqrtsoftplus",
        norm_topk_prob=True,routed_scaling_factor=1.,swiglu_limit=10.,hidden_size=5120,
        compress_ratios=[2]*20+[1]*20,engram_layer_ids=[1,14],candidate_source_layer_id=14)


def construct_layer(monkeypatch,declaration,*,output=True,**changes):
    from ignition.recipes.deepseek41 import decoder
    leaves(monkeypatch,decoder)
    options=dict(shard=SimpleNamespace(world_size=4),mode="prefill",runtime=declaration.runtime,
        coordinator=None,session=None,name="layer1",workspace=SimpleNamespace(overlap=False),carry=False,
        defer_post=False,output_rows=declaration.owner if output else None)
    options.update(changes)
    return decoder.PreparedLayer(1,layer_weights(),None,declaration.scratch,declaration.metadata,
        SimpleNamespace(config=SimpleNamespace(text=text())),None,**options)


def test_normal_layer_constructor_reduces_and_posts_only_owner_rows(monkeypatch,declaration):
    layer=construct_layer(monkeypatch,declaration)
    assert layer.row_shard is None and not layer.attention_shard
    assert layer.attention_reduce.kind=="PreparedAllReduce"
    assert layer.ffn_hc.args[2] is declaration.scratch.ffn_hc
    assert layer.experts.args[0].shape[0]==8192
    assert layer.shared.args[0].shape[0]==8192
    assert layer.mlp_reduce.kind=="PreparedReduceScatter"
    assert layer.mlp_reduce.args[0].shape==(8192,5120)
    assert layer.mlp_reduce.args[1].shape==(2048,5120)
    assert [tuple(v.shape) for v in layer.post.args[:5]]==[
        (2048,5120),(2048,4,5120),(2048,4),(2048,4,4),(2048,4,5120)]
    assert layer.output_row_indices is declaration.owner.rows_index
    assert not any(value is declaration.owner for value in layer.__dict__.values())


def test_ordinary_constructor_keeps_full_output(monkeypatch,declaration):
    layer=construct_layer(monkeypatch,declaration,output=False)
    assert layer.output_row_indices is None
    assert layer.mlp_reduce.kind=="PreparedAllReduce"
    assert layer.post.args[0].shape[0]==8192 and layer.post.args[-1].shape[0]==8192


@pytest.mark.parametrize("changes",({"defer_post":True},{"row_shard":object()},{"boundary":object()},{"mode":"decode"}))
def test_constructor_rejects_conflicting_output_layout(monkeypatch,declaration,changes):
    with pytest.raises(ValueError,match="owner FFN output requires"):
        construct_layer(monkeypatch,declaration,**changes)


@dataclass
class Weights:
    engram:object
    attention:dict


@pytest.mark.parametrize("disable",(False,True))
def test_normal_decoder_handoff_preserves_selection_and_skips_only_reseed(monkeypatch,declaration,disable):
    from ignition.recipes.deepseek41 import decoder
    leaves(monkeypatch,decoder)
    monkeypatch.setattr(decoder,"Rotary",lambda *args:object())
    monkeypatch.setattr(decoder,"AttentionIndices",lambda *args,**kwargs:Leaf("indices"))
    monkeypatch.setattr(decoder.PreparedDecoder,"_prepare_head",lambda *args,**kwargs:None)
    class Layer:
        def __init__(self,index,weights,tables,scratch,metadata,pool,rotary,**kwargs):
            self.options=kwargs;self.row_shard=kwargs["row_shard"]
            self.post=None if kwargs["defer_post"] else object()
            self.residual_gather=self.mix_gather=(object() if self.row_shard is not None and self.post is not None else None)
    monkeypatch.setattr(decoder,"PreparedLayer",Layer)
    cfg=SimpleNamespace(text=text(),layers=40,decoder_start=20,sources=(),draft=SimpleNamespace(target_layers=(33,34,35)))
    cfg.prefill_stop=lambda scored:39 if scored else 19
    cfg.shardable_attention_layers=lambda stop,aux:tuple(i for i in range(stop) if i not in cfg.text["engram_layer_ids"] and i not in aux)
    cfg.index_source=lambda index:None if index<2 else 2 if index<14 else 14
    weights=SimpleNamespace(config=cfg,layers=[Weights({} if i in (1,14) else None,{}) for i in range(40)],
        draft=object(),embedding=torch.empty((129280,5120),dtype=torch.bfloat16,device="meta"),
        head=None,shard=SimpleNamespace(world_size=4))
    scoring=SimpleNamespace(prepare=lambda *args,**kwargs:object()) if disable else None
    model=decoder.PreparedDecoder(weights,declaration.tables if hasattr(declaration,"tables") else SimpleNamespace(shard=SimpleNamespace(world_size=4)),
        declaration.scratch,declaration.metadata,SimpleNamespace(config=cfg),mode="prefill",runtime=declaration.runtime,
        vocabulary_runtime=declaration.runtime,coordinator=None,session=None,name="decoder",workspace=SimpleNamespace(overlap=False),
        row_shard_attention=True,scoring=scoring)
    for index in (1,14):
        assert ("output_rows" in model.layers[index].options) is not disable
        assert model.layers[index+1].options["reseed"] is disable
        assert model.layers[index].options["row_shard"] is None
    move=model.layers[15].options["selection_move"]
    assert move is not None and move.selected=="full_to_shard" and move.source==14
