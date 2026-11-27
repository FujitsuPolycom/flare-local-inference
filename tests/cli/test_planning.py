# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe-free planning, ParameterGroup hooks, and PreparationManifest serialization."""

from dataclasses import dataclass, replace
import json
from types import MappingProxyType, SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.errors import CapacityError, UsageError
from ignition.parameters import (
    ParameterContext,
    ParameterGroup,
    ParameterScope,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    param,
)
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto

GIB = 1 << 30
MIB = 1 << 20
DRAFTING = SpeculationParameters(draft_tokens=3, max_draft_tokens=7)


@dataclass(frozen=True)
class ToyPrecisionGroup(ParameterGroup):
    """Fixture group exercising TARGET_WEIGHT + DRAFT_WEIGHT manifest emission."""

    online_mxfp8: bool = param(
        False,
        flag="--toy-mxfp8",
        scope=ParameterScope.TARGET_WEIGHT,
        help="enable toy MXFP8 conversion",
    )
    mxfp8_scope: str = param(
        "all",
        flag="--toy-mxfp8-scope",
        choices=("all", "partial"),
        scope=ParameterScope.TARGET_WEIGHT,
        help="toy conversion scope",
    )

    def manifest_parameters(self, context: ParameterContext):
        entries = super().manifest_parameters(context)
        root = context.root
        if (root is not None and root.speculation is not None
                and root.speculation.method == "dflash2"
                and self.online_mxfp8 and self.mxfp8_scope == "all"):
            entries.append(("speculation.draft_body_quantization", ParameterScope.DRAFT_WEIGHT, "mxfp8"))
        return entries


@dataclass(frozen=True)
class ToyShardingParameters(RecipeParameters):
    """Fixture subclass exercising world-size filtering and replicated weight charges."""

    sharded_prefill: bool = param(
        False,
        flag="--sharded-prefill",
        scope=ParameterScope.DEPLOYMENT,
        help="enable toy row-sharded prefill",
    )
    sharded_attention: bool = param(
        False,
        flag="--sharded-attention",
        scope=ParameterScope.DEPLOYMENT,
        help="enable toy row-sharded attention",
    )
    precision: ToyPrecisionGroup | None = None
    sharded_world_sizes: tuple[int, ...] = ()
    replicated_bytes_per_layer: int = 0

    def filter_world_sizes(self, world_sizes: tuple[int, ...], context: ParameterContext) -> tuple[int, ...]:
        sizes = super().filter_world_sizes(world_sizes, context)
        if self.sharded_prefill:
            return tuple(size for size in sizes if size in self.sharded_world_sizes)
        return sizes

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.sharded_attention and not self.sharded_prefill:
            raise UsageError("--sharded-attention requires --sharded-prefill")
        if self.sharded_attention and self.replicated_bytes_per_layer <= 0:
            raise UsageError(f"--sharded-attention is not supported by {context.alias}")
        if self.sharded_prefill:
            if not self.sharded_world_sizes:
                raise UsageError(f"--sharded-prefill is not supported by {context.alias}")
            if context.world_size is not None:
                if context.world_size < 2:
                    raise UsageError("--sharded-prefill requires tensor parallelism")
                if context.world_size not in self.sharded_world_sizes:
                    supported = ", ".join(map(str, self.sharded_world_sizes))
                    raise UsageError(
                        f"--sharded-prefill is not supported at TP{context.world_size} for {context.alias};"
                        f" supported tensor-parallel sizes: {supported}"
                    )
            if any(gpu.capability == (10, 3) for gpu in context.gpus):
                raise UsageError("--sharded-prefill is not qualified on SM103; select SM120 or SM121 GPUs")

    def adjust_weights_bytes(self, weights_bytes: int, context: ParameterContext) -> int:
        weights_bytes = super().adjust_weights_bytes(weights_bytes, context)
        if self.sharded_attention and context.world_size and context.world_size >= 2:
            prompt_lp = bool(self.prompt_logprobs and self.prompt_logprobs.enabled)
            mtp_only = bool(self.speculation and self.speculation.mtp_only)
            layers = {(False, False): 17, (False, True): 19, (True, False): 37, (True, True): 35}[(prompt_lp, mtp_only)]
            return weights_bytes + layers * self.replicated_bytes_per_layer
        return weights_bytes


def _recipe(name="fixture", *, device_gib=60, ranks=(1, 2, 4), parameters=None):
    params = (
        ToyShardingParameters(
            max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096,
            speculation=DRAFTING, prompt_logprobs=PromptLogprobsParameters(),
        )
        if parameters is None
        else parameters
    )
    checkpoint = Checkpoint(
        alias=f"{name}-alias",
        repository=f"example/{name}",
        revision="ab12cd34" + "0" * 32,
        architecture="FixtureForCausalLM",
        stored_bytes=device_gib * GIB,
        device_bytes=device_gib * GIB,
        draft_bytes=2 * GIB if params.speculation is not None else 0,
    )
    return Recipe(
        name=name,
        served_model_name=name,
        checkpoints=(checkpoint,),
        context=262144,
        page_tokens=64,
        parameters=params,
        ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB) for size in ranks}),
        plain_ranks=MappingProxyType(
            {size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB) for size in ranks}
            if params.speculation is not None else {}
        ),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
    )


def _gpu(index, free_gib=95, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, int(free_gib * GIB), capability)


def test_custom_parameter_hooks_validation_automatic_sizing_and_manifest(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    unsupported = _recipe("plain", ranks=(1, 2, 4), parameters=ToyShardingParameters(
        max_concurrency=4, speculation=DRAFTING, sharded_world_sizes=(),
    ))
    supported = _recipe("sharded", ranks=(1, 2, 4), parameters=ToyShardingParameters(
        max_concurrency=4, speculation=DRAFTING, sharded_world_sizes=(2, 4),
    ))
    tp34 = _recipe("tp34", ranks=(3, 4), parameters=ToyShardingParameters(
        max_concurrency=4, speculation=DRAFTING, sharded_world_sizes=(4,),
    ))

    pair_gpus = (_gpu(0), _gpu(1))
    ordinary = planning.plan(supported, supported.checkpoints[0], pair_gpus, selected=(0, 1))
    sharded_params = replace(supported.parameters, sharded_prefill=True)
    sharded = planning.plan(supported, supported.checkpoints[0], pair_gpus, selected=(0, 1), parameters=sharded_params)
    assert not ordinary.parameters.sharded_prefill
    assert sharded.parameters.sharded_prefill
    assert sharded.to_dict()["sharded_prefill"] is True
    assert replace(sharded, parameters=replace(sharded.parameters, sharded_prefill=False)) == ordinary

    fitted = planning.fit(supported, supported.checkpoints[0], pair_gpus, parameters=sharded_params)
    assert fitted.parameters.sharded_prefill

    for candidate in (ordinary, sharded, fitted):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), candidate)
        assert ("--sharded-prefill" in manifest["reproduction_args"]) == candidate.parameters.sharded_prefill
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
        keys = [parameter.key for parameter in decoded.parameters]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        by_key = {parameter.key: parameter for parameter in decoded.parameters}
        assert by_key["sharded_prefill"].scope == proto.PREPARATION_PARAMETER_SCOPE_DEPLOYMENT
        assert by_key["sharded_prefill"].value.bool_value == candidate.parameters.sharded_prefill
        assert by_key["speculation.has_draft"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
        assert by_key["speculation.has_draft"].value.bool_value is True
        # Unrelated parameters absent from this recipe are never emitted.
        assert "mla_compute_mode" not in by_key
        assert "online_mxfp8" not in by_key

    with pytest.raises(UsageError, match="--sharded-prefill is not supported by plain-alias"):
        planning.plan(unsupported, unsupported.checkpoints[0], pair_gpus, selected=(0, 1),
                      parameters=replace(unsupported.parameters, sharded_prefill=True))

    with pytest.raises(UsageError, match="--sharded-prefill requires tensor parallelism"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0),), selected=(0,), parameters=sharded_params)
    with pytest.raises(UsageError, match="--sharded-prefill requires tensor parallelism"):
        planning.plan(tp34, tp34.checkpoints[0], tuple(_gpu(i) for i in range(4)), selected=(0,),
                      parameters=replace(tp34.parameters, sharded_prefill=True))

    four_gpus = tuple(_gpu(i) for i in range(4))
    tp34_sharded = replace(tp34.parameters, sharded_prefill=True)
    with pytest.raises(UsageError, match="--sharded-prefill is not supported at TP3 for tp34-alias; supported tensor-parallel sizes: 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=(0, 1, 2), parameters=tp34_sharded)
    auto_tp4 = planning.plan(tp34, tp34.checkpoints[0], four_gpus, selected=None, parameters=tp34_sharded)
    assert auto_tp4.world_size == 4 and auto_tp4.parameters.sharded_prefill
    with pytest.raises(CapacityError, match="tp34-alias needs 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=None, parameters=tp34_sharded)

    with pytest.raises(UsageError, match="SM103"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0, 95, (10, 3)), _gpu(1, 95, (10, 3))),
                      selected=(0, 1), parameters=sharded_params)


def test_preparation_manifest_resolves_scopes_and_heliopause_round_trip(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    dflash = SpeculationParameters(draft_tokens=3, max_draft_tokens=7, method="dflash2")
    params = ToyShardingParameters(
        max_concurrency=4,
        speculation=dflash,
        precision=ToyPrecisionGroup(),
    )
    recipe = _recipe("dflash", ranks=(1,), parameters=params)
    gpu = (_gpu(0),)

    def decode_params(plan):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan)
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
        keys = [parameter.key for parameter in decoded.parameters]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        return {parameter.key: parameter for parameter in decoded.parameters}

    inherited = planning.plan(
        recipe, recipe.checkpoints[0], gpu, selected=(0,),
        parameters=replace(params, precision=ToyPrecisionGroup(online_mxfp8=True, mxfp8_scope="all")),
    )
    decoded = decode_params(inherited)
    assert decoded["speculation.draft_body_quantization"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
    assert decoded["speculation.draft_body_quantization"].value.string_value == "mxfp8"
    assert decoded["speculation.has_draft"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
    assert decoded["speculation.has_draft"].value.bool_value is True
    assert decoded["online_mxfp8"].scope == proto.PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT

    scoped = planning.plan(
        recipe, recipe.checkpoints[0], gpu, selected=(0,),
        parameters=replace(params, precision=ToyPrecisionGroup(online_mxfp8=True, mxfp8_scope="partial")),
    )
    scoped_params = decode_params(scoped)
    assert "speculation.draft_body_quantization" not in scoped_params
    assert scoped_params["mxfp8_scope"].value.string_value == "partial"

    target_only = planning.plan(
        recipe, recipe.checkpoints[0], gpu, selected=(0,),
        parameters=replace(params, speculation=None, precision=ToyPrecisionGroup(online_mxfp8=True, mxfp8_scope="all")),
    )
    target_params = decode_params(target_only)
    assert not any(
        parameter.scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
        for parameter in target_params.values()
    )

    import flare_heliopause

    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), inherited)
    server = {
        "model_id": "example/dflash",
        "frontend": {"eos_token_ids": [1]},
        "sampling_defaults": {
            "seed": 0, "temperature": 1.0, "top_k": -1, "top_p": 1.0,
            "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "assets": {"chat_template": "t", "tokenizer": "k"},
        "preparation_manifest": manifest,
    }
    encoded_index = flare_heliopause.export_serving(
        proto.ModelIndex(version=4).SerializeToString(),
        json.dumps(server),
    )
    decoded_index = proto.ModelIndex.FromString(encoded_index)
    assert decoded_index.HasField("preparation")
    assert not decoded_index.HasField("rebuild")
    expected = json_format.ParseDict(manifest, proto.PreparationManifest())
    assert decoded_index.preparation.SerializeToString(deterministic=True) == expected.SerializeToString(deterministic=True)


def test_weight_adjustment_hook_sizes_gpus_and_records_manifest(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    per_layer = 128 * MIB
    supported = _recipe(
        "attn",
        device_gib=60,
        ranks=(2, 4),
        parameters=ToyShardingParameters(
            max_concurrency=4,
            speculation=DRAFTING,
            prompt_logprobs=PromptLogprobsParameters(),
            sharded_world_sizes=(2, 4),
            replicated_bytes_per_layer=per_layer,
        ),
    )
    pair_gpus = (_gpu(0), _gpu(1))
    draft_only = replace(DRAFTING, mtp_only=True)

    for (prompt_lp, spec), expected_layers in (
        ((False, DRAFTING), 17),
        ((False, draft_only), 19),
        ((True, DRAFTING), 37),
        ((True, draft_only), 35),
    ):
        base_params = replace(
            supported.parameters,
            speculation=spec,
            prompt_logprobs=PromptLogprobsParameters(enabled=prompt_lp),
            sharded_prefill=True,
        )
        base = planning.plan(supported, supported.checkpoints[0], pair_gpus, selected=(0, 1), parameters=base_params)
        attn = planning.plan(supported, supported.checkpoints[0], pair_gpus, selected=(0, 1),
                             parameters=replace(base_params, sharded_attention=True))
        assert attn.weights_bytes - base.weights_bytes == expected_layers * per_layer
        assert attn.pages < base.pages
        assert attn.to_dict()["sharded_attention"] is True

    # Automatic GPU sizing steps up from TP2 to TP4 when replicated weights exceed TP2 capacity.
    tight = _recipe(
        "tight",
        device_gib=60,
        ranks=(2, 4),
        parameters=ToyShardingParameters(
            max_concurrency=4,
            speculation=DRAFTING,
            sharded_world_sizes=(2, 4),
            replicated_bytes_per_layer=4 * GIB // 17,
        ),
    )
    tight_gpus = tuple(_gpu(i, free_gib=39) for i in range(4))
    tp2 = planning.plan(tight, tight.checkpoints[0], tight_gpus, selected=None,
                        parameters=replace(tight.parameters, sharded_prefill=True))
    tp4 = planning.plan(tight, tight.checkpoints[0], tight_gpus, selected=None,
                        parameters=replace(tight.parameters, sharded_prefill=True, sharded_attention=True))
    assert tp2.world_size == 2
    assert tp4.world_size == 4 and tp4.parameters.sharded_attention

    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), tp4)
    assert "--sharded-prefill" in manifest["reproduction_args"]
    assert "--sharded-attention" in manifest["reproduction_args"]
    encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
    by_key = {p.key: p for p in encoded.parameters}
    assert by_key["sharded_attention"].scope == proto.PREPARATION_PARAMETER_SCOPE_DEPLOYMENT
    assert by_key["sharded_attention"].value.bool_value is True
