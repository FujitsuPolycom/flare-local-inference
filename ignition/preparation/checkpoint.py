# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate checkpoint metadata without importing torch or reading tensor payloads.

Tensor transfer belongs to b12x.loader.DirectWeightSession. Its manifest reader
requires a CUDA session; this preflight runs before CUDA initialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType


class CheckpointError(ValueError):
    """Checkpoint metadata does not satisfy the prepared model's contract."""


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shard: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def component(self) -> str:
        if self.name.startswith("mtp."):
            return "mtp"
        if ".ngram_embedding." in self.name:
            return "ple_tables"
        if self.name.startswith("model.visual."):
            return "vision"
        if ".mlp.experts." in self.name:
            return "decoder_experts"
        return "decoder_other"


@dataclass(frozen=True, slots=True)
class ShardMetadata:
    path: str
    file_size: int
    header_size: int
    header_sha256: str
    tensor_count: int


@dataclass(frozen=True, slots=True)
class IndexedCheckpoint:
    snapshot: Path
    tensors: Mapping[str, TensorMetadata]
    shards: tuple[ShardMetadata, ...]

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(self.snapshot / shard.path for shard in self.shards)

    @property
    def index_path(self) -> Path | None:
        path = self.snapshot / "model.safetensors.index.json"
        return path if path.is_file() else None

    @property
    def metadata_path(self) -> Path:
        return self.snapshot


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, source: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise CheckpointError(f"{source}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"{source}: expected a JSON object")
    return value


_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1,
                "I8": 1, "U8": 1, "U16": 2, "I32": 4, "U32": 4, "I64": 8}


def _read_header(path: Path, shard: str) -> tuple[ShardMetadata, dict[str, TensorMetadata]]:
    from .checkpoint_source import source_for

    remote = source_for(path)
    if remote is not None:
        name = remote.name(path)
        size = remote.files[name]["bytes"]
        content = remote.headers[name]
        prefix, raw = content[:8], content[8:]
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or length == 0 or length > 100 << 20 or length > size - 8 or len(raw) != length:
            raise CheckpointError(f"{shard}: invalid remote safetensors header")
    else:
        with path.open("rb", buffering=0) as source:
            size = os.fstat(source.fileno()).st_size
            prefix = source.read(8)
            if len(prefix) != 8:
                raise CheckpointError(f"{shard}: truncated safetensors length")
            length = int.from_bytes(prefix, "little")
            if length == 0 or length > 100 << 20 or length > size - 8:
                raise CheckpointError(f"{shard}: invalid safetensors header length")
            raw = source.read(length)
            if len(raw) != length:
                raise CheckpointError(f"{shard}: truncated safetensors header")
    header = _parse_json(raw, shard)
    tensors = {}
    data_start = 8 + length
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                raise CheckpointError(f"{shard}: invalid safetensors metadata")
            continue
        if not name or not isinstance(entry, dict):
            raise CheckpointError(f"{shard}: invalid tensor entry {name!r}")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        dtype = entry.get("dtype")
        if (
            not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(v) is not int or v < 0 for v in offsets)
            or not isinstance(dtype, str) or dtype not in _DTYPE_BYTES
        ):
            raise CheckpointError(f"{shard}: invalid shape, dtype, or offsets for {name}")
        start, end = offsets
        nbytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - start != nbytes or end > size - data_start:
            raise CheckpointError(f"{shard}: invalid byte range for {name}")
        tensors[name] = TensorMetadata(
            name, shard, data_start + start, nbytes, tuple(shape), dtype
        )
    end = data_start
    for tensor in sorted(tensors.values(), key=lambda value: (value.offset, value.nbytes)):
        if tensor.offset != end:
            raise CheckpointError(f"{shard}: overlap or hole at {tensor.name}")
        end += tensor.nbytes
    if end != size:
        raise CheckpointError(f"{shard}: tensor ranges do not cover the file")
    metadata = ShardMetadata(shard, size, length, hashlib.sha256(raw).hexdigest(), len(tensors))
    return metadata, tensors


def inspect_indexed_checkpoint(snapshot: str | Path) -> IndexedCheckpoint:
    """Read indexed safetensors geometry and validate byte ranges without CUDA.

    Model recipes validate configuration, tensor names and expected precision.
    The direct loader revalidates source file identity before moving payloads.
    """
    snapshot = Path(snapshot).expanduser().absolute()
    from .checkpoint_source import source_for

    remote = source_for(snapshot)
    container = snapshot / "manifest.json"
    if container.is_file():
        description = _parse_json(container.read_bytes(), "manifest.json")
        if description.get("schema") == "lil-mxfp4-csf-checkpoint/1":
            from .csf_checkpoint import inspect_mxfp4_csf

            return inspect_mxfp4_csf(snapshot, description)
    if not (snapshot / "model.safetensors.index.json").exists() and (
            (snapshot / "model.safetensors").is_file() or remote is not None and "model.safetensors" in remote.files):
        shard, tensors = _read_header(snapshot / "model.safetensors", "model.safetensors")
        return IndexedCheckpoint(snapshot, MappingProxyType(tensors), (shard,))
    index = _parse_json((snapshot / "model.safetensors.index.json").read_bytes(),
                        "model.safetensors.index.json")
    return _inspect_indexed_checkpoint(snapshot, index)


def _inspect_indexed_checkpoint(snapshot: Path, index: dict) -> IndexedCheckpoint:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("model.safetensors.index.json: missing weight_map")
    shard_names = set()
    for name, shard in weight_map.items():
        if not name or not isinstance(shard, str):
            raise CheckpointError(f"invalid indexed shard for {name}: {shard!r}")
        if shard not in shard_names:
            path = PurePosixPath(shard)
            if path.is_absolute() or ".." in path.parts or not shard.endswith(".safetensors"):
                raise CheckpointError(f"invalid indexed shard for {name}: {shard!r}")
            shard_names.add(shard)
    tensors, shards = {}, []
    for shard in sorted(shard_names):
        record, entries = _read_header(snapshot / shard, shard)
        for name in entries:
            if name in tensors:
                raise CheckpointError(f"duplicate tensor across shards: {name}")
            if weight_map.get(name) != shard:
                raise CheckpointError(f"index/shard mismatch for {name}: found in {shard}")
        tensors.update(entries)
        shards.append(record)
    missing = weight_map.keys() - tensors.keys()
    if missing:
        raise CheckpointError(f"indexed tensor missing from shards: {min(missing)}")
    total = sum(tensor.nbytes for tensor in tensors.values())
    declared = index["metadata"].get("total_size") if isinstance(index.get("metadata"), dict) else None
    if type(declared) is not int or declared < total:
        raise CheckpointError(f"index total_size is invalid for tensor payload bytes: {total}")
    return IndexedCheckpoint(snapshot, MappingProxyType(tensors), tuple(shards))
