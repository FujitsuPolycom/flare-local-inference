# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition GPU KV-cache codec cubin compilation metadata."""

from ignition.preparation import runtime_core_pb2
from ignition.preparation.kv_codec import GPU_CODEC_MODULE_NAME, _target_arch


def test_target_arch_formatting():
    assert _target_arch((10, 3)) == "sm_103a"
    assert _target_arch((12, 0)) == "sm_120a"
    assert _target_arch((12, 1)) == "sm_121a"
    assert _target_arch((9, 0)) == "sm_90a"
    assert _target_arch((8, 9)) == "sm_89"


def test_codec_layout_proto_gpu_module_optional():
    empty = runtime_core_pb2.CodecLayout(revision=1)
    assert not empty.HasField("gpu_module")
    roundtrip_empty = runtime_core_pb2.CodecLayout()
    roundtrip_empty.ParseFromString(empty.SerializeToString())
    assert not roundtrip_empty.HasField("gpu_module")

    populated = runtime_core_pb2.CodecLayout(revision=1, gpu_module=GPU_CODEC_MODULE_NAME)
    assert populated.HasField("gpu_module")
    assert populated.gpu_module == GPU_CODEC_MODULE_NAME
    roundtrip_populated = runtime_core_pb2.CodecLayout()
    roundtrip_populated.ParseFromString(populated.SerializeToString())
    assert roundtrip_populated.HasField("gpu_module")
    assert roundtrip_populated.gpu_module == GPU_CODEC_MODULE_NAME


def test_gpu_codec_source_packaged():
    from ignition.preparation import kv_codec

    assert kv_codec._SOURCE.is_file(), f"missing packaged CUDA source: {kv_codec._SOURCE}"


def test_compiled_gpu_codec_cubin_declares_no_elf_globals(tmp_path):
    import os
    import shutil
    import subprocess
    import pytest

    if not os.environ.get("FLARE_RUN_CUDA_TESTS"):
        pytest.skip("set FLARE_RUN_CUDA_TESTS=1 to compile and inspect kv_codec_gpu.cubin")
    if shutil.which("readelf") is None:
        pytest.skip("readelf not available")

    from ignition.preparation.kv_codec import compile_gpu_codec_cubin

    cubin = compile_gpu_codec_cubin((12, 0), cache_dir=tmp_path)
    out = subprocess.run(
        ["readelf", "-sW", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    globals_defined = [
        line
        for line in out.splitlines()
        if len(line.split()) >= 8 and line.split()[3] == "OBJECT" and line.split()[6] != "UND"
    ]
    assert globals_defined == [], f"standalone cubin must declare zero ELF globals: {globals_defined}"
