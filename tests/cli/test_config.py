# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Configuration file validation and precedence."""

from types import SimpleNamespace

import pytest

from ignition.cli.config import Configuration, Setting, parse_size
from ignition.errors import UsageError
from ignition.parameters import ParameterContext, RecipeParameters


def load(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return Configuration.load(path)


def test_absent_default_file_yields_built_in_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    configuration = Configuration.load()
    assert not configuration.exists and configuration.path == tmp_path / ".config/flare/config.toml"
    assert configuration.get("server.host") == Setting("127.0.0.1", "default")
    assert configuration.get("server.port") == Setting(8000, "default")
    assert configuration.get("server.thinking_token_budget") == Setting(None, "default")
    assert configuration.reserve_bytes == 0 and configuration.model is None
    assert configuration.directory("hub_cache") == tmp_path / ".cache/huggingface/hub"
    assert configuration.directory("kernel_cache") == tmp_path / ".cache/flare/kernels"
    assert configuration.directory("cuda") is None


def test_an_explicit_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="does not exist"):
        Configuration.load(tmp_path / "absent.toml")


def test_core_cache_accepts_legacy_setting_without_moving_files(tmp_path):
    cache = tmp_path / "downloaded"
    cache.mkdir()
    artifact = cache / "model.safetensors"
    artifact.write_bytes(b"retained model")
    configuration = load(tmp_path, f'[paths]\npackage_cache = "{cache.as_posix()}"\n')
    assert configuration.directory("core_cache") == cache
    assert artifact.read_bytes() == b"retained model"
    with pytest.raises(UsageError, match="use only core_cache"):
        load(tmp_path, '[paths]\ncore_cache = "one"\npackage_cache = "two"\n')


def test_core_cache_defaults_find_existing_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    configuration = Configuration.load()
    canonical = tmp_path / ".cache/flare/cores"
    legacy = canonical.with_name("packages")
    assert configuration.directory("core_cache") == canonical
    legacy.mkdir(parents=True)
    assert configuration.directory("core_cache") == legacy
    canonical.mkdir()
    assert configuration.directory("core_cache") == canonical
    explicit = load(tmp_path, '[paths]\ncore_cache = "D:/FlareCores"\n')
    assert explicit.directory("core_cache").as_posix() == "D:/FlareCores"


def test_flags_override_the_file_which_overrides_the_recipe(tmp_path):
    configuration = load(tmp_path, """
[server]
host = "0.0.0.0"
port = 9000
thinking_token_budget = 1536

[defaults]
model = "qwen3.8-flash"
max_concurrency = 6

[memory]
reserve = "16G"

[models."qwen3.8-flash"]
max_concurrency = 2
""")
    assert configuration.get("server.host") == Setting("0.0.0.0", "[server]")
    assert configuration.get("server.port") == Setting(9000, "[server]")
    assert configuration.get("server.port", 7000) == Setting(7000, "flag")
    assert configuration.get("server.thinking_token_budget") == Setting(1536, "[server]")
    assert configuration.model == "qwen3.8-flash" and configuration.reserve_bytes == 16 << 30

    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(2, '[models."qwen3.8-flash"]')
    assert configuration.max_concurrency("glm-5.3-flash", 4) == Setting(6, "[defaults]")
    assert configuration.max_concurrency("qwen3.8-flash", 4, 9) == Setting(9, "flag")


def test_prefix_cache_settings_validate(tmp_path):
    configuration = load(tmp_path, "[prefix_cache]\nmemory = \"4G\"\n")
    assert configuration.get("prefix_cache.memory") == Setting("4G", "[prefix_cache]")
    assert configuration.directory("prefix_cache") is None
    with pytest.raises(UsageError, match="prefix_cache disk_limit"):
        load(tmp_path, "[prefix_cache]\ndisk_limit = \"lots\"\n")
    for transport in ("nvme", "staged", "direct"):
        with pytest.raises(UsageError, match="disk_transport"):
            load(tmp_path, f'[prefix_cache]\ndisk_transport = "{transport}"\n')

    assert configuration.get("prefix_cache.max_age") == Setting("off", "default")
    for value in ("off", "2h", "0.5s"):
        assert load(tmp_path, f'[prefix_cache]\nmax_age="{value}"\n').get("prefix_cache.max_age").value == value
    for value in ("on", "0", "-1s", "NaN", "2d"):
        with pytest.raises(UsageError, match="max_age"):
            load(tmp_path, f'[prefix_cache]\nmax_age="{value}"\n')


def test_recipe_supplies_concurrency_when_the_file_is_silent(tmp_path):
    configuration = load(tmp_path, "[server]\nport = 8001\n")
    assert configuration.max_concurrency("qwen3.8-flash", 4) == Setting(4, "recipe")


def test_compiler_worker_flag_overrides_configured_host_budget(tmp_path):
    base = RecipeParameters()
    ctx = ParameterContext(root=base, defaults=base)
    config = load(tmp_path, "[preparation]\ncompile_workers = 24\n")
    assert base.resolve(SimpleNamespace(), config, "m", ctx).compile_workers == 24
    assert base.resolve(SimpleNamespace(compile_workers=32), config, "m", ctx).compile_workers == 32
    assert base.resolve(SimpleNamespace(), load(tmp_path, ""), "m", ctx).compile_workers is None
    with pytest.raises(UsageError, match="positive integer"):
        base.resolve(SimpleNamespace(compile_workers=0), config, "m", ctx)
    resolved_neg = base.resolve(SimpleNamespace(compile_workers=-1), config, "m", ctx)
    with pytest.raises(UsageError, match="positive integer"):
        resolved_neg.validate(ctx)
    with pytest.raises(UsageError, match="at least 1"):
        load(tmp_path, "[preparation]\ncompile_workers = 0\n")


@pytest.mark.parametrize("text", [
    '[preparation]\nforce_moe_decode_a16 = true\n',
    '[defaults]\nimage_concurrency = 4\n',
    '[defaults]\nspeculative_token_capacity = 32\n',
    '[defaults]\nspeculative_graph_budget = 64\n',
    '[defaults]\nspeculative_method = "dflash2"\n',
    '[models."demo"]\nimage_concurrency = 2\n',
    '[models."demo"]\nspeculative_token_capacity = 32\n',
    '[models."demo"]\nspeculative_graph_budget = 32\n',
    '[models."demo"]\nspeculative_method = "mtp"\n',
])
def test_export_recipe_parameter_keys_are_rejected_in_config_toml(tmp_path, text):
    with pytest.raises(UsageError, match="unknown key"):
        load(tmp_path, text)


@pytest.mark.parametrize("text, message", [
    ("[memory]\nfraction = 0.9\n", "unknown key fraction"),
    ("[engine]\npages = 4\n", "unknown section"),
    ("[server]\nport = \"8000\"\n", "port in .* must be int"),
    ("[server]\nport = 70000\n", "1 through 65535"),
    ("[server]\nthinking_token_budget = 0\n", "positive 32-bit integer"),
    ("[server]\nthinking_token_budget = -1\n", "positive 32-bit integer"),
    ("[server]\nthinking_token_budget = 4294967296\n", "positive 32-bit integer"),
    ("[server]\nthinking_token_budget = 1.5\n", "must be int"),
    ("[server]\nthinking_token_budget = true\n", "must be int"),
    ("[server]\nthinking_token_budget = \"1536\"\n", "must be int"),
    ("[defaults]\nmax_concurrency = true\n", "must be int"),
    ("[defaults]\nmax_concurrency = 0\n", "at least 1"),
    ("[memory]\nconversation_cache = 16\n", "unknown key conversation_cache"),
    ("[memory]\nreserve = \"lots\"\n", "memory size"),
    ("[memory]\nreserve = 16\n", "must be str"),
    ("[models.\"qwen3.8-flash\"]\ncontext = 4096\n", "unknown key context"),
    ("server = 3\n", "must be a table"),
    ("[server\n", "config.toml"),
])
def test_invalid_files_are_usage_errors(tmp_path, text, message):
    with pytest.raises(UsageError, match=message):
        load(tmp_path, text)


@pytest.mark.parametrize("text, expected", [
    ("0", 0), ("4096", 4096), ("16G", 16 << 30), ("16g", 16 << 30), ("1.5G", 3 << 29), ("512M", 512 << 20),
    ("8 GiB", 8 << 30), ("2T", 2 << 40), ("64K", 64 << 10),
])
def test_sizes_use_binary_units(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "G", "-1G", "16X", "1e3"])
def test_malformed_sizes_are_usage_errors(text):
    with pytest.raises(UsageError):
        parse_size(text)


def test_shard_size_is_not_a_publishing_setting(tmp_path):
    with pytest.raises(UsageError, match='unknown key.*shard_size'):
        load(tmp_path, '[publishing]\nshard_size = "2G"\n')


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_autotune_effort_configuration_and_flag_precedence(tmp_path, effort):
    base = RecipeParameters()
    ctx = ParameterContext(root=base, defaults=base)
    assert base.resolve(SimpleNamespace(), load(tmp_path, ""), "m", ctx).autotune_effort == "default"
    config = load(tmp_path, f'[preparation]\nautotune_effort = "{effort}"\n')
    assert base.resolve(SimpleNamespace(), config, "m", ctx).autotune_effort == effort
    assert base.resolve(SimpleNamespace(autotune_effort="off"), config, "m", ctx).autotune_effort == "off"
    assert base.resolve(SimpleNamespace(autotune_effort="audit"), config, "m", ctx).autotune_effort == "audit"
    for legacy in ("exhaustive", "medium", "max"):
        with pytest.raises(UsageError, match="autotune_effort"):
            load(tmp_path, f'[preparation]\nautotune_effort = "{legacy}"\n')


@pytest.mark.parametrize("value", ["on", "off", "0", "90s", "1.5m", "20ms", "2h", "0.001"])
def test_shutdown_drain_configuration(tmp_path, value):
    config = load(tmp_path, f'[prefix_cache]\ndrain_on_shutdown = "{value}"\n')
    assert config.get("prefix_cache.drain_on_shutdown").value == value
    assert config.get("prefix_cache.drain_on_shutdown", "off").value == "off"


@pytest.mark.parametrize("value", ["-1", "NaN", "inf", "1e30h", "later", "0s"])
def test_invalid_shutdown_drain_configuration(tmp_path, value):
    with pytest.raises(UsageError, match="drain_on_shutdown"):
        load(tmp_path, f'[prefix_cache]\ndrain_on_shutdown = "{value}"\n')


@pytest.mark.parametrize("value", [0, 0.05, 1])
def test_checkpoint_extra_work_fraction_configuration(tmp_path, value):
    key = "server.scheduler_checkpoint_extra_work_fraction"
    configuration = load(tmp_path, f"[server]\nscheduler_checkpoint_extra_work_fraction = {value}\n")
    assert configuration.get(key) == Setting(value, "[server]")
    assert configuration.get(key, 0) == Setting(0, "flag")


@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan", "inf", "true", '\"0.05\"'])
def test_checkpoint_extra_work_fraction_rejects_invalid_configuration(tmp_path, value):
    with pytest.raises(UsageError, match="scheduler_checkpoint_extra_work_fraction"):
        load(tmp_path, f"[server]\nscheduler_checkpoint_extra_work_fraction = {value}\n")
