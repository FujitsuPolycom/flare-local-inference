# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The `ignition` preparation and diagnostic command line."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

from ignition.errors import FlareError, UsageError
from ignition.parameters import (
    OptionArity,
    ParameterContext,
    RecipeParameters,
    known_core_abi_flags,
)

_PREPARATION_COMMANDS = ("export", "debug-serve", "bench")


def _add_preparation_base_arguments(subparser: argparse.ArgumentParser, *, command: str,
                                    recipes=(), prog="ignition", recipe=None, checkpoint=None) -> None:
    from .help import catalog_epilog, model_epilog

    subparser.usage = "%(prog)s [MODEL] [options]"
    subparser.epilog = (
        model_epilog(recipe, checkpoint, prog=prog)
        if recipe is not None and checkpoint is not None
        else catalog_epilog(recipes=recipes, prog=prog)
    )
    subparser.add_argument(
        "model", nargs="?", metavar="MODEL",
        help="recipe alias, repository id, or checkpoint directory (default: defaults.model)",
    )
    subparser.add_argument("--official-source", metavar="URL", help="published checkpoint repository URL")
    subparser.add_argument(
        "--gpus", metavar="LIST",
        help="GPU indices relative to CUDA_VISIBLE_DEVICES when set, otherwise NVML indices;"
             " such as 0, 0,1, or 0-3;"
             " their count is the tensor-parallel size (default: first GPUs that fit)",
    )
    subparser.add_argument("--verbose", action="store_true", help="per-phase detail on stderr")

    include_multi_host = command in ("export", "debug-serve")
    params = RecipeParameters() if recipe is None else recipe.parameters
    context = ParameterContext(recipe=recipe, checkpoint=checkpoint, root=params, defaults=params, command=command)
    params.add_arguments(subparser, context, include_multi_host=include_multi_host)

    advanced = next((g for g in subparser._action_groups if g.title == "advanced options"), None)
    if advanced is None:
        advanced = subparser.add_argument_group("advanced options")
    advanced.add_argument(
        "--tuning-gpus", metavar="LIST",
        help="local GPUs for kernel tuning, starting with --gpus and followed by complete"
             " TP replicas; serving TP size is unchanged (default: serving GPUs)",
    )

    if include_multi_host:
        hosts = next((g for g in subparser._action_groups if g.title == "multi-host tensor parallelism"), None)
        if hosts is None:
            hosts = subparser.add_argument_group("multi-host tensor parallelism")
        hosts.add_argument(
            "--transfer-streams", type=int, choices=range(1, 65), metavar="N",
            help="checkpoint and export fabric TCP streams per peer, 1..64 (default: 8)",
        )
        hosts.add_argument(
            "--ranks", type=int, default=1, choices=range(1, 17), metavar="N",
            help="tensor-parallel ranks, one per host; this process is rank zero (default: 1);"
                 " an export publishes all ranks to the shared model cache on rank zero",
        )
        hosts.add_argument(
            "--rendezvous", metavar="ADDR:PORT",
            help="IPv4 address and port on which rank zero admits its peers",
        )
    else:
        subparser.set_defaults(roce_ring=False, roce_chain=False)


def _parser(*, recipes=(), prog="ignition", recipe=None, checkpoint=None) -> argparse.ArgumentParser:
    from . import bench, publishing, serve
    from .help import catalog_epilog

    recipes = tuple(recipes)
    parser = argparse.ArgumentParser(
        prog=prog, allow_abbrev=False,
        description="Prepare native runtime cores for NVIDIA SM103, SM120, and SM121 GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=catalog_epilog(recipes=recipes, prog=prog),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, handler, summary):
        subparser = subparsers.add_parser(
            name, help=summary, description=summary, allow_abbrev=False,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparser.set_defaults(handler=handler)
        subparser.add_argument("--config", type=Path, metavar="PATH",
                               help="configuration file (default: ~/.config/flare/config.toml)")
        return subparser

    exporting = command("export", serve.export, "prepare a native model core in the shared cache")
    weights = exporting.add_mutually_exclusive_group()
    weights.add_argument("--no-weights", action="store_true",
                         help="require cached stored-data blobs; export execution resources and metadata")
    weights.add_argument("--reuse-weights", action="store_true",
                         help="reuse the cached default version's weight hashes and layouts without hashing live weights")
    exporting.add_argument("--publisher-key", dest="publisher_keys", action="append", type=Path, default=[],
                           help="public publisher profile authenticating the cached weight source")
    exporting.add_argument("--repo-id", metavar="NAMESPACE/REPOSITORY",
                           help="Hugging Face target used to organize local exports (default: recipe target)")
    _add_preparation_base_arguments(exporting, command="export", recipes=recipes, prog=prog,
                                    recipe=recipe, checkpoint=checkpoint)

    benchmarking = command(
        "bench", bench.run,
        "measure complete prepared GPU graph latency without export or HTTP; not serving throughput",
    )
    _add_preparation_base_arguments(benchmarking, command="bench", recipes=recipes, prog=prog,
                                    recipe=recipe, checkpoint=checkpoint)
    bench.arguments(benchmarking)

    serving = command(
        "debug-serve", serve.debug_serve,
        "prepare and run diagnostic HTTP requests; not for actual serving or performance"
        " comparisons against other runtimes. Use ignition export and flare run for those workflows.",
    )
    serving.add_argument("--host", metavar="ADDR", help="listening address (default: server.host, else 127.0.0.1)")
    serving.add_argument("--port", type=int, metavar="N", help="listening port (default: server.port, else 8000)")
    _add_preparation_base_arguments(serving, command="debug-serve", recipes=recipes, prog=prog,
                                    recipe=recipe, checkpoint=checkpoint)
    serving.add_argument("--prefix-cache-states", type=int, metavar="N",
                         help="advanced: cached request states kept resident for prefix reuse; fewer leave more"
                              " memory for KV pages (default: sized from free memory)")
    serving.add_argument("--history-drafts", action=argparse.BooleanOptionalAction, default=None,
                         help="augment learned drafts with continuations from shared decoded history;"
                              " requires prepared history inputs (default: off)")
    diagnostics = serving.add_argument_group(
        "diagnostics",
        "Ignition's direct execution reproduces model-level faults without Helios. It is a"
        " diagnostic and qualification path, not production serving.",
    )
    diagnostics.add_argument("--direct", action="store_true",
                             help="schedule and pack each step in Python and replay its captured graph on every"
                                  " rank, with no Helios scheduler or native server; non-streaming chat only")
    diagnostics.add_argument("--eager", action="store_true",
                             help="with --direct: call the forward each graph captured instead of replaying it")
    diagnostics.add_argument("--graph-times", type=Path, metavar="PATH",
                             help="with --direct: graph replay saves its estimate of each graph's time here; with"
                                  " --eager, decode rounds are chosen from the times saved here")

    joining = command("join", serve.join, "run a peer rank for rank zero's rendezvous on another host")
    joining.add_argument("rendezvous", metavar="ADDR:PORT", help="rank zero's rendezvous address")
    joining.add_argument("--rank", type=int, required=True, metavar="R", help="this host's rank, 1 through N-1")
    joining.add_argument("--ranks", type=int, required=True, choices=range(2, 17), metavar="N", help="tensor-parallel ranks")
    joining.add_argument("--gpu", type=int, default=0, metavar="INDEX", help="this host's GPU (default: 0)")
    keys = command("keygen", publishing.keygen, "create an encrypted production signing key and public profile")
    keys.add_argument("--directory", required=True, type=Path, help="new private directory; must not exist")
    keys.add_argument("--name", required=True, help="publisher display name")
    keys.add_argument("--namespace", required=True, help="publisher Hugging Face namespace")
    password = command("key-password", publishing.key_password, "change a signing key's passphrase")
    password.add_argument("--signing-key", required=True, type=Path, help="Ed25519 PEM key to re-encrypt in place")
    signing = command("sign", publishing.sign, "verify and sign a development core for production")
    signing.add_argument("core", type=Path, help="signed development .safetensors core")
    signing.add_argument("--output", required=True, type=Path, help="destination core; must not exist")
    signing.add_argument("--signing-key", type=Path, help="Ed25519 PEM key (default: publishing.signing_key)")
    signing.add_argument("--namespace", help="publisher namespace (default: publishing.namespace)")
    signing.add_argument("--official-source", help="published repository URL (default: publishing.official_source)")
    signing.add_argument("--development-publisher", type=Path, help="source public profile (default: beside config)")
    editing = command("edit", publishing.edit, "edit core frontend metadata without rebuilding the model")
    editing.add_argument("core", type=Path, help="signed payload core file or directory")
    editing.add_argument("--output", required=True, type=Path, help="destination directory; must not exist")
    editing.add_argument("--served-model-name", help="replace the served alias for every deployment")
    editing.add_argument("--chat-template", type=Path, help="replacement UTF-8 Jinja template with message-boundary marks")
    editing.add_argument("--publisher-keys", type=Path, action="append", default=[],
                         help="public publisher profiles authenticating the source; repeatable")
    editing.add_argument("--development-publisher", type=Path, help="source development public profile (default: beside config)")
    combining = command("combine", publishing.combine, "combine prepared TP cores into one signed core")
    combining.add_argument("sources", nargs="+", type=Path, help="prepared payload-format cores")
    combining.add_argument("--output", required=True, type=Path, help="destination core directory; must not exist")
    combining.add_argument("--signing-key", type=Path, help="Ed25519 PEM key (default: publishing.signing_key)")
    combining.add_argument("--namespace", help="publisher namespace (default: publishing.namespace)")
    combining.add_argument("--official-source", help="published repository URL (default: publishing.official_source)")
    combining.add_argument("--publisher-keys", type=Path, action="append", default=[],
                           help="trusted public publisher profiles of the sources; repeatable")
    converting = command("convert", publishing.convert, "convert a core to the selected canonical container form")
    converting.add_argument("core", type=Path, help="existing signed core file or directory, including a cache entry")
    converting.add_argument("--output", type=Path, help="standalone destination (default: shared model cache entry)")
    converting.add_argument("--publisher-keys", type=Path, action="append", default=[],
                            help="trusted public publisher profiles; repeatable")
    converting.add_argument("--form", choices=("directory", "packed"), default="directory",
                            help="payload container form: a directory of companions or one packed .safetensors file (default: directory)")
    return parser


def _subcommand_parser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise RuntimeError(f"missing subcommand parser for {name}")


def _parser_option_arities(subparser: argparse.ArgumentParser) -> dict[str, OptionArity]:
    arities: dict[str, OptionArity] = {}
    for action in subparser._actions:
        if not action.option_strings:
            continue
        if action.nargs == 0:
            arity = OptionArity(0, 0)
        elif action.nargs == "?":
            choices = frozenset(map(str, action.choices)) if action.choices else frozenset()
            arity = OptionArity(0, 1, choices)
        else:
            arity = OptionArity(1, 1)
        for flag in action.option_strings:
            arities[flag] = arity
    return arities


def _all_recipe_option_arities(recipes: Sequence, *, include_multi_host: bool) -> dict[str, OptionArity]:
    arities: dict[str, OptionArity] = {}
    for recipe in recipes:
        checkpoints = recipe.checkpoints or (None,)
        for checkpoint in checkpoints:
            ctx = ParameterContext(recipe=recipe, checkpoint=checkpoint, root=recipe.parameters, defaults=recipe.parameters)
            arities.update(recipe.parameters.option_arities(ctx, include_multi_host=include_multi_host))
    return arities


def _prescan(tokens: Sequence[str], arities: Mapping[str, OptionArity]) -> tuple[bool, Path | None, str | None]:
    """Scan subcommand tokens to detect `--help`, `--config`, and positional `MODEL`."""
    wants_help = False
    config_path: Path | None = None
    raw_model: str | None = None
    i = 0
    n = len(tokens)
    after_double_dash = False
    while i < n:
        tok = tokens[i]
        if after_double_dash:
            if raw_model is None:
                raw_model = tok
            i += 1
            continue
        if tok == "--":
            after_double_dash = True
            i += 1
            continue
        if tok in ("-h", "--help"):
            wants_help = True
            i += 1
            continue
        if tok.startswith("--"):
            flag, eq, inline = tok.partition("=")
            if flag == "--config":
                if eq:
                    config_path = Path(inline)
                    i += 1
                elif i + 1 < n:
                    config_path = Path(tokens[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            arity = arities.get(flag)
            if arity is not None:
                if eq:
                    i += 1
                elif arity.min_args == 1:
                    i += 2 if i + 1 < n else 1
                elif arity.max_args == 1 and arity.optional_choices is not None:
                    if i + 1 < n and not tokens[i + 1].startswith("-") and tokens[i + 1] in arity.optional_choices:
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
                continue
            i += 1
            continue
        if tok.startswith("-") and tok != "-":
            i += 1
            continue
        if raw_model is None:
            raw_model = tok
        i += 1
    return wants_help, config_path, raw_model


def _match_offline(model: str, recipes: Sequence):
    """Match an alias or declared repository offline without touching config, disk, or network."""
    repository, _, revision = model.partition("@")
    for recipe in recipes:
        for checkpoint in recipe.checkpoints:
            if model == checkpoint.alias:
                return recipe, checkpoint
            matches = (
                checkpoint.repository == repository
                if Path(checkpoint.repository).is_absolute()
                else checkpoint.repository.casefold() == repository.casefold()
            )
            if matches and revision in ("", checkpoint.revision):
                return recipe, checkpoint
    return None


def parse_args(argv=None, *, recipes=(), prog="ignition"):
    """Parse `ignition` CLI arguments with dynamic recipe-parameter registration and help."""
    from ignition.registry import resolve
    from .config import Configuration

    tokens = list(sys.argv[1:] if argv is None else argv)
    recipes = tuple(recipes)
    if not tokens or tokens[0] not in _PREPARATION_COMMANDS:
        return _parser(recipes=recipes, prog=prog).parse_args(tokens)

    command = tokens[0]
    rest = tokens[1:]
    include_multi_host = command in ("export", "debug-serve")

    bare_root = _parser(recipes=recipes, prog=prog)
    bare_sub = _subcommand_parser(bare_root, command)
    base_arities = _parser_option_arities(bare_sub)
    recipe_arities = _all_recipe_option_arities(recipes, include_multi_host=include_multi_host)
    combined_arities = {**recipe_arities, **base_arities}

    wants_help, config_path, raw_model = _prescan(rest, combined_arities)
    if wants_help:
        matched = _match_offline(raw_model, recipes) if raw_model is not None else None
        if matched is not None:
            recipe, checkpoint = matched
            dynamic_root = _parser(recipes=recipes, prog=prog, recipe=recipe, checkpoint=checkpoint)
            sub = _subcommand_parser(dynamic_root, command)
            sub.print_help()
            sub.exit(0)
        bare_sub.print_help()
        bare_sub.exit(0)

    configuration = Configuration.load(config_path)
    model_name = raw_model if raw_model is not None else configuration.model
    if model_name is None:
        return bare_root.parse_args(tokens)

    matched = _match_offline(model_name, recipes)
    if matched is not None:
        recipe, checkpoint = matched
    else:
        resolved = resolve(model_name, configuration.directory("hub_cache"), recipes=recipes)
        recipe, checkpoint = resolved.recipe, resolved.checkpoint
    dynamic_root = _parser(recipes=recipes, prog=prog, recipe=recipe, checkpoint=checkpoint)
    dynamic_sub = _subcommand_parser(dynamic_root, command)
    args, unknown = dynamic_root.parse_known_args(tokens)
    if unknown:
        known_other_flags = (set(recipe_arities) | set(known_core_abi_flags())) - set(_parser_option_arities(dynamic_sub))
        for token in unknown:
            flag = token.partition("=")[0]
            if flag in known_other_flags:
                raise UsageError(f"{flag} is not supported by {checkpoint.alias}")
        dynamic_sub.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def main(argv=None, *, recipes=(), prog="ignition", source_roots=()) -> int:
    from ignition.preparation.source_identity import source_roots as bind_source_roots
    from ignition.registry import active_recipes

    recipes = tuple(recipes)
    try:
        with active_recipes(recipes), bind_source_roots(source_roots):
            args = parse_args(argv, recipes=recipes, prog=prog)
            return args.handler(args) or 0
    except FlareError as error:
        from . import display

        display.error(str(error))
        return error.exit_code
