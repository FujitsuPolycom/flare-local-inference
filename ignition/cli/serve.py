# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare one checkpoint for native serving or runtime-core export."""

import ipaddress
import json
import socket
import sys
import traceback
from dataclasses import replace

from ignition import planning
from ignition.errors import CapacityError, FlareError, PreparationError, UsageError
from ignition.parameters import explicit_flags_from_args

from . import environment
from .config import Configuration
from .preparation import Host, derive, fetch, resolve_model, resolve_parameters, tuning_devices

# Request bytes allowed per context token: JSON-escaped text (a \uXXXX escape takes six bytes per
# UTF-16 unit) and message structure stay well within this for prompts that fill the context.
_BODY_BYTES_PER_TOKEN = 32


def _checkpoint_fabric(args):
    if args.ranks < 2:
        if args.transfer_streams is not None:
            raise UsageError("--transfer-streams requires multiple host ranks")
        return None
    return dict(ring=args.roce_ring, chain=args.roce_chain,
                streams=8 if args.transfer_streams is None else args.transfer_streams)


def _address(host: str, port: int) -> str:
    """Validate a listening address and confirm that nothing else holds it."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise UsageError(f"--host must be an IP address; got {host!r}") from error
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    try:
        socket.create_server((host, port), family=family).close()
    except OSError as error:
        raise UsageError(f"cannot listen on {host} port {port}: {error.strerror or error}") from error
    return f"[{host}]:{port}" if address.version == 6 else f"{host}:{port}"


def _keep_reserve(plan, usage) -> None:
    """Require that preparation left the configured reserve free on every GPU."""
    free = min(rank.free_bytes for rank in usage)
    if free < plan.reserve_bytes:
        gib = 1 << 30
        raise CapacityError(f"preparation left {free / gib:.1f} GiB free;\n"
                            f"[memory] reserve requires {plan.reserve_bytes / gib:.1f} GiB.")


def _server_options(configuration, model, plan) -> dict:
    """Derive native server settings independently of preparation's physical KV backing."""
    recipe, checkpoint = plan.recipe, plan.checkpoint
    return {
        "model_id": model.model_id, "served_names": [recipe.served_name(checkpoint)],
        "reserve_bytes": plan.reserve_bytes,
        "api_key": configuration.get("server.api_key").value,
        "queue_capacity": 4 * plan.concurrency, "frontend_workers": plan.concurrency,
        "max_body_bytes": plan.max_images * recipe.image_bytes + recipe.context * _BODY_BYTES_PER_TOKEN,
        "context_hint": "context is limited by the checkpoint and available cache capacity",
        "capacity_hint": "--max-concurrency sets the request limit; a lower [memory] reserve leaves more KV pages",
        **({"decode_weight_bytes": checkpoint.decode_weight_bytes} if checkpoint.decode_weight_bytes > 0 else {}),
    }


def _exhausted(error, plan) -> CapacityError | None:
    """Describe a preparation that ran out of GPU memory, or return None for any other failure."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(error, torch.OutOfMemoryError):
        return None
    relief = ["fewer concurrent requests", "more GPUs"]
    if plan.speculation is not None:
        relief.insert(1, "--no-mtp")
    return CapacityError(
        f"preparation ran out of GPU memory: {str(error).split(' Including non-PyTorch memory')[0]}\n"
        f"{', '.join(relief[:-1]).capitalize()}, or {relief[-1]} need less memory.")


def _publisher(configuration, args, model=None, hub_cache=None):
    """The signing publisher; with `model`, carrying the license its cached model card declares."""
    from ignition.preparation.protection import Publisher
    values = configuration.values.get("publishing", {})
    try:
        if not any(key in values for key in ("signing_key", "namespace", "official_source")) and not args.official_source:
            from ignition.development import publisher
            signer = publisher(configuration)
        else:
            signer = Publisher.load(values["signing_key"], values["namespace"],
                                    args.official_source or values["official_source"])
    except (KeyError, ValueError, OSError) as error:
        raise UsageError(f"publishing requires an Ed25519 signing_key, namespace, and official_source: {error}") from error
    if model is None:
        return signer
    from ignition.preparation.model_card import model_license
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    try:
        metadata = inspect_indexed_checkpoint(model.snapshot(hub_cache)).metadata_path
        license = model_license(metadata,
                                None if model.directory is not None else model.checkpoint.repository,
                                model.checkpoint.revision)
    except ValueError as error:
        raise UsageError(str(error)) from error
    return replace(signer, license=license)


def _check_host_shape(args) -> None:
    """Reject a host shape the rank count cannot form, before any model or GPU work."""
    from ignition.planning import host_shape_error

    error = host_shape_error(args.roce_ring, args.roce_chain, args.ranks if args.ranks > 1 else None)
    if error:
        raise UsageError(error)


def _skip_weight_export_flags(args) -> tuple[bool, bool]:
    from ignition.preparation.core_snapshot import _OPTIONS

    active = _OPTIONS.get()
    no_weights = bool(getattr(args, "no_weights", False) or (active is not None and active.no_weights))
    reuse_weights = bool(getattr(args, "reuse_weights", False) or (active is not None and active.reuse_weights is not None))
    return no_weights, reuse_weights


def export(args) -> int:
    from ignition.preparation.core_snapshot import export_options
    from .publishing import validate_repository
    configuration = Configuration.load(args.config)
    _check_host_shape(args)
    model = resolve_model(configuration, args.model)
    repository = getattr(args, 'repo_id', None) or model.recipe.export_repository
    if repository:
        validate_repository(repository)
    _checkpoint_fabric(args)
    if args.ranks > 1 and args.tuning_gpus is not None:
        raise UsageError("--tuning-gpus applies to local preparation; multi-host ranks retain their sharding")
    cache = configuration.directory('core_cache')
    if getattr(args, 'no_weights', False) and not (cache / 'blobs').is_dir():
        raise UsageError('--no-weights requires an existing blob cache; export or convert a core first')
    weight_source = reuse_weights = None
    if getattr(args, 'reuse_weights', False):
        import flare_heliopause
        from .publishing import _public_profiles

        try:
            weight_source = flare_heliopause.export_reuse_weights(
                str(cache), model.model_id, json.dumps(_public_profiles(args, configuration)))
            reuse_weights = json.loads(weight_source.metadata())
        except (ValueError, OSError) as error:
            raise UsageError(str(error)) from error
    with export_options(cache, no_weights=getattr(args, 'no_weights', False) or reuse_weights is not None,
                        reuse_weights=reuse_weights,
                        kernel_cache=configuration.directory('kernel_cache')):
        if args.ranks > 1:
            code, result = _export_hosts(args)
        else:
            code, result = _prepare(args)
    print(f'Exported cache entry: {result.entry_path}')
    if repository:
        print(f'Hugging Face target: {repository}')
    return code


def _export_hosts(args):
    """Prepare one rank per host and collect a shared core snapshot on rank zero."""
    import uuid

    from ignition.progress import phase
    from ignition.preparation.host_ranks import HostRendezvous, local_topology, source_identity, topology_gpu

    _check_host_shape(args)
    if args.rendezvous is None:
        raise UsageError("--ranks greater than one requires --rendezvous ADDRESS:PORT for the peer hosts")
    progress = _serving_progress(args.verbose)
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    model = resolve_model(configuration, args.model)
    if not model.cached(hub_cache):
        raise UsageError(f"{model.checkpoint.alias} is not cached; a single-host export downloads it")
    parameters = resolve_parameters(model, args, configuration, multi_host=True)
    core_cache = configuration.directory("core_cache")
    export_staging = configuration.directory("export_staging")
    no_weights, reuse_weights = _skip_weight_export_flags(args)
    if args.ranks in model.recipe.checkpoint_tensor_parallel(model.checkpoint):
        planned_weights = model.recipe.weights_bytes(model.checkpoint, args.ranks, parameters)
        planned_graphs = model.recipe.costs(args.ranks, parameters.speculation, model.checkpoint).graph_bytes
        planning.check_export_disk_space(
            None,
            model.model_id,
            checkpoint=model.checkpoint,
            core_cache=core_cache,
            export_staging=export_staging,
            no_weights=no_weights,
            reuse_weights=reuse_weights,
            world_size=args.ranks,
            weights_bytes=planned_weights,
            graph_bytes=planned_graphs,
        )
    host = Host.discover()
    selected = planning.parse_gpus(args.gpus or "0")
    if len(selected) != 1 or selected[0] >= len(host.gpus):
        raise UsageError("multi-host ranks each use one GPU; --gpus names this host's")
    local = (host.gpus[selected[0]],)
    local_storage = planning.probe_export_storage(core_cache, export_staging)
    rendezvous = HostRendezvous(args.rendezvous, args.ranks, identity=source_identity(),
                                topology=local_topology(local[0], storage=local_storage),
                                checkpoint_fabric=_checkpoint_fabric(args),
                                checkpoint_token_file=configuration.get("preparation.checkpoint_token_file").value)
    try:
        with phase(progress, "Wait for tensor-parallel hosts") as stage:
            topologies = rendezvous.accept(report=stage.update)
        gpus = tuple(topology_gpu(topology, index) for index, topology in enumerate(topologies))
        plan = planning.plan(
            model.recipe,
            model.checkpoint,
            gpus,
            selected=tuple(range(len(gpus))),
            parameters=parameters,
            reserve_bytes=configuration.reserve_bytes,
            explicit_flags=explicit_flags_from_args(args),
        )
        planning.check_export_disk_space(
            plan,
            model.model_id,
            core_cache=core_cache,
            export_staging=export_staging,
            no_weights=no_weights,
            reuse_weights=reuse_weights,
            topologies=topologies,
        )
    except BaseException:
        rendezvous.abort()
        raise
    plan = replace(plan, hub_cache=hub_cache)
    environment.apply(configuration, local, plan.recipe.b12x_environment)
    options = _server_options(configuration, model, plan)
    options.pop("api_key")
    print(json.dumps({"plan": plan.to_dict()}), file=sys.stderr, flush=True)
    import flare_heliopause

    signals = flare_heliopause.CliSignals()
    signals.__enter__()
    try:
        import torch

        from ignition.preparation.module_images import ModuleImages
        from ignition.preparation.core_storage import CoreStorage

        recorder = ModuleImages()
        with recorder:
            torch.cuda.set_device(0)
            storage = CoreStorage(world=plan.world_size, local_ranks=1)
            with storage:
                prepared = model.recipe.prepare(
                    model.snapshot(hub_cache), plan, compile_forwards=plan.parameters.compiled_forwards,
                    kernel_cache=configuration.directory("kernel_cache"), progress=progress,
                    retain_graphs=True, core_storage=storage, ranks=rendezvous)
        with prepared:
            with phase(progress, "Export native cores", live=False) as stage:
                from ignition.preparation.rebuild_manifest import from_plan

                options['preparation_manifest'] = from_plan(model, plan)
                result = prepared.export(None, options=options, export_id=uuid.uuid4().hex, images=recorder,
                                         storage=storage, publisher=_publisher(configuration, args, model, hub_cache),
                                         staging=configuration.directory("export_staging"),
                                         progress=progress)
                stage.update(f"{result.entry_path} ({plan.world_size} ranks)")
        for region in storage.host_regions:
            region.close()
        return 0, result
    finally:
        signals.__exit__(None, None, None)


def _prepare(args):
    from ignition.progress import StartupProgress, phase

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if args.verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    progress = Progress()
    configuration = Configuration.load(args.config)
    _publisher(configuration, args)
    hub_cache = configuration.directory("hub_cache")
    with phase(progress, "Resolve checkpoint") as stage:
        model = resolve_model(configuration, args.model)
        cached = model.cached(hub_cache)
        stage.update(f"{model.checkpoint.alias} · {model.source} · {'cached' if cached else 'not cached'}")
    with phase(progress, "Select GPUs") as stage:
        parameters = resolve_parameters(model, args, configuration, multi_host=False)
        host = Host.discover()
        plan = derive(
            configuration, model, host, gpus=args.gpus, parameters=parameters,
            tuning_gpus=args.tuning_gpus, explicit_flags=explicit_flags_from_args(args),
        )
        indices = planning.format_gpus(gpu.index for gpu in plan.gpus)
        # A caller's restriction changes which GPU each index names.
        restricted = "" if host.restriction is None else f" of CUDA_VISIBLE_DEVICES={host.restriction}"
        parts = [f"GPU{'s' if plan.world_size > 1 else ''} {indices}{restricted}"]
        parts.extend(plan.parameters.status_parts(plan.parameter_context))
        if plan.tuning_gpus:
            parts.append("tuning GPUs " + planning.format_gpus(gpu.index for gpu in plan.tuning_gpus))
        stage.update(" · ".join(parts))
    no_weights, reuse_weights = _skip_weight_export_flags(args)
    planning.check_export_disk_space(
        plan,
        model.model_id,
        core_cache=configuration.directory("core_cache"),
        export_staging=configuration.directory("export_staging"),
        no_weights=no_weights,
        reuse_weights=reuse_weights,
    )
    if args.config is None and not configuration.exists:
        configuration.write_initial(model.checkpoint.alias, plan.concurrency)
    if not cached:
        if model.directory is not None:
            raise UsageError(f"{model.directory} lacks tensor files named by its safetensors index")
        with phase(progress, "Download checkpoint"):
            fetch(model, hub_cache, progress_bars=False)
            model.verify(hub_cache)

    environment.apply(configuration, plan.gpus, plan.recipe.b12x_environment)
    import flare_heliopause

    # Preparation runs native CUDA work that cannot unwind; a signal ends the process at once.
    signals = flare_heliopause.CliSignals()
    signals.__enter__()
    try:
        try:
            from ignition.preparation.module_images import ModuleImages
            from ignition.preparation.tuning_replicas import prepare_recipe

            recorder = ModuleImages()
            preparation_options = {"retain_graphs": True}
            with recorder:
                import torch
                from ignition.preparation.core_storage import CoreStorage

                torch.cuda.set_device(0)
                storage = CoreStorage(world=plan.world_size)
                preparation_options["core_storage"] = storage
                with storage:
                    prepared = prepare_recipe(
                        model.recipe.prepare,
                        model.snapshot(hub_cache), plan, compile_forwards=plan.parameters.compiled_forwards,
                        kernel_cache=configuration.directory("kernel_cache"), progress=progress,
                        **preparation_options)
        except FlareError:
            raise
        except MemoryError as error:
            raise CapacityError(f"insufficient memory during model preparation: {error}") from error
        except Exception as error:  # noqa: BLE001 - Every other preparation failure maps to one exit code.
            if args.verbose:
                traceback.print_exc()
            raise (_exhausted(error, plan)
                   or PreparationError(f"preparation failed: {type(error).__name__}: {error}")) from error
        with prepared:
            usage = model.recipe.usage(prepared)
            _keep_reserve(plan, usage)
            options = _server_options(configuration, model, plan)
            from ignition.preparation.model_core import export_core

            publisher = _publisher(configuration, args, model, hub_cache)
            with phase(progress, "Export native core", live=False) as stage:
                try:
                    options.pop("api_key")
                    from ignition.preparation.rebuild_manifest import from_plan

                    options['preparation_manifest'] = from_plan(model, plan)
                    result = export_core(prepared, recorder, storage, None, options=options, publisher=publisher,
                                         progress=progress, console=progress.console,
                                         staging=configuration.directory("export_staging"))
                except Exception as error:
                    if args.verbose:
                        traceback.print_exc()
                    raise PreparationError(f"native core export failed: {error}") from error
                stage.update(str(result.entry_path))
            return 0, result
    finally:
        signals.__exit__(None, None, None)


def _serving_progress(verbose):
    from ignition.progress import StartupProgress

    class Progress(StartupProgress):
        def __call__(self, message):
            super().__call__(message)
            if verbose:
                self.console.print(f"  {message}", style="dim", highlight=False)

    return Progress()


def debug_serve(args) -> int:
    """Prepare in this process for diagnostic HTTP requests, optionally across ranks.

    Requires heliopause's qualification build, which embeds the HTTP server in Python. With
    --direct, Ignition's diagnostic step loop serves instead, without Helios or that build.
    Actual serving and comparisons against other runtimes use exported cores with flare run.
    """
    from ignition.progress import phase

    _check_host_shape(args)
    _checkpoint_fabric(args)
    if args.ranks > 1 and args.tuning_gpus is not None:
        raise UsageError("--tuning-gpus applies to local preparation; multi-host ranks retain their sharding")
    if args.eager and not args.direct:
        raise UsageError("--eager applies only to --direct execution")
    if args.graph_times is not None and not args.direct:
        raise UsageError("--graph-times applies only to --direct execution")
    if args.eager and args.graph_times is not None and not args.graph_times.is_file():
        raise UsageError(f"--graph-times {args.graph_times} does not exist; graph replay with --graph-times saves it")
    if args.direct and args.history_drafts:
        raise UsageError("--history-drafts requires native serving; it does not apply to --direct")
    if not args.direct:
        requirement = ("ignition debug-serve requires Heliopause's qualification Cargo feature; rebuild ./heliopause "
                       "with --config-settings 'build-args=--features=qualification'")
        try:
            import flare_heliopause
        except ImportError as error:
            raise UsageError(requirement) from error
        if not hasattr(flare_heliopause, "Server"):
            raise UsageError(requirement)
    progress = _serving_progress(args.verbose)
    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    model = resolve_model(configuration, args.model)
    if not model.cached(hub_cache):
        raise UsageError(f"{model.checkpoint.alias} is not cached; ignition export downloads it")
    parameters = resolve_parameters(model, args, configuration, multi_host=(args.ranks > 1))
    host = Host.discover()
    bind = _address(configuration.get("server.host", args.host).value,
                    configuration.get("server.port", args.port).value)
    rendezvous = None
    if args.ranks == 1:
        gpus = host.gpus if args.gpus is None else tuple(host.gpus[i] for i in planning.parse_gpus(args.gpus))
        local = gpus
    else:
        from ignition.preparation.host_ranks import HostRendezvous, local_topology, source_identity, topology_gpu

        if args.rendezvous is None:
            raise UsageError("--ranks greater than one requires --rendezvous ADDRESS:PORT for the peer hosts")
        selected = planning.parse_gpus(args.gpus or "0")
        if len(selected) != 1 or selected[0] >= len(host.gpus):
            raise UsageError("multi-host ranks each use one GPU; --gpus names this host's")
        local = (host.gpus[selected[0]],)
        rendezvous = HostRendezvous(args.rendezvous, args.ranks, identity=source_identity(),
                                    topology=local_topology(local[0]),
                                    checkpoint_fabric=_checkpoint_fabric(args),
                                    checkpoint_token_file=configuration.get("preparation.checkpoint_token_file").value)
        with phase(progress, "Wait for tensor-parallel hosts") as stage:
            topologies = rendezvous.accept(report=stage.update)
        gpus = tuple(topology_gpu(topology, index) for index, topology in enumerate(topologies))
    plan = planning.plan(
        model.recipe,
        model.checkpoint,
        gpus,
        selected=None if args.ranks == 1 and args.gpus is None else tuple(range(len(gpus))),
        parameters=parameters,
        reserve_bytes=configuration.reserve_bytes,
        cache_states=args.prefix_cache_states,
        explicit_flags=explicit_flags_from_args(args),
    )
    plan = replace(plan, hub_cache=hub_cache)
    if rendezvous is None:
        plan = tuning_devices(plan, host, args.tuning_gpus)
        local = plan.gpus
    environment.apply(configuration, local, plan.recipe.b12x_environment)
    options = _server_options(configuration, model, plan)
    options["thinking_token_budget"] = configuration.get("server.thinking_token_budget").value
    options["scheduler_checkpoint_extra_work_fraction"] = configuration.get("server.scheduler_checkpoint_extra_work_fraction").value
    print(json.dumps({"plan": plan.to_dict()}), file=sys.stderr, flush=True)
    if args.history_drafts is not None:
        if args.history_drafts and (plan.speculation is None or plan.speculation.draft_tokens == 0):
            raise UsageError("--history-drafts requires enabled learned speculation")
        options["history_drafts"] = args.history_drafts
    preparation_options = {}
    storage = None
    if rendezvous is None:
        import torch
        from ignition.preparation.core_storage import CoreStorage

        torch.cuda.set_device(0)
        storage = CoreStorage(world=plan.world_size)
        preparation_options["core_storage"] = storage
    elif rendezvous is not None:
        preparation_options["ranks"] = rendezvous
    from contextlib import nullcontext
    from ignition.preparation.tuning_replicas import prepare_recipe

    with storage if storage is not None else nullcontext():
        prepared = prepare_recipe(
            model.recipe.prepare,
            model.snapshot(hub_cache), plan, compile_forwards=plan.parameters.compiled_forwards,
            kernel_cache=configuration.directory("kernel_cache"), progress=progress,
            **preparation_options)
    if plan.speculation is not None and plan.speculation.packed_verification:
        local_model = prepared.model if hasattr(prepared, "model") else prepared
        if not any(signature.packed_token_capacity for signature in local_model.signatures.values()):
            prepared.close()
            raise UsageError("this prepared recipe does not declare packed verification capacities")
        if plan.speculation.draft_prefix_selection == "gpu" and not any(
                signature.kind == "speculative" and signature.confidence_width > 0 for signature in local_model.signatures.values()):
            prepared.close()
            raise UsageError("this prepared recipe does not declare GPU prefix-selection rounds")
    if args.direct:
        from ignition.direct.server import serve_direct

        code = serve_direct(prepared, bind=bind, eager=args.eager, verbose=args.verbose, graph_times=args.graph_times,
                            served_names=(*options["served_names"], options["model_id"]))
    else:
        code = _serve_prepared(prepared, bind, options)
    # Native replay has drained before core-backed host mappings are released.
    if storage is not None:
        for region in storage.host_regions:
            region.close()
    print("Stopped; every rank drained and released its resources.", flush=True)
    return code


def _diagnostic_options(prepared, options):
    """Limit diagnostic admission to the physical backing retained by preparation."""
    from ignition.components.capacity import request_capacity

    model = getattr(prepared, "model", prepared)
    rows, pages = model.buffers["memory_capacity"].cpu().tolist()
    graphs = model.server_config["model_contract"]["graphs"]
    execution_rows = request_capacity(model.max_active) * max(graph["state_rows_per_request"] for graph in graphs)
    if rows <= execution_rows or pages <= 0:
        raise PreparationError("diagnostic backing cannot hold the prepared request concurrency")
    # Row zero is reserved. Export retains the declared capacities; only this
    # diagnostic server is restricted to the backing that already exists.
    snapshots = min(model.server_config["snapshots"], rows - 1 - execution_rows)
    return {**options, "pages": min(model.server_config["pages"], pages), "snapshots": snapshots}


def _serve_prepared(prepared, bind, options):
    import signal

    with prepared:
        ranks = [{key: value for key, value in rank.items() if key != "tuning"}
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,))]
        print(json.dumps({"memory": ranks}, default=str), file=sys.stderr, flush=True)
        limited = _diagnostic_options(prepared, options)
        print(json.dumps({"diagnostic_capacity": {key: limited[key] for key in ("pages", "snapshots")}}),
              file=sys.stderr, flush=True)
        server = prepared.start_server(bind=bind, options=limited)
        print(f"Listening on {server.address}", flush=True)

        def stop(signum, frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            while server.wait(1000):
                pass
            raise PreparationError("native HTTP server stopped unexpectedly")
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous)


def join(args) -> int:
    """Run one peer rank on this host for rank zero's rendezvous on another host."""
    import time

    from ignition.preparation.host_ranks import knock, local_topology, serve_peer, source_identity

    started = time.monotonic()

    def progress(message):
        print(f"{time.monotonic() - started:8.1f}s rank {args.rank}: {message}", flush=True)

    configuration = Configuration.load(args.config)
    hub_cache = configuration.directory("hub_cache")
    host = Host.discover()
    if not 0 <= args.gpu < len(host.gpus):
        raise UsageError(f"--gpu {args.gpu} names a GPU this host does not have")
    gpu = host.gpus[args.gpu]
    environment.apply(configuration, (gpu,))
    identity = source_identity()
    core_cache = configuration.directory("core_cache")
    export_staging = configuration.directory("export_staging")
    local_storage = planning.probe_export_storage(core_cache, export_staging)
    control, executor = knock(args.rendezvous, args.rank, args.ranks, identity=identity,
                              topology=local_topology(gpu, storage=local_storage), report=progress)
    serve_peer(control, executor, args.rank, args.ranks, hub_cache=hub_cache, progress=progress,
               export_staging=export_staging, identity=identity,
               core_cache=core_cache,
               checkpoint_token_file=configuration.get("preparation.checkpoint_token_file").value)
    return 0
