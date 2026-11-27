"""Export prepared column gathers for native replay after Python exits."""

from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from collective_core_fixture import _configuration, _process
from signing import publisher


def _rank(rank, directory):
    # Compiled forwards are pickled to compile workers, so their classes must live in an importable module.
    from b12x.comm.pcie import DcpAllToAll
    from tests.components.gather_core_forwards import (
        FixtureAllGather, FixtureVarlenGather, GatherForward, RecordForward, VARLEN_BANK_BYTES)
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.model import PreparedModel
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.runtime_core import export_model
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(directory / "gloo").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=5))
    folder = directory / f"rank-{rank}"
    folder.mkdir()
    runtimes, graphs = [], {}
    prepared = session = None
    try:
        with ModuleImages() as images, CoreStorage(rank=rank, world=4, device=rank) as storage:
            try:
                coordinator = CollectivePreparation(dist.group.WORLD)
                runtimes = [DcpAllToAll.from_exchange_group(exchange_group=dist.group.WORLD, device=device,
                    max_batch_size=256, total_heads=columns // 64, head_dim=64, stream_affine=False,
                    ipc=storage.peer_storage) for columns in (256, 2048, 256, 16384)]
                session = RecordedPreparationSession(device=device, checkpoint="gather-core",
                    workload_schema="tp4-column-bits", numerical_policy="exact", coordinator=coordinator,
                    compile_workers=4, benchmark_storage=storage)
                shapes = {"init": (1, torch.bfloat16, 0, 128), "decode": (1, torch.float32, 0, 128),
                          "prefill64": (16, torch.float8_e4m3fn, 1, 512),
                          "prefill128": (4, torch.bfloat16, 3, 8192),
                          "resume": (1, torch.bfloat16, 2, 128)}
                sources, outputs, operations = {}, {}, {}
                for name, (rows, dtype, channel, width) in shapes.items():
                    source = sources[name] = torch.zeros((rows, width // dtype.itemsize), dtype=dtype, device=device)
                    output = outputs[name] = torch.empty((rows, source.shape[1] * 4), dtype=dtype, device=device)
                    operations[name] = FixtureAllGather(source, output, runtime=runtimes[channel], session=session,
                        coordinator=coordinator, name=name)
                runtimes.append(DcpAllToAll.from_exchange_group(exchange_group=dist.group.WORLD, device=device,
                    max_batch_size=1, total_heads=(VARLEN_BANK_BYTES // 2) // 64, head_dim=64,
                    stream_affine=False, ipc=storage.peer_storage))
                window = 16
                cache = sources["prefill256"] = torch.zeros((16, 64, 528), dtype=torch.uint8, device=device)
                table_values = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
                table = table_values.to(device)
                output = outputs["prefill256"] = torch.empty((256, 528), dtype=torch.uint8, device=device)
                records = torch.empty((4, window * 528), dtype=torch.uint8, device=device)
                counts = torch.zeros(4, dtype=torch.int64, device=device)
                windows = tuple((first, FixtureVarlenGather(records, counts, runtime=runtimes[-1],
                    session=session, coordinator=coordinator, name=f"records{first}")) for first in range(0, 64, window))
                rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
                record_call = RecordForward(windows, rank_tensor, 4, cache, table, output, window)
                session.tune(show_progress=False)
                signatures = {name: GraphSignature("prefill" if name.startswith("prefill") else name, 1,
                    int(name.removeprefix("prefill")) if name.startswith("prefill") else 1) for name in shapes}
                signatures["prefill256"] = GraphSignature("prefill", 1, 256)
                signatures.update({name: GraphSignature(name, 1, 1) for name in ("restore", "snapshot")})
                functions = {name: GatherForward(operations[name if name in operations else "init"],
                    sources[name if name in operations else "init"]) for name in signatures}
                functions["prefill256"] = record_call
                stream = torch.cuda.Stream(device=device)
                # Every forward is a class instance, so capture compiles them all as serving does.
                graphs, _ = capture_forwards(functions, signatures, session=session, device=device, stream=stream,
                    pool=storage.graph_pool.id, coordinator=coordinator, compile_decoder=True, retain_graphs=True)
                session.freeze()
                state = storage.device_allocate("continuations", (4, 32), torch.uint8,
                    group="STATE_ROWS", maximum=4, units=4)
                pages = storage.device_allocate("pages", (16, 16), torch.uint8,
                    group="KV_PAGES", maximum=16, units=16)
                buffers = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 24), dtype=torch.int64, device=device),
                    "state_pool": state, "memory_capacity": torch.tensor([4, 16], device=device), "pages": pages}
                records = buffers["completion_records"]
                records[..., 4] = -1
                records[..., 12:14] = -1
                records[..., 15] = 1
                records[..., 16:] = -1
                buffers.update({f"source-{name}": value.view(torch.uint8) for name, value in sources.items()})
                buffers.update({f"output-{name}": value.view(torch.uint8) for name, value in outputs.items()})
                configuration = dict(_configuration(folder, signatures), prefill_chunk_size=256, max_sequence_tokens=256)
                prepared = PreparedModel(device=rank, graphs=graphs, buffers=buffers, signatures=signatures,
                    max_active=1, session=session, memory_report={}, server_config=configuration,
                    close_resources=runtimes, resources=(operations, sources, outputs, storage, stream, record_call))
                cases = []
                for step in range(3):
                    for name, (rows, _, _, width) in shapes.items():
                        values = ((torch.arange(4 * rows * width).reshape(4, rows, width) + step * 53) % 256).to(torch.uint8)
                        expected = values.permute(1, 0, 2).contiguous()
                        input_path, expected_path = f"input-{name}-{step}.bin", f"expected-{name}-{step}.bin"
                        (folder / input_path).write_bytes(values[rank].numpy().tobytes())
                        (folder / expected_path).write_bytes(expected.numpy().tobytes())
                        cases.append(dict(graph=name, source=f"source-{name}", output=f"output-{name}",
                            input=f"rank-{rank}/{input_path}", expected=f"rank-{rank}/{expected_path}", step=step))
                    values = ((torch.arange(4 * 16 * 64 * 528).reshape(4, 16, 64, 528)
                        + torch.arange(4).view(4, 1, 1, 1) * 37 + step * 53) % 256).to(torch.uint8)
                    expected = torch.zeros((256, 528), dtype=torch.uint8)
                    for token in range(128):
                        peer = (token // 4) % 4
                        local = (token // 16) * 4 + token % 4
                        expected[token] = values[peer].view(1024, 528)[table_values[local // 256] * 256 + local % 256]
                    input_path, expected_path = f"input-records-{step}.bin", f"expected-records-{step}.bin"
                    (folder / input_path).write_bytes(values[rank].numpy().tobytes())
                    (folder / expected_path).write_bytes(expected.numpy().tobytes())
                    cases.append(dict(graph="prefill256", source="source-prefill256", output="output-prefill256",
                        input=f"rank-{rank}/{input_path}", expected=f"rank-{rank}/{expected_path}", step=step))
                torch.cuda.synchronize(device)
                dist.barrier()
                storage.peer_storage.reset()
                torch.cuda.synchronize(device)
                dist.barrier()
                export_model(prepared, images, storage, directory / f"rank-{rank}.json",
                    options={}, bundle_only=rank != 0, sources=True)
                (folder / "metadata.json").write_text(json.dumps(dict(cases=cases,
                    process=_process(), owned=list(storage.peer_storage.owned.items()))))
                dist.barrier()
            finally:
                if prepared is not None:
                    prepared.close()
                else:
                    for graph in graphs.values():
                        graph.reset()
                    if session is not None:
                        session.close()
                    for runtime in runtimes:
                        runtime.close()
    finally:
        dist.destroy_process_group()


def prepare(directory):
    from ignition.preparation.core_file import combine

    directory.mkdir(exist_ok=True)
    mp.spawn(_rank, args=(directory,), nprocs=4, join=True)
    ranks = [json.loads((directory / f"rank-{rank}/metadata.json").read_text()) for rank in range(4)]
    peers = [dict(owner=rank, address=address, nbytes=size, readers=list(range(4)))
             for rank, metadata in enumerate(ranks) for address, size in metadata["owned"]]
    combine(directory / "core", [directory / f"rank-{rank}.json" for rank in range(4)],
            peers, publisher=publisher())
    (directory / "collective-expected.json").write_text(json.dumps(dict(ranks=ranks,
        preparation_processes=[_process(), *(rank["process"] for rank in ranks)])))
    print("exported TP4 gather core; all four preparation ranks exited", flush=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve())
