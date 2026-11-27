# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank zero admits each peer rank once, from its own host, running rank zero's sources."""

import threading

import pytest

from ignition.preparation import host_ranks
from ignition.preparation.host_ranks import HostRendezvous, knock


def topology(boot):
    return {"host": boot, "boot_id": boot, "rdma": [],
            "gpu": {"name": "NVIDIA GB10", "uuid": f"GPU-{boot}", "total_bytes": 1 << 37,
                    "free_bytes": 1 << 36, "capability": [12, 1]}}


def rendezvous(world=2):
    owner = HostRendezvous("127.0.0.1:0", world, identity="same", topology=topology("zero"))
    return owner, f"127.0.0.1:{owner.listener.getsockname()[1]}"


def join(address, rank, world=2, *, identity="same", boot=None, results=None):
    try:
        sockets = knock(address, rank, world, identity=identity, topology=topology(boot or f"peer{rank}"),
                        timeout=10)
        if results is not None:
            results.append(sockets)
    except RuntimeError as error:
        if results is not None:
            results.append(error)


@pytest.mark.parametrize("world", [3, 4, 6])
def test_admits_every_peer_rank_with_its_executor_connection(world):
    owner, address = rendezvous(world)
    results = []
    peers = [threading.Thread(target=join, args=(address, rank, world), kwargs={"results": results})
             for rank in reversed(range(1, world))]
    for peer in peers:
        peer.start()
    topologies = owner.accept(timeout=10)
    for peer in peers:
        peer.join()
    expected = ("zero", *(f"peer{rank}" for rank in range(1, world)))
    assert tuple(item["boot_id"] for item in topologies) == expected
    assert owner.hosts == expected
    assert all(isinstance(result, tuple) for result in results)
    group = host_ranks.HostRankGroup.__new__(host_ranks.HostRankGroup)
    group.rendezvous = owner
    assert len(group.connections) == len(group.controls) == world - 1


@pytest.mark.parametrize("arguments, reason", [
    ({"identity": "other"}, "source identity"),
    ({"world": 4}, "world size"),
])
def test_rejects_peers_that_differ_from_rank_zero(arguments, reason):
    owner, address = rendezvous()
    results = []
    peer = threading.Thread(target=join, args=(address, 1), kwargs={**arguments, "results": results})
    peer.start()
    with pytest.raises(TimeoutError):
        owner.accept(timeout=2)
    peer.join()
    assert isinstance(results[0], RuntimeError) and reason in str(results[0])


def test_rejects_a_second_process_for_an_admitted_rank():
    owner, address = rendezvous(4)
    failures = []

    def admit():
        try:
            owner.accept(timeout=3)
        except TimeoutError as error:
            failures.append(error)

    accepting = threading.Thread(target=admit)
    accepting.start()
    results = []
    join(address, 1, 4, results=results)
    join(address, 1, 4, results=results)
    accepting.join()
    assert isinstance(results[0], tuple) and failures
    assert isinstance(results[1], RuntimeError) and "already joined" in str(results[1])


def test_requires_one_rank_per_host():
    owner, address = rendezvous()
    peer = threading.Thread(target=join, args=(address, 1), kwargs={"boot": "zero"})
    peer.start()
    with pytest.raises(ValueError, match="different host"):
        owner.accept(timeout=10)
    peer.join()


def test_finds_the_interface_holding_an_address():
    assert host_ranks.interface_of("127.0.0.1") == "lo"
    with pytest.raises(RuntimeError):
        host_ranks.interface_of("203.0.113.7")


def test_multi_host_preparation_requires_a_hub_checkpoint(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="hub cache"):
        owner.prepare(tmp_path / "local-model", None, None,
                      checkpoint=SimpleNamespace(repository="org/model", revision="rev"))


def test_local_checkpoint_source_must_match_its_declared_directory(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="declared source"):
        owner.prepare(tmp_path / "source", None, None,
                      checkpoint=SimpleNamespace(repository=str(tmp_path / "other"), revision="path-id"))



def test_export_manifest_transfers_only_absent_files(tmp_path):
    from ignition.preparation.core_file import container_header, read_sources, write_sources
    from ignition.preparation.runtime_core_pb2 import ModelIndex, BundleIndex, Span
    from tests.preparation.test_export_transfer import transfer
    import hashlib

    data = container_header(8) + b'weights!'
    weights = tmp_path / (hashlib.sha256(data).hexdigest() + '.safetensors')
    weights.write_bytes(data)
    code = tmp_path / 'code.bin'
    code.write_bytes(b'kernel')
    bundle = BundleIndex(resources={'weights': Span(nbytes=8), 'kernel': Span(offset=4096, nbytes=6)})
    bundle.allocations.add(name='weights', file='weights')
    bundle.modules.add(name='kernel', file='kernel')
    manifest = tmp_path / 'rank.json'
    write_sources(manifest, ModelIndex(version=4, ranks=[bundle]),
                  [(0, weights, 4096, 8), (4096, code, 0, 6)])
    for iteration in range(2):
        destination = tmp_path / f'received-{iteration}' / 'rank.json'
        report = transfer(manifest, destination, tmp_path / 'cache', streams=2, no_weights=iteration == 1)
        assert report['files'] == (2 if iteration == 0 else 1)
        assert report['payload_bytes'] == (len(data) + 6 if iteration == 0 else 6)
        index, locate = read_sources(destination)
        for name, expected in [('weights', b'weights!'), ('kernel', b'kernel')]:
            path, offset, count = locate(index.ranks[0].resources[name])[0]
            assert path.read_bytes()[offset:offset + count] == expected


def test_export_subprocess_reopens_more_sources_than_descriptor_limit(tmp_path):
    import subprocess
    import sys
    from contextlib import ExitStack
    from tests.preparation.test_export_transfer import channels, manifest, receive, sockets, assert_contents

    path, contents = manifest(tmp_path / 'peer', 256, blob=False)
    code = """
import resource, socket, sys
from ignition.preparation.host_ranks import send_export_sources
from ignition.preparation.checkpoint_source import Channel
from tests.preparation.test_export_transfer import core
resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
with socket.socket(fileno=int(sys.argv[1])) as control:
    control.settimeout(10)
    connections = [socket.socket(fileno=int(fd)) for fd in sys.argv[3:]]
    channels = [Channel(connection, b'k' * 32, server=False) for connection in connections]
    send_export_sources(control, core(sys.argv[2]), {}, channels=channels)
"""
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 2)
        inherited = (peer.fileno(), *(channel.connection.fileno() for channel in peers))
        child = subprocess.Popen([sys.executable, '-c', code, str(peer.fileno()), str(path),
                                  *(str(channel.connection.fileno()) for channel in peers)],
                                 pass_fds=inherited, stderr=subprocess.PIPE)
        peer.close()
        for channel in peers:
            channel.connection.close()
        try:
            report = receive(control, roots, tmp_path / 'received.json', tmp_path / 'cache')
            _, errors = child.communicate(timeout=10)
            assert child.returncode == 0, errors.decode()
            assert report['files'] == 256
            assert_contents(tmp_path / 'received.json', contents)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_cluster_export_combines_received_sources_into_payload_core(tmp_path):
    from helios.tests.signing import publisher
    from ignition.preparation.core_file import combine
    from ignition.preparation.core_snapshot import export_options
    from tests.preparation.test_export_transfer import transfer
    from tests.preparation.payload_fixture import rank_source, manifest, schema
    from safetensors import safe_open

    local = rank_source(tmp_path / 'local', data=b'weights-0')
    remote = rank_source(tmp_path / 'remote', data=b'weights-1')
    received = tmp_path / 'received' / 'rank-1.json'
    report = transfer(remote, received, tmp_path / 'cache', streams=2)
    assert report['files'] == 3
    output = tmp_path / 'core'
    with export_options(tmp_path / 'cache'):
        combine(output, [local, received], [], publisher=publisher(), cluster=(2, 'fixture-cluster'))
    envelope = manifest(output)
    deployment = schema()('DeploymentBody').FromString(envelope.deployments[0].body)
    assert deployment.cluster.world == 2 and deployment.cluster.export_id == 'fixture-cluster'
    assert len(envelope.ranks) == 2
    contents = set()
    for path in output.rglob('*.safetensors'):
        with safe_open(path, framework='numpy') as file:
            for key in file.keys():
                contents.add(file.get_tensor(key).tobytes())
    assert {b'weights-0', b'weights-1'} <= contents


def test_host_export_cache_only_uses_native_workspace_and_staging(tmp_path, monkeypatch):
    """Cache-only host export passes no destination, propagates staging, and sets cache_only."""
    from pathlib import Path
    from types import SimpleNamespace

    import flare_heliopause
    from ignition.preparation import host_ranks, core_file, runtime_core

    calls = {}

    class Stage:
        def __init__(self, destination, staging, cache):
            calls['stage'] = (destination, staging, cache)
            self.path = str(tmp_path / 'workspace')
            (tmp_path / 'workspace').mkdir(exist_ok=True)

        def rank_path(self, rank):
            return str(tmp_path / 'workspace' / f'rank-{rank}.json')

        def retain_sources(self, paths):
            calls.setdefault('retained', []).append(list(paths))

        def close(self):
            calls['closed'] = True

    def fake_export_model(model, images, storage, path, **kwargs):
        Path(path).write_text('{}')
        return SimpleNamespace(memory_inventory={})

    def fake_combine(destination, roots, peers, **kwargs):
        calls['combine'] = (destination, list(roots), kwargs)
        return SimpleNamespace(model='m', entry='e')

    monkeypatch.setattr(flare_heliopause, 'NativeExportStage', Stage)
    monkeypatch.setattr(runtime_core, 'export_model', fake_export_model)
    monkeypatch.setattr(core_file, 'combine', fake_combine)
    monkeypatch.setattr(host_ranks, 'send_object', lambda control, value: None)
    from ignition.preparation import memory_inventory
    monkeypatch.setattr(memory_inventory, 'print_inventory', lambda *args, **kwargs: None)
    monkeypatch.setattr(host_ranks, '_export_channels', lambda channels: [])
    monkeypatch.setattr(
        flare_heliopause, 'export_receive_ranks',
        lambda requests, cache, no_weights, owners, stage: __import__('json').dumps(
            [{'inventory': {}, 'channels': [{'sent': 0, 'received': 0}], 'stats': {
                'payload_bytes': 0, 'payload_seconds': 0.0, 'payload_gbps': 0.0,
                'streams': 1, 'transport': 'fabric-tcp'}}]))

    group = object.__new__(host_ranks.HostRankGroup)
    control = SimpleNamespace(fileno=lambda: 7)
    group.rendezvous = SimpleNamespace(world=2, peers={1: SimpleNamespace(control=control)})
    group.model = object()
    channel = SimpleNamespace(sent=0, received=0)
    group.checkpoint_fabric = SimpleNamespace(
        readers={(1, 0): SimpleNamespace(channel=channel)}, streams=1, close=lambda: None)
    group._started = False
    group._closed = False
    group._close_checkpoint_fabric = lambda: None
    group.close = lambda: None
    storage = SimpleNamespace(host_regions=[])
    staging = tmp_path / 'staging'

    options = {}

    from ignition.preparation.core_snapshot import export_options
    with export_options(tmp_path / 'cache'):
        result = host_ranks.HostRankGroup.export(
            group, None, options=options, export_id='export', images=None,
            storage=storage, publisher=object(), staging=staging, form='directory')
    assert calls['stage'] == (None, staging, tmp_path / 'cache')
    assert calls['closed'] is True
    destination, _roots, kwargs = calls['combine']
    assert destination is None
    assert kwargs['cache_only'] is True
    assert result.model == 'm'


def test_host_export_without_configured_cache_fails_before_peer_commands(tmp_path, monkeypatch):
    """Missing core-cache configuration fails before any workspace or peer side effect."""
    from types import SimpleNamespace

    import flare_heliopause
    from ignition.preparation import host_ranks

    side_effects = []

    class Stage:
        def __init__(self, *args):
            side_effects.append(('stage', args))

    monkeypatch.setattr(flare_heliopause, 'NativeExportStage', Stage)
    monkeypatch.setattr(host_ranks, 'send_object',
                        lambda control, value: side_effects.append(('send', value)))

    control = SimpleNamespace(fileno=lambda: 7)
    group = object.__new__(host_ranks.HostRankGroup)
    group.rendezvous = SimpleNamespace(world=2, peers={1: SimpleNamespace(control=control)})
    group.model = object()
    group.checkpoint_fabric = SimpleNamespace(readers={}, streams=1, close=lambda: None)
    group._started = False
    group._closed = False

    # No export_options context: settings are absent.
    with pytest.raises(RuntimeError, match='configured core cache'):
        host_ranks.HostRankGroup.export(
            group, None, options={}, export_id='export', images=None,
            storage=SimpleNamespace(host_regions=[]), publisher=object())
    assert side_effects == [], f"peer or workspace side effects: {side_effects}"


def test_host_export_overlaps_local_export_with_peer_receive_and_aborts_sockets_on_failure(tmp_path, monkeypatch):
    import contextlib
    import json
    from pathlib import Path
    import socket
    import struct
    import threading
    import time
    from types import SimpleNamespace

    import flare_heliopause
    from ignition.preparation import checkpoint_source, core_file, host_ranks, memory_inventory, runtime_core
    from ignition.preparation.core_snapshot import export_options

    # Verify RootSource pre-indexes safetensors header extents and preserves single-row and strided reads.
    snap = tmp_path / "snap"
    snap.mkdir()
    st_header = json.dumps({"w": {"dtype": "U8", "shape": [2, 4], "data_offsets": [0, 8]}}).encode()
    (snap / "model.safetensors").write_bytes(struct.pack("<Q", len(st_header)) + st_header + bytes(range(8)))
    (snap / "config.json").write_text("{}")
    root_src = checkpoint_source.RootSource(snap, SimpleNamespace(repository=str(snap), revision="r0"))
    try:
        start = 8 + len(st_header)
        assert root_src._header_bytes == {"config.json": None, "model.safetensors": start}
        assert root_src.read("model.safetensors", start, 8) == bytes(range(8))
        assert root_src.read("model.safetensors", start, 4, rows=2, stride=4) == bytes(range(8))
        assert root_src.read("model.safetensors", start + 1, 2, rows=2, stride=4) == bytes((1, 2, 5, 6))
    finally:
        root_src.close()

    class Stage:
        def __init__(self, destination, staging, cache):
            (tmp_path / "ws").mkdir(exist_ok=True)

        def rank_path(self, rank):
            return str(tmp_path / "ws" / f"rank-{rank}.json")

        def retain_sources(self, paths):
            pass

        def close(self):
            pass

    monkeypatch.setattr(flare_heliopause, "NativeExportStage", Stage)
    monkeypatch.setattr(host_ranks, "send_object", lambda control, value: None)
    monkeypatch.setattr(host_ranks, "_export_channels", lambda channels: [])
    monkeypatch.setattr(memory_inventory, "print_inventory", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_file, "combine", lambda *args, **kwargs: SimpleNamespace(ok=True))

    # 1. Prove concurrent overlap: export_model and export_receive_ranks synchronize on a 2-party barrier.
    overlap_barrier = threading.Barrier(2, timeout=5.0)
    fabric_closed = []

    def overlapping_export_model(model, images, storage, path, **kwargs):
        overlap_barrier.wait()
        Path(path).write_text("{}")
        return SimpleNamespace(memory_inventory={"rank": 0})

    def overlapping_receive_ranks(requests, cache, no_weights, owners, stage):
        overlap_barrier.wait()
        return json.dumps([{
            "inventory": {"rank": 1},
            "channels": [{"sent": 3, "received": 4}],
            "stats": {
                "payload_bytes": 1024,
                "payload_seconds": 0.01,
                "payload_gbps": 1.0,
                "streams": 1,
                "transport": "fabric-tcp",
            },
        }])

    monkeypatch.setattr(runtime_core, "export_model", overlapping_export_model)
    monkeypatch.setattr(flare_heliopause, "export_receive_ranks", overlapping_receive_ranks)

    group = object.__new__(host_ranks.HostRankGroup)
    control = SimpleNamespace(fileno=lambda: 7)
    channel = SimpleNamespace(sent=0, received=0)
    group.rendezvous = SimpleNamespace(world=2, peers={1: SimpleNamespace(control=control)}, stopping=threading.Event())
    group.model = object()
    group.checkpoint_fabric = SimpleNamespace(
        readers={(1, 0): SimpleNamespace(channel=channel)},
        streams=1,
        close=lambda: fabric_closed.append(True),
    )
    group._started = False
    group._closed = False
    group.close = lambda: None

    with export_options(tmp_path / "cache"):
        result = host_ranks.HostRankGroup.export(
            group,
            None,
            options={},
            export_id="exp-overlap",
            images=None,
            storage=SimpleNamespace(host_regions=[]),
            publisher=object(),
        )
    assert result.ok is True
    assert (channel.sent, channel.received) == (3, 4)
    assert fabric_closed == [True]

    # 2. Prove fast socket abort when export_model fails while export_receive_ranks is blocked on peer I/O,
    # with real _watch_until_closed and real HostRankGroup.close().
    ctrl_root, ctrl_peer = socket.socketpair()
    chan_root, chan_peer = socket.socketpair()
    exec_root, exec_peer = socket.socketpair()
    ctrl_peer.settimeout(5.0)
    chan_peer.settimeout(5.0)
    receive_started = threading.Event()
    restored_channels = []
    monkeypatch.setattr(host_ranks, "_restore_export_channels", lambda *args: restored_channels.append(args))

    def blocking_receive_ranks(requests, cache, no_weights, owners, stage):
        receive_started.set()
        # Blocks until coordinator shuts down ctrl_root and chan_root on export_model failure.
        assert ctrl_peer.recv(1) == b""
        assert chan_peer.recv(1) == b""
        raise ConnectionError("peer export socket shut down")

    def failing_export_model(model, images, storage, path, **kwargs):
        assert receive_started.wait(5.0)
        raise RuntimeError("rank zero local export failed")

    monkeypatch.setattr(flare_heliopause, "export_receive_ranks", blocking_receive_ranks)
    monkeypatch.setattr(runtime_core, "export_model", failing_export_model)

    abort_group = object.__new__(host_ranks.HostRankGroup)
    abort_channel = SimpleNamespace(connection=chan_root, sent=1, received=2)
    stopping_event = threading.Event()
    suppress_event = threading.Event()
    abort_group.rendezvous = SimpleNamespace(
        world=2,
        peers={1: SimpleNamespace(control=ctrl_root, executor=exec_root)},
        stopping=stopping_event,
        suppress_watch=suppress_event,
        store=object(),
    )
    model_closed = []
    abort_group.model = SimpleNamespace(close=lambda: model_closed.append(True))
    abort_closed = []
    abort_group.checkpoint_fabric = SimpleNamespace(
        readers={(1, 0): SimpleNamespace(channel=abort_channel)},
        streams=1,
        close=lambda: abort_closed.append(True),
    )
    abort_group._started = False
    abort_group._closed = False
    abort_group._aborted = False
    watcher = host_ranks._watch_until_closed(
        [ctrl_root], stopping_event, "peer disconnected", suppress=suppress_event)

    try:
        with export_options(tmp_path / "cache"):
            with pytest.raises(RuntimeError, match="rank zero local export failed") as caught:
                with abort_group:
                    host_ranks.HostRankGroup.export(
                        abort_group,
                        None,
                        options={},
                        export_id="exp-abort",
                        images=None,
                        storage=SimpleNamespace(host_regions=[]),
                        publisher=object(),
                    )
        watcher.join(timeout=2.0)
        assert not watcher.is_alive()
        assert stopping_event.is_set()
        assert abort_group._aborted is True
        assert abort_group._closed is True
        assert model_closed == [True]
        assert abort_closed == [True]
        assert restored_channels == []
        assert any("peer export receive aborted" in note for note in getattr(caught.value, "__notes__", ()))
    finally:
        stopping_event.set()
        for sock in (ctrl_root, ctrl_peer, chan_root, chan_peer, exec_root, exec_peer):
            with contextlib.suppress(OSError):
                sock.close()

    # 3. Prove _watch_until_closed and HostRankGroup.close() survive when export_receive_ranks fails and shuts
    # down control sockets >200ms before export_model finishes.
    ctrl_root2, ctrl_peer2 = socket.socketpair()
    chan_root2, chan_peer2 = socket.socketpair()
    exec_root2, exec_peer2 = socket.socketpair()
    peer_failed = threading.Event()

    def failing_receive_ranks(requests, cache, no_weights, owners, stage):
        ctrl_root2.shutdown(socket.SHUT_RDWR)
        peer_failed.set()
        raise RuntimeError("peer export stream failed")

    def slow_export_model(model, images, storage, path, **kwargs):
        assert peer_failed.wait(5.0)
        # Exceed _watch_until_closed's 200ms poll window while control has POLLHUP.
        time.sleep(0.35)
        Path(path).write_text("{}")
        return SimpleNamespace(memory_inventory={"rank": 0})

    monkeypatch.setattr(flare_heliopause, "export_receive_ranks", failing_receive_ranks)
    monkeypatch.setattr(runtime_core, "export_model", slow_export_model)

    recv_fail_group = object.__new__(host_ranks.HostRankGroup)
    stopping_event2 = threading.Event()
    suppress_event2 = threading.Event()
    recv_fail_group.rendezvous = SimpleNamespace(
        world=2,
        peers={1: SimpleNamespace(control=ctrl_root2, executor=exec_root2)},
        stopping=stopping_event2,
        suppress_watch=suppress_event2,
        store=object(),
    )
    model_closed2 = []
    recv_fail_group.model = SimpleNamespace(close=lambda: model_closed2.append(True))
    recv_fail_group.checkpoint_fabric = SimpleNamespace(
        readers={(1, 0): SimpleNamespace(channel=SimpleNamespace(connection=chan_root2, sent=0, received=0))},
        streams=1,
        close=lambda: None,
    )
    recv_fail_group._started = False
    recv_fail_group._closed = False
    recv_fail_group._aborted = False
    watcher2 = host_ranks._watch_until_closed(
        [ctrl_root2], stopping_event2, "peer disconnected", suppress=suppress_event2)

    try:
        with export_options(tmp_path / "cache"):
            with pytest.raises(RuntimeError, match="peer export stream failed"):
                with recv_fail_group:
                    host_ranks.HostRankGroup.export(
                        recv_fail_group,
                        None,
                        options={},
                        export_id="exp-recv-fail",
                        images=None,
                        storage=SimpleNamespace(host_regions=[]),
                        publisher=object(),
                    )
        watcher2.join(timeout=2.0)
        assert not watcher2.is_alive()
        assert recv_fail_group._aborted is True
        assert recv_fail_group._closed is True
        assert model_closed2 == [True]
    finally:
        stopping_event2.set()
        for sock in (ctrl_root2, ctrl_peer2, chan_root2, chan_peer2, exec_root2, exec_peer2):
            with contextlib.suppress(OSError):
                sock.close()


def test_export_suppresses_broken_pipe_teardown_on_abort_and_checks_clean_exit(tmp_path, monkeypatch):
    import contextlib
    import json
    import socket
    import threading
    from types import SimpleNamespace
    import flare_heliopause
    from ignition.preparation import checkpoint_fabric, host_ranks, memory_inventory, runtime_core
    from ignition.preparation.core_snapshot import export_options

    # 1. FabricSource.close forwards check=False to its underlying FabricTransport.
    seen_checks = []
    fake_source = object.__new__(checkpoint_fabric.FabricSource)
    fake_source.stopping = threading.Event()
    fake_source.transport = SimpleNamespace(close=lambda *, check=True: seen_checks.append(check))
    fake_source.thread = SimpleNamespace(join=lambda: None)
    fake_source.readers = {}
    fake_source.close(check=False)
    fake_source.close(check=True)
    assert seen_checks == [False, True]

    class FakeNativeStage:
        def __init__(self, destination, staging, cache):
            self.dir = tmp_path / f"stage-{id(self)}"
            self.dir.mkdir(parents=True, exist_ok=True)

        def rank_path(self, rank):
            return str(self.dir / f"rank-{rank}.json")

        def retain_sources(self, paths):
            return len(paths)

        def close(self):
            pass

    monkeypatch.setattr(flare_heliopause, "NativeExportStage", FakeNativeStage)
    monkeypatch.setattr(host_ranks, "send_object", lambda sock, value: None)
    monkeypatch.setattr(host_ranks, "_export_channels", lambda channels: [{"fd": 9}])
    monkeypatch.setattr(memory_inventory, "print_inventory", lambda *args, **kwargs: None)
    monkeypatch.setattr("ignition.preparation.core_file.combine", lambda *args, **kwargs: SimpleNamespace(ok=True))

    def make_group(*, broken_pipe_on_checked_close=True):
        ctrl_root, ctrl_peer = socket.socketpair()
        chan_root, chan_peer = socket.socketpair()
        exec_root, exec_peer = socket.socketpair()
        close_calls = []

        def close_fabric(*, check=True):
            close_calls.append(check)
            if check and broken_pipe_on_checked_close:
                raise RuntimeError("checkpoint fabric helper failed (1): Error: Broken pipe (os error 32)")

        group = object.__new__(host_ranks.HostRankGroup)
        group.rendezvous = SimpleNamespace(
            world=2,
            peers={1: SimpleNamespace(control=ctrl_root, executor=exec_root)},
            stopping=threading.Event(),
            suppress_watch=threading.Event(),
            store=object(),
        )
        group.model = SimpleNamespace(close=lambda: None)
        group.checkpoint_fabric = SimpleNamespace(
            readers={(1, 0): SimpleNamespace(channel=SimpleNamespace(connection=chan_root, sent=0, received=0))},
            streams=1,
            close=close_fabric,
        )
        group._started = False
        group._closed = False
        group._aborted = False
        sockets = (ctrl_root, ctrl_peer, chan_root, chan_peer, exec_root, exec_peer)
        return group, close_calls, sockets

    # 2. export_receive_ranks ENOSPC failure propagates cleanly without secondary Broken pipe chaining.
    monkeypatch.setattr(
        flare_heliopause,
        "export_receive_ranks",
        lambda *args: (_ for _ in ()).throw(OSError(28, "No space left on device")),
    )
    monkeypatch.setattr(
        runtime_core,
        "export_model",
        lambda model, images, storage, path, **kw: SimpleNamespace(memory_inventory={"rank": 0}),
    )
    group_recv, calls_recv, socks_recv = make_group()
    try:
        with export_options(tmp_path / "cache"):
            with pytest.raises(OSError, match="No space left on device") as caught:
                with group_recv:
                    host_ranks.HostRankGroup.export(
                        group_recv, None, options={}, export_id="exp-enospc",
                        images=None, storage=SimpleNamespace(host_regions=[]), publisher=object(),
                    )
        assert caught.value.__context__ is None
        assert group_recv._aborted is True
        assert calls_recv == [False]
    finally:
        for sock in socks_recv:
            with contextlib.suppress(OSError):
                sock.close()

    # 3. Pre-thread failure (_export_channels raising before receiver_thread starts) sets _aborted and suppresses Broken pipe.
    monkeypatch.setattr(host_ranks, "_export_channels", lambda channels: (_ for _ in ()).throw(RuntimeError("bad channel fd")))
    group_pre, calls_pre, socks_pre = make_group()
    try:
        with export_options(tmp_path / "cache"):
            with pytest.raises(RuntimeError, match="bad channel fd") as caught_pre:
                with group_pre:
                    host_ranks.HostRankGroup.export(
                        group_pre, None, options={}, export_id="exp-pre",
                        images=None, storage=SimpleNamespace(host_regions=[]), publisher=object(),
                    )
        assert caught_pre.value.__context__ is None
        assert group_pre._aborted is True
        assert calls_pre == [False]
    finally:
        for sock in socks_pre:
            with contextlib.suppress(OSError):
                sock.close()

    # 4. Happy path still passes check=True so a broken transport helper surfaces when export itself did not abort.
    monkeypatch.setattr(host_ranks, "_export_channels", lambda channels: [{"fd": 9}])
    monkeypatch.setattr(
        flare_heliopause,
        "export_receive_ranks",
        lambda *args: json.dumps([{
            "channels": [{"sent": 1, "received": 1}],
            "inventory": {"rank": 1},
            "stats": {"payload_bytes": 0, "payload_seconds": 0.1, "payload_gbps": 0.0, "streams": 1, "transport": "fabric-tcp"},
        }]),
    )
    group_happy, calls_happy, socks_happy = make_group(broken_pipe_on_checked_close=True)
    try:
        with export_options(tmp_path / "cache"):
            with pytest.raises(RuntimeError, match="checkpoint fabric helper failed"):
                host_ranks.HostRankGroup.export(
                    group_happy, None, options={}, export_id="exp-happy-check",
                    images=None, storage=SimpleNamespace(host_regions=[]), publisher=object(),
                )
        assert calls_happy == [True]
    finally:
        for sock in socks_happy:
            with contextlib.suppress(OSError):
                sock.close()

    # 5. HostRendezvous.abort closes listener and admitted peer sockets immediately.
    listener = socket.create_server(("127.0.0.1", 0))
    peer_ctrl_a, peer_ctrl_b = socket.socketpair()
    peer_exec_a, peer_exec_b = socket.socketpair()
    try:
        rv = object.__new__(host_ranks.HostRendezvous)
        rv.stopping = threading.Event()
        rv.listener = listener
        rv.peers = {1: SimpleNamespace(control=peer_ctrl_a, executor=peer_exec_a)}
        rv.abort()
        assert rv.stopping.is_set()
        assert listener.fileno() == -1
        assert peer_ctrl_a.fileno() == -1
        assert peer_exec_a.fileno() == -1
    finally:
        for sock in (listener, peer_ctrl_a, peer_ctrl_b, peer_exec_a, peer_exec_b):
            with contextlib.suppress(OSError):
                sock.close()
