# SPDX-License-Identifier: Apache-2.0
"""Partial tensor-parallel tuning compares complete local components before distributing a race."""

from types import SimpleNamespace

import pytest
import torch
from b12x._lib.compile_pool import CompileJob
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import DetectedDevice, DeviceIdentity, Knob, MemoryRequirements, Plan, TuningContract

from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.tuning_shards import component_key, local_components, materialization_key


DEVICE = DetectedDevice(0, DeviceIdentity('nvidia', (12, 0), 188, 'RTX PRO 6000 Blackwell'))


def request(name, size, *, dependencies=(), memory=0, factory_value=0, shared=False, override=None):
    contract = TuningContract(component_id='test.projection', query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({'size'}), config_fields=frozenset({'width'}),
        encode_query=lambda q: {'size': q}, encode_config=lambda c: {'width': c}, decode_config=lambda d: d['width'],
        validate_query=lambda q, d: None, validate_config=lambda q, c, d: None, default_config=lambda q, d: 1,
        knobs=(Knob(name='width', values=(1, 2)),))
    plan = Plan(contract=contract, query=size, shared=shared, override=override,
        _compile_jobs=lambda c, d: (CompileJob.create('example:compile', size, c, factory_value if c == 2 else 0),),
        _memory_requirements=lambda c, d: MemoryRequirements(scratch=(
            ScratchBufferSpec(name='tiles', shape=(size + (memory if c == 2 else 0),), dtype=torch.uint8, device=torch.device('cuda', d.ordinal)),)),
        _materialize=lambda s, d: s.config)
    return plan.request(name=name, dependencies=dependencies, prepare_call=lambda s: None,
        benchmark_call=lambda s: None)


def test_local_components_keep_aliases_dependencies_and_collective_neighbors():
    from dataclasses import replace
    a, b, alias = request('input', 32), request('consume', 64, dependencies=('input',)), request('input-alias', 32)
    collective = replace(request('exchange', 128), collective=SimpleNamespace(ranks=(0,1)))
    downstream = request('after-exchange', 256, dependencies=('exchange',))
    groups = local_components((b, downstream, collective, alias, a))
    assert [[r.name for r in group] for group in groups] == [['consume', 'input', 'input-alias']]
    assert component_key(groups[0]) == component_key(local_components((a, alias, b))[0])
    unequal_input = request('input', 48)
    assert component_key(groups[0]) != component_key(local_components((unequal_input, b))[0])
    with pytest.raises(ValueError, match='missing'):
        local_components((request('broken', 32, dependencies=('missing',)),))


def test_component_keys_include_pins_and_every_candidate_binding():
    def key(**kwargs):
        return component_key([request('projection', 32, **kwargs)])
    assert key(override=2) != key()
    assert key(shared=True) != key()
    session=SimpleNamespace(device=DEVICE)
    fingerprint=lambda **kwargs: materialization_key(session, [request('projection',32,**kwargs)])
    assert fingerprint(memory=4) != fingerprint()
    assert fingerprint(factory_value=8) != fingerprint()
    shared=''.join(['repeated', ' factory argument'])
    assert fingerprint(factory_value=(shared,shared)) == fingerprint(factory_value=(shared,(' '+shared)[1:]))


def test_metadata_failure_is_reported_from_the_failing_rank(monkeypatch):
    coordinator=object.__new__(CollectivePreparation)
    coordinator.group,coordinator.ranks=object(),(2,5)
    def gather(output,local,*,group):
        assert group is coordinator.group
        output[:] = [('ok',None),local]
    monkeypatch.setattr(torch.distributed,'all_gather_object',gather)
    def fail():
        raise ValueError('invalid dependency')
    with pytest.raises(RuntimeError,match='5.*ValueError: invalid dependency') as error:
        coordinator.gather_tuning_metadata(fail)
    assert isinstance(error.value.__cause__,ValueError)


def _gloo_mixed_batch_worker(rank, store_path, queue):
    from datetime import timedelta
    import torch.distributed as dist
    from ignition.preparation.tuning import RecordedPreparationSession

    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        shard_history = []
        prepared_calls = []

        class _FakeSession:
            _coordinator = coordinator
            device = DEVICE
            autotune = True

            def configure_tuning_shard(self, shard_rank, shard_ranks):
                shard_history.append((shard_rank, tuple(shard_ranks)))

            def _prepare_job(self, reqs, *, progress, autotune):
                names = tuple(r.name for r in reqs)
                prepared_calls.append(("local", names))
                closed = []
                return SimpleNamespace(names=names, closed=closed, close=lambda: closed.append(True))

        session = _FakeSession()
        orig_prepare = coordinator.prepare

        def fake_prepare(sess, reqs, *, progress, autotune):
            names = tuple(r.name for r in reqs)
            prepared_calls.append(("sharded", names))
            return SimpleNamespace(names=names, close=lambda: None)

        coordinator.prepare = fake_prepare
        try:
            common = request("shared_proj", 32)
            reqs = (common, request("rank0_only", 64)) if rank == 0 else (common,)
            batches = RecordedPreparationSession._mixed_tuning_batches(session, reqs)
            assert [(tuple(r.name for r in b), mode) for b, mode in batches] == [
                (("shared_proj",), "sharded"),
                ((("rank0_only",) if rank == 0 else ()), "independent"),
            ]
            results = [RecordedPreparationSession._prepare_mixed_batch(session, b, mode, None)
                       for b, mode in batches]
            assert shard_history == [(rank, (0, 1)), (0, (0,))]
            assert results[0].names == ("shared_proj",)
            assert (results[1].names == ("rank0_only",)) if rank == 0 else (results[1] is None)

            # Failure on rank 0 during an independent remainder propagates to rank 1 and closes peer result.
            def fail_on_rank1(reqs_inner, *, progress, autotune):
                if rank == 1:
                    raise ValueError("remainder boom")
                closed = []
                res = SimpleNamespace(names=tuple(r.name for r in reqs_inner), closed=closed,
                                      close=lambda: closed.append(True))
                prepared_calls.append(("cleanup_target", res))
                return res

            session._prepare_job = fail_on_rank1
            with pytest.raises(RuntimeError, match="1.*ValueError: remainder boom"):
                RecordedPreparationSession._prepare_mixed_batch(
                    session, (request("fail_rem", 16),), "independent", None)
            if rank == 0:
                target = next(v for tag, v in prepared_calls if tag == "cleanup_target")
                assert target.closed == [True]
        finally:
            coordinator.prepare = orig_prepare
        queue.put((rank, "ok"))
    except BaseException as exc:
        queue.put((rank, repr(exc)))
        raise
    finally:
        dist.destroy_process_group()


def test_mixed_tuning_batches_shard_common_components_and_run_remainders_over_gloo(tmp_path):
    ctx = torch.multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    store = tmp_path / "gloo_store"
    procs = [ctx.Process(target=_gloo_mixed_batch_worker, args=(rank, str(store), queue)) for rank in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
    assert [p.exitcode for p in procs] == [0, 0]
    results = dict(queue.get(timeout=5) for _ in range(2))
    assert results == {0: "ok", 1: "ok"}
