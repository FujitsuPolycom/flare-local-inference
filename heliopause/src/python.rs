/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#[cfg(target_os = "linux")]
use std::os::fd::BorrowedFd;
use std::sync::Mutex;
#[cfg(any(target_os = "linux", feature = "qualification"))]
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::execution::{
    BufferDescriptor, Dtype, ExecutionManifest, ExecutionWorker, GraphDescriptor, GraphKind,
    InputBinding, OwnedUpload, PersistentWorkerDescriptor, SamplingMode, TunedBinding,
    TuningGraphDescriptor,
};
use crate::lifetime::OwnerLease;
#[cfg(feature = "qualification")]
use crate::{BufferSpan, GraphBundle, ReplayWorker};

#[cfg(feature = "qualification")]
#[pyclass(module = "flare_heliopause")]
struct ReplayCompletion {
    #[pyo3(get)]
    ticket: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    repetitions: u32,
    #[pyo3(get)]
    error: Option<String>,
}

/// Borrow a prepared model's executable graphs and replay them on one native stream.
///
/// The owner must retain every captured allocation, executable, and CUDA context.
/// Graph handles must not be reinstantiated and tensor storage must not be resized
/// or mutated concurrently. Captured PyTorch generator-backed RNG is unsupported.
#[cfg(feature = "qualification")]
#[pyclass(module = "flare_heliopause")]
struct GraphReplayWorker {
    worker: ReplayWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
}

#[cfg(feature = "qualification")]
#[pymethods]
impl GraphReplayWorker {
    #[new]
    #[pyo3(signature = (owner, context, device, graph_execs, buffers, queue_capacity=8))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        context: usize,
        device: i32,
        graph_execs: Vec<usize>,
        buffers: Vec<(usize, usize)>,
        queue_capacity: usize,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the prepared graph resources",
            ));
        }
        let bundle = GraphBundle {
            context,
            device,
            graph_execs,
            buffers: buffers
                .into_iter()
                .map(|(address, nbytes)| BufferSpan { address, nbytes })
                .collect(),
        };
        bundle
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation supplies ready handles and retains their
                // captured allocations in owner. This reference stays live during
                // startup; OwnerLease retains it afterward until confirmed drain.
                unsafe { ReplayWorker::new(bundle, queue_capacity) }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
        })
    }

    #[pyo3(signature = (graph_index, repetitions=1))]
    fn submit(&self, graph_index: usize, repetitions: u32) -> PyResult<u64> {
        self.worker
            .submit(graph_index, repetitions)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (timeout_ms=0))]
    fn poll(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<ReplayCompletion>> {
        py.detach(|| self.worker.poll(Duration::from_millis(timeout_ms)))
            .map(|completion| {
                completion.map(|value| ReplayCompletion {
                    ticket: value.ticket,
                    graph_index: value.graph_index,
                    repetitions: value.repetitions,
                    error: value.error.map(|error| error.to_string()),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self) -> usize {
        self.worker.outstanding()
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "qualification")]
impl Drop for GraphReplayWorker {
    fn drop(&mut self) {
        let shutdown = Python::try_attach(|py| py.detach(|| self.worker.close()))
            .unwrap_or_else(|| self.worker.close());
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pyfunction]
fn current_context() -> PyResult<(usize, i32)> {
    crate::current_context().map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "qualification")]
#[pyfunction]
fn validate_prepared_contract(manifest: &Bound<'_, PyAny>, contract_json: &str) -> PyResult<()> {
    let manifest = execution_manifest(manifest)?;
    manifest.validate().map_err(|error| PyValueError::new_err(error.to_string()))?;
    let contract: crate::model::PreparedContract = serde_json::from_str(contract_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    contract
        .validate(&manifest)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(all(feature = "server", feature = "qualification"))]
#[pyfunction]
fn validate_prepared_catalog(
    py: Python<'_>,
    manifest: &Bound<'_, PyAny>,
    config_json: &str,
) -> PyResult<()> {
    let manifest = execution_manifest(manifest)?;
    py.detach(|| crate::server::Bootstrap::load(config_json, &manifest))
        .map(|_| ())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn execution_manifest(value: &Bound<'_, PyAny>) -> PyResult<ExecutionManifest> {
    macro_rules! field {
        ($value:expr, $name:literal, $kind:ty) => {
            $value.get_item($name)?.extract::<$kind>()?
        };
    }
    let mut buffers = Vec::new();
    for buffer in value.get_item("buffers")?.try_iter()? {
        let buffer = buffer?;
        buffers.push(BufferDescriptor {
            name: field!(buffer, "name", String),
            address: field!(buffer, "address", usize),
            nbytes: field!(buffer, "nbytes", usize),
            dtype: Dtype::try_from(field!(buffer, "dtype", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            shape: field!(buffer, "shape", Vec<usize>),
            strides: field!(buffer, "strides", Vec<usize>),
        });
    }
    let mut graphs = Vec::new();
    for graph in value.get_item("graphs")?.try_iter()? {
        let graph = graph?;
        let mut input_bindings = Vec::new();
        for binding in graph.get_item("input_bindings")?.try_iter()? {
            let binding = binding?;
            input_bindings.push(InputBinding {
                buffer_name: field!(binding, "buffer_name", String),
                host_offset: field!(binding, "host_offset", usize),
                nbytes: field!(binding, "nbytes", usize),
            });
        }
        let mut tuned_bindings = Vec::new();
        for binding in graph.get_item("tuned_bindings")?.try_iter()? {
            let binding = binding?;
            tuned_bindings.push(TunedBinding {
                buffer: field!(binding, "buffer", String),
                tuning_graph: field!(binding, "tuning_graph", String),
            });
        }
        let kind = GraphKind::try_from(field!(graph, "kind", String).as_str())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let draft_source = match graph.get_item("draft_source").ok()
            .map(|value| value.extract::<u32>()).transpose()? {
            Some(0) => None,
            Some(1) => Some(helios_core::scheduler::speculation::ProposalSource::Mtp),
            Some(2) => Some(helios_core::scheduler::speculation::ProposalSource::Copy),
            Some(_) => return Err(PyValueError::new_err("unsupported graph draft_source")),
            None => (kind == GraphKind::Speculative)
                .then_some(helios_core::scheduler::speculation::ProposalSource::Mtp),
        };
        graphs.push(GraphDescriptor {
            confidence_width: field!(graph, "confidence_width", usize),
            draft_tokens: if kind == GraphKind::Speculative { field!(graph, "draft_tokens", usize) } else { 0 },
            draft_source,
            name: field!(graph, "name", String),
            kind,
            sampling_mode: SamplingMode::try_from(field!(graph, "sampling_mode", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            handle: field!(graph, "handle", usize),
            batch_capacity: field!(graph, "batch_capacity", usize),
            token_capacity: field!(graph, "token_capacity", usize),
            packed_token_capacity: graph.get_item("packed_token_capacity").ok()
                .map(|value| value.extract::<usize>()).transpose()?.unwrap_or(0),
            prefill_capacity: field!(graph, "prefill_capacity", usize),
            estimated_duration_ns: field!(graph, "estimated_duration_ns", u64),
            input_bindings,
            tuned_bindings,
        });
    }
    let mut tuning_graphs = Vec::new();
    for graph in value.get_item("tuning_graphs")?.try_iter()? {
        let graph = graph?;
        tuning_graphs.push(TuningGraphDescriptor {
            name: field!(graph, "name", String),
            handle: field!(graph, "handle", usize),
            outputs: field!(graph, "outputs", Vec<String>),
            status_buffer: field!(graph, "status_buffer", String),
        });
    }
    let mut persistent_workers = Vec::new();
    match value.get_item("persistent_workers") {
        Ok(workers) if !workers.is_none() => {
            for pw in workers.try_iter()? {
                let pw = pw?;
                let stop_flag = pw.get_item("stop_flag")?;
                let stop_flag_allocation = field!(stop_flag, "allocation", String);
                let stop_flag_offset = match stop_flag.get_item("offset") {
                    Ok(v) if !v.is_none() => v.extract::<usize>()?,
                    Ok(_) => 0,
                    Err(error)
                        if error.is_instance_of::<pyo3::exceptions::PyKeyError>(value.py()) =>
                    {
                        0
                    }
                    Err(error) => return Err(error),
                };
                let stop_buf = buffers
                    .iter()
                    .find(|b| b.name == stop_flag_allocation)
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "persistent worker stop_flag allocation is not a registered buffer",
                        )
                    })?;
                let stop_flag_end = stop_flag_offset.checked_add(4).ok_or_else(|| {
                    PyValueError::new_err("persistent worker stop_flag offset overflow")
                })?;
                if stop_flag_end > stop_buf.nbytes {
                    return Err(PyValueError::new_err(
                        "persistent worker stop_flag exceeds its registered buffer",
                    ));
                }
                let stop_flag_address = stop_buf
                    .address
                    .checked_add(stop_flag_offset)
                    .ok_or_else(|| {
                        PyValueError::new_err("persistent worker stop_flag address overflow")
                    })?;
                persistent_workers.push(PersistentWorkerDescriptor {
                    worker_sm_count: field!(pw, "worker_sm_count", u32),
                    graph: field!(pw, "graph", String),
                    handle: pw
                        .get_item("handle")
                        .ok()
                        .and_then(|v| v.extract::<usize>().ok())
                        .unwrap_or(0),
                    stop_flag_allocation,
                    stop_flag_offset,
                    stop_flag_address,
                    metrics_allocation: None,
                    metrics_offset: 0,
                    metrics_address: 0,
                    metrics_count: 0,
                    control: None,
                });
            }
        }
        Ok(_) => {}
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(value.py()) => {}
        Err(error) => return Err(error),
    }
    Ok(ExecutionManifest {
        collectives: None,
        engram: crate::embedding_tables::binding(value)?,
        cache: None,
        offload: None,
        abi_version: field!(value, "abi_version", u32),
        completion_token_capacity: helios_core::execution::resolve_completion_token_capacity(match value.get_item("completion_token_capacity") {
            Ok(value) => value.extract::<u32>()?,
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(value.py()) => 0,
            Err(error) => return Err(error),
        }),
        context: field!(value, "context", usize),
        device: field!(value, "device", i32),
        max_active: field!(value, "max_active", usize),
        input_frame_bytes: field!(value, "input_frame_bytes", usize),
        buffers,
        graphs,
        tuning_graphs,
        persistent_workers,
    })
}

#[pyclass(module = "flare_heliopause")]
struct ExecutionCompletion {
    #[pyo3(get)]
    fence: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    proposal_confidence: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    speculative_survival: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    records: Vec<Vec<i64>>,
    #[pyo3(get)]
    logprobs: Option<Vec<Vec<Option<(u32, f32, Vec<(u32, f32)>)>>>>,
    #[pyo3(get)]
    gpu_ns: [Option<u64>; 16],
    #[pyo3(get)]
    readiness_wait_ns: Option<u64>,
    #[pyo3(get)]
    wall_ns: u64,
    #[pyo3(get)]
    queued: bool,
}

#[pyclass(name = "GraphExecutor", module = "flare_heliopause")]
struct PreparedGraphExecutor {
    worker: ExecutionWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}

fn reject_in_process_persistent_worker(manifest: &ExecutionManifest) -> PyResult<()> {
    if !manifest.persistent_workers.is_empty() {
        return Err(PyValueError::new_err(
            "persistent worker requires green-context runtime control (export a core and serve with flare run)",
        ));
    }
    Ok(())
}

#[pymethods]
impl PreparedGraphExecutor {
    #[new]
    #[pyo3(signature = (owner, manifest, *, peer_fds=None))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        manifest: &Bound<'_, PyAny>,
        peer_fds: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain prepared execution resources",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        manifest
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        reject_in_process_persistent_worker(&manifest)?;
        let peers = peer_fds
            .map(|fds| {
                fds.into_iter()
                    .map(rank_socket)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation's owner retains the validated buffers,
                // executable graphs, and context with exclusive native access.
                // Startup holds owner locally; OwnerLease releases it only after
                // close confirms both streams drained, including failed submits.
                match peers {
                    Some(peers) => unsafe { ExecutionWorker::new_parallel(manifest, peers) },
                    None => unsafe { ExecutionWorker::new(manifest) },
                }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[pyo3(signature = (graph_index, uploads, grammar_constrained=false))]
    fn submit(
        &self,
        py: Python<'_>,
        graph_index: usize,
        uploads: Vec<(String, usize, Vec<u8>)>,
        grammar_constrained: bool,
    ) -> PyResult<Option<u64>> {
        let uploads = uploads
            .into_iter()
            .map(|(buffer, offset, bytes)| OwnedUpload {
                buffer,
                offset,
                bytes,
            })
            .collect();
        py.detach(|| {
            self.worker
                .submit(graph_index, uploads, grammar_constrained)
        })
        .map(|fence| fence.map(|fence| fence.0))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn poll(&self, py: Python<'_>) -> PyResult<Option<ExecutionCompletion>> {
        py.detach(|| self.worker.poll())
            .map(|completion| {
                completion.map(|completion| ExecutionCompletion {
                    fence: completion.fence.0,
                    graph_index: completion.graph_index,
                    logprobs: completion.logprobs.map(|slots| slots.into_iter().map(|positions|
                        positions.into_iter().map(|entry| entry.map(|entry|
                            (entry.token, entry.logprob, entry.top))).collect()).collect()),
                    proposal_confidence: completion.proposal_confidence
                        .map(|rows| rows[..completion.records.len()].to_vec()),
                    speculative_survival: completion.speculative_survival
                        .map(|rows| rows[..completion.records.len()].to_vec()),
                    gpu_ns: completion.timing.gpu_ns,
                    readiness_wait_ns: completion.timing.readiness.map(|sample| sample.wait_ns()),
                    wall_ns: completion.timing.wall_ns,
                    queued: completion.timing.queued,
                    records: completion
                        .records
                        .into_iter()
                        .map(|record| record.fields())
                        .collect(),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.worker.outstanding())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

#[cfg(target_os = "linux")]
fn rank_socket(fd: i32) -> PyResult<crate::execution::RankStream> {
    if fd < 0 {
        return Err(PyValueError::new_err(
            "rank socket descriptor must be nonnegative",
        ));
    }
    // SAFETY: Python supplies a live socket descriptor and retains it during this
    // call. Duplication gives Rust an independent owner of the same connection.
    let socket = unsafe { BorrowedFd::borrow_raw(fd) }
        .try_clone_to_owned()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    crate::execution::RankStream::from_socket(socket)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(target_os = "windows")]
fn rank_socket(_fd: i32) -> PyResult<crate::execution::RankStream> {
    Err(PyRuntimeError::new_err(
        "tensor-parallel preparation requires Linux rank sockets",
    ))
}

#[pyfunction]
fn serve_tensor_parallel_rank(
    py: Python<'_>,
    owner: Py<PyAny>,
    manifest: &Bound<'_, PyAny>,
    socket_fd: i32,
    rank: usize,
    world: usize,
) -> PyResult<()> {
    if owner.bind(py).is_none() {
        return Err(PyValueError::new_err(
            "owner must retain prepared rank resources",
        ));
    }
    let manifest = execution_manifest(manifest)?;
    manifest
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    reject_in_process_persistent_worker(&manifest)?;
    let socket = rank_socket(socket_fd)?;
    let mut owner = OwnerLease(Some(owner));
    // SAFETY: The lease retains every graph and IPC mapping until the native
    // coordinator confirms that all ranks drained, or intentionally retains them.
    let shutdown =
        py.detach(|| unsafe { crate::execution::serve_rank(manifest, socket, rank, world) });
    if shutdown.drained {
        drop(owner.take_after_drain());
    }
    match shutdown.error {
        Some(error) => {
            let error = PyRuntimeError::new_err(error.to_string());
            error
                .value(py)
                .setattr("resources_retained", !shutdown.drained)?;
            Err(error)
        }
        None => Ok(()),
    }
}

/// Monitor owned child ranks in Rust; unexpected peer exit terminates this process.
#[cfg(target_os = "linux")]
#[pyclass(name = "RankSupervisor", module = "flare_heliopause")]
struct NativeRankSupervisor {
    supervisor: Mutex<crate::processes::RankSupervisor>,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl NativeRankSupervisor {
    #[new]
    fn new(pids: Vec<i32>) -> PyResult<Self> {
        Ok(Self {
            supervisor: Mutex::new(
                crate::processes::RankSupervisor::new(&pids)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        })
    }

    #[pyo3(signature = (timeout_ms=30000))]
    fn begin_shutdown(&self, timeout_ms: u64) -> PyResult<()> {
        self.supervisor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .begin_shutdown(Duration::from_millis(timeout_ms))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn finish_shutdown(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.supervisor
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .finish_shutdown()
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn abort(&self, reason: &str) {
        self.supervisor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .abort(reason)
    }
}

#[cfg(target_os = "linux")]
#[pyfunction]
fn bind_parent_lifetime(parent_pid: i32) -> PyResult<()> {
    if parent_pid <= 0 {
        return Err(PyValueError::new_err("parent PID must be positive"));
    }
    crate::processes::bind_parent_lifetime(parent_pid)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyclass(name = "CliSignals", module = "flare_heliopause", unsendable)]
struct NativeCliSignals {
    signals: Option<crate::processes::CliSignals>,
}

#[pymethods]
impl NativeCliSignals {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            signals: Some(
                crate::processes::CliSignals::new()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
            ),
        })
    }
    fn __enter__(&self) {}
    fn __exit__(&mut self, _kind: Py<PyAny>, _value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.signals.take();
    }
}

impl Drop for PreparedGraphExecutor {
    fn drop(&mut self) {
        let shutdown = Python::try_attach(|py| py.detach(|| self.worker.close()))
            .unwrap_or_else(|| self.worker.close());
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pymodule]
fn flare_heliopause(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::export::binding::register(module)?;
    #[cfg(feature = "qualification")]
    {
        module.add_class::<GraphReplayWorker>()?;
        module.add_class::<ReplayCompletion>()?;
        module.add_function(wrap_pyfunction!(validate_prepared_contract, module)?)?;
    }
    #[cfg(all(feature = "server", feature = "qualification"))]
    module.add_function(wrap_pyfunction!(validate_prepared_catalog, module)?)?;
    module.add_class::<PreparedGraphExecutor>()?;
    module.add_class::<crate::embedding_tables::NativeEmbeddingTables>()?;
    module.add_class::<crate::embedding_tables::NativeEmbeddingBinding>()?;
    module.add_class::<ExecutionCompletion>()?;
    #[cfg(target_os = "linux")]
    module.add_class::<NativeRankSupervisor>()?;
    module.add_class::<NativeCliSignals>()?;
    #[cfg(all(feature = "server", feature = "qualification"))]
    module.add_class::<NativeServer>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(memory_inventory, module)?)?;
    module.add_function(wrap_pyfunction!(roce_gid_index, module)?)?;
    module.add_function(wrap_pyfunction!(serve_tensor_parallel_rank, module)?)?;
    #[cfg(target_os = "linux")]
    module.add_function(wrap_pyfunction!(bind_parent_lifetime, module)?)?;
    Ok(())
}

/// Snapshot device-wide usage and the core allocator's physical mappings.
///
/// Call after preparation has materialized scratch. Imported peer mappings,
/// dynamically backed caches, and allocations outside this allocator are not
/// included in `mappings`. Device-wide usage includes other CUDA processes.
#[pyfunction]
fn memory_inventory(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let inventory = crate::allocator::memory_inventory()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let result = pyo3::types::PyDict::new(py);
    result.set_item("device", inventory.device)?;
    result.set_item("free_bytes", inventory.free)?;
    result.set_item("total_bytes", inventory.total)?;
    result.set_item("mappings", inventory.mappings)?;
    Ok(result)
}

/// This host's RoCE GID index by the rule native serving uses: 3 when every active device has a
/// RoCEv2 IPv4 GID there, otherwise the lowest index at which one does.
#[pyfunction]
fn roce_gid_index() -> PyResult<u32> {
    #[cfg(target_os = "linux")]
    {
        crate::execution::roce_fabric::discover()
            .map(|fabric| fabric.gid_index)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[cfg(not(target_os = "linux"))]
    Err(PyRuntimeError::new_err("RoCE preparation requires Linux"))
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn raw_replay_exports_require_qualification_feature() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "flare_heliopause").unwrap();
            flare_heliopause(&module).unwrap();
            for name in ["GraphReplayWorker", "ReplayCompletion"] {
                assert_eq!(
                    module.hasattr(name).unwrap(),
                    cfg!(feature = "qualification")
                );
            }
            assert!(module.hasattr("GraphExecutor").unwrap());
            assert!(module.hasattr("current_context").unwrap());
            assert_eq!(
                module.hasattr("Server").unwrap(),
                cfg!(all(feature = "server", feature = "qualification"))
            );
        });
    }

    #[test]
    fn in_process_loader_rejects_persistent_workers_manifest() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "flare_heliopause").unwrap();
            flare_heliopause(&module).unwrap();
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("mod", &module).unwrap();
            let manifest = py
                .eval(
                    c"dict(
                        abi_version=6,
                        completion_token_capacity=8,
                        context=0x1234,
                        device=0,
                        max_active=4,
                        input_frame_bytes=16,
                        buffers=[
                            dict(name='frame_index', address=0x10000, nbytes=4, dtype='int32', shape=[1], strides=[1]),
                            dict(name='submission_sequence', address=0x20000, nbytes=8, dtype='int64', shape=[1], strides=[1]),
                            dict(name='completion_records', address=0x30000, nbytes=2*4*24*8, dtype='int64', shape=[2, 4, 24], strides=[96, 24, 1]),
                            dict(name='worker_ctrl', address=0x40000, nbytes=128, dtype='uint8', shape=[128], strides=[1]),
                        ],
                        graphs=[
                            dict(
                                name='decode',
                                kind='decode',
                                sampling_mode='general',
                                handle=101,
                                batch_capacity=4,
                                token_capacity=1,
                                prefill_capacity=0,
                                confidence_width=0,
                                estimated_duration_ns=0,
                                input_bindings=[
                                    dict(buffer_name='frame_index', host_offset=0, nbytes=4),
                                    dict(buffer_name='submission_sequence', host_offset=8, nbytes=8),
                                ],
                                tuned_bindings=[],
                            )
                        ],
                        tuning_graphs=[],
                        persistent_workers=[dict(
                            worker_sm_count=8,
                            graph='worker_launch',
                            handle=202,
                            stop_flag=dict(allocation='worker_ctrl', offset=0),
                        )],
                    )",
                    None,
                    Some(&locals),
                )
                .unwrap();

            // Native manifest parsing retains persistent_workers and passes structural validation.
            let parsed = execution_manifest(&manifest).unwrap();
            assert_eq!(parsed.persistent_workers.len(), 1);
            assert_eq!(parsed.persistent_workers[0].worker_sm_count, 8);
            assert_eq!(parsed.persistent_workers[0].graph, "worker_launch");
            assert_eq!(
                parsed.persistent_workers[0].stop_flag_allocation,
                "worker_ctrl"
            );
            assert_eq!(parsed.persistent_workers[0].stop_flag_offset, 0);
            assert_eq!(parsed.persistent_workers[0].stop_flag_address, 0x40000);
            parsed.validate().unwrap();

            // In-process GraphExecutor and serve_tensor_parallel_rank both reject persistent_workers explicitly.
            let owner = py.eval(c"object()", None, None).unwrap();
            let exec_err = module
                .getattr("GraphExecutor")
                .unwrap()
                .call1((&owner, &manifest))
                .unwrap_err();
            assert!(exec_err.is_instance_of::<PyValueError>(py));
            assert!(
                exec_err.to_string().contains("green-context runtime control"),
                "unexpected GraphExecutor error: {exec_err}"
            );

            let rank_err = module
                .getattr("serve_tensor_parallel_rank")
                .unwrap()
                .call1((&owner, &manifest, 0, 0usize, 1usize))
                .unwrap_err();
            assert!(rank_err.is_instance_of::<PyValueError>(py));
            assert!(
                rank_err.to_string().contains("green-context runtime control"),
                "unexpected serve_tensor_parallel_rank error: {rank_err}"
            );

            // More than one persistent worker fails manifest validation.
            let multi_manifest = manifest
                .cast::<pyo3::types::PyDict>()
                .unwrap()
                .copy()
                .unwrap();
            let workers = py
                .eval(
                    c"[
                        dict(worker_sm_count=8, graph='worker_a', handle=202, stop_flag=dict(allocation='worker_ctrl', offset=0)),
                        dict(worker_sm_count=8, graph='worker_b', handle=203, stop_flag=dict(allocation='worker_ctrl', offset=4)),
                    ]",
                    None,
                    None,
                )
                .unwrap();
            multi_manifest
                .set_item("persistent_workers", workers)
                .unwrap();
            let multi_parsed = execution_manifest(multi_manifest.as_any()).unwrap();
            let multi_err = multi_parsed.validate().unwrap_err().to_string();
            assert!(multi_err.contains("at most one persistent worker"), "{multi_err}");

            // Unregistered stop_flag allocation is rejected during manifest parsing.
            let bad_manifest = manifest
                .cast::<pyo3::types::PyDict>()
                .unwrap()
                .copy()
                .unwrap();
            let bad_workers = py
                .eval(
                    c"[dict(worker_sm_count=8, graph='worker_launch', handle=202, stop_flag=dict(allocation='missing_ctrl', offset=0))]",
                    None,
                    None,
                )
                .unwrap();
            bad_manifest
                .set_item("persistent_workers", bad_workers)
                .unwrap();
            let parse_err = execution_manifest(bad_manifest.as_any()).unwrap_err();
            assert!(parse_err.is_instance_of::<PyValueError>(py));

            // Out-of-bounds stop_flag offset is rejected during manifest parsing.
            let oob_manifest = manifest
                .cast::<pyo3::types::PyDict>()
                .unwrap()
                .copy()
                .unwrap();
            let oob_workers = py
                .eval(
                    c"[dict(worker_sm_count=8, graph='worker_launch', handle=202, stop_flag=dict(allocation='worker_ctrl', offset=128))]",
                    None,
                    None,
                )
                .unwrap();
            oob_manifest
                .set_item("persistent_workers", oob_workers)
                .unwrap();
            let oob_err = execution_manifest(oob_manifest.as_any()).unwrap_err();
            assert!(oob_err.is_instance_of::<PyValueError>(py));
        });
    }
}

#[cfg(all(feature = "server", feature = "qualification"))]
#[pyclass(name = "Server", module = "flare_heliopause")]
struct NativeServer {
    server: crate::server::Server,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}
#[cfg(all(feature = "server", feature = "qualification"))]
#[pymethods]
impl NativeServer {
    #[new]
    #[pyo3(signature = (owner, manifest, config_json, *, peer_fds=None))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        manifest: &Bound<'_, PyAny>,
        config_json: &str,
        peer_fds: Option<Vec<i32>>,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the complete prepared model",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        reject_in_process_persistent_worker(&manifest)?;
        let peers = peer_fds
            .map(|fds| {
                fds.into_iter()
                    .map(rank_socket)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let bootstrap = py
            .detach(|| crate::server::Bootstrap::load(config_json, &manifest))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let server = match py.detach(|| {
            // SAFETY: The prepared owner retains every captured allocation and
            // handle and relinquishes external mutation during native execution.
            // It stays live across startup, is retained on an undrained failure,
            // and is otherwise transferred into OwnerLease until confirmed drain.
            unsafe {
                crate::server::Server::start(
                    manifest,
                    bootstrap.model,
                    bootstrap.frontend,
                    bootstrap.images,
                    bootstrap.server,
                    peers.map(crate::execution::RankPeers::Processes),
                )
            }
        }) {
            Ok(server) => server,
            Err(error) => {
                if !error.drained {
                    std::mem::forget(owner);
                }
                let exception = PyRuntimeError::new_err(error.message.to_string());
                exception
                    .value(py)
                    .setattr("resources_retained", !error.drained)?;
                return Err(exception);
            }
        };
        Ok(Self {
            server,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }
    #[getter]
    fn address(&self) -> String {
        self.server.address().to_string()
    }
    #[getter]
    fn healthy(&self) -> bool {
        self.server.healthy()
    }
    fn start_dashboard(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.server.start_dashboard())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[pyo3(signature = (timeout_ms=1000))]
    fn wait(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if timeout_ms > 60_000 {
            return Err(PyValueError::new_err(
                "wait timeout must not exceed 60000 ms",
            ));
        }
        let healthy = py.detach(|| self.server.wait(Duration::from_millis(timeout_ms)));
        py.check_signals()?;
        Ok(healthy)
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.server.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        shutdown.error.map_or(Ok(()), |error| {
            Err(PyRuntimeError::new_err(error.to_string()))
        })
    }
}
#[cfg(all(feature = "server", feature = "qualification"))]
impl Drop for NativeServer {
    fn drop(&mut self) {
        let shutdown = Python::try_attach(|py| py.detach(|| self.server.close()))
            .unwrap_or_else(|| self.server.close());
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}
