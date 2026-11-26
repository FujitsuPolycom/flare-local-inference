/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{NativeResources, format::invalid, protobuf::size};
use crate::{
    Result,
    execution::{
        BufferDescriptor, ExecutionManifest, GraphDescriptor, InputBinding,
        PersistentWorkerDescriptor, TunedBinding, TuningGraphDescriptor,
    },
};

impl NativeResources {
    pub fn execution_manifest(&self) -> Result<ExecutionManifest> {
        let execution = self
            .execution
            .as_ref()
            .ok_or_else(|| invalid("core has no serving contract"))?;
        let buffers = execution
            .buffers
            .iter()
            .map(|b| {
                let reference = b
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("missing execution buffer reference"))?;
                let allocation = self.allocation(&reference.allocation)?;
                let offset = size(reference.offset)?;
                let nbytes = size(b.nbytes)?;
                if offset
                    .checked_add(nbytes)
                    .is_none_or(|end| end > allocation.nbytes)
                {
                    return Err(invalid("execution buffer exceeds its allocation"));
                }
                Ok(BufferDescriptor {
                    name: b.name.clone(),
                    address: allocation.address + offset,
                    nbytes,
                    dtype: b.dtype.as_str().try_into()?,
                    shape: b.shape.iter().copied().map(size).collect::<Result<_>>()?,
                    strides: b.strides.iter().copied().map(size).collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let graphs = execution
            .graphs
            .iter()
            .map(|g| {
                let index = self
                    .names
                    .iter()
                    .position(|name| *name == g.name)
                    .ok_or_else(|| invalid("execution references an unknown graph"))?;
                let draft_source = match g.draft_source {
                    0 => None,
                    1 => Some(crate::scheduler::speculation::ProposalSource::Mtp),
                    2 => Some(crate::scheduler::speculation::ProposalSource::Copy),
                    _ => return Err(invalid("unknown graph draft source")),
                };
                Ok(GraphDescriptor {
                    tuned_bindings: g.tuned_bindings.iter().map(|binding| TunedBinding {
                        buffer: binding.buffer.clone(), tuning_graph: binding.tuning_graph.clone(),
                    }).collect(),
                    confidence_width: g.confidence_width as usize,
                    draft_tokens: if g.kind == "speculative" {
                        g.draft_tokens.ok_or_else(|| {
                            invalid("speculative graph lacks proposal depth")
                        })? as usize
                    } else {
                        0
                    },
                    draft_source,
                    packed_token_capacity: size(g.packed_token_capacity)?,
                    name: g.name.clone(),
                    kind: g.kind.as_str().try_into()?,
                    sampling_mode: g.sampling_mode.as_str().try_into()?,
                    handle: self.executables[index] as usize,
                    batch_capacity: size(g.batch_capacity)?,
                    token_capacity: size(g.token_capacity)?,
                    prefill_capacity: size(g.prefill_capacity)?,
                    estimated_duration_ns: g.estimated_duration_ns,
                    input_bindings: g
                        .input_bindings
                        .iter()
                        .map(|b| {
                            Ok(InputBinding {
                                buffer_name: b.buffer_name.clone(),
                                host_offset: size(b.host_offset)?,
                                nbytes: size(b.nbytes)?,
                            })
                        })
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tuning_graphs = execution.tuning_graphs.iter().map(|graph| {
            let index = self.names.iter().position(|name| *name == graph.name)
                .ok_or_else(|| invalid("tuning metadata references an unknown graph"))?;
            Ok(TuningGraphDescriptor {
                name: graph.name.clone(), handle: self.executables[index] as usize,
                outputs: graph.outputs.clone(), status_buffer: graph.status_buffer.clone(),
            })
        }).collect::<Result<Vec<_>>>()?;
        if execution.persistent_workers.len() > 1 {
            return Err(invalid(
                "at most one persistent worker is supported per rank",
            ));
        }
        let persistent_workers = execution
            .persistent_workers
            .iter()
            .map(|worker| -> Result<PersistentWorkerDescriptor> {
                let index = self
                    .names
                    .iter()
                    .position(|name| *name == worker.graph)
                    .ok_or_else(|| invalid("persistent worker references an unknown graph"))?;
                let stop_flag = worker
                    .stop_flag
                    .as_ref()
                    .ok_or_else(|| invalid("persistent worker requires a stop_flag reference"))?;
                let stop_alloc = self.allocation(&stop_flag.allocation)?;
                let stop_flag_offset = size(stop_flag.offset)?;
                if stop_flag_offset % 4 != 0
                    || stop_flag_offset
                        .checked_add(4)
                        .is_none_or(|end| end > stop_alloc.nbytes)
                {
                    return Err(invalid(
                        "persistent worker stop flag offset must be 4-byte aligned and in bounds",
                    ));
                }
                let (metrics_allocation, metrics_offset, metrics_address, metrics_count) =
                    match (worker.metrics.as_ref(), worker.metrics_count) {
                        (None, 0) => (None, 0usize, 0usize, 0usize),
                        (Some(_), 0) => {
                            return Err(invalid(
                                "persistent worker metrics reference requires a positive metrics_count",
                            ));
                        }
                        (None, _) => {
                            return Err(invalid(
                                "persistent worker metrics_count requires a metrics reference",
                            ));
                        }
                        (Some(metrics), count) => {
                            let metrics_alloc = self.allocation(&metrics.allocation)?;
                            let metrics_offset = size(metrics.offset)?;
                            let count = count as usize;
                            let metrics_bytes = count
                                .checked_mul(std::mem::size_of::<u64>())
                                .ok_or_else(|| {
                                    invalid("persistent worker metrics size overflow")
                                })?;
                            if metrics_offset % std::mem::align_of::<u64>() != 0
                                || metrics_offset
                                    .checked_add(metrics_bytes)
                                    .is_none_or(|end| end > metrics_alloc.nbytes)
                            {
                                return Err(invalid(
                                    "persistent worker metrics offset must be 8-byte aligned and in bounds",
                                ));
                            }
                            let address = metrics_alloc
                                .address
                                .checked_add(metrics_offset)
                                .ok_or_else(|| {
                                    invalid("persistent worker metrics address overflow")
                                })?;
                            (Some(metrics.allocation.clone()), metrics_offset, address, count)
                        }
                    };
                Ok(PersistentWorkerDescriptor {
                    worker_sm_count: worker.worker_sm_count,
                    graph: worker.graph.clone(),
                    handle: self.executables[index] as usize,
                    stop_flag_allocation: stop_flag.allocation.clone(),
                    stop_flag_offset,
                    stop_flag_address: stop_alloc.address + stop_flag_offset,
                    metrics_allocation,
                    metrics_offset,
                    metrics_address,
                    metrics_count,
                    control: self.persistent_worker_control(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = ExecutionManifest {
            tuning_graphs,
            persistent_workers,
            collectives: None,
            engram: None,
            cache: self.cache.clone(),
            offload: None,
            abi_version: execution.abi_version,
            completion_token_capacity: crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity),
            context: self.execution_context() as usize,
            device: self.device,
            max_active: size(execution.max_active)?,
            input_frame_bytes: size(execution.input_frame_bytes)?,
            buffers,
            graphs,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}
