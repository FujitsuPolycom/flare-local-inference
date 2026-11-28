/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native ownership and reconstruction of prepared CUDA graph cores.

mod archive;
mod cache_memory;
#[cfg(test)]
mod cooperative_tests;
mod describe;
pub use describe::InspectionOptions;
pub mod address_space;
mod device_memory;
pub mod driver;
mod engram;
#[cfg(test)]
mod event_tests;
mod format;
#[cfg(any(windows, test))]
mod host;
mod limits;
mod metadata;
#[cfg(test)]
mod programmatic_edge_tests;
pub use metadata::{unverified_metadata, unverified_metadata_selected};
mod memory;
mod model;
mod payload;
mod peer_access;
mod persistent_worker;
pub use persistent_worker::WorkerState;
pub mod preparation;
mod protection;
mod protobuf;
pub use protobuf::wire;
pub use preparation::{
    PREPARATION_MANIFEST_VERSION, compatible_for_combine, compatible_for_weight_reuse,
    from_legacy_rebuild_manifest, resolve_preparation, validate_preparation_manifest,
};
pub fn validate_sampling_defaults(value: &wire::DefaultSamplingParameters) -> crate::Result<()> {
    protobuf::sampling_defaults(value.clone()).map(|_| ())
}
pub const WIRE_DESCRIPTOR: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/runtime_core_descriptor.bin"));
mod publishers;
pub fn embedded_publishers() -> Result<Vec<Publisher>> {
    publishers::embedded()
}
#[cfg(any(test, feature = "test-tools"))]
pub mod test_support;
pub use protection::{
    AuthorizedCore, CoreClaim, CoreLicense, PayloadAuthorizedCore, PayloadSource,
    PayloadVerifiedCore, Publisher, VerifiedCore, verify_core_metadata,
};
use std::path::Path;
mod selection;
mod serving;
#[cfg(test)]
mod shared_memory_tests;
pub use archive::{
    CoreFile, LegacyHeader, LegacyLayout, MAX_LEGACY_HEADER, MAX_LEGACY_INDEX, MAX_LEGACY_PAYLOAD,
    core_companions, is_snapshot_blob_name, read_legacy_header, validate_companion_filename,
};
pub use limits::{KvCacheLimit, MemoryLimit};
pub use model::{
    CacheAgreement, ClusterRank, NativeModel, RankNeeds, RoceSpec, core_cluster, rank_needs,
    rank_ranges, reserve_bytes,
};
pub use selection::LoadOptions;

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::{null, null_mut};
use std::sync::Arc;

use crate::{
    BufferSpan, GraphBundle, Result,
    startup::{Phase, StartupProgress},
};
use driver::{
    AllocationProperties, CopyEndpoint, CopyParams, Driver, Handle, KernelParams, Location,
    MemsetParams, TensorMap,
};
use format::{Blob, Core, Memory, Node, Reference, Relocation, RelocationKind, invalid};

struct Storage {
    base: u64,
    address: u64,
    nbytes: usize,
    memory: Memory,
    fixed: bool,
    mappings: Vec<(u64, usize)>,
}

/// Owns a CUDA context and every allocation, module, and graph loaded into it.
/// Borrowed handles remain valid until this owner is dropped, after replay drains.
pub struct NativeResources {
    driver: Arc<Driver>,
    context: Handle,
    primary_context: bool,
    device: i32,
    rank: usize,
    // Allocation base -> (local rank, CUDA device) readers, including the owner.
    peer_readers: HashMap<u64, Vec<(usize, i32)>>,
    allocations: HashMap<String, Storage>,
    modules: Vec<Handle>,
    functions: HashMap<usize, Handle>,
    templates: Vec<Handle>,
    executables: Vec<Handle>,
    names: Vec<String>,
    reservations: Vec<(u64, usize)>,
    execution: Option<protobuf::wire::Execution>,
    // An integrated GPU (GB10) registers host memory without pinning it.
    integrated: bool,
    events: HashMap<String, Handle>,
    cache: Option<flare_selene::CacheGeometry>,
    #[cfg(windows)]
    host_mappings: Vec<(u64, usize)>,
    green_partition: Option<persistent_worker::GreenContextPartition>,
    persistent_worker: Option<Arc<persistent_worker::PersistentWorkerRuntime>>,
}

// SAFETY: CUDA contexts may move between threads. Resource operations bind the
// owning context, and mutable ownership serializes loading and destruction.
unsafe impl Send for NativeResources {}

fn clamp_cooperative_grid(mut grid: [u32; 3], capacity: u64) -> Result<[u32; 3]> {
    let axis = (0..3).max_by_key(|&axis| grid[axis]).unwrap();
    let plane = u64::from(grid[(axis + 1) % 3]) * u64::from(grid[(axis + 2) % 3]);
    let max_workers = capacity / plane;
    if max_workers == 0 {
        return Err(invalid(format!(
            "cooperative kernel grid {grid:?} cannot fit {capacity} resident CTAs by reducing one dimension",
        )));
    }
    grid[axis] = u64::from(grid[axis]).min(max_workers) as u32;
    Ok(grid)
}

/// An accepted core from either on-disk format, ready for a loader entry point.
pub enum ModelCore {
    Legacy(AuthorizedCore),
    Payload(PayloadAuthorizedCore),
}

impl ModelCore {
    /// Open and authorize a core, selecting the deployment and rank to serve.
    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        selection: crate::core::protection::Selection,
        accepted: &str,
        verify_weights: bool,
    ) -> Result<Self> {
        match NativeResources::core_format(path)? {
            CoreFormat::Payload => {
                let verified = PayloadVerifiedCore::open(path, publishers, selection)?;
                Ok(Self::Payload(verified.authorize(accepted, verify_weights)?))
            }
            CoreFormat::Legacy => {
                let rank = match selection {
                    crate::core::protection::Selection::Deployment {
                        rank: Some(rank), ..
                    } => rank as usize,
                    _ => 0,
                };
                let verified = VerifiedCore::open(path, publishers, Some(rank))?;
                Ok(Self::Legacy(verified.authorize(accepted, verify_weights)?))
            }
        }
    }
}

/// On-disk prepared-core formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreFormat {
    /// Signed format-4 and format-6 logical-resource cores read through the archive.
    Legacy,
    /// Payload-identity cores read through pinned payload containers.
    Payload,
}

fn payload_selection(deployment: usize, rank: u32) -> crate::core::protection::Selection {
    crate::core::protection::Selection::Deployment {
        deployment,
        rank: Some(rank),
    }
}

impl NativeResources {
    /// Load compiled code and persistent storage, then bind and instantiate graphs.
    ///
    /// # Safety
    /// The core must be trusted. Its kernels and declared bindings must obey
    /// CUDA's memory and execution contracts, including any mapped host access.
    /// Cooperative kernels must tolerate fewer worker CTAs in the largest grid dimension.
    /// This method makes the owner's context current on the calling thread.
    pub unsafe fn load(authorized: AuthorizedCore, ordinal: i32) -> Result<Self> {
        let (archive, mut index) = match authorized.kind {
            protection::AuthorizedKind::Legacy { archive, index } => (archive, index),
            protection::AuthorizedKind::Payload(payload) => {
                if payload.rank_count() != 1 {
                    return Err(invalid(
                        "payload core holds several ranks; use the model loader",
                    ));
                }
                return unsafe { payload.load(0, ordinal) };
            }
        };
        if index.ranks.len() != 1 || !index.peer_memory.is_empty() || index.engram.is_some() {
            return Err(invalid("resource loader requires one independent rank"));
        }
        let core = protobuf::decode(index.ranks.remove(0))?;
        unsafe { Self::load_prepared(&archive, core, ordinal) }
    }

    /// Load an accepted core from either format.
    ///
    /// # Safety
    /// The core must be trusted and the selected device must match its architecture.
    pub unsafe fn load_any(core: ModelCore, ordinal: i32) -> Result<Self> {
        match core {
            ModelCore::Legacy(authorized) => unsafe { Self::load(authorized, ordinal) },
            ModelCore::Payload(payload) => {
                if payload.rank_count() != 1 {
                    return Err(invalid(
                        "payload core holds several ranks; use the model loader",
                    ));
                }
                unsafe { payload.load(0, ordinal) }
            }
        }
    }

    /// Detect the carrier from the safetensors header before applying its reader.
    /// Logical-resource and payload readers validate their own framing and metadata.
    pub fn core_format(path: &Path) -> Result<CoreFormat> {
        let path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_owned()
        };
        match crate::core::reader::declared_format(&path)? {
            Some(format) if format == crate::core::container::PAYLOAD_FORMAT => {
                Ok(CoreFormat::Payload)
            }
            _ => Ok(CoreFormat::Legacy),
        }
    }

    /// Load one rank of the selected deployment from either format.
    ///
    /// Each carrier authenticates the selected rank and validates its prepared
    /// execution contract before replay.
    ///
    /// # Safety
    /// The core must be trusted and the selected device must match its architecture.
    pub unsafe fn load_selected(
        path: &Path,
        publishers: &[Publisher],
        selection: crate::core::protection::Selection,
        accepted: &str,
        verify_weights: bool,
        ordinal: i32,
    ) -> Result<Self> {
        match Self::core_format(path)? {
            CoreFormat::Payload => {
                let selected = match selection {
                    crate::core::protection::Selection::Deployment {
                        deployment,
                        rank: Some(rank),
                    } => payload_selection(deployment, rank),
                    crate::core::protection::Selection::Deployment { rank: None, .. } => {
                        return Err(invalid(
                            "loading one rank requires a selected rank; use the model loader",
                        ));
                    }
                    crate::core::protection::Selection::Catalog => {
                        return Err(invalid("catalog verification cannot be loaded"));
                    }
                };
                let core = ModelCore::open(path, publishers, selected, accepted, verify_weights)?;
                unsafe { Self::load_any(core, ordinal) }
            }
            CoreFormat::Legacy => {
                let rank = match selection {
                    crate::core::protection::Selection::Deployment {
                        rank: Some(rank), ..
                    } => rank as usize,
                    _ => 0,
                };
                let verified = VerifiedCore::open(path, publishers, Some(rank))?;
                let authorized = verified.authorize(accepted, verify_weights)?;
                unsafe { Self::load(authorized, ordinal) }
            }
        }
    }

    /// Load one already-adapted rank description from files the caller opened and pinned.
    ///
    /// This is the payload-format entry point: the description came from an authenticated core
    /// body, and every resource key resolves inside one of the caller's containers.
    ///
    /// # Safety
    /// The core must be trusted and obey the CUDA execution contract, and the files it reads
    /// must stay open for the returned owner's lifetime.
    pub(in crate::runtime_core) unsafe fn load_prepared<F: payload::Files>(
        files: &F,
        core: Core,
        ordinal: i32,
    ) -> Result<Self> {
        core.validate()?;
        if core.allocations.iter().any(|a| a.sizing.is_some()) {
            return Err(invalid(
                "runtime-sized storage requires a memory model plan",
            ));
        }
        let phase = StartupProgress::silent().phase("Load model", None);
        let mut owner = unsafe { Self::initialize(&core, ordinal)? };
        unsafe {
            owner.reserve(&core)?;
            owner.allocate(&core, &phase)?;
            owner.load_payload(files, &core, &phase, 1)?;
            owner.prepare_kernels(files, &core, &phase, None)?;
            owner.prepare_graphs(files, &core, &phase)?;
        }
        phase.finish("");
        Ok(owner)
    }

    unsafe fn initialize(core: &Core, ordinal: i32) -> Result<Self> {
        let driver = Arc::new(Driver::load()?);
        // SAFETY: Driver signatures are checked against cuda.h; output storage is
        // valid, and CUDA initialization precedes all context/resource operations.
        unsafe {
            driver.check("cuInit", (driver.init)(0))?;
            let mut device = 0;
            driver.check("cuDeviceGet", (driver.device)(&mut device, ordinal))?;
            for (attribute, expected) in [75, 76].into_iter().zip(core.architecture) {
                let mut value = 0;
                driver.check(
                    "cuDeviceGetAttribute",
                    (driver.device_attribute)(&mut value, attribute, device),
                )?;
                if value != expected {
                    return Err(invalid("core GPU architecture does not match device"));
                }
            }
            let mut integrated = 0;
            driver.check(
                "cuDeviceGetAttribute",
                (driver.device_attribute)(&mut integrated, 18, device),
            )?;
            if let Some(execution) = core.execution.as_ref()
                && execution.persistent_workers.len() > 1
            {
                return Err(invalid(
                    "at most one persistent worker is supported per rank",
                ));
            }
            let wants_worker = core
                .execution
                .as_ref()
                .and_then(|execution| execution.persistent_workers.first());
            let mut context = null_mut();
            let primary_context = if wants_worker.is_some() {
                if driver.green_ctx.is_none() {
                    return Err(invalid(
                        "persistent worker requires CUDA driver Green Context APIs",
                    ));
                }
                // PyTorch records deprecated CU_CTX_MAP_HOST (0x08) from cuCtxGetFlags();
                // CUDA 11+ contexts always map host memory and cuDevicePrimaryCtxSetFlags_v2
                // rejects the bit with CUDA_ERROR_INVALID_VALUE.
                const CU_CTX_MAP_HOST: u32 = 0x08;
                let primary_flags = core.context.flags & !CU_CTX_MAP_HOST;
                if primary_flags != 0 {
                    let status = (driver.primary_ctx_set_flags)(device, primary_flags);
                    if status != 0 && status != driver::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE {
                        driver.check("cuDevicePrimaryCtxSetFlags_v2", status)?;
                    }
                }
                driver.check(
                    "cuDevicePrimaryCtxRetain",
                    (driver.primary_ctx_retain)(&mut context, device),
                )?;
                true
            } else {
                driver.check(
                    "cuCtxCreate_v2",
                    (driver.context_create)(&mut context, core.context.flags, device),
                )?;
                false
            };
            let owner = Self {
                driver,
                context,
                primary_context,
                device,
                rank: 0,
                peer_readers: HashMap::new(),
                allocations: HashMap::new(),
                modules: vec![],
                functions: HashMap::new(),
                templates: vec![],
                executables: vec![],
                names: vec![],
                reservations: vec![],
                execution: core.execution.clone(),
                integrated: integrated != 0,
                events: HashMap::new(),
                cache: None,
                #[cfg(windows)]
                host_mappings: Vec::new(),
                green_partition: None,
                persistent_worker: None,
            };
            if !primary_context {
                let configured = (|| -> Result<()> {
                    for [limit, value] in &core.context.limits {
                        owner.driver.check(
                            "cuCtxSetLimit",
                            (owner.driver.context_limit)(*limit as i32, *value),
                        )?;
                    }
                    owner.driver.check(
                        "cuCtxSetCacheConfig",
                        (owner.driver.context_cache)(core.context.cache),
                    )
                })();
                // Creation pushes the context; later operations bind it without
                // retaining creation-stack entries that could outlive the owner.
                let mut popped = null_mut();
                let restored = owner.driver.check(
                    "cuCtxPopCurrent_v2",
                    (owner.driver.context_pop)(&mut popped),
                );
                configured?;
                restored?;
                if popped != owner.context {
                    return Err(invalid(
                        "CUDA context creation stack differs from its owner",
                    ));
                }
            }
            Ok(owner)
        }
    }

    pub(super) fn pci_bus_id(&self) -> Result<String> {
        let mut buf = [0 as std::ffi::c_char; 32];
        // SAFETY: `self.driver` initialized CUDA and `self.device` is a valid device handle.
        unsafe {
            self.driver.check(
                "cuDeviceGetPCIBusId",
                (self.driver.device_pci_bus_id)(
                    buf.as_mut_ptr(),
                    buf.len() as std::ffi::c_int,
                    self.device,
                ),
            )?;
            Ok(std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_string_lossy()
                .into_owned())
        }
    }

    fn address(&self, reference: &Reference) -> u64 {
        self.allocations[&reference.allocation].address + reference.offset as u64
    }

    fn relocate(&self, data: &mut [u8], relocation: &Relocation, offset: usize) -> Result<()> {
        let address = self.address(&relocation.target);
        match relocation.kind {
            RelocationKind::Pointer => {
                data[offset..offset + 8].copy_from_slice(&address.to_le_bytes())
            }
            RelocationKind::TensorMap => {
                let mut descriptor = TensorMap(data[offset..offset + 128].try_into().unwrap());
                // SAFETY: TensorMap has CUDA's required 64-byte alignment. The
                // descriptor and replacement allocation were validated before use.
                self.driver.check("cuTensorMapReplaceAddress", unsafe {
                    (self.driver.tensor_map_replace)(&mut descriptor, address as Handle)
                })?;
                data[offset..offset + 128].copy_from_slice(&descriptor.0);
            }
        }
        Ok(())
    }

    fn bind(&self, blob: &Blob) -> Result<Vec<u8>> {
        let mut data = blob.data.clone();
        for relocation in &blob.relocations {
            self.relocate(&mut data, relocation, relocation.offset)?;
        }
        Ok(data)
    }

    unsafe fn reserve(&mut self, core: &Core) -> Result<()> {
        let driver = &self.driver;
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let bits = address_space::virtual_address_bits(self.device);
            for reservation in &core.reservations {
                address_space::check_fits(
                    "core reservation",
                    reservation.address,
                    reservation.nbytes as u64,
                    bits,
                )
                .map_err(|error| invalid(format!("{error}; re-export the core for this GPU")))?;
                let mut address = 0;
                driver.check(
                    "cuMemAddressReserve",
                    (driver.reserve)(&mut address, reservation.nbytes, 0, reservation.address, 0),
                )?;
                self.reservations.push((address, reservation.nbytes));
                if address != reservation.address {
                    return Err(invalid(format!(
                        "CUDA address collision: requested {:#x}, reserved {address:#x}",
                        reservation.address
                    )));
                }
            }
        }
        Ok(())
    }

    unsafe fn allocate(&mut self, core: &Core, phase: &Phase) -> Result<()> {
        #[cfg(windows)]
        if self.host_mappings.is_empty() {
            unsafe { self.allocate_host(core)? };
        }
        let driver = self.driver.clone();
        // SAFETY: Core validation bounds all indexes and relocation fields.
        // Every CUDA resource is immediately retained by self for error cleanup.
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let footprint = core
                .allocations
                .iter()
                .filter(|allocation| {
                    matches!(allocation.memory, Memory::Device)
                        && !self
                            .allocations
                            .get(&allocation.name)
                            .is_some_and(|a| a.nbytes > 0)
                })
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.nbytes)
                })
                .ok_or_else(|| invalid("core device footprint overflows address space"))?;
            let free = self.available_memory()?;
            if footprint > free {
                return Err(invalid(format!(
                    "core needs {footprint} device bytes; only {free} are free"
                )));
            }
            for allocation in &core.allocations {
                if self
                    .allocations
                    .get(&allocation.name)
                    .is_some_and(|a| a.nbytes > 0)
                {
                    continue;
                }
                if allocation.sizing.is_some() {
                    self.allocations.insert(
                        allocation.name.clone(),
                        Storage {
                            base: 0,
                            address: allocation.address.unwrap(),
                            nbytes: 0,
                            memory: allocation.memory,
                            fixed: true,
                            mappings: Vec::new(),
                        },
                    );
                    continue;
                }
                let mut base = 0;
                let size = allocation.nbytes + allocation.alignment - 1;
                let address = if let Some(address) = allocation.address {
                    match allocation.memory {
                        Memory::Device => {
                            let properties = AllocationProperties {
                                kind: 1,
                                handles: 0,
                                location: Location {
                                    kind: 1,
                                    device: self.device,
                                },
                                windows: null_mut(),
                                flags: [0; 8],
                            };
                            let mut physical = 0;
                            driver.check(
                                "cuMemCreate",
                                (driver.physical_create)(
                                    &mut physical,
                                    allocation.nbytes,
                                    &properties,
                                    0,
                                ),
                            )?;
                            let mapped = (driver.map)(address, allocation.nbytes, 0, physical, 0);
                            (driver.physical_release)(physical);
                            driver.check("cuMemMap", mapped)?;
                        }
                        Memory::MappedHost => {
                            #[cfg(unix)]
                            {
                                let mapped = libc::mmap(
                                    address as Handle,
                                    allocation.nbytes,
                                    libc::PROT_READ | libc::PROT_WRITE,
                                    libc::MAP_PRIVATE
                                        | libc::MAP_ANONYMOUS
                                        | libc::MAP_FIXED_NOREPLACE,
                                    -1,
                                    0,
                                );
                                if mapped != address as Handle {
                                    return Err(invalid(format!(
                                        "host address collision at {address:#x}: {}",
                                        std::io::Error::last_os_error()
                                    )));
                                }
                                // Registration does not pin host memory for an integrated GPU, and a
                                // swapped-out page turns each GPU access during replay into a fault.
                                if self.integrated && libc::mlock(mapped, allocation.nbytes) != 0 {
                                    let error = std::io::Error::last_os_error();
                                    libc::munmap(mapped, allocation.nbytes);
                                    return Err(invalid(format!(
                                        "cannot lock {} bytes of mapped host storage: {error}; raise RLIMIT_MEMLOCK",
                                        allocation.nbytes
                                    )));
                                }
                                let status = (driver.host_register)(mapped, allocation.nbytes, 2);
                                if status != 0 {
                                    libc::munmap(mapped, allocation.nbytes);
                                }
                                driver.check("cuMemHostRegister_v2", status)?;
                            }
                        }
                        // The owning rank maps peer storage and grants this rank access.
                        Memory::Peer => {}
                    }
                    address
                } else {
                    driver.check("cuMemAlloc_v2", (driver.allocate)(&mut base, size))?;
                    (base + allocation.alignment as u64 - 1) & !(allocation.alignment as u64 - 1)
                };
                self.allocations.insert(
                    allocation.name.clone(),
                    Storage {
                        base,
                        address,
                        nbytes: allocation.nbytes,
                        memory: allocation.memory,
                        fixed: allocation.address.is_some(),
                        mappings: if allocation.address.is_some()
                            && matches!(allocation.memory, Memory::Device)
                        {
                            vec![(address, allocation.nbytes)]
                        } else {
                            Vec::new()
                        },
                    },
                );
                if allocation.address.is_some() {
                    match allocation.memory {
                        Memory::Device => {
                            self.grant_storage_access(&allocation.name)?;
                        }
                        Memory::MappedHost => {
                            #[cfg(unix)]
                            {
                                let mut gpu = 0;
                                driver.check(
                                    "cuMemHostGetDevicePointer_v2",
                                    (driver.host_device)(&mut gpu, address as Handle, 0),
                                )?;
                                if gpu != address {
                                    return Err(invalid(
                                        "mapped host address differs from recorded CUDA address",
                                    ));
                                }
                            }
                        }
                        Memory::Peer => {}
                    }
                }
                if allocation.stored_bytes() < allocation.nbytes {
                    match allocation.memory {
                        Memory::MappedHost => {
                            std::ptr::write_bytes(address as *mut u8, 0, allocation.nbytes)
                        }
                        Memory::Device => driver.check(
                            "cuMemsetD8_v2",
                            (driver.zero)(address, 0, allocation.nbytes),
                        )?,
                        Memory::Peer => {}
                    }
                }
                phase.advance(allocation.nbytes as u64);
            }
            driver.check("cuCtxSynchronize", (driver.context_sync)())?;
        }
        Ok(())
    }

    unsafe fn load_payload<F: payload::Files>(
        &mut self,
        files: &F,
        core: &Core,
        phase: &Phase,
        ranks: usize,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            payload::load(self, files, core, phase, ranks)?;
            for allocation in &core.allocations {
                for relocation in &allocation.relocations {
                    let address =
                        self.allocations[&allocation.name].address + relocation.offset as u64;
                    let mut bytes = vec![0u8; relocation.width()];
                    driver.check(
                        "cuMemcpyDtoH_v2",
                        (driver.download)(bytes.as_mut_ptr().cast(), address, bytes.len()),
                    )?;
                    self.relocate(&mut bytes, relocation, 0)?;
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, bytes.as_ptr().cast(), bytes.len()),
                    )?;
                }
            }
        }
        Ok(())
    }

    unsafe fn prepare_kernels<F: payload::Files>(
        &mut self,
        files: &F,
        core: &Core,
        phase: &Phase,
        skip_module: Option<&str>,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            // Configure recorded limits and Green Context partitions after VA
            // reservations and mappings so cuCtxSetLimit cannot claim core VA.
            if self.primary_context {
                apply_context_settings(&driver, self.context, &core.context)?;
                if let Some(worker) = core
                    .execution
                    .as_ref()
                    .and_then(|execution| execution.persistent_workers.first())
                {
                    let partition = persistent_worker::GreenContextPartition::create(
                        &driver,
                        self.device,
                        worker.worker_sm_count,
                    )?;
                    let (worker_context, compute_context) =
                        (partition.worker_context, partition.compute_context);
                    self.green_partition = Some(partition);
                    apply_context_settings(&driver, worker_context, &core.context)?;
                    apply_context_settings(&driver, compute_context, &core.context)?;
                }
            }
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let mut modules = HashMap::new();
            for module in &core.modules {
                if skip_module == Some(module.name.as_str()) {
                    phase.advance(1);
                    continue;
                }
                let image = files.read(core, &module.file)?;
                if !image.starts_with(b"\x7fELF") {
                    return Err(invalid("module image is not a compiled cubin"));
                }
                let mut handle = null_mut();
                driver.check(
                    "cuModuleLoadData",
                    (driver.module_load)(&mut handle, image.as_ptr().cast()),
                )?;
                self.modules.push(handle);
                modules.insert(&module.name, handle);
                for global in &module.globals {
                    let symbol = CString::new(global.symbol.as_str()).unwrap();
                    let (mut address, mut size) = (0, 0);
                    driver.check(
                        "cuModuleGetGlobal_v2",
                        (driver.module_global)(&mut address, &mut size, handle, symbol.as_ptr()),
                    )?;
                    let mut contents = files.read(core, &global.file)?;
                    if size != global.nbytes || contents.len() != size {
                        return Err(invalid(format!(
                            "module global size mismatch for {}",
                            global.symbol
                        )));
                    }
                    for relocation in &global.relocations {
                        self.relocate(&mut contents, relocation, relocation.offset)?;
                    }
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, contents.as_ptr().cast(), size),
                    )?;
                }
                phase.advance(1);
            }
            for function in &core.functions {
                let symbol = CString::new(function.symbol.as_str()).unwrap();
                let mut handle = null_mut();
                driver.check(
                    "cuModuleGetFunction",
                    (driver.function)(&mut handle, modules[&function.module], symbol.as_ptr()),
                )?;
                driver.check("cuFuncLoad", (driver.function_load)(handle))?;
                for (index, expected) in function.parameter_sizes.iter().enumerate() {
                    let (mut offset, mut size) = (0, 0);
                    driver.check(
                        "cuFuncGetParamInfo",
                        (driver.parameter_info)(handle, index, &mut offset, &mut size),
                    )?;
                    if size != *expected {
                        return Err(invalid("loaded kernel parameter layout differs from core"));
                    }
                }
                let (mut offset, mut size) = (0, 0);
                let status = (driver.parameter_info)(
                    handle,
                    function.parameter_sizes.len(),
                    &mut offset,
                    &mut size,
                );
                if status != 1 {
                    return Err(invalid("loaded kernel parameter count differs from core"));
                }
                for [attribute, value] in &function.attributes {
                    driver.check(
                        "cuFuncSetAttribute",
                        (driver.function_attribute)(handle, *attribute, *value),
                    )?;
                }
                self.functions.insert(function.id, handle);
                phase.advance(1);
            }
        }
        Ok(())
    }

    unsafe fn prepare_graphs<F: payload::Files>(
        &mut self,
        files: &F,
        core: &Core,
        phase: &Phase,
    ) -> Result<()> {
        let driver = self.driver.clone();
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            for name in &core.events {
                let mut event = null_mut();
                driver.check("cuEventCreate", (driver.event_create)(&mut event, 2))?;
                self.events.insert(name.clone(), event);
            }
            let mut dynamic_shared = HashMap::new();
            for &function in self.functions.values() {
                let mut bytes = 0;
                driver.check(
                    "cuFuncGetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    (driver.function_get_attribute)(&mut bytes, 8, function),
                )?;
                dynamic_shared.insert(function as usize, bytes as u32);
            }
            let worker_graph_name = core
                .execution
                .as_ref()
                .and_then(|execution| execution.persistent_workers.first())
                .map(|worker| worker.graph.as_str());
            for graph in &core.graphs {
                let (graph_context, upload_stream, partition_sms) =
                    if let Some(partition) = self.green_partition.as_ref() {
                        if worker_graph_name == Some(graph.name.as_str()) {
                            (
                                partition.worker_context,
                                partition.worker_stream,
                                Some(partition.worker_sm_count),
                            )
                        } else {
                            (
                                partition.compute_context,
                                partition.compute_stream,
                                Some(partition.compute_sm_count),
                            )
                        }
                    } else {
                        (self.context, null_mut(), None)
                    };
                driver.check("cuCtxSetCurrent", (driver.context_set)(graph_context))?;
                let mut handle = null_mut();
                driver.check("cuGraphCreate", (driver.graph_create)(&mut handle, 0))?;
                self.templates.push(handle);
                let mut nodes = Vec::with_capacity(graph.node_count);
                let mut records = files.records(core, &graph.resource)?;
                for _ in 0..graph.node_count {
                    let node = protobuf::node(records.read()?)?;
                    core.validate_node(&node)?;
                    if let Node::Kernel { function, .. } = &node {
                        if !graph.functions.contains(function) {
                            return Err(invalid(
                                "graph kernel is absent from its dependency index",
                            ));
                        }
                    }
                    let node = &node;
                    let mut result = null_mut();
                    match node {
                        Node::Kernel {
                            function,
                            grid,
                            block,
                            shared,
                            arguments,
                            attributes,
                        } => {
                            let function_handle = self.functions[function];
                            let allowance =
                                dynamic_shared.get_mut(&(function_handle as usize)).unwrap();
                            // Captured nodes retain their launch requirements even if later
                            // preparation lowers the function's dynamic shared-memory allowance.
                            if *shared > *allowance {
                                let bytes = i32::try_from(*shared).map_err(|_| {
                                    invalid(
                                        "kernel shared-memory size exceeds CUDA's attribute range",
                                    )
                                })?;
                                driver.check(
                                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                                    (driver.function_attribute)(function_handle, 8, bytes),
                                )?;
                                *allowance = *shared;
                            }
                            let mut storage = arguments
                                .iter()
                                .map(|a| self.bind(a))
                                .collect::<Result<Vec<_>>>()?;
                            let mut pointers: Vec<_> =
                                storage.iter_mut().map(|a| a.as_mut_ptr().cast()).collect();
                            let mut params = KernelParams {
                                function: function_handle,
                                grid: *grid,
                                block: *block,
                                shared: *shared,
                                arguments: pointers.as_mut_ptr(),
                                extra: null_mut(),
                                kernel: null_mut(),
                                context: null_mut(),
                            };
                            let status =
                                (driver.graph_kernel)(&mut result, handle, null(), 0, &params);
                            if status != 0 {
                                let symbol = core
                                    .functions
                                    .iter()
                                    .find(|entry| entry.id == *function)
                                    .map(|entry| entry.symbol.as_str())
                                    .unwrap_or("unknown");
                                driver.check(
                                    &format!(
                                        "cuGraphAddKernelNode_v2 for graph {} node {} kernel {symbol} (grid {grid:?}, block {block:?}, shared {shared})",
                                        graph.name, nodes.len()
                                    ),
                                    status,
                                )?;
                            }
                            for attribute in attributes {
                                let bytes = self.bind(&attribute.value)?;
                                let mut value = [0u64; 8];
                                std::ptr::copy_nonoverlapping(
                                    bytes.as_ptr(),
                                    value.as_mut_ptr().cast(),
                                    64,
                                );
                                let mut status = (driver.graph_attribute)(
                                    result,
                                    attribute.id,
                                    value.as_ptr().cast(),
                                );
                                if attribute.id == 2
                                    && value[0] as u32 != 0
                                    && (status == 720 || (status == 0 && partition_sms.is_some()))
                                {
                                    self.clamp_cooperative_launch(&mut params, partition_sms)?;
                                    driver.check(
                                        "cuGraphKernelNodeSetParams_v2",
                                        (driver.graph_kernel_params)(result, &params),
                                    )?;
                                    status = (driver.graph_attribute)(
                                        result,
                                        attribute.id,
                                        value.as_ptr().cast(),
                                    );
                                }
                                driver.check("cuGraphKernelNodeSetAttribute", status)?;
                            }
                        }
                        Node::Memset {
                            destination,
                            pitch,
                            value,
                            element_size,
                            width,
                            height,
                        } => {
                            let params = MemsetParams {
                                destination: self.address(destination),
                                pitch: *pitch,
                                value: *value,
                                element_size: *element_size,
                                width: *width,
                                height: *height,
                            };
                            let status = (driver.graph_memset)(
                                &mut result,
                                handle,
                                null(),
                                0,
                                &params,
                                graph_context,
                            );
                            if status == 1 {
                                // Tegra CUDA 12.6 rejects VMM addresses in cuGraphAddMemsetNode
                                // directly, but accepts the same memset captured as a child graph.
                                let mut stream = null_mut();
                                driver.check("cuStreamCreate", (driver.stream_create)(&mut stream, 1))?;
                                let mut child = null_mut();
                                let captured = (|| -> Result<()> {
                                    driver.check(
                                        "cuStreamBeginCapture_v2",
                                        (driver.stream_begin_capture)(stream, 2),
                                    )?;
                                    let memset_status = if params.height <= 1 {
                                        match params.element_size {
                                            1 => (driver.memset_d8_async)(
                                                params.destination,
                                                params.value as u8,
                                                params.width,
                                                stream,
                                            ),
                                            2 => (driver.memset_d16_async)(
                                                params.destination,
                                                params.value as u16,
                                                params.width,
                                                stream,
                                            ),
                                            4 => (driver.memset_d32_async)(
                                                params.destination,
                                                params.value,
                                                params.width,
                                                stream,
                                            ),
                                            _ => 1,
                                        }
                                    } else {
                                        match params.element_size {
                                            1 => (driver.memset_d2d8_async)(
                                                params.destination,
                                                params.pitch,
                                                params.value as u8,
                                                params.width,
                                                params.height,
                                                stream,
                                            ),
                                            2 => (driver.memset_d2d16_async)(
                                                params.destination,
                                                params.pitch,
                                                params.value as u16,
                                                params.width,
                                                params.height,
                                                stream,
                                            ),
                                            4 => (driver.memset_d2d32_async)(
                                                params.destination,
                                                params.pitch,
                                                params.value,
                                                params.width,
                                                params.height,
                                                stream,
                                            ),
                                            _ => 1,
                                        }
                                    };
                                    let end_status = (driver.stream_end_capture)(stream, &mut child);
                                    driver.check("cuMemsetAsync(graph capture)", memset_status)?;
                                    driver.check("cuStreamEndCapture", end_status)?;
                                    driver.check(
                                        "cuGraphAddChildGraphNode",
                                        (driver.graph_child)(&mut result, handle, null(), 0, child),
                                    )
                                })();
                                if !child.is_null() {
                                    let _ = (driver.graph_destroy)(child);
                                }
                                let _ = (driver.stream_destroy)(stream);
                                captured?;
                            } else {
                                driver.check("cuGraphAddMemsetNode", status)?;
                            }
                        }
                        Node::Empty => {
                            driver.check(
                                "cuGraphAddEmptyNode",
                                (driver.graph_empty)(&mut result, handle, null(), 0),
                            )?;
                        }
                        Node::EventWait { event } => {
                            driver.check(
                                "cuGraphAddEventWaitNode",
                                (driver.graph_event_wait)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    self.events[event],
                                ),
                            )?;
                        }
                        Node::Copy {
                            source,
                            destination,
                            nbytes,
                        } => {
                            let endpoint = |reference| CopyEndpoint {
                                memory_type: 2,
                                device: self.address(reference),
                                pitch: *nbytes,
                                height: 1,
                                ..CopyEndpoint::default()
                            };
                            let params = CopyParams {
                                source: endpoint(source),
                                destination: endpoint(destination),
                                width: *nbytes,
                                height: 1,
                                depth: 1,
                            };
                            driver.check(
                                "cuGraphAddMemcpyNode",
                                (driver.graph_copy)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    &params,
                                    graph_context,
                                ),
                            )?;
                        }
                    }
                    nodes.push(result);
                    phase.advance(1);
                }
                let mut dependencies = format::Dependencies::new(graph.node_count);
                let add_dependencies = |sources: &mut Vec<Handle>,
                                        destinations: &mut Vec<Handle>,
                                        data: &mut Vec<[u8; 8]>|
                 -> Result<()> {
                    if sources.is_empty() {
                        return Ok(());
                    }
                    driver.check(
                        "cuGraphAddDependencies_v2",
                        (driver.graph_edges)(
                            handle,
                            sources.as_ptr(),
                            destinations.as_ptr(),
                            data.as_ptr(),
                            data.len(),
                        ),
                    )?;
                    sources.clear();
                    destinations.clear();
                    data.clear();
                    Ok(())
                };
                for begin in (0..graph.edge_count).step_by(4096) {
                    let mut sources = Vec::new();
                    let mut destinations = Vec::new();
                    let mut data = Vec::new();
                    for _ in begin..(begin + 4096).min(graph.edge_count) {
                        let edge = protobuf::edge(records.read()?)?;
                        dependencies.add(&edge)?;
                        if data.last().is_some_and(|previous| previous != &edge.data) {
                            // A mixed batch can apply its first annotation to every
                            // edge. Flush at each change so programmatic ports and
                            // types survive without changing captured edge order.
                            add_dependencies(&mut sources, &mut destinations, &mut data)?;
                        }
                        sources.push(nodes[edge.source]);
                        destinations.push(nodes[edge.destination]);
                        data.push(edge.data);
                    }
                    add_dependencies(&mut sources, &mut destinations, &mut data)?;
                }
                dependencies.finish()?;
                records.finish()?;
                let mut executable = null_mut();
                driver.check(
                    "cuGraphInstantiateWithFlags",
                    (driver.graph_instantiate)(&mut executable, handle, graph.flags),
                )?;
                self.executables.push(executable);
                self.names.push(graph.name.clone());
                driver.check(
                    "cuGraphUpload",
                    (driver.graph_upload)(executable, upload_stream),
                )?;
                if upload_stream.is_null() {
                    driver.check("cuCtxSynchronize", (driver.context_sync)())?;
                } else {
                    driver.check("cuStreamSynchronize", (driver.stream_sync)(upload_stream))?;
                }
                driver.check("cuGraphDestroy", (driver.graph_destroy)(handle))?;
                self.templates.pop();
                phase.advance(1);
            }
            self.functions.clear();
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            driver.check("cuCtxSynchronize", (driver.context_sync)())?;
            if let Some(worker) = core
                .execution
                .as_ref()
                .and_then(|execution| execution.persistent_workers.first())
            {
                let partition = self.green_partition.as_ref().ok_or_else(|| {
                    invalid("persistent worker requires an initialized green context partition")
                })?;
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
                let stop_offset = protobuf::size(stop_flag.offset)?;
                if stop_offset % 4 != 0
                    || stop_offset
                        .checked_add(4)
                        .is_none_or(|end| end > stop_alloc.nbytes)
                {
                    return Err(invalid(
                        "persistent worker stop flag offset must be 4-byte aligned and in bounds",
                    ));
                }
                let stop_flag_address = stop_alloc.address + stop_offset;
                let (metrics_address, metrics_count) = match (
                    worker.metrics.as_ref(),
                    worker.metrics_count,
                ) {
                    (None, 0) => (0usize, 0usize),
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
                        let metrics_alloc =
                            self.allocations.get(&metrics.allocation).ok_or_else(|| {
                                invalid(format!("unknown allocation {:?}", metrics.allocation))
                            })?;
                        if !matches!(metrics_alloc.memory, Memory::MappedHost) {
                            return Err(invalid(
                                "persistent worker metrics must reside in a mapped host allocation",
                            ));
                        }
                        let metrics_offset = protobuf::size(metrics.offset)?;
                        let count = count as usize;
                        let metrics_bytes = count
                            .checked_mul(std::mem::size_of::<u64>())
                            .ok_or_else(|| invalid("persistent worker metrics size overflow"))?;
                        if metrics_offset % std::mem::align_of::<u64>() != 0
                            || metrics_offset
                                .checked_add(metrics_bytes)
                                .is_none_or(|end| end > metrics_alloc.nbytes)
                        {
                            return Err(invalid(
                                "persistent worker metrics offset must be 8-byte aligned and in bounds",
                            ));
                        }
                        let address = (metrics_alloc.address as usize)
                            .checked_add(metrics_offset)
                            .ok_or_else(|| {
                                invalid("persistent worker metrics address overflow")
                            })?;
                        (address, count)
                    }
                };
                self.persistent_worker = Some(persistent_worker::PersistentWorkerRuntime::new(
                    worker.graph.clone(),
                    partition.worker_sm_count,
                    self.driver.clone(),
                    partition.worker_context,
                    partition.worker_stream,
                    self.executables[index],
                    stop_flag_address,
                    metrics_address,
                    metrics_count,
                ));
            }
        }
        Ok(())
    }

    unsafe fn clamp_cooperative_launch(
        &self,
        params: &mut KernelParams,
        partition_sms: Option<u32>,
    ) -> Result<()> {
        let threads = params
            .block
            .iter()
            .try_fold(1i32, |n, &dim| n.checked_mul(i32::try_from(dim).ok()?))
            .ok_or_else(|| invalid("cooperative kernel block size overflow"))?;
        let (mut blocks_per_sm, mut sms) = (0, 0);
        unsafe {
            self.driver.check(
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                (self.driver.occupancy)(
                    &mut blocks_per_sm,
                    params.function,
                    threads,
                    params.shared as usize,
                ),
            )?;
            if let Some(partition_sms) = partition_sms {
                sms = i32::try_from(partition_sms)
                    .map_err(|_| invalid("partition SM count exceeds CUDA attribute range"))?;
            } else if let Some(partition) = self.green_partition.as_ref() {
                sms = partition.compute_sm_count as i32;
            } else {
                self.driver.check(
                    "cuDeviceGetAttribute",
                    (self.driver.device_attribute)(&mut sms, 16, self.device),
                )?;
            }
        }
        if blocks_per_sm <= 0 || sms <= 0 {
            return Err(invalid("cooperative kernel has no resident CTA capacity"));
        }
        let capacity = blocks_per_sm as u64 * sms as u64;
        params.grid = clamp_cooperative_grid(params.grid, capacity)?;
        Ok(())
    }

    pub fn graph_names(&self) -> &[String] {
        &self.names
    }

    /// The rank's prefix checkpoint storage, present after cache allocation.
    pub fn cache_geometry(&self) -> Option<&flare_selene::CacheGeometry> {
        self.cache.as_ref()
    }

    pub(super) fn context_handle(&self) -> (usize, i32) {
        (self.context as usize, self.device)
    }

    pub(super) fn execution_context(&self) -> Handle {
        self.green_partition
            .as_ref()
            .map_or(self.context, |partition| partition.compute_context)
    }

    /// Launch the prepared persistent worker graph once on its green-context worker stream.
    /// No-op when the core declares no persistent worker or the worker is already running.
    pub fn start_persistent_worker(&self) -> Result<()> {
        match self.persistent_worker.as_ref() {
            Some(worker) => worker.start(),
            None => Ok(()),
        }
    }

    /// Signal the mapped-host stop flag and drain the persistent worker stream.
    /// Idempotent when already stopped or when no persistent worker was configured.
    pub fn stop_persistent_worker(&self) -> Result<()> {
        match self.persistent_worker.as_ref() {
            Some(worker) => worker.stop(),
            None => Ok(()),
        }
    }

    /// Current lifecycle state of the rank's persistent worker, if declared.
    pub fn persistent_worker_state(&self) -> Option<WorkerState> {
        self.persistent_worker.as_ref().map(|worker| worker.state())
    }

    /// Lock-free telemetry snapshot from the rank's persistent worker, if declared.
    pub fn persistent_worker_snapshot(
        &self,
    ) -> Option<crate::execution::PersistentWorkerSnapshot> {
        self.persistent_worker
            .as_ref()
            .map(|worker| worker.snapshot())
    }
    pub(super) fn persistent_worker_control(
        &self,
    ) -> Option<crate::execution::PersistentWorkerControl> {
        self.persistent_worker
            .as_ref()
            .map(persistent_worker::PersistentWorkerRuntime::control)
    }

    pub fn graph_bundle(&self) -> GraphBundle {
        GraphBundle {
            context: self.execution_context() as usize,
            device: self.device,
            graph_execs: self.executables.iter().map(|v| *v as usize).collect(),
            buffers: self
                .allocations
                .values()
                .filter(|a| !matches!(a.memory, Memory::Peer))
                .map(|a| BufferSpan {
                    address: a.address as usize,
                    nbytes: a.nbytes,
                })
                .collect(),
        }
    }

    pub fn allocation(&self, name: &str) -> Result<BufferSpan> {
        let a = self
            .allocations
            .get(name)
            .ok_or_else(|| invalid(format!("unknown allocation {name:?}")))?;
        Ok(BufferSpan {
            address: a.address as usize,
            nbytes: a.nbytes,
        })
    }

    /// Copy an allocation after the caller has drained every replay that uses it.
    pub fn download(&self, name: &str) -> Result<Vec<u8>> {
        if matches!(
            self.persistent_worker_state(),
            Some(WorkerState::Running | WorkerState::Draining | WorkerState::FailedDrain)
        ) {
            return Err(invalid(
                "cannot synchronize primary context while persistent worker is running",
            ));
        }
        let a = self.allocation(name)?;
        let mut bytes = vec![0u8; a.nbytes];
        // SAFETY: The allocation remains owned by self and the output is sized to it.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyDtoH_v2",
                (self.driver.download)(bytes.as_mut_ptr().cast(), a.address as u64, bytes.len()),
            )?;
        }
        Ok(bytes)
    }

    /// Replace allocation contents after draining replay on the owned context.
    pub fn upload(&self, name: &str, bytes: &[u8]) -> Result<()> {
        if matches!(
            self.persistent_worker_state(),
            Some(WorkerState::Running | WorkerState::Draining | WorkerState::FailedDrain)
        ) {
            return Err(invalid(
                "cannot synchronize primary context while persistent worker is running",
            ));
        }
        let a = self.allocation(name)?;
        if bytes.len() != a.nbytes {
            return Err(invalid("upload size differs from allocation"));
        }
        // SAFETY: The owned allocation and input span have equal validated sizes.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyHtoD_v2",
                (self.driver.upload)(a.address as u64, bytes.as_ptr().cast(), bytes.len()),
            )?;
        }
        Ok(())
    }
}

impl NativeResources {
    /// Phase 1 of teardown: stop any persistent worker, synchronize this rank's context,
    /// and destroy all graphs, modules, events, streams, device/host mappings, and green contexts
    /// while peer contexts and shared VA reservations stay alive.
    /// Returns `false` when the worker or context cannot be drained, retaining the driver until process exit.
    pub(super) fn release_execution_resources(&mut self) -> bool {
        if self.context.is_null() {
            return true;
        }
        if self.stop_persistent_worker().is_err() {
            std::mem::forget(self.driver.clone());
            self.context = null_mut();
            return false;
        }
        unsafe {
            let mut previous = null_mut();
            let had_previous = (self.driver.context_get)(&mut previous) == 0;
            if (self.driver.context_set)(self.context) != 0 || (self.driver.context_sync)() != 0 {
                if had_previous && previous != self.context {
                    let _ = (self.driver.context_set)(previous);
                }
                std::mem::forget(self.driver.clone());
                self.context = null_mut();
                return false;
            }
            if let Some(partition) = self.green_partition.as_mut() {
                partition.destroy_streams(&self.driver);
            }
            for executable in self.executables.drain(..).rev() {
                (self.driver.executable_destroy)(executable);
            }
            for graph in self.templates.drain(..).rev() {
                (self.driver.graph_destroy)(graph);
            }
            for (_, event) in std::mem::take(&mut self.events) {
                (self.driver.event_destroy)(event);
            }
            for module in self.modules.drain(..).rev() {
                (self.driver.module_unload)(module);
            }
            if let Some(worker) = &self.persistent_worker {
                worker.invalidate_metrics();
            }
            for (_, allocation) in std::mem::take(&mut self.allocations) {
                if allocation.nbytes == 0 {
                    continue;
                }
                match (allocation.memory, allocation.fixed) {
                    (Memory::Device, false) => {
                        (self.driver.free)(allocation.base);
                    }
                    (Memory::Device, true) => {
                        for &(address, bytes) in &allocation.mappings {
                            (self.driver.unmap)(address, bytes);
                        }
                    }
                    (Memory::MappedHost, _) => {
                        #[cfg(unix)]
                        {
                            (self.driver.host_unregister)(allocation.address as Handle);
                            libc::munmap(allocation.address as Handle, allocation.nbytes);
                        }
                    }
                    (Memory::Peer, _) => {}
                }
            }
            #[cfg(windows)]
            for (address, bytes) in self.host_mappings.drain(..).rev() {
                (self.driver.unmap)(address, bytes);
            }
            self.persistent_worker = None;
            let mut previous_destroyed = false;
            if let Some(mut partition) = self.green_partition.take() {
                if previous == partition.worker_context || previous == partition.compute_context {
                    previous_destroyed = true;
                }
                partition.destroy_green_contexts(&self.driver);
            }
            if had_previous && previous != self.context && !previous_destroyed {
                (self.driver.context_set)(previous);
            }
            true
        }
    }

    /// Phase 2 of teardown: release VA reservations and destroy or release this rank's CUDA context.
    pub(super) fn release_reservations_and_context(&mut self) {
        if self.context.is_null() {
            return;
        }
        let context = std::mem::replace(&mut self.context, null_mut());
        unsafe {
            let mut previous = null_mut();
            let had_previous = (self.driver.context_get)(&mut previous) == 0;
            if (self.driver.context_set)(context) != 0 {
                std::mem::forget(self.driver.clone());
                return;
            }
            for (address, size) in self.reservations.drain(..).rev() {
                (self.driver.release_address)(address, size);
            }
            if self.primary_context {
                if !had_previous || previous == context {
                    let _ = (self.driver.context_set)(null_mut());
                }
                (self.driver.primary_ctx_release)(self.device);
            } else {
                (self.driver.context_destroy)(context);
            }
            if had_previous && previous != context {
                (self.driver.context_set)(previous);
            }
        }
    }
}

impl Drop for NativeResources {
    fn drop(&mut self) {
        // SAFETY: This owner is the sole resource destroyer. On a failed drain,
        // retain the driver and all GPU resources until process termination.
        if self.release_execution_resources() {
            self.release_reservations_and_context();
        }
    }
}

unsafe fn apply_context_settings(
    driver: &Driver,
    context: Handle,
    settings: &format::Context,
) -> Result<()> {
    unsafe {
        let mut previous = null_mut();
        let had_previous = (driver.context_get)(&mut previous) == 0;
        driver.check("cuCtxSetCurrent", (driver.context_set)(context))?;
        let configured = (|| -> Result<()> {
            for [limit, value] in &settings.limits {
                driver.check(
                    "cuCtxSetLimit",
                    (driver.context_limit)(*limit as i32, *value),
                )?;
            }
            driver.check(
                "cuCtxSetCacheConfig",
                (driver.context_cache)(settings.cache),
            )
        })();
        if had_previous && previous != context {
            let _ = (driver.context_set)(previous);
        }
        configured
    }
}
