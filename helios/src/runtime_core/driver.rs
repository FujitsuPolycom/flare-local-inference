/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use libloading::Library;

use crate::{CudaError, ExecutionError, Result};

pub type Handle = *mut c_void;

#[repr(C)]
pub struct KernelParams {
    pub function: Handle,
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared: c_uint,
    pub arguments: *mut Handle,
    pub extra: *mut Handle,
    pub kernel: Handle,
    pub context: Handle,
}

#[repr(C)]
pub struct MemsetParams {
    pub destination: u64,
    pub pitch: usize,
    pub value: c_uint,
    pub element_size: c_uint,
    pub width: usize,
    pub height: usize,
}

#[repr(C)]
#[derive(Default)]
pub struct CopyEndpoint {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub lod: usize,
    pub memory_type: c_int,
    pub host: Handle,
    pub device: u64,
    pub array: Handle,
    pub reserved: Handle,
    pub pitch: usize,
    pub height: usize,
}

#[repr(C)]
pub struct CopyParams {
    pub source: CopyEndpoint,
    pub destination: CopyEndpoint,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[repr(C, align(64))]
pub struct TensorMap(pub [u8; 128]);

#[repr(C)]
pub struct Location {
    pub kind: c_int,
    pub device: c_int,
}

#[repr(C)]
pub struct AllocationProperties {
    pub kind: c_int,
    pub handles: c_int,
    pub location: Location,
    pub windows: Handle,
    pub flags: [u8; 8],
}

#[repr(C)]
pub struct Access {
    pub location: Location,
    pub flags: c_int,
}

/// `CU_DEV_RESOURCE_TYPE_SM`: streaming-multiprocessor device resource.
pub const CU_DEV_RESOURCE_TYPE_SM: c_int = 1;
/// `CU_DEV_SM_RESOURCE_GROUP_BACKFILL`: include remaining SMs beyond co-scheduled multiples
/// in a structured SM resource group rather than dropping them to an unstructured remainder.
pub const CU_DEV_SM_RESOURCE_GROUP_BACKFILL: c_uint = 1;
/// `CU_GREEN_CTX_DEFAULT_STREAM`: required creation flag for CUDA green contexts.
pub const CU_GREEN_CTX_DEFAULT_STREAM: c_uint = 1;
/// `CU_STREAM_NON_BLOCKING`: required flag for streams created on a green context.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
/// `CUDA_ERROR_NOT_READY`: asynchronous stream or event operation has not completed.
pub const CUDA_ERROR_NOT_READY: c_int = 600;
/// `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`: primary context flags were already fixed by an active user.
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: c_int = 708;

/// `CUdevSmResource`: output-only SM partition geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevSmResource {
    pub sm_count: c_uint,
    pub min_sm_partition_size: c_uint,
    pub sm_coscheduled_alignment: c_uint,
    pub flags: c_uint,
}

/// `CU_DEV_SM_RESOURCE_GROUP_PARAMS`: structured SM group specification for `cuDevSmResourceSplit`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevSmResourceGroupParams {
    pub sm_count: c_uint,
    pub coscheduled_sm_count: c_uint,
    pub preferred_coscheduled_sm_count: c_uint,
    pub flags: c_uint,
    pub reserved: [c_uint; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union DevResourceData {
    pub sm: DevSmResource,
    pub raw: [u8; 40],
}

/// `CUdevResource` (ABI v1): tagged union with driver-private padding.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DevResource {
    pub kind: c_int,
    pub _internal_padding: [u8; 92],
    pub data: DevResourceData,
    pub next_resource: *mut DevResource,
}

impl DevResource {
    pub fn zeroed() -> Self {
        Self {
            kind: 0,
            _internal_padding: [0; 92],
            data: DevResourceData { raw: [0; 40] },
            next_resource: std::ptr::null_mut(),
        }
    }

    pub fn sm(&self) -> Option<DevSmResource> {
        if self.kind == CU_DEV_RESOURCE_TYPE_SM {
            // SAFETY: `kind == CU_DEV_RESOURCE_TYPE_SM` selects the `sm` union member.
            Some(unsafe { self.data.sm })
        } else {
            None
        }
    }
}

/// Optional CUDA driver Green Context entrypoints; `None` on drivers that do not export them.
pub struct GreenCtxDriver {
    pub device_get_dev_resource:
        unsafe extern "C" fn(c_int, *mut DevResource, c_int) -> c_int,
    pub dev_sm_resource_split: unsafe extern "C" fn(
        *mut DevResource,
        c_uint,
        *const DevResource,
        *mut DevResource,
        c_uint,
        *mut DevSmResourceGroupParams,
    ) -> c_int,
    pub dev_resource_generate_desc:
        unsafe extern "C" fn(*mut Handle, *mut DevResource, c_uint) -> c_int,
    pub green_ctx_create: unsafe extern "C" fn(*mut Handle, Handle, c_int, c_uint) -> c_int,
    pub green_ctx_destroy: unsafe extern "C" fn(Handle) -> c_int,
    pub green_ctx_get_dev_resource:
        unsafe extern "C" fn(Handle, *mut DevResource, c_int) -> c_int,
    pub ctx_from_green_ctx: unsafe extern "C" fn(*mut Handle, Handle) -> c_int,
    pub green_ctx_stream_create:
        unsafe extern "C" fn(*mut Handle, Handle, c_uint, c_int) -> c_int,
}

impl GreenCtxDriver {
    unsafe fn try_load(library: &Library) -> Option<Self> {
        macro_rules! opt_symbol {
            ($name:literal) => {
                *library.get(concat!($name, "\0").as_bytes()).ok()?
            };
        }
        // SAFETY: Signatures match cuda.h; `Driver` retains `library` for the full lifetime.
        unsafe {
            Some(Self {
                device_get_dev_resource: opt_symbol!("cuDeviceGetDevResource"),
                dev_sm_resource_split: opt_symbol!("cuDevSmResourceSplit"),
                dev_resource_generate_desc: opt_symbol!("cuDevResourceGenerateDesc"),
                green_ctx_create: opt_symbol!("cuGreenCtxCreate"),
                green_ctx_destroy: opt_symbol!("cuGreenCtxDestroy"),
                green_ctx_get_dev_resource: opt_symbol!("cuGreenCtxGetDevResource"),
                ctx_from_green_ctx: opt_symbol!("cuCtxFromGreenCtx"),
                green_ctx_stream_create: opt_symbol!("cuGreenCtxStreamCreate"),
            })
        }
    }
}

macro_rules! driver {
    ($($field:ident: $symbol:literal($($arg:ty),*) );* $(;)?) => {
        pub struct Driver {
            _library: Library,
            pub green_ctx: Option<GreenCtxDriver>,
            $(pub $field: unsafe extern "C" fn($($arg),*) -> c_int,)*
        }

        impl Driver {
            pub fn load() -> Result<Self> {
                // SAFETY: These signatures match the CUDA driver ABI; the library
                // remains loaded for the lifetime of all resolved entrypoints.
                unsafe {
                    let library = crate::os::cuda_library().map_err(|e| e.to_string())?;
                    $(let $field = *library.get(concat!($symbol, "\0").as_bytes())
                        .map_err(|e| e.to_string())?;)*
                    let green_ctx = GreenCtxDriver::try_load(&library);
                    Ok(Self { _library: library, green_ctx, $($field,)* })
                }
            }
        }
    };
}

driver! {
    init: "cuInit"(c_uint);
    device: "cuDeviceGet"(*mut c_int, c_int);
    device_total_memory: "cuDeviceTotalMem_v2"(*mut usize, c_int);
    device_pci_bus_id: "cuDeviceGetPCIBusId"(*mut c_char, c_int, c_int);
    device_attribute: "cuDeviceGetAttribute"(*mut c_int, c_int, c_int);
    primary_ctx_retain: "cuDevicePrimaryCtxRetain"(*mut Handle, c_int);
    primary_ctx_release: "cuDevicePrimaryCtxRelease_v2"(c_int);
    primary_ctx_set_flags: "cuDevicePrimaryCtxSetFlags_v2"(c_int, c_uint);
    context_create: "cuCtxCreate_v2"(*mut Handle, c_uint, c_int);
    context_set: "cuCtxSetCurrent"(Handle);
    context_pop: "cuCtxPopCurrent_v2"(*mut Handle);
    context_get: "cuCtxGetCurrent"(*mut Handle);
    context_sync: "cuCtxSynchronize"();
    context_destroy: "cuCtxDestroy_v2"(Handle);
    context_limit: "cuCtxSetLimit"(c_int, usize);
    context_get_limit: "cuCtxGetLimit"(*mut usize, c_int);
    context_cache: "cuCtxSetCacheConfig"(c_int);
    allocate: "cuMemAlloc_v2"(*mut u64, usize);
    memory_info: "cuMemGetInfo_v2"(*mut usize, *mut usize);
    free: "cuMemFree_v2"(u64);
    reserve: "cuMemAddressReserve"(*mut u64, usize, usize, u64, u64);
    release_address: "cuMemAddressFree"(u64, usize);
    physical_create: "cuMemCreate"(*mut u64, usize, *const AllocationProperties, u64);
    physical_release: "cuMemRelease"(u64);
    physical_retain: "cuMemRetainAllocationHandle"(*mut u64, *mut c_void);
    granularity: "cuMemGetAllocationGranularity"(*mut usize, *const AllocationProperties, c_int);
    map: "cuMemMap"(u64, usize, usize, u64, u64);
    unmap: "cuMemUnmap"(u64, usize);
    access: "cuMemSetAccess"(u64, usize, *const Access, usize);
    host_register: "cuMemHostRegister_v2"(Handle, usize, c_uint);
    host_unregister: "cuMemHostUnregister"(Handle);
    host_device: "cuMemHostGetDevicePointer_v2"(*mut u64, Handle, c_uint);
    host_allocate: "cuMemHostAlloc"(*mut Handle, usize, c_uint);
    host_free: "cuMemFreeHost"(Handle);
    upload: "cuMemcpyHtoD_v2"(u64, *const c_void, usize);
    upload_async: "cuMemcpyHtoDAsync_v2"(u64, *const c_void, usize, Handle);
    stream_create: "cuStreamCreate"(*mut Handle, c_uint);
    stream_query: "cuStreamQuery"(Handle);
    stream_sync: "cuStreamSynchronize"(Handle);
    stream_destroy: "cuStreamDestroy_v2"(Handle);
    stream_begin_capture: "cuStreamBeginCapture_v2"(Handle, c_int);
    stream_end_capture: "cuStreamEndCapture"(Handle, *mut Handle);
    event_create: "cuEventCreate"(*mut Handle, c_uint);
    event_record: "cuEventRecord"(Handle, Handle);
    event_sync: "cuEventSynchronize"(Handle);
    event_destroy: "cuEventDestroy_v2"(Handle);
    download: "cuMemcpyDtoH_v2"(*mut c_void, u64, usize);
    zero: "cuMemsetD8_v2"(u64, u8, usize);
    memset_d8_async: "cuMemsetD8Async"(u64, u8, usize, Handle);
    memset_d16_async: "cuMemsetD16Async"(u64, u16, usize, Handle);
    memset_d32_async: "cuMemsetD32Async"(u64, c_uint, usize, Handle);
    memset_d2d8_async: "cuMemsetD2D8Async"(u64, usize, u8, usize, usize, Handle);
    memset_d2d16_async: "cuMemsetD2D16Async"(u64, usize, u16, usize, usize, Handle);
    memset_d2d32_async: "cuMemsetD2D32Async"(u64, usize, c_uint, usize, usize, Handle);
    module_load: "cuModuleLoadData"(*mut Handle, *const c_void);
    module_global: "cuModuleGetGlobal_v2"(*mut u64, *mut usize, Handle, *const c_char);
    module_unload: "cuModuleUnload"(Handle);
    function: "cuModuleGetFunction"(*mut Handle, Handle, *const c_char);
    parameter_info: "cuFuncGetParamInfo"(Handle, usize, *mut usize, *mut usize);
    function_attribute: "cuFuncSetAttribute"(Handle, c_int, c_int);
    function_get_attribute: "cuFuncGetAttribute"(*mut c_int, c_int, Handle);
    function_load: "cuFuncLoad"(Handle);
    occupancy: "cuOccupancyMaxActiveBlocksPerMultiprocessor"(*mut c_int, Handle, c_int, usize);
    graph_upload: "cuGraphUpload"(Handle, Handle);
    graph_launch: "cuGraphLaunch"(Handle, Handle);
    graph_create: "cuGraphCreate"(*mut Handle, c_uint);
    graph_destroy: "cuGraphDestroy"(Handle);
    graph_kernel: "cuGraphAddKernelNode_v2"(*mut Handle, Handle, *const Handle, usize, *const KernelParams);
    graph_kernel_params: "cuGraphKernelNodeSetParams_v2"(Handle, *const KernelParams);
    graph_memset: "cuGraphAddMemsetNode"(*mut Handle, Handle, *const Handle, usize, *const MemsetParams, Handle);
    graph_copy: "cuGraphAddMemcpyNode"(*mut Handle, Handle, *const Handle, usize, *const CopyParams, Handle);
    graph_empty: "cuGraphAddEmptyNode"(*mut Handle, Handle, *const Handle, usize);
    graph_child: "cuGraphAddChildGraphNode"(*mut Handle, Handle, *const Handle, usize, Handle);
    graph_event_wait: "cuGraphAddEventWaitNode"(*mut Handle, Handle, *const Handle, usize, Handle);
    graph_attribute: "cuGraphKernelNodeSetAttribute"(Handle, c_int, *const c_void);
    graph_edges: "cuGraphAddDependencies_v2"(Handle, *const Handle, *const Handle, *const [u8; 8], usize);
    graph_instantiate: "cuGraphInstantiateWithFlags"(*mut Handle, Handle, u64);
    executable_destroy: "cuGraphExecDestroy"(Handle);
    tensor_map_replace: "cuTensorMapReplaceAddress"(*mut TensorMap, *mut c_void);
    error_name: "cuGetErrorName"(c_int, *mut *const c_char);
    error_string: "cuGetErrorString"(c_int, *mut *const c_char);
}

impl Driver {
    pub fn check(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let text = |function: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int| {
            let mut value = std::ptr::null();
            // SAFETY: CUDA owns the returned string, which is copied before return.
            unsafe {
                if function(status, &mut value) == 0 && !value.is_null() {
                    CStr::from_ptr(value).to_string_lossy().into_owned()
                } else {
                    "unknown CUDA error".to_owned()
                }
            }
        };
        Err(ExecutionError::Driver(CudaError {
            operation: operation.into(),
            code: status,
            name: text(self.error_name),
            description: text(self.error_string),
        })
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_resource_layout_matches_cuda_abi() {
        assert_eq!(std::mem::size_of::<DevSmResource>(), 16);
        assert_eq!(std::mem::size_of::<DevResource>(), 144);
        assert_eq!(std::mem::align_of::<DevResource>(), 8);
        assert_eq!(std::mem::offset_of!(DevResource, kind), 0);
        assert_eq!(std::mem::offset_of!(DevResource, data), 96);
        assert_eq!(std::mem::offset_of!(DevResource, next_resource), 136);
        let mut resource = DevResource::zeroed();
        assert!(resource.sm().is_none());
        resource.kind = CU_DEV_RESOURCE_TYPE_SM;
        resource.data = DevResourceData {
            sm: DevSmResource {
                sm_count: 144,
                min_sm_partition_size: 8,
                sm_coscheduled_alignment: 8,
                flags: 0,
            },
        };
        assert_eq!(resource.sm().unwrap().sm_count, 144);
    }
}
