/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Access permissions for owned VMM mappings, including cache backing aliases.

use super::{
    NativeResources,
    driver::{Access, Location},
};
use crate::Result;

fn grant_mappings(
    owner: usize,
    name: &str,
    mappings: &[(u64, usize)],
    readers: &[(usize, i32)],
    mut grant: impl FnMut(&str, u64, usize, &[Access]) -> Result<()>,
) -> Result<()> {
    let access: Vec<_> = readers
        .iter()
        .map(|&(_, device)| Access {
            location: Location { kind: 1, device },
            flags: 3,
        })
        .collect();
    for &(address, nbytes) in mappings {
        let operation = format!(
            "cuMemSetAccess owner_rank={owner} allocation={name:?} address={address:#x} bytes={nbytes} readers(rank,device)={readers:?}"
        );
        grant(&operation, address, nbytes, &access)?;
    }
    Ok(())
}

impl NativeResources {
    pub(super) unsafe fn grant_storage_access(&self, name: &str) -> Result<()> {
        let storage = &self.allocations[name];
        let local = [(self.rank, self.device)];
        let readers = self
            .peer_readers
            .get(&storage.address)
            .map_or(local.as_slice(), Vec::as_slice);
        grant_mappings(
            self.rank,
            name,
            &storage.mappings,
            readers,
            |operation, address, nbytes, access| unsafe {
                self.driver.check(
                    operation,
                    (self.driver.access)(address, nbytes, access.as_ptr(), access.len()),
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, ExecutionError, error::CudaError};

    #[test]
    fn each_backed_mapping_gets_one_combined_owner_and_importer_grant() {
        let mappings = [(0x200000, 0x200000), (0x400000, 0x200000)];
        let readers = [(8, 3), (9, 1)];
        let mut calls = Vec::new();
        grant_mappings(
            8,
            "latent-kv",
            &mappings,
            &readers,
            |_, address, bytes, access| {
                calls.push((address, bytes));
                assert_eq!(
                    access.iter().map(|a| a.location.device).collect::<Vec<_>>(),
                    [3, 1]
                );
                assert!(access.iter().all(|a| a.location.kind == 1 && a.flags == 3));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls, mappings);
        // Unbacked runtime allocations have no mapping to grant during fixed allocation.
        grant_mappings(8, "latent-kv", &[], &readers, |_, _, _, _| {
            panic!("unbacked grant")
        })
        .unwrap();
        // A continuation alias of shared KV keeps its own owner-only permissions.
        grant_mappings(
            8,
            "state",
            &[(0x800000, 0x200000)],
            &[(8, 3)],
            |_, _, _, access| {
                assert_eq!(access.len(), 1);
                assert_eq!(access[0].location.device, 3);
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn failing_grant_identifies_the_range_and_preserves_driver_status() {
        let error = grant_mappings(
            8,
            "latent-kv",
            &[(0x200000, 0x200000), (0x400000, 0x200000)],
            &[(8, 3), (9, 1)],
            |operation, address, _, _| {
                if address == 0x200000 {
                    return Ok(());
                }
                Err(ExecutionError::Driver(CudaError {
                    operation: operation.into(),
                    code: 2,
                    name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
                    description: "out of memory".into(),
                })
                .into())
            },
        )
        .unwrap_err();
        let Error::Execution(ExecutionError::Driver(driver)) = &error else {
            panic!("lost driver error")
        };
        assert_eq!(driver.code, 2);
        let message = error.to_string();
        for expected in [
            "cuMemSetAccess",
            "owner_rank=8",
            "latent-kv",
            "address=0x400000",
            "bytes=2097152",
            "[(8, 3), (9, 1)]",
            "CUDA_ERROR_OUT_OF_MEMORY",
        ] {
            assert!(message.contains(expected), "{message}");
        }
    }
}
