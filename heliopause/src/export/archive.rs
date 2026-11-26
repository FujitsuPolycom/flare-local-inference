/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Safetensors metadata and pinned logical core sources.

use super::{
    ExportResult as Result,
    io::{self, Identity, Source},
};
#[cfg(test)]
pub(crate) use helios_core::runtime_core::is_snapshot_blob_name as blob_name;
pub(crate) use helios_core::runtime_core::{
    LegacyLayout as Layout, MAX_LEGACY_HEADER as MAX_HEADER,
};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn validate_filename(name: &str) -> Result<()> {
    helios_core::runtime_core::validate_companion_filename(name)
        .map_err(|e| e.to_string().into())
}

pub(crate) struct Container {
    pub(crate) header_json: serde_json::Value,
    pub(crate) file: Arc<File>,
    pub(crate) base: u64,
    pub(crate) payload_length: u64,
    pub(crate) envelope: Vec<u8>,
    pub(crate) layout: Option<Layout>,
    pub(crate) format: String,
}
impl Container {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        Self::from_file(Source::open(path, 0, 0)?.file)
    }
    pub(crate) fn from_file(file: Arc<File>) -> Result<Self> {
        let before = Identity::of(&file)?;
        let header = helios_core::runtime_core::read_legacy_header(&file)
            .map_err(|e| e.to_string())?;
        let header_json = serde_json::from_slice(&header.raw_header)?;
        let mut envelope = Vec::new();
        if let Some(length) = header.index_length {
            envelope.resize(length as usize, 0);
            io::read_exact_at(&file, &mut envelope, header.data_start + header.payload_length)?;
        }
        let layout = Layout::read_from_header(&file, &header)
            .map_err(|e| e.to_string())?;
        if before != Identity::of(&file)? {
            return Err("core metadata changed while reading".into());
        }
        Ok(Self {
            header_json,
            file,
            base: header.data_start,
            payload_length: header.payload_length,
            envelope,
            layout,
            format: header.format,
        })
    }
}

#[derive(Clone)]
pub(crate) enum SourcePiece {
    Data(Source),
    Zero(u64),
}
pub(crate) struct CoreSources {
    pub(crate) container: Container,
    directory: PathBuf,
    physical: BTreeMap<String, Source>,
    _directory: File,
}
impl CoreSources {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let directory = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let directory_handle = open_directory(directory)?;
        let stable = pinned_directory(&directory_handle, directory);
        let container =
            Container::open(&stable.join(path.file_name().ok_or("missing primary filename")?))?;
        let mut physical = BTreeMap::new();
        physical.insert(
            String::new(),
            Source::from_file(
                container.file.clone(),
                container.base,
                container.payload_length,
            )?,
        );
        Ok(Self {
            container,
            directory: stable,
            physical,
            _directory: directory_handle,
        })
    }
    fn physical(&mut self, name: &str) -> Result<Source> {
        if let Some(source) = self.physical.get(name) {
            return Ok(source.clone());
        }
        validate_filename(name)?;
        if self.physical.len()
            >= if self.container.layout.is_some() {
                65537
            } else {
                100
            }
        {
            return Err("too many physical core files".into());
        }
        let file = Container::open(&self.directory.join(name))?;
        if file.format != "flare-data" {
            return Err("invalid core companion format".into());
        }
        let source = Source::from_file(file.file, file.base, file.payload_length)?;
        self.physical.insert(name.to_owned(), source.clone());
        Ok(source)
    }
    pub(crate) fn logical_length(&mut self, name: &str) -> Result<u64> {
        if let Some(layout) = &self.container.layout {
            return Ok(layout
                .files
                .get(name)
                .ok_or("missing logical core file")?
                .iter()
                .map(|p| p.0)
                .sum());
        }
        Ok(self.physical(name)?.length)
    }
    pub(crate) fn sources(
        &mut self,
        name: &str,
        offset: u64,
        length: u64,
    ) -> Result<Vec<SourcePiece>> {
        let end = offset.checked_add(length).ok_or("logical range overflow")?;
        if end > self.logical_length(name)? {
            return Err("resource exceeds logical payload".into());
        }
        let Some(layout) = &self.container.layout else {
            return Ok(vec![SourcePiece::Data(
                self.physical(name)?.slice(offset, length)?,
            )]);
        };
        let parts = layout
            .files
            .get(name)
            .ok_or("missing logical core file")?
            .clone();
        let mut cursor = 0;
        let mut pieces = Vec::new();
        for (size, blob) in parts {
            let stop = cursor + size;
            if cursor < end && offset < stop {
                let first = offset.max(cursor);
                let count = end.min(stop) - first;
                pieces.push(if blob.is_empty() {
                    SourcePiece::Zero(count)
                } else {
                    let source = self.physical(&blob)?;
                    if source.offset != 4096 || source.length != size {
                        return Err("snapshot blob length or alignment mismatch".into());
                    }
                    SourcePiece::Data(source.slice(first - cursor, count)?)
                });
            }
            cursor = stop;
        }
        Ok(pieces)
    }
    pub(crate) fn read_at(&mut self, name: &str, offset: u64, output: &mut [u8]) -> Result<()> {
        let mut cursor = 0;
        for piece in self.sources(name, offset, output.len() as u64)? {
            let count = match &piece {
                SourcePiece::Data(source) => source.length,
                SourcePiece::Zero(count) => *count,
            } as usize;
            match piece {
                SourcePiece::Data(source) => {
                    source.read_at(0, &mut output[cursor..cursor + count])?
                }
                SourcePiece::Zero(_) => output[cursor..cursor + count].fill(0),
            }
            cursor += count;
        }
        if cursor != output.len() {
            return Err("truncated logical core range".into());
        }
        Ok(())
    }
}

fn encode_header(raw: Vec<u8>) -> Result<Vec<u8>> {
    let size = (raw.len() + 8).next_multiple_of(4096) - 8;
    if size > MAX_HEADER as usize {
        return Err("core header exceeds its size limit".into());
    }
    let mut header = Vec::with_capacity(size + 8);
    header.extend_from_slice(&(size as u64).to_le_bytes());
    header.extend_from_slice(&raw);
    header.resize(size + 8, b' ');
    Ok(header)
}
pub(crate) fn container_header(length: u64, index_size: u64, format: &str) -> Result<Vec<u8>> {
    let mut raw = format!(
        "{{\"__metadata__\":{{\"format\":{}}},\"flare.payload\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[0,{}]}}",
        serde_json::to_string(format)?,
        length,
        length
    );
    if format != "flare-data" {
        raw.push_str(&format!(
            ",\"flare.metadata\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            index_size,
            length,
            length
                .checked_add(index_size)
                .ok_or("container length overflow")?
        ));
    }
    raw.push('}');
    encode_header(raw.into_bytes())
}
pub(crate) fn temporary(directory: &Path, prefix: &str) -> Result<(File, PathBuf)> {
    for _ in 0..100 {
        let path = directory.join(format!(
            ".{prefix}-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((file, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err("cannot create unique export temporary file".into())
}
pub(crate) fn install_blob(temporary: &Path, destination: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(temporary.as_os_str().as_bytes())?;
        let target = std::ffi::CString::new(destination.as_os_str().as_bytes())?;
        if unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW};
        let source: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
        let target: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        if unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
pub(crate) fn remove_temporary(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::fs::remove_file(path)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
        use windows_sys::Win32::Storage::FileSystem::*;
        let file = OpenOptions::new()
            .access_mode(DELETE)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)?;
        let info = FILE_DISPOSITION_INFO_EX {
            Flags: FILE_DISPOSITION_FLAG_DELETE
                | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS
                | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,
        };
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileDispositionInfoEx,
                (&info as *const FILE_DISPOSITION_INFO_EX).cast(),
                std::mem::size_of_val(&info) as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
fn open_directory(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::*;
        options
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    }
    Ok(options.open(path)?)
}
fn pinned_directory(file: &File, path: &Path) -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let _ = path;
        PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
    }
    #[cfg(windows)]
    {
        let _ = file;
        path.to_owned()
    }
}
pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        open_directory(path)?.sync_all()?;
    }
    #[cfg(windows)]
    {
        use std::{
            ffi::c_void,
            os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
        };
        use windows_sys::Win32::{Foundation::HANDLE, Storage::FileSystem::*};
        #[repr(C)]
        struct IoStatus {
            status: isize,
            information: usize,
        }
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtFlushBuffersFileEx(
                file: HANDLE,
                flags: u32,
                parameters: *mut c_void,
                size: u32,
                status: *mut IoStatus,
            ) -> i32;
            fn RtlNtStatusToDosError(status: i32) -> u32;
        }
        let directory = OpenOptions::new()
            .read(true)
            .write(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?;
        let mut status = IoStatus {
            status: 0,
            information: 0,
        };
        let result = unsafe {
            NtFlushBuffersFileEx(
                directory.as_raw_handle(),
                0,
                std::ptr::null_mut(),
                0,
                &mut status,
            )
        };
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(
                unsafe { RtlNtStatusToDosError(result) } as i32,
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blob_header_preserves_python_content_identity() {
        use sha2::{Digest, Sha256};
        let header = container_header(3, 0, "flare-data").unwrap();
        assert_eq!(header.len(), 4096);
        assert_eq!(&header[..8], &4088u64.to_le_bytes());
        let mut hash = Sha256::new();
        hash.update(header);
        hash.update(b"abc");
        assert_eq!(
            super::super::blob::hex(&hash.finalize()),
            "8f56e06cc2a240781ab8020dcd057b2d0f80df7599500615e6f43cc3b0a92885"
        );
    }
    #[test]
    fn duplicate_files_and_conflicting_blob_sizes_are_rejected() {
        assert!(serde_json::from_str::<Layout>(r#"{"version":1,"files":{"":[],"":[]}}"#).is_err());
        let name = format!("{}.safetensors", "a".repeat(64));
        let layout = Layout {
            version: 1,
            files: BTreeMap::from([(String::new(), vec![(1, name.clone()), (2, name)])]),
        };
        assert!(layout.validate().is_err());
    }
    #[test]
    fn snapshot_resolves_data_and_padding_without_overwrite() {
        let directory = std::env::temp_dir().join(format!(
            "flare-layout-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let blob = format!("{}.safetensors", "a".repeat(64));
        let mut bytes = container_header(3, 0, "flare-data").unwrap();
        bytes.extend_from_slice(b"abc");
        std::fs::write(directory.join(&blob), bytes).unwrap();
        let layout = Layout {
            version: 1,
            files: BTreeMap::from([(
                String::new(),
                vec![(2, String::new()), (3, blob), (1, String::new())],
            )]),
        };
        let path = directory.join("model.safetensors");
        let envelope = b"signed envelope";
        let encoded = serde_json::to_vec(&layout).unwrap();
        let header = serde_json::json!({
            "__metadata__": {"format": "flare-snapshot"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
            "flare.layout": {"dtype": "U8", "shape": [encoded.len()], "data_offsets": [envelope.len(), envelope.len() + encoded.len()]},
        });
        let mut bytes = encode_header(serde_json::to_vec(&header).unwrap()).unwrap();
        bytes.extend_from_slice(envelope);
        bytes.extend_from_slice(&encoded);
        std::fs::write(&path, bytes).unwrap();
        let mut sources = CoreSources::open(&path).unwrap();
        let mut output = [0xff; 6];
        sources.read_at("", 0, &mut output).unwrap();
        assert_eq!(&output, b"\0\0abc\0");
        assert!(sources.read_at("", 5, &mut [0; 2]).is_err());
        assert_eq!(&sources.container.envelope, b"signed envelope");
        drop(sources);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn staging_and_companion_sources_keep_payload_bounds() {
        let directory = std::env::temp_dir().join(format!(
            "flare-staging-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let primary = directory.join("model.safetensors");
        let mut data = container_header(3, 1, "flare-staging").unwrap();
        data.extend_from_slice(b"abcx");
        std::fs::write(&primary, data).unwrap();
        let mut companion = container_header(2, 0, "flare-data").unwrap();
        companion.extend_from_slice(b"yz");
        std::fs::write(directory.join("data.safetensors"), companion).unwrap();
        let mut sources = CoreSources::open(&primary).unwrap();
        let mut output = [0; 2];
        sources.read_at("data.safetensors", 0, &mut output).unwrap();
        assert_eq!(&output, b"yz");
        assert!(sources.sources("", 2, 2).is_err());
        assert_eq!(sources.container.format, "flare-staging");
        drop(sources);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
