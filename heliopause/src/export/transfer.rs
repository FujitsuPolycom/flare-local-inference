/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Authenticated preparation-rank export handoff over established fabric streams.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
#[cfg(test)]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::Instant;

use ring::{
    digest::{Context, SHA256},
    hmac,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{RankOutput, Result, invalid};

const BLOCK_BYTES: usize = 4 << 20;
const METADATA_BYTES: usize = 256 << 20;
const PROTOCOL: u32 = 2;
const EXPORT_IDLE_TIMEOUT_MS: i32 = 2 * 60 * 60 * 1000;

#[derive(Clone, Copy)]
enum FileDirection {
    Send,
    Receive,
}
struct FileTransactions {
    send: usize,
    receive: usize,
    reserved: usize,
    capacity: usize,
}
static FILE_TRANSACTIONS: (Mutex<FileTransactions>, Condvar) = (
    Mutex::new(FileTransactions {
        send: 0,
        receive: 0,
        reserved: 0,
        capacity: 0,
    }),
    Condvar::new(),
);
struct FileTransaction {
    direction: FileDirection,
    descriptors: usize,
}

fn descriptor_headroom() -> Result<usize> {
    let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let limit = unsafe { limit.assume_init() }.rlim_cur;
    let open = fs::read_dir("/proc/self/fd")?
        .collect::<std::io::Result<Vec<_>>>()?
        .len()
        .saturating_sub(1);
    Ok(usize::try_from(limit)
        .unwrap_or(usize::MAX)
        .saturating_sub(open)
        .saturating_sub(2))
}
struct AbortScope {
    sockets: Vec<Socket>,
    aborted: AtomicBool,
}
impl AbortScope {
    fn new(sockets: Vec<Socket>) -> Self {
        Self {
            sockets,
            aborted: AtomicBool::new(false),
        }
    }
    fn trigger(&self) {
        if !self.aborted.swap(true, Ordering::AcqRel) {
            for socket in &self.sockets {
                socket.abort();
            }
            FILE_TRANSACTIONS.1.notify_all();
        }
    }
    fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }
}

impl FileTransaction {
    fn acquire(direction: FileDirection, descriptors: usize) -> Result<Self> {
        Self::acquire_with_abort(direction, descriptors, None)
    }

    fn acquire_with_abort(
        direction: FileDirection,
        descriptors: usize,
        abort: Option<&AbortScope>,
    ) -> Result<Self> {
        let (lock, ready) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        loop {
            if abort.is_some_and(AbortScope::is_aborted) {
                return Err(invalid("rank export aborted"));
            }
            if state.reserved == 0 {
                state.capacity = descriptor_headroom()?;
            }
            // Network writes hold no file transaction; receivers leave room for source reads.
            let peer_reserve = match direction {
                FileDirection::Receive => 3,
                FileDirection::Send => 0,
            };
            if state.reserved + descriptors + peer_reserve <= state.capacity {
                state.reserved += descriptors;
                match direction {
                    FileDirection::Send => state.send += 1,
                    FileDirection::Receive => state.receive += 1,
                }
                ready.notify_all();
                return Ok(Self {
                    direction,
                    descriptors,
                });
            }
            if state.reserved == 0 {
                return Err(invalid(format!(
                    "insufficient file-descriptor headroom for rank export: {} available",
                    state.capacity
                )));
            }
            state = ready.wait(state).unwrap();
        }
    }

    fn note_retained_descriptors(count: usize) {
        let (lock, _) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        if state.reserved > 0 {
            state.capacity = state.capacity.saturating_sub(count);
        }
    }

    fn release_retained_descriptors(count: usize) {
        let (lock, ready) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        if state.reserved > 0 {
            state.capacity = state.capacity.saturating_add(count);
        }
        ready.notify_all();
    }
}
impl Drop for FileTransaction {
    fn drop(&mut self) {
        let (lock, ready) = &FILE_TRANSACTIONS;
        let mut state = lock.lock().unwrap();
        state.reserved -= self.descriptors;
        match self.direction {
            FileDirection::Send => state.send -= 1,
            FileDirection::Receive => state.receive -= 1,
        }
        ready.notify_all();
    }
}

/// Retains installed `PayloadObject` leases during one rank receive and restores
/// their descriptor headroom if the receive fails before ownership transfers out.
struct RetainedLeases {
    objects: Mutex<Vec<helios_core::core::cache::PayloadObject>>,
    retained_descriptors: AtomicUsize,
}
impl RetainedLeases {
    fn new() -> Self {
        Self {
            objects: Mutex::new(Vec::new()),
            retained_descriptors: AtomicUsize::new(0),
        }
    }

    fn push(&self, object: helios_core::core::cache::PayloadObject) -> Result<()> {
        self.objects
            .lock()
            .map_err(|_| invalid("payload lease lock poisoned"))?
            .push(object);
        self.retained_descriptors.fetch_add(2, Ordering::AcqRel);
        FileTransaction::note_retained_descriptors(2);
        Ok(())
    }

    fn into_leases(self) -> Result<crate::export::PayloadLeases> {
        self.retained_descriptors.store(0, Ordering::Release);
        let mut guard = self
            .objects
            .lock()
            .map_err(|_| invalid("payload lease lock poisoned"))?;
        Ok(crate::export::PayloadLeases(std::mem::take(&mut *guard)))
    }
}
impl Drop for RetainedLeases {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.objects.lock() {
            guard.clear();
        }
        let released = self.retained_descriptors.swap(0, Ordering::AcqRel);
        if released > 0 {
            FileTransaction::release_retained_descriptors(released);
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ChannelState {
    pub fd: i32,
    pub key: Vec<u8>,
    pub send_role: String,
    pub receive_role: String,
    pub sent: u64,
    pub received: u64,
}

// The binding retains the Python socket owners until every scoped worker has joined.
#[derive(Clone)]
struct Socket(i32);

impl Socket {
    fn borrow(fd: i32) -> Result<Self> {
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self(fd))
    }
    fn abort(&self) {
        unsafe {
            libc::shutdown(self.0, libc::SHUT_RDWR);
        }
    }
    fn ready(&self, events: i16) -> std::io::Result<i16> {
        let mut poll = libc::pollfd {
            fd: self.0,
            events,
            revents: 0,
        };
        loop {
            // Peers may be hashing sources, verifying cache entries, or draining bounded disk writers.
            let result = unsafe { libc::poll(&mut poll, 1, EXPORT_IDLE_TIMEOUT_MS) };
            if result > 0 {
                return Ok(poll.revents);
            }
            if result == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "rank export socket timed out",
                ));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}
impl Read for Socket {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        loop {
            self.ready(libc::POLLIN)?;
            let size = unsafe {
                libc::recv(
                    self.0,
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                    libc::MSG_DONTWAIT,
                )
            };
            if size >= 0 {
                return Ok(size as usize);
            }
            let error = std::io::Error::last_os_error();
            if !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(error);
            }
        }
    }
}
impl Write for Socket {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        loop {
            let revents = self.ready(libc::POLLOUT | libc::POLLRDHUP)?;
            let size = unsafe {
                libc::send(
                    self.0,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                )
            };
            if size >= 0 {
                if (revents & libc::POLLRDHUP) != 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "rank export peer closed connection",
                    ));
                }
                return Ok(size as usize);
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock
                && (revents & (libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR)) != 0
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "rank export peer closed connection",
                ));
            }
            if !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(error);
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Channel {
    socket: Socket,
    state: ChannelState,
}
impl Channel {
    fn new(state: ChannelState) -> Result<Self> {
        if state.key.len() != 32
            || !matches!(
                (state.send_role.as_str(), state.receive_role.as_str()),
                ("root", "peer") | ("peer", "root")
            )
        {
            return Err(invalid("invalid authenticated export channel handoff"));
        }
        Ok(Self {
            socket: Socket::borrow(state.fd)?,
            state,
        })
    }
    fn send(&mut self, mut header: Value, payload: &[u8]) -> Result<()> {
        if payload.len() > BLOCK_BYTES {
            return Err(invalid("export frame exceeds capacity"));
        }
        header["sequence"] = self.state.sent.into();
        header["bytes"] = payload.len().into();
        let header = serde_json::to_vec(&header)?;
        if header.len() > 65536 {
            return Err(invalid("export frame header exceeds capacity"));
        }
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.state.key);
        let mut signature = hmac::Context::with_key(&key);
        signature.update(self.state.send_role.as_bytes());
        signature.update(&header);
        signature.update(payload);
        self.socket
            .write_all(&(header.len() as u32).to_le_bytes())?;
        self.socket.write_all(&header)?;
        self.socket.write_all(payload)?;
        self.socket.write_all(signature.sign().as_ref())?;
        self.state.sent = self
            .state
            .sent
            .checked_add(1)
            .ok_or_else(|| invalid("export sequence overflow"))?;
        Ok(())
    }
    fn header(&mut self) -> Result<(Value, Vec<u8>, usize)> {
        let mut length = [0; 4];
        self.socket.read_exact(&mut length)?;
        let length = u32::from_le_bytes(length) as usize;
        if length > 65536 {
            return Err(invalid("export frame header exceeds capacity"));
        }
        let mut header = vec![0; length];
        self.socket.read_exact(&mut header)?;
        let value: Value = serde_json::from_slice(&header)?;
        let length = value["bytes"]
            .as_u64()
            .filter(|&n| n <= BLOCK_BYTES as u64)
            .ok_or_else(|| invalid("invalid export frame length"))? as usize;
        Ok((value, header, length))
    }
    fn payload(
        &mut self,
        kind: &str,
        value: &Value,
        header: &[u8],
        payload: &mut [u8],
    ) -> Result<()> {
        let mut signature = [0; 32];
        self.socket.read_exact(payload)?;
        self.socket.read_exact(&mut signature)?;
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.state.key);
        let mut signed = hmac::Context::with_key(&key);
        signed.update(self.state.receive_role.as_bytes());
        signed.update(header);
        signed.update(payload);
        #[allow(deprecated)]
        ring::constant_time::verify_slices_are_equal(signed.sign().as_ref(), &signature)
            .map_err(|_| invalid("rank export message authentication failed"))?;
        if value["sequence"].as_u64() != Some(self.state.received) {
            return Err(invalid("rank export message sequence differs"));
        }
        if value["kind"] != kind {
            return Err(invalid(format!(
                "expected export {kind}, received {}",
                value["kind"]
            )));
        }
        self.state.received = self
            .state
            .received
            .checked_add(1)
            .ok_or_else(|| invalid("export sequence overflow"))?;
        Ok(())
    }
    fn receive_into(&mut self, kind: &str, payload: &mut [u8]) -> Result<Value> {
        let (value, header, length) = self.header()?;
        if length != payload.len() {
            return Err(invalid("rank export payload range differs"));
        }
        self.payload(kind, &value, &header, payload)?;
        Ok(value)
    }
    #[cfg(test)]
    fn receive(&mut self, kind: &str) -> Result<(Value, Vec<u8>)> {
        let (value, header, length) = self.header()?;
        let mut payload = vec![0; length];
        self.payload(kind, &value, &header, &mut payload)?;
        Ok((value, payload))
    }
}

fn send_control(socket: &mut Socket, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > METADATA_BYTES {
        return Err(invalid("rank export manifest exceeds capacity"));
    }
    socket.write_all(b"N")?;
    socket.write_all(&(bytes.len() as u64).to_le_bytes())?;
    socket.write_all(&bytes)?;
    Ok(())
}
fn receive_control<T: serde::de::DeserializeOwned>(socket: &mut Socket) -> Result<T> {
    let mut header = [0; 9];
    socket.read_exact(&mut header)?;
    let length = u64::from_le_bytes(header[1..].try_into()?);
    if header[0] != b'N' || length > METADATA_BYTES as u64 {
        return Err(invalid("incompatible native rank export protocol"));
    }
    let mut bytes = vec![0; length as usize];
    socket.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn digest_bytes(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 {
        return Err(invalid("invalid rank payload digest"));
    }
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)?;
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Source {
    size: u64,
    sha256: String,
    blob: bool,
    /// Hex payload identity when the bytes are a cache object rather than a plain file.
    #[serde(default)]
    payload: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    output: RankOutput,
    files: BTreeMap<PathBuf, Source>,
    inventory: Value,
}
#[derive(Serialize, Deserialize)]
struct Request {
    version: u32,
    files: Vec<PathBuf>,
}

fn streams<T: Send>(
    states: Vec<ChannelState>,
    action: impl Fn(usize, &mut Channel) -> Result<T> + Sync,
) -> Result<(Vec<T>, Vec<ChannelState>)> {
    if states.is_empty() || states.len() > 64 {
        return Err(invalid(
            "rank export requires 1..64 established fabric streams",
        ));
    }
    let channels: Vec<_> = states
        .into_iter()
        .map(Channel::new)
        .collect::<Result<_>>()?;
    streams_ready(channels, None, action)
}

fn streams_ready<T: Send>(
    channels: Vec<Channel>,
    external_abort: Option<&AbortScope>,
    action: impl Fn(usize, &mut Channel) -> Result<T> + Sync,
) -> Result<(Vec<T>, Vec<ChannelState>)> {
    let abort: Vec<_> = channels
        .iter()
        .map(|channel| channel.socket.clone())
        .collect();
    let failure = Mutex::new(None);
    std::thread::scope(|scope| {
        let handles: Vec<_> = channels
            .into_iter()
            .enumerate()
            .map(|(stream, mut channel)| {
                let action = &action;
                let failure = &failure;
                let abort = &abort;
                scope.spawn(move || match action(stream, &mut channel) {
                    Ok(value) => Some((value, channel.state)),
                    Err(error) => {
                        let mut first = failure.lock().unwrap();
                        if first.is_none() {
                            eprintln!("rank export failed: {error}");
                            *first = Some(error);
                            for socket in abort {
                                socket.abort();
                            }
                            if let Some(scope) = external_abort {
                                scope.trigger();
                            }
                        }
                        None
                    }
                })
            })
            .collect();
        let mut values = Vec::new();
        let mut states = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Some((value, state))) => {
                    values.push(value);
                    states.push(state);
                }
                Ok(None) => {}
                Err(_) => {
                    for socket in &abort {
                        socket.abort();
                    }
                    if let Some(scope) = external_abort {
                        scope.trigger();
                    }
                    return Err(invalid("rank export worker panicked"));
                }
            }
        }
        if let Some(error) = failure.lock().unwrap().take() {
            return Err(error);
        }
        Ok((values, states))
    })
}

fn completion(
    channel: &mut Channel,
    names: &[PathBuf],
    files: &BTreeMap<PathBuf, Source>,
    sender: bool,
) -> Result<()> {
    let stats = json!({"files": names.len(), "payload_bytes": names.iter().map(|n| files[n].size).sum::<u64>()});
    if sender {
        channel.send(json!({"kind":"export-finished", "stats": stats}), &[])?;
    }
    let message = channel.receive_into("export-finished", &mut [])?;
    if message["stats"] != stats {
        return Err(invalid("rank export completion accounting differs"));
    }
    if !sender {
        channel.send(json!({"kind":"export-finished", "stats": stats}), &[])?;
    }
    Ok(())
}

/// Declared payload objects resolved from the rank's live leases.
///
/// Declared weight groups hold their bytes in the shared model cache under a
/// payload identity (`WeightPayloadGroup.payload_id`). Capture keeps those
/// objects leased in `RankOutput.payload_leases`, so the sender reads the exact
/// leased files rather than reopening paths from an identity string.
pub(crate) fn cache_payload_sources(output: &RankOutput) -> Result<Vec<(PathBuf, String, u64)>> {
    let mut sources = Vec::new();
    let Some(weight) = output.weight_payload.as_ref() else {
        return Ok(sources);
    };
    for group in weight
        .groups
        .iter()
        .chain(std::iter::once(&weight.initial_state))
    {
        if group.payload_id.is_empty() {
            continue;
        }
        if weight.cached.iter().any(|payload| payload.id == group.payload_id && payload.nbytes == group.nbytes) {
            continue;
        }
        let id: [u8; 32] = group
            .payload_id
            .as_slice()
            .try_into()
            .map_err(|_| invalid("declared payload identity is not 32 bytes"))?;
        let object = output
            .payload_leases
            .0
            .iter()
            .find(|object| object.id().as_bytes() == &id)
            .ok_or_else(|| {
                invalid(format!(
                    "declared payload group {:?} has no leased cache object",
                    group.name
                ))
            })?;
        if object.length() != group.nbytes {
            return Err(invalid(format!(
                "declared payload group {:?} length differs from its leased object",
                group.name
            )));
        }
        let hex: String = id.iter().map(|byte| format!("{byte:02x}")).collect();
        sources.push((object.path().to_owned(), hex, group.nbytes));
    }
    Ok(sources)
}

pub(crate) fn send_rank(
    output: RankOutput,
    inventory: Value,
    control_fd: i32,
    channels: Vec<ChannelState>,
) -> Result<Vec<ChannelState>> {
    let abort_fds: Vec<i32> = std::iter::once(control_fd)
        .chain(channels.iter().map(|channel| channel.fd))
        .collect();
    let result = send_rank_inner(output, inventory, control_fd, channels);
    if result.is_err() {
        for fd in abort_fds {
            if let Ok(socket) = Socket::borrow(fd) {
                socket.abort();
            }
        }
    }
    result
}

fn send_rank_inner(
    output: RankOutput,
    inventory: Value,
    control_fd: i32,
    channels: Vec<ChannelState>,
) -> Result<Vec<ChannelState>> {
    // Hold the leases for the whole transfer: the rank's declared payload objects
    // must outlive the manifest, the data channels, and the completion exchange.
    let payload_leases = output.payload_leases.0.clone();
    let _leased = payload_leases;
    if channels.is_empty() {
        return Err(invalid(
            "rank export requires established fabric channels; management payload transfer is forbidden",
        ));
    }
    let budget = super::io::IoBudget::for_ranks(1)?;
    let progress = helios_core::startup::StartupProgress::silent();
    let scan_phase = progress.phase("Hash rank export sources", None);
    // Cache-backed declared payloads resolve one leased object per declared group.
    let objects = cache_payload_sources(&output)?;
    let mut seen = BTreeSet::new();
    let mut scan_targets: Vec<(PathBuf, Option<String>)> = Vec::new();
    for part in &output.payload {
        if seen.insert(part.source.path.clone()) {
            scan_targets.push((part.source.path.clone(), None));
        }
    }
    for (path, hex, _length) in objects {
        if seen.insert(path.clone()) {
            scan_targets.push((path, Some(hex)));
        }
    }
    let hashed_bytes = AtomicU64::new(0);
    let mut files = BTreeMap::new();
    let mut identities = BTreeMap::new();
    if !scan_targets.is_empty() {
        let workers = (descriptor_headroom().unwrap_or(3) / 4)
            .clamp(1, 4)
            .min(scan_targets.len());
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let failure: Mutex<Option<Box<dyn std::error::Error + Send + Sync>>> = Mutex::new(None);
        let scanned: Mutex<Vec<(PathBuf, Source, super::io::Identity)>> =
            Mutex::new(Vec::with_capacity(scan_targets.len()));
        let scan_one = |(path, payload): &(PathBuf, Option<String>)| -> Result<(PathBuf, Source, super::io::Identity)> {
            let _transaction = FileTransaction::acquire(FileDirection::Send, 3)?;
            let file = Arc::new(super::io::open_read(path)?);
            let before = super::io::Identity::of(&file)?;
            let size = file.metadata()?.len();
            let mut digest = Context::new(&SHA256);
            let source = super::io::Source::from_file(file.clone(), 0, size)?;
            budget.scan(&source, |bytes| {
                if failed.load(Ordering::Relaxed) {
                    return Err(invalid("rank export scan aborted"));
                }
                digest.update(bytes);
                let count = bytes.len() as u64;
                let total = hashed_bytes.fetch_add(count, Ordering::Relaxed) + count;
                if payload.is_none() {
                    scan_phase.advance(count);
                    scan_phase.detail(format!("{total} bytes hashed"));
                }
                Ok(())
            })?;
            if super::io::Identity::of(&file)? != before {
                return Err(invalid(if payload.is_some() {
                    "cached payload object changed during metadata scan"
                } else {
                    "rank export source changed during metadata scan"
                }));
            }
            let sha256 = super::blob::hex(digest.finish().as_ref());
            let blob = payload.is_none()
                && path.file_name().and_then(|n| n.to_str())
                    == Some(&format!("{sha256}.safetensors"));
            drop(source);
            drop(file);
            Ok((
                path.clone(),
                Source {
                    size,
                    sha256,
                    blob,
                    payload: payload.clone(),
                },
                before,
            ))
        };
        let worker_loop = || {
            while !failed.load(Ordering::Acquire) {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= scan_targets.len() {
                    break;
                }
                match scan_one(&scan_targets[index]) {
                    Ok(entry) => scanned.lock().unwrap().push(entry),
                    Err(error) => {
                        failed.store(true, Ordering::Release);
                        let mut first = failure.lock().unwrap();
                        if first.is_none() {
                            *first = Some(error);
                        }
                        break;
                    }
                }
            }
        };
        if workers == 1 {
            worker_loop();
        } else {
            std::thread::scope(|scope| -> Result<()> {
                let handles: Vec<_> = (1..workers).map(|_| scope.spawn(worker_loop)).collect();
                worker_loop();
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| invalid("rank export scan worker panicked"))?;
                }
                Ok(())
            })?;
        }
        if let Some(error) = failure.into_inner().unwrap() {
            return Err(error);
        }
        for (path, info, before) in scanned.into_inner().unwrap() {
            files.insert(path.clone(), info);
            identities.insert(path, before);
        }
    }
    scan_phase.finish(format!(
        "{} files, {} bytes hashed",
        files.len(),
        hashed_bytes.load(Ordering::Relaxed)
    ));
    let mut control = Socket::borrow(control_fd)?;
    send_control(
        &mut control,
        &Manifest {
            version: PROTOCOL,
            output,
            files: files.clone(),
            inventory,
        },
    )?;
    let request: Request = receive_control(&mut control)?;
    let unique: BTreeSet<_> = request.files.iter().collect();
    if request.version != PROTOCOL
        || unique.len() != request.files.len()
        || request.files.iter().any(|n| !files.contains_key(n))
    {
        return Err(invalid("invalid requested rank export sources"));
    }
    let count = channels.len();
    let total: u64 = request.files.iter().map(|name| files[name].size).sum();
    let phase = progress.phase("Send export payload", Some(total));
    let (_, states) = streams(channels, |stream, channel| {
        let names: Vec<_> = request
            .files
            .iter()
            .skip(stream)
            .step_by(count)
            .cloned()
            .collect();
        let mut buffer = budget.buffer()?;
        for name in &names {
            let size = files[name].size;
            let mut offset = 0;
            while offset < size {
                let length = (size - offset).min(BLOCK_BYTES as u64) as usize;
                {
                    let _transaction = FileTransaction::acquire(FileDirection::Send, 3)?;
                    let source = Arc::new(super::io::open_read(name)?);
                    if super::io::Identity::of(&source)? != identities[name] {
                        return Err(invalid("rank export source changed before transfer"));
                    }
                    let pinned = super::io::Source::from_file(source.clone(), 0, size)?;
                    pinned.read_at(offset, &mut buffer.bytes_mut()[..length])?;
                    if super::io::Identity::of(&source)? != identities[name] {
                        return Err(invalid("rank export source changed during transfer"));
                    }
                }
                channel.send(
                    json!({"kind":"export-data", "name":name, "offset":offset}),
                    &buffer.bytes()[..length],
                )?;
                offset += length as u64;
                phase.advance_bytes(length as u64);
            }
        }
        completion(channel, &names, &files, true)
    })?;
    phase.finish(format!("{total} bytes sent over {count} streams"));
    Ok(states)
}

#[derive(Deserialize)]
pub(crate) struct ReceiveRequest {
    pub control_fd: i32,
    pub channels: Vec<ChannelState>,
    pub destination: PathBuf,
}
#[derive(Serialize)]
pub(crate) struct ReceiveOutput {
    #[cfg(test)]
    #[serde(skip_serializing)]
    pub output: RankOutput,
    /// Live leases for the payload objects this receive installed.
    ///
    /// The coordinator must move these into its retained export stage before the
    /// outputs are dropped, otherwise a later `combine` can find the objects gone.
    #[serde(skip)]
    pub leases: crate::export::PayloadLeases,
    pub inventory: Value,
    pub channels: Vec<ChannelState>,
    pub stats: Value,
}

struct ReceiveProgress {
    phase: helios_core::startup::Phase,
    bytes: Mutex<u64>,
}
impl ReceiveProgress {
    fn advance(&self, count: u64) {
        let mut bytes = self.bytes.lock().unwrap();
        *bytes += count;
        self.phase.advance(count);
        self.phase.detail(format!("{} bytes received", *bytes));
    }
}

/// Verify an already-present cache object by payload identity and declared length.
///
/// A cached container may be any valid wrapper of the same payload bytes, so the
/// payload digest and the declared group length are authoritative here; the
/// transport container digest only proves what this transfer carried.
fn verify_cached_payload(
    cache: &Path,
    id: helios_core::core::PayloadId,
    length: u64,
) -> Result<helios_core::core::cache::PayloadObject> {
    let root = helios_core::core::cache::Root::open(cache)?;
    let object = root
        .payload(id)?
        .ok_or_else(|| invalid("cached payload object disappeared"))?;
    if object.length() != length {
        return Err(invalid("cached payload object has the wrong length"));
    }
    if object.slice().digest()? != id {
        return Err(invalid("cached payload object does not match its identity"));
    }
    Ok(object)
}

/// Adopt a received cache payload object under its signed payload identity.
///
/// The staged file is a complete, synced companion container; its payload bytes
/// are verified against the declared identity by the cache insertion path, so a
/// corrupt or mislabelled object never enters the cache. A canonical wrapper is
/// linked into the cache as is; otherwise, or when the filesystem refuses the
/// link, the slice is copied.
fn install_payload_object(
    cache: &Path,
    id: helios_core::core::PayloadId,
    length: u64,
    partial: &Path,
) -> Result<helios_core::core::cache::PayloadObject> {
    let root = helios_core::core::cache::Root::open(cache)?;
    let container = helios_core::core::container::Container::open(partial)?;
    let slice = container.tensor(helios_core::core::container::COMPANION_TENSOR)?;
    drop(container);
    if slice.len() != length {
        return Err(invalid(
            "received payload object length differs from its declared group",
        ));
    }
    let object = match root.adopt_companion(id, &slice, &partial.display().to_string())? {
        Some(object) => object,
        None => root.store_slice(Some(id), &slice)?,
    };
    if object.length() != length {
        return Err(invalid("received payload object has the wrong length"));
    }
    Ok(object)
}

/// Adopt a synced canonical companion whose payload digest was verified while
/// streaming the transfer, falling back to slice installation when the framing
/// is non-canonical or hard-linking is refused.
fn install_verified_payload_object(
    cache: &Path,
    id: helios_core::core::PayloadId,
    length: u64,
    partial: &Path,
    expected_dev_ino: (u64, u64),
) -> Result<helios_core::core::cache::PayloadObject> {
    let root = helios_core::core::cache::Root::open(cache)?;
    if let Some(object) = root.adopt_verified_companion(id, length, partial, expected_dev_ino)? {
        if object.length() != length {
            return Err(invalid("received payload object has the wrong length"));
        }
        return Ok(object);
    }
    install_payload_object(cache, id, length, partial)
}

fn receive_rank(
    request: ReceiveRequest,
    mut control: Socket,
    channels: Vec<Channel>,
    cache: &Path,
    no_weights: bool,
    budget: &Arc<super::io::IoBudget>,
    aggregate: &ReceiveProgress,
    abort: &AbortScope,
) -> Result<ReceiveOutput> {
    if request.channels.is_empty() {
        return Err(invalid(
            "rank export requires established fabric channels; management payload transfer is forbidden",
        ));
    }
    let mut manifest: Manifest = receive_control(&mut control)?;
    if manifest.version != PROTOCOL {
        return Err(invalid("incompatible native rank export protocol version"));
    }
    let mut logical_end = 0;
    for part in &manifest.output.payload {
        if part.offset < logical_end || part.source.offset.checked_add(part.source.nbytes).is_none()
        {
            return Err(invalid(
                "rank export payload ranges must be ordered and disjoint",
            ));
        }
        logical_end = part
            .offset
            .checked_add(part.source.nbytes)
            .ok_or_else(|| invalid("rank export payload length overflow"))?;
    }
    super::core::validate_index(&manifest.output.index, logical_end)?;
    fs::create_dir_all(
        request
            .destination
            .parent()
            .ok_or_else(|| invalid("rank export destination has no parent"))?,
    )?;
    let store = super::blob::BlobStore::new(cache, budget.clone())?;
    let installed = RetainedLeases::new();
    let progress = helios_core::startup::StartupProgress::silent();
    let cache_phase = progress.phase("Verify rank export cache", None);
    let mut cached_bytes = 0;
    let mut cached_files = 0;
    let mut targets = BTreeMap::new();
    let mut requested = Vec::new();
    // Declared cache payload identities, parsed before any path is formed.
    let mut declared_objects: BTreeMap<[u8; 32], u64> = BTreeMap::new();
    let mut declared_weights = BTreeSet::new();
    if let Some(sources) = manifest.output.weight_payload.as_ref() {
        declared_weights.extend(sources.groups.iter().map(|group| group.payload_id.clone()));
        for group in sources
            .groups
            .iter()
            .chain(std::iter::once(&sources.initial_state))
        {
            if group.payload_id.is_empty() {
                continue;
            }
            let id: [u8; 32] = group
                .payload_id
                .as_slice()
                .try_into()
                .map_err(|_| invalid("declared payload identity is not 32 bytes"))?;
            // Byte-identical groups legitimately share one payload identity; only
            // conflicting lengths for one identity are invalid.
            match declared_objects.entry(id) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(group.nbytes);
                }
                std::collections::btree_map::Entry::Occupied(slot) => {
                    if *slot.get() != group.nbytes {
                        return Err(invalid(
                            "declared payload identity has conflicting group lengths",
                        ));
                    }
                }
            }
        }
    }
    let mut object_sources: BTreeMap<[u8; 32], &Source> = BTreeMap::new();
    for (name, info) in &manifest.files {
        let Some(hex) = &info.payload else {
            continue;
        };
        let id = helios_core::core::PayloadId::parse(hex)
            .map_err(|_| invalid("rank export payload identity is invalid"))?;
        let Some(length) = declared_objects.get(id.as_bytes()) else {
            return Err(invalid("rank export carries an undeclared payload object"));
        };
        if *length == 0 || info.size == 0 || info.size > 64 << 30 {
            return Err(invalid("invalid declared payload object size"));
        }
        let _ = name;
        object_sources.insert(*id.as_bytes(), info);
    }
    let mut cached_objects = BTreeSet::new();
    if let Some(sources) = manifest.output.weight_payload.as_ref() {
        for descriptor in &sources.cached {
            let id: [u8; 32] = descriptor.id.as_slice().try_into()
                .map_err(|_| invalid("cached rank payload identity is not 32 bytes"))?;
            if declared_objects.get(&id) != Some(&descriptor.nbytes) || object_sources.contains_key(&id) {
                return Err(invalid("cached rank payload differs from its declared group"));
            }
            if cached_objects.insert(id) {
                let _transaction =
                    FileTransaction::acquire_with_abort(FileDirection::Receive, 3, Some(abort))?;
                let root = helios_core::core::cache::Root::open(cache)?;
                let object = root.payload(helios_core::core::PayloadId::from_bytes(id))?
                    .ok_or_else(|| invalid("reused payload object disappeared"))?;
                if object.length() != descriptor.nbytes {
                    return Err(invalid("reused payload object has the wrong length"));
                }
                installed.push(object)?;
                cached_bytes += descriptor.nbytes;
                cached_files += 1;
            }
        }
    }
    if object_sources.len() + cached_objects.len() != declared_objects.len() {
        return Err(invalid(
            "declared payload objects are missing from the rank export inventory",
        ));
    }
    let referenced: BTreeSet<_> = manifest
        .output
        .payload
        .iter()
        .map(|p| p.source.path.clone())
        .collect();
    // Cache-backed declared payload objects are inventory entries that no logical
    // payload part references, so both sets together must describe the manifest.
    let objects: BTreeSet<_> = manifest
        .files
        .iter()
        .filter(|(_, info)| info.payload.is_some())
        .map(|(name, _)| name.clone())
        .collect();
    let declared: BTreeSet<_> = referenced.union(&objects).cloned().collect();
    if declared != manifest.files.keys().cloned().collect() {
        return Err(invalid(
            "rank export file inventory differs from payload sources",
        ));
    }
    for (name, info) in &manifest.files {
        if info.sha256.len() != 64
            || !info
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || info.size == 0
            || info.size > 64 << 30
        {
            return Err(invalid("invalid rank export file digest or size"));
        }
        let target = if let Some(hex) = &info.payload {
            cache.join("blobs").join(format!("{hex}.safetensors"))
        } else if info.blob {
            store.root.join(format!("{}.safetensors", info.sha256))
        } else {
            request
                .destination
                .parent()
                .unwrap()
                .join(format!("{}.source", info.sha256))
        };
        if target.try_exists()? {
            let _transaction =
                FileTransaction::acquire_with_abort(FileDirection::Receive, 3, Some(abort))?;
            if let Some(hex) = &info.payload {
                let id = helios_core::core::PayloadId::parse(hex)?;
                let length = declared_objects[id.as_bytes()];
                let object = verify_cached_payload(cache, id, length)?;
                installed.push(object)?;
            } else {
                super::blob::verify_file(&target, digest_bytes(&info.sha256)?, info.size, &budget)?;
            }
            cached_bytes += info.size;
            cached_files += 1;
            cache_phase.advance(info.size);
            cache_phase.detail(format!(
                "{cached_files} cached files, {cached_bytes} bytes available"
            ));
        } else {
            if info.blob && no_weights {
                return Err(invalid(
                    "--no-weights requires the peer stored-data blob in the coordinator cache",
                ));
            }
            if info.payload.as_ref().is_some_and(|hex| no_weights &&
                helios_core::core::PayloadId::parse(hex).is_ok_and(|id| declared_weights.contains(id.as_bytes().as_slice()))) {
                return Err(invalid(
                    "--no-weights requires the peer payload object in the coordinator cache",
                ));
            }
            requested.push(name.clone());
        }
        targets.insert(name.clone(), target);
    }
    for part in &manifest.output.payload {
        if part
            .source
            .offset
            .checked_add(part.source.nbytes)
            .filter(|&end| end <= manifest.files[&part.source.path].size)
            .is_none()
        {
            return Err(invalid("rank export source range exceeds its file"));
        }
    }
    cache_phase.finish(format!(
        "{cached_files} cached files, {cached_bytes} bytes available"
    ));
    send_control(
        &mut control,
        &Request {
            version: PROTOCOL,
            files: requested.clone(),
        },
    )?;
    let started = Instant::now();
    let count = request.channels.len();
    let total: u64 = requested.iter().map(|name| manifest.files[name].size).sum();
    let phase = progress.phase("Receive export payload", Some(total));
    let (_, channels) = streams_ready(channels, Some(abort), |stream, channel| {
        let names: Vec<_> = requested
            .iter()
            .skip(stream)
            .step_by(count)
            .cloned()
            .collect();
        for name in &names {
            let info = &manifest.files[name];
            let target = &targets[name];
            channel.socket.ready(libc::POLLIN)?;
            let _transaction =
                FileTransaction::acquire_with_abort(FileDirection::Receive, 4, Some(abort))?;
            let (file, partial) =
                super::archive::temporary(target.parent().unwrap(), ".rank-blob-")?;
            let result = (|| -> Result<()> {
                struct StreamedCompanion {
                    expected_prefix: Vec<u8>,
                    header: Vec<u8>,
                    payload_digest: Context,
                    next_offset: u64,
                }
                let mut declared_payload = if let Some(hex) = info.payload.as_deref() {
                    let id = helios_core::core::PayloadId::parse(hex)?;
                    let length = declared_objects[id.as_bytes()];
                    let streamed = helios_core::core::cache::companion_prefix(length)
                        .ok()
                        .filter(|prefix| {
                            (prefix.len() as u64).checked_add(length) == Some(info.size)
                        })
                        .map(|expected_prefix| StreamedCompanion {
                            header: Vec::with_capacity(expected_prefix.len()),
                            expected_prefix,
                            payload_digest: Context::new(&SHA256),
                            next_offset: 0,
                        });
                    Some((id, length, streamed))
                } else {
                    None
                };
                let mut digest = Context::new(&SHA256);
                budget.write(&file, 0, info.size, |position, bytes| {
                    let mut filled = 0;
                    while filled < bytes.len() {
                        let offset = position + filled as u64;
                        let length = (info.size - offset).min(BLOCK_BYTES as u64) as usize;
                        if length > bytes.len() - filled {
                            return Err(invalid("rank export payload range differs"));
                        }
                        let payload = &mut bytes[filled..filled + length];
                        let message = channel.receive_into("export-data", payload)?;
                        if message["name"].as_str() != name.to_str()
                            || message["offset"].as_u64() != Some(offset)
                        {
                            return Err(invalid("rank export payload range differs"));
                        }
                        digest.update(payload);
                        if let Some((_, _, Some(verifier))) = &mut declared_payload {
                            if offset != verifier.next_offset {
                                return Err(invalid("rank export payload range differs"));
                            }
                            let prefix_len = verifier.expected_prefix.len() as u64;
                            if offset < prefix_len {
                                let header_take = ((prefix_len - offset) as usize).min(payload.len());
                                verifier.header.extend_from_slice(&payload[..header_take]);
                                if header_take < payload.len() {
                                    verifier.payload_digest.update(&payload[header_take..]);
                                }
                            } else {
                                verifier.payload_digest.update(payload);
                            }
                            verifier.next_offset = offset + payload.len() as u64;
                        }
                        filled += length;
                        phase.advance_bytes(length as u64);
                        aggregate.advance(length as u64);
                    }
                    Ok(())
                })?;
                if super::blob::hex(digest.finish().as_ref()) != info.sha256 {
                    return Err(invalid("rank payload digest mismatch"));
                }
                if let Some((declared_id, declared_len, streamed)) = declared_payload {
                    if let Some(verifier) = streamed
                        && verifier.next_offset == info.size
                        && verifier.header == verifier.expected_prefix
                    {
                        let actual_id = helios_core::core::PayloadId::from_digest(
                            verifier.payload_digest.finish(),
                        );
                        if actual_id != declared_id {
                            return Err(invalid(
                                "received payload digest does not match its declared identity",
                            ));
                        }
                        #[cfg(unix)]
                        file.set_permissions({
                            use std::os::unix::fs::PermissionsExt;
                            fs::Permissions::from_mode(0o444)
                        })?;
                        file.sync_all()?;
                        #[cfg(unix)]
                        let dev_ino = {
                            use std::os::unix::fs::MetadataExt;
                            let metadata = file.metadata()?;
                            (metadata.dev(), metadata.ino())
                        };
                        #[cfg(not(unix))]
                        let dev_ino = (0, 0);
                        drop(file);
                        let object = install_verified_payload_object(
                            cache,
                            declared_id,
                            declared_len,
                            &partial,
                            dev_ino,
                        )?;
                        installed.push(object)?;
                        return Ok(());
                    }
                    file.sync_all()?;
                    drop(file);
                    let object = install_payload_object(cache, declared_id, declared_len, &partial)?;
                    installed.push(object)?;
                    return Ok(());
                }
                file.sync_all()?;
                if info.blob {
                    file.set_permissions({
                        use std::os::unix::fs::PermissionsExt;
                        fs::Permissions::from_mode(0o444)
                    })?;
                }
                // The synced file may become a cache object itself: close the
                // writable descriptor before it is published.
                drop(file);
                match fs::hard_link(&partial, target) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        super::blob::verify_file(
                            target,
                            digest_bytes(&info.sha256)?,
                            info.size,
                            &budget,
                        )?;
                    }
                    Err(error) => return Err(error.into()),
                }
                Ok(())
            })();
            let _ = fs::remove_file(&partial);
            result?;
        }
        completion(channel, &names, &manifest.files, false)
    })?;
    for part in &mut manifest.output.payload {
        part.source.path = targets[&part.source.path].clone();
    }
    manifest.output.staging = None;
    {
        let _transaction =
            FileTransaction::acquire_with_abort(FileDirection::Receive, 1, Some(abort))?;
        super::core::write_sources(&request.destination, &manifest.output)?;
    }
    let leases = installed.into_leases()?;
    let seconds = started.elapsed().as_secs_f64();
    let size: u64 = requested.iter().map(|n| manifest.files[n].size).sum();
    phase.finish(format!(
        "{size} bytes received, {cached_files} cached files ({cached_bytes} bytes)"
    ));
    Ok(ReceiveOutput {
        #[cfg(test)]
        output: manifest.output,
        leases,
        inventory: manifest.inventory,
        channels,
        stats: json!({"payload_bytes":size,"payload_seconds":seconds,"payload_gbps":size as f64 * 8.0 / seconds.max(f64::MIN_POSITIVE) / 1e9,"files":requested.len(),"streams":count,"transport":"fabric-tcp","cached_bytes":cached_bytes,"cached_files":cached_files}),
    })
}

pub(crate) fn receive_ranks(
    requests: Vec<ReceiveRequest>,
    cache: &Path,
    no_weights: bool,
) -> Result<Vec<ReceiveOutput>> {
    let budget = super::io::IoBudget::for_ranks(1)?;
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let aggregate = ReceiveProgress {
        phase: progress.phase("Receive rank export payloads", None),
        bytes: Mutex::new(0),
    };
    let jobs = requests
        .into_iter()
        .map(|request| -> Result<_> {
            if request.channels.len() > 64 {
                return Err(invalid(
                    "rank export requires 1..64 established fabric streams",
                ));
            }
            let control = Socket::borrow(request.control_fd)?;
            let channels = request
                .channels
                .iter()
                .cloned()
                .map(Channel::new)
                .collect::<Result<Vec<_>>>()?;
            Ok((request, control, channels))
        })
        .collect::<Result<Vec<_>>>()?;
    let abort = AbortScope::new(
        jobs.iter()
            .flat_map(|(_, control, channels)| {
                std::iter::once(control.clone())
                    .chain(channels.iter().map(|channel| channel.socket.clone()))
            })
            .collect(),
    );
    let failure = Mutex::new(None);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .into_iter()
            .map(|(request, control, channels)| {
                let failure = &failure;
                let abort = &abort;
                let budget = &budget;
                let aggregate = &aggregate;
                scope.spawn(move || {
                    match receive_rank(
                        request, control, channels, cache, no_weights, budget, aggregate, abort,
                    ) {
                        Ok(output) => Some(output),
                        Err(error) => {
                            abort.trigger();
                            let mut first = failure.lock().unwrap();
                            if first.is_none() {
                                eprintln!("rank export failed: {error}");
                                *first = Some(error);
                            }
                            None
                        }
                    }
                })
            })
            .collect();
        let mut outputs = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Some(output)) => outputs.push(output),
                Ok(None) => {}
                Err(_) => {
                    abort.trigger();
                    return Err(invalid("rank export receiver panicked"));
                }
            }
        }
        if let Some(error) = failure.lock().unwrap().take() {
            return Err(error);
        }
        Ok(outputs)
    })?;
    let received: u64 = outputs
        .iter()
        .map(|output| output.stats["payload_bytes"].as_u64().unwrap_or(0))
        .sum();
    let cached: u64 = outputs
        .iter()
        .map(|output| output.stats["cached_bytes"].as_u64().unwrap_or(0))
        .sum();
    aggregate.phase.finish(format!(
        "{received} bytes received, {cached} cached bytes across {} ranks",
        outputs.len()
    ));
    Ok(outputs)
}

#[pyo3::pyfunction]
fn export_send_rank(
    py: pyo3::Python<'_>,
    output: pyo3::PyRef<'_, super::binding::NativeRankOutput>,
    inventory: &str,
    control_fd: i32,
    channels: &str,
    owners: pyo3::Py<pyo3::PyAny>,
) -> pyo3::PyResult<String> {
    use pyo3::types::PyAnyMethods;
    if owners.bind(py).is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rank export requires retained socket owners with exclusive use until return",
        ));
    }
    let parse = || -> Result<_> {
        Ok((
            serde_json::from_str(inventory)?,
            serde_json::from_str(channels)?,
        ))
    };
    let (inventory, channels) =
        parse().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let output = output.output.clone();
    let mut lease = crate::lifetime::OwnerLease(Some(owners));
    let result = py.detach(move || -> Result<String> {
        Ok(serde_json::to_string(&send_rank(
            output, inventory, control_fd, channels,
        )?)?)
    });
    drop(lease.take_after_drain());
    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyo3::pyfunction]
fn export_receive_ranks(
    py: pyo3::Python<'_>,
    requests: &str,
    cache: PathBuf,
    no_weights: bool,
    owners: pyo3::Py<pyo3::PyAny>,
    // Required, not optional: a caller without a stage would silently drop the
    // received payload leases before `combine` runs.
    stage: pyo3::PyRef<'_, super::binding::NativeExportStage>,
) -> pyo3::PyResult<String> {
    use pyo3::types::PyAnyMethods;
    if owners.bind(py).is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rank export requires retained socket owners with exclusive use until return",
        ));
    }
    let requests = serde_json::from_str(requests)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut lease = crate::lifetime::OwnerLease(Some(owners));
    let result = py.detach(move || receive_ranks(requests, &cache, no_weights));
    drop(lease.take_after_drain());
    let mut outputs =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    // Move received payload leases into the retained export stage while the
    // outputs still own them, so `combine` cannot lose the objects afterwards.
    for output in outputs.iter_mut() {
        stage
            .retain_payload_leases(std::mem::take(&mut output.leases))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    }
    serde_json::to_string(&outputs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

pub(crate) fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    module.add_function(pyo3::wrap_pyfunction!(export_send_rank, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(export_receive_ranks, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    static SERIAL: AtomicU64 = AtomicU64::new(0);
    fn sha256_hex(bytes: &[u8]) -> String {
        super::super::blob::hex(ring::digest::digest(&SHA256, bytes).as_ref())
    }
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-export-transfer-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn sockets() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (root, _) = listener.accept().unwrap();
        (root, peer)
    }
    fn state(socket: &TcpStream, server: bool) -> ChannelState {
        ChannelState {
            fd: socket.as_raw_fd(),
            key: vec![23; 32],
            send_role: if server { "root" } else { "peer" }.into(),
            receive_role: if server { "peer" } else { "root" }.into(),
            sent: 17,
            received: 17,
        }
    }
    fn output(path: PathBuf, length: u64) -> RankOutput {
        RankOutput {
            payload_leases: Default::default(),
            hardware: None,
            index: super::super::wire::ModelIndex {
                version: 4,
                ranks: vec![Default::default()],
                ..Default::default()
            },
            payload: vec![super::super::FilePart {
                offset: 0,
                source: super::super::FileSource {
                    path,
                    offset: 0,
                    nbytes: length,
                },
            }],
            allocations: vec![],
            module_globals: vec![],
            external: vec![],
            staging: None,
            weight_payload: None,
        }
    }
    fn roundtrip(
        directory: &Directory,
        name: &str,
        source: &RankOutput,
        no_weights: bool,
    ) -> Result<ReceiveOutput> {
        let (control_root, control_peer) = sockets();
        let (root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    source.clone(),
                    json!({"test":true}),
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let receiver = receive_ranks(
                vec![ReceiveRequest {
                    control_fd: control_root.as_raw_fd(),
                    channels: vec![state(&root, true)],
                    destination: directory.0.join(name),
                }],
                &directory.0.join("cache"),
                no_weights,
            );
            let sent = sender.join().unwrap();
            match receiver {
                Ok(mut outputs) => {
                    sent?;
                    Ok(outputs.remove(0))
                }
                Err(error) => Err(error),
            }
        })
    }
    /// Roundtrip of a rank whose declared payload objects are leased by the sender.
    fn roundtrip_with_cache(
        directory: &Directory,
        name: &str,
        source: &RankOutput,
        no_weights: bool,
    ) -> Result<ReceiveOutput> {
        let (control_root, control_peer) = sockets();
        let (root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    source.clone(),
                    json!({"test":true}),
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let receiver = receive_ranks(
                vec![ReceiveRequest {
                    control_fd: control_root.as_raw_fd(),
                    channels: vec![state(&root, true)],
                    destination: directory.0.join(name),
                }],
                &directory.0.join("cache"),
                no_weights,
            );
            let sent = sender.join().unwrap();
            match receiver {
                Ok(mut outputs) => {
                    sent?;
                    Ok(outputs.remove(0))
                }
                Err(error) => Err(error),
            }
        })
    }

    fn cache_backed_output(
        directory: &Directory,
        payload: &[u8],
    ) -> (RankOutput, helios_core::core::PayloadId) {
        let sender_cache = directory.0.join("sender-cache");
        let root = helios_core::core::cache::Root::open(&sender_cache).unwrap();
        let raw = directory.0.join("group.payload");
        fs::write(&raw, payload).unwrap();
        let slice = helios_core::core::container::Slice::new(
            std::sync::Arc::new(fs::File::open(&raw).unwrap()),
            0,
            payload.len() as u64,
        )
        .unwrap();
        let object = root.store_slice(None, &slice).unwrap();
        let id = object.id();
        let staged = directory.0.join("staged-resource");
        fs::write(&staged, b"rank-resource").unwrap();
        let mut output = output(staged.clone(), 13);
        output.weight_payload = Some(crate::export::WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![crate::export::WeightPayloadGroup {
                name: "layer-000".into(),
                storage: "stable".into(),
                nbytes: payload.len() as u64,
                spans: Vec::new(),
                payload_id: object.id().as_bytes().to_vec(),
            }],
            initial_state: crate::export::WeightPayloadGroup::default(),
            views: Vec::new(),
        });
        output.payload_leases = crate::export::PayloadLeases(vec![object]);
        (output, id)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn received_canonical_wrapper_is_adopted_and_others_are_copied() {
        use std::os::unix::fs::MetadataExt;
        let directory = Directory::new();
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 97) as u8).collect();
        let (_, id) = cache_backed_output(&directory, &payload);
        let sent = directory.0.join("sender-cache").join("blobs").join(id.file_name());
        let cache = directory.0.join("cache");
        helios_core::core::cache::Root::open(&cache).unwrap();

        // The receiver writes the sender's object file under the cache's blobs directory.
        let partial = cache.join("blobs").join(".rank-blob-canonical");
        fs::copy(&sent, &partial).unwrap();
        let object = install_payload_object(&cache, id, payload.len() as u64, &partial).unwrap();
        assert_eq!(
            fs::metadata(object.path()).unwrap().ino(),
            fs::metadata(&partial).unwrap().ino(),
            "a canonical received wrapper was copied instead of adopted"
        );
        drop(object);

        // Another framing of the same payload is stored through its slice in a new file.
        let other = directory.0.join("other-cache");
        helios_core::core::cache::Root::open(&other).unwrap();
        let framed = other.join("blobs").join(".rank-blob-framed");
        let slice = helios_core::core::container::Container::open(&sent)
            .unwrap()
            .tensor(helios_core::core::container::COMPANION_TENSOR)
            .unwrap();
        let mut writer = helios_core::core::container::Writer::new(
            fs::File::create(&framed).unwrap(),
            helios_core::core::container::PAYLOAD_FORMAT,
        )
        .unwrap();
        writer
            .add(
                helios_core::core::container::COMPANION_TENSOR,
                slice.len(),
                helios_core::core::container::Source::Slice(slice),
            )
            .unwrap();
        writer
            .add("extra.tail", 3, helios_core::core::container::Source::Bytes(vec![1, 2, 3]))
            .unwrap();
        writer.finish().unwrap();
        let copied = install_payload_object(&other, id, payload.len() as u64, &framed).unwrap();
        assert_ne!(
            fs::metadata(copied.path()).unwrap().ino(),
            fs::metadata(&framed).unwrap().ino()
        );
    }

    #[test]
    fn repeated_identical_payload_groups_share_one_object_and_round_trip() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 89) as u8).collect();
        let (mut source, id) = cache_backed_output(&directory, &payload);
        // A second semantic group legitimately holds the identical bytes.
        let weight = source.weight_payload.as_mut().unwrap();
        let mut twin = weight.groups[0].clone();
        twin.name = "layer-000-scales".into();
        weight.groups.push(twin);

        let received = roundtrip_with_cache(&directory, "rank.json", &source, false).unwrap();
        let groups = received.output.weight_payload.as_ref().unwrap();
        assert_eq!(groups.groups.len(), 2, "a declared group was dropped");
        assert_eq!(groups.groups[0].payload_id, id.as_bytes().to_vec());
        assert_eq!(groups.groups[1].payload_id, id.as_bytes().to_vec());
        let store = helios_core::core::cache::Root::open(&directory.0.join("cache")).unwrap();
        assert!(store.payload(id).unwrap().is_some());
        assert_eq!(received.leases.0.len(), 1, "one object per unique identity");
    }

    #[test]
    fn native_stage_retains_received_leases_and_rejects_a_closed_stage() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 59) as u8).collect();
        let (source, id) = cache_backed_output(&directory, &payload);
        let mut received = roundtrip_with_cache(&directory, "rank.json", &source, false).unwrap();
        let cache = directory.0.join("cache");
        let store = helios_core::core::cache::Root::open(&cache).unwrap();
        let stage = crate::export::binding::NativeExportStage::for_test(
            Some(cache.clone()),
            false,
        );

        // Binding order: move the leases into the stage, then let outputs drop.
        stage
            .retain_payload_leases(std::mem::take(&mut received.leases))
            .unwrap();
        drop(received);
        assert_eq!(stage.retained_for_test(), 1);
        assert_eq!(
            store.collect().unwrap().objects,
            0,
            "the stage did not keep the received object"
        );
        assert!(store.payload(id).unwrap().is_some());
        drop(stage);
        assert!(store.collect().unwrap().objects >= 1);

        // A closed stage refuses retention rather than silently dropping leases.
        let mut received = roundtrip_with_cache(&directory, "rank2.json", &source, false).unwrap();
        let closed = crate::export::binding::NativeExportStage::for_test(
            Some(cache),
            true,
        );
        assert!(closed
            .retain_payload_leases(std::mem::take(&mut received.leases))
            .is_err());
        assert_eq!(closed.retained_for_test(), 0);
    }

    #[test]
    fn retained_receive_leases_outlive_the_receive_outputs() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..1536u32).map(|i| (i % 71) as u8).collect();
        let (source, id) = cache_backed_output(&directory, &payload);
        let mut received = roundtrip_with_cache(&directory, "rank.json", &source, false).unwrap();
        let cache = directory.0.join("cache");
        let store = helios_core::core::cache::Root::open(&cache).unwrap();

        // The binding moves leases into the retained stage before the outputs drop.
        let retained = std::mem::take(&mut received.leases);
        assert_eq!(retained.0.len(), 1);
        drop(received);
        assert_eq!(
            store.collect().unwrap().objects,
            0,
            "retained stage lost the object"
        );
        assert!(store.payload(id).unwrap().is_some());

        drop(retained);
        assert!(store.collect().unwrap().objects >= 1);
        assert!(store.payload(id).unwrap().is_none());
    }

    #[test]
    fn warm_receive_retains_the_object_lease_until_the_output_is_dropped() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..2048u32).map(|i| (i % 113) as u8).collect();
        let (source, id) = cache_backed_output(&directory, &payload);

        let cold = roundtrip_with_cache(&directory, "rank-cold.json", &source, false).unwrap();
        let warm = roundtrip_with_cache(&directory, "rank-warm.json", &source, true).unwrap();
        let cache = directory.0.join("cache");
        let store = helios_core::core::cache::Root::open(&cache).unwrap();

        // Both receiver outputs hold live leases for the warm-hit object.
        assert_eq!(warm.leases.0.len(), 1);
        assert_eq!(cold.leases.0.len(), 1);
        assert_eq!(store.collect().unwrap().objects, 0, "leased object reclaimed");

        drop(cold);
        drop(warm);
        assert!(store.collect().unwrap().objects >= 1, "object not reclaimed");
        assert!(store.payload(id).unwrap().is_none());
    }

    #[test]
    fn cache_backed_payloads_transfer_and_reuse_the_receiver_cache() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let (source, id) = cache_backed_output(&directory, &payload);

        let received = roundtrip_with_cache(&directory, "rank.json", &source, false).unwrap();
        assert_eq!(received.inventory["test"], true);
        let receiver_cache = directory.0.join("cache");
        let store = helios_core::core::cache::Root::open(&receiver_cache).unwrap();
        assert!(store.payload(id).unwrap().is_some(), "payload object missing");

        // A second transfer with --no-weights reuses the object already present.
        let received = roundtrip_with_cache(&directory, "rank2.json", &source, true).unwrap();
        assert_eq!(received.inventory["test"], true);
        let store = helios_core::core::cache::Root::open(&receiver_cache).unwrap();
        assert!(store.payload(id).unwrap().is_some());
    }

    #[test]
    fn reused_weight_references_need_no_sender_object() {
        let directory = Directory::new();
        let bytes = vec![0x61; 512];
        let (mut source, id) = cache_backed_output(&directory, &bytes);
        let received = roundtrip_with_cache(&directory, "seed.json", &source, false).unwrap();
        source.weight_payload.as_mut().unwrap().cached =
            vec![helios_core::core::schema::wire::PayloadDescriptor {
                id: id.as_bytes().to_vec(),
                nbytes: bytes.len() as u64,
            }];
        source.payload_leases = Default::default();
        assert!(cache_payload_sources(&source).unwrap().is_empty());
        let watched = vec![received.leases.0[0].path().to_path_buf()];
        let transfer = || roundtrip_with_cache(&directory, "reused.json", &source, true);
        #[cfg(unix)]
        let (reused, bytes_read) = helios_core::core::container::count_payload_reads(&watched, transfer);
        #[cfg(not(unix))]
        let reused = transfer();
        let reused = reused.unwrap();
        #[cfg(unix)]
        assert_eq!(bytes_read, 0, "cached weight transfer references must not read payload bytes");
        assert_eq!(reused.leases.0.len(), 1);
        assert_eq!(reused.leases.0[0].id(), id);
        drop(received);
        drop(reused);
        let root = helios_core::core::cache::Root::open(directory.0.join("cache")).unwrap();
        let path = root.payload(id).unwrap().unwrap().path().to_path_buf();
        fs::remove_file(path).unwrap();
        assert!(roundtrip_with_cache(&directory, "missing.json", &source, true).is_err());
    }
    #[test]
    fn cache_only_weights_allow_fresh_initial_state_from_peers() {
        let directory = Directory::new();
        let bytes = vec![0x71; 128];
        let (mut source, id) = cache_backed_output(&directory, &bytes);
        let weights = source.weight_payload.as_mut().unwrap();
        weights.initial_state = weights.groups.pop().unwrap();
        weights.initial_state.name = "initial-state".into();
        weights.initial_state.storage = "state".into();
        let received = roundtrip_with_cache(&directory, "fresh-state.json", &source, true).unwrap();
        assert_eq!(received.leases.0[0].id(), id);
    }

    #[test]
    fn declared_payload_object_must_exist_in_the_sender_cache() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..512u32).map(|i| (i % 31) as u8).collect();
        let (mut source, id) = cache_backed_output(&directory, &payload);
        source.weight_payload.as_mut().unwrap().groups[0].payload_id = id.as_bytes().to_vec();
        // Dropping the lease models a capture that lost its object: the sender must
        // refuse before any control manifest is written.
        source.payload_leases = Default::default();
        let error = cache_payload_sources(&source)
            .expect_err("declared payload without a lease was accepted");
        assert!(
            error.to_string().contains("leased cache object"),
            "{error}"
        );
    }

    #[test]
    fn cache_backed_payloads_require_the_object_for_no_weights_and_reject_corruption() {
        let directory = Directory::new();
        let payload: Vec<u8> = (0..2048u32).map(|i| (i % 97) as u8).collect();
        let (source, id) = cache_backed_output(&directory, &payload);
        let object_path = source.payload_leases.0[0].path().to_owned();

        // --no-weights with an empty receiver cache must fail, not reconstruct.
        let error = match roundtrip_with_cache(&directory, "rank.json", &source, true)
        {
            Ok(_) => panic!("--no-weights accepted a missing payload object"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("--no-weights"),
            "{error}"
        );
        let receiver_cache = directory.0.join("cache");
        let store = helios_core::core::cache::Root::open(&receiver_cache).unwrap();
        assert!(store.payload(id).unwrap().is_none());

        // Corrupting the sender object is detected before it can be adopted.
        let _sender_cache = object_path.parent().unwrap().parent().unwrap().to_path_buf();
        let mut permissions = fs::metadata(&object_path).unwrap().permissions();
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o644);
        }
        fs::set_permissions(&object_path, permissions).unwrap();
        let mut bytes = fs::read(&object_path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&object_path, &bytes).unwrap();
        let fresh = Directory::new();
        assert!(roundtrip_with_cache(&fresh, "rank3.json", &source, false).is_err());
        let store = helios_core::core::cache::Root::open(&receiver_cache).unwrap();
        assert!(store.payload(id).unwrap().is_none());
    }

    #[test]
    fn loopback_streams_payload_and_preserves_sequence() {
        let directory = Directory::new();
        let bytes = vec![71; BLOCK_BYTES + 37];
        let path = directory.0.join("resource");
        fs::write(&path, &bytes).unwrap();
        let received = roundtrip(
            &directory,
            "rank.json",
            &output(path, bytes.len() as u64),
            false,
        )
        .unwrap();
        assert_eq!(
            fs::read(&received.output.payload[0].source.path).unwrap(),
            bytes
        );
        assert_eq!(received.stats["payload_bytes"], bytes.len());
        assert_eq!(received.channels[0].received, 20);
        assert_eq!(received.channels[0].sent, 18);
        assert!(directory.0.join("rank.json").exists());
    }
    #[test]
    fn rank_export_waits_for_control_preparation() {
        for delay_sender in [true, false] {
            let directory = Directory::new();
            let bytes = b"prepared rank payload";
            let path = directory.0.join("resource");
            fs::write(&path, bytes).unwrap();
            let (control_root, control_peer) = sockets();
            let (root, peer) = sockets();
            std::thread::scope(|scope| {
                let sender = scope.spawn(|| {
                    if delay_sender {
                        std::thread::sleep(std::time::Duration::from_secs(31));
                    }
                    send_rank(
                        output(path.clone(), bytes.len() as u64),
                        Value::Null,
                        control_peer.as_raw_fd(),
                        vec![state(&peer, false)],
                    )
                });
                if !delay_sender {
                    std::thread::sleep(std::time::Duration::from_secs(31));
                }
                let received = receive_ranks(
                    vec![ReceiveRequest {
                        control_fd: control_root.as_raw_fd(),
                        channels: vec![state(&root, true)],
                        destination: directory.0.join("rank.json"),
                    }],
                    &directory.0.join("cache"),
                    false,
                );
                let sent = sender.join().unwrap();
                let received = received.unwrap();
                sent.unwrap();
                assert_eq!(
                    fs::read(&received[0].output.payload[0].source.path).unwrap(),
                    bytes
                );
            });
        }
    }
    #[test]
    fn rank_export_waits_for_payload_completion() {
        let directory = Directory::new();
        let bytes = b"prepared rank payload";
        let path = directory.0.join("resource");
        fs::write(&path, bytes).unwrap();
        let (control_root, control_peer) = sockets();
        let (root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    output(path.clone(), bytes.len() as u64),
                    Value::Null,
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let mut control = Socket::borrow(control_root.as_raw_fd()).unwrap();
            let manifest: Manifest = receive_control(&mut control).unwrap();
            send_control(
                &mut control,
                &Request {
                    version: PROTOCOL,
                    files: manifest.files.keys().cloned().collect(),
                },
            )
            .unwrap();
            let mut channel = Channel::new(state(&root, true)).unwrap();
            let (_, received) = channel.receive("export-data").unwrap();
            assert_eq!(received, bytes);
            let (finished, empty) = channel.receive("export-finished").unwrap();
            assert!(empty.is_empty());
            std::thread::sleep(std::time::Duration::from_secs(31));
            channel
                .send(
                    json!({"kind":"export-finished", "stats":finished["stats"]}),
                    &[],
                )
                .unwrap();
            sender.join().unwrap().unwrap();
        });
    }
    #[test]
    fn multiple_ranks_and_streams_share_cache_and_drain() {
        let directory = Directory::new();
        let shared = vec![87; 16_385];
        let shared_path = directory
            .0
            .join(format!("{}.safetensors", sha256_hex(&shared)));
        fs::write(&shared_path, &shared).unwrap();
        let mut requests = Vec::new();
        let mut peers = Vec::new();
        let mut retained = Vec::new();
        for rank in 0..2 {
            let (control, peer_control) = sockets();
            let mut roots = Vec::new();
            let mut rank_peers = Vec::new();
            for _ in 0..2 {
                let (root, peer) = sockets();
                roots.push(state(&root, true));
                retained.push(root);
                rank_peers.push(peer);
            }
            requests.push(ReceiveRequest {
                control_fd: control.as_raw_fd(),
                channels: roots,
                destination: directory.0.join(format!("rank-{rank}.json")),
            });
            retained.push(control);
            let mut prepared = output(shared_path.clone(), shared.len() as u64);
            let mut cursor = shared.len() as u64;
            for item in 0..2 {
                let data = vec![rank as u8 + item as u8; 8193];
                let path = directory.0.join(format!("rank-{rank}-resource-{item}"));
                fs::write(&path, &data).unwrap();
                prepared.payload.push(super::super::FilePart {
                    offset: cursor,
                    source: super::super::FileSource {
                        path,
                        offset: 0,
                        nbytes: data.len() as u64,
                    },
                });
                cursor += data.len() as u64;
            }
            peers.push((peer_control, rank_peers, prepared, rank));
        }
        std::thread::scope(|scope| {
            let senders: Vec<_> = peers
                .into_iter()
                .map(|(control, channels, prepared, rank)| {
                    scope.spawn(move || {
                        let states = channels.iter().map(|c| state(c, false)).collect();
                        send_rank(prepared, json!({"rank":rank}), control.as_raw_fd(), states)
                    })
                })
                .collect();
            let received = receive_ranks(requests, &directory.0.join("cache"), false).unwrap();
            for sender in senders {
                assert_eq!(sender.join().unwrap().unwrap().len(), 2);
            }
            assert_eq!(received.len(), 2);
            for (rank, result) in received.iter().enumerate() {
                assert_eq!(result.inventory["rank"], rank);
                assert_eq!(result.channels.len(), 2);
                assert_eq!(
                    fs::read(&result.output.payload[0].source.path).unwrap(),
                    shared
                );
            }
            assert_eq!(
                received[0].output.payload[0].source.path,
                received[1].output.payload[0].source.path
            );
        });
    }
    #[test]
    fn verified_cache_hit_transfers_no_weights() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let name = format!("{}.safetensors", sha256_hex(&bytes));
        let path = directory.0.join(name);
        fs::write(&path, &bytes).unwrap();
        let rank = output(path, bytes.len() as u64);
        roundtrip(&directory, "first.json", &rank, false).unwrap();
        let result = roundtrip(&directory, "second.json", &rank, true).unwrap();
        assert_eq!(result.stats["payload_bytes"], 0);
        assert_eq!(result.channels[0].received, 18);
    }
    #[test]
    fn missing_weights_fail_without_receiving_payload() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let path = directory
            .0
            .join(format!("{}.safetensors", sha256_hex(&bytes)));
        fs::write(&path, &bytes).unwrap();
        let error = roundtrip(
            &directory,
            "rank.json",
            &output(path, bytes.len() as u64),
            true,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("--no-weights"));
        assert!(!directory.0.join("rank.json").exists());
    }
    #[test]
    fn corrupt_cached_blob_is_rejected() {
        let directory = Directory::new();
        let bytes = vec![19; 8193];
        let name = format!("{}.safetensors", sha256_hex(&bytes));
        let path = directory.0.join(&name);
        fs::write(&path, &bytes).unwrap();
        fs::create_dir_all(directory.0.join("cache/blobs")).unwrap();
        fs::write(
            directory.0.join("cache/blobs").join(name),
            vec![20; bytes.len()],
        )
        .unwrap();
        assert!(
            roundtrip(
                &directory,
                "rank.json",
                &output(path, bytes.len() as u64),
                false
            )
            .is_err()
        );
        assert!(!directory.0.join("rank.json").exists());
    }
    #[test]
    fn duplicate_requested_sources_are_rejected() {
        let directory = Directory::new();
        let path = directory.0.join("source");
        fs::write(&path, b"payload").unwrap();
        let (control_root, control_peer) = sockets();
        let (_root, peer) = sockets();
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| {
                send_rank(
                    output(path.clone(), 7),
                    Value::Null,
                    control_peer.as_raw_fd(),
                    vec![state(&peer, false)],
                )
            });
            let mut control = Socket::borrow(control_root.as_raw_fd()).unwrap();
            let manifest: Manifest = receive_control(&mut control).unwrap();
            let name = manifest.files.keys().next().unwrap().clone();
            send_control(
                &mut control,
                &Request {
                    version: PROTOCOL,
                    files: vec![name.clone(), name],
                },
            )
            .unwrap();
            assert!(
                sender
                    .join()
                    .unwrap()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("invalid requested")
            );
        });
    }
    #[test]
    fn incompatible_export_version_is_rejected() {
        let directory = Directory::new();
        let (control_root, control_peer) = sockets();
        let (root, _peer) = sockets();
        let mut control = Socket::borrow(control_peer.as_raw_fd()).unwrap();
        send_control(
            &mut control,
            &Manifest {
                version: PROTOCOL + 1,
                output: output(directory.0.join("unused"), 7),
                files: BTreeMap::new(),
                inventory: Value::Null,
            },
        )
        .unwrap();
        let error = receive_ranks(
            vec![ReceiveRequest {
                control_fd: control_root.as_raw_fd(),
                channels: vec![state(&root, true)],
                destination: directory.0.join("rank.json"),
            }],
            &directory.0.join("cache"),
            false,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("protocol version"));
    }
    #[test]
    fn incorrect_authentication_is_rejected() {
        let (root, peer) = sockets();
        let mut sender = Channel::new(state(&peer, false)).unwrap();
        let mut root_state = state(&root, true);
        root_state.key[0] ^= 1;
        let mut receiver = Channel::new(root_state).unwrap();
        sender.send(json!({"kind":"export-data"}), b"test").unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("authentication")
        );
    }
    #[test]
    fn replayed_sequence_is_rejected() {
        let (root, peer) = sockets();
        let mut sender = Channel::new(state(&peer, false)).unwrap();
        let mut receiver = Channel::new(state(&root, true)).unwrap();
        sender
            .send(json!({"kind":"export-data"}), b"first")
            .unwrap();
        receiver.receive("export-data").unwrap();
        sender.state.sent -= 1;
        sender
            .send(json!({"kind":"export-data"}), b"replay")
            .unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("sequence")
        );
    }
    #[test]
    fn truncated_frame_and_disconnect_are_rejected() {
        for bytes in [&b""[..], &b"\x08\x00\x00\x00{}"[..]] {
            let (root, mut peer) = sockets();
            let mut receiver = Channel::new(state(&root, true)).unwrap();
            peer.write_all(bytes).unwrap();
            drop(peer);
            assert!(receiver.receive("export-data").is_err());
        }
    }
    #[test]
    fn damaged_payloads_never_publish_files() {
        for truncated in [false, true] {
            let directory = Directory::new();
            let (control_root, control_peer) = sockets();
            let (root, peer) = sockets();
            let source = directory.0.join("source");
            let bytes = vec![41; 8193];
            let digest = sha256_hex(&bytes);
            let mut files = BTreeMap::new();
            files.insert(
                source.clone(),
                Source {
                    size: bytes.len() as u64,
                    sha256: digest.clone(),
                    blob: false,
                    payload: None,
                },
            );
            std::thread::scope(|scope| {
                let sender = scope.spawn(move || {
                    let mut control = Socket::borrow(control_peer.as_raw_fd()).unwrap();
                    send_control(&mut control, &Manifest { version: PROTOCOL, output: output(source.clone(), bytes.len() as u64), files, inventory: Value::Null }).unwrap();
                    let _: Request = receive_control(&mut control).unwrap();
                    let mut channel = Channel::new(state(&peer, false)).unwrap();
                    if truncated {
                        let header = serde_json::to_vec(&json!({"kind":"export-data","name":source,"offset":0,"sequence":17,"bytes":bytes.len()})).unwrap();
                        channel.socket.write_all(&(header.len() as u32).to_le_bytes()).unwrap();
                        channel.socket.write_all(&header).unwrap();
                        channel.socket.write_all(b"truncated").unwrap();
                    } else {
                        channel.send(json!({"kind":"export-data","name":source,"offset":0}), &vec![42; bytes.len()]).unwrap();
                    }
                });
                let result = receive_ranks(
                    vec![ReceiveRequest {
                        control_fd: control_root.as_raw_fd(),
                        channels: vec![state(&root, true)],
                        destination: directory.0.join("rank.json"),
                    }],
                    &directory.0.join("cache"),
                    false,
                );
                sender.join().unwrap();
                assert!(result.is_err());
                if !truncated {
                    assert!(
                        result
                            .err()
                            .unwrap()
                            .to_string()
                            .contains("digest mismatch")
                    );
                }
            });
            assert!(!directory.0.join("rank.json").exists());
            assert!(!directory.0.join(format!("{digest}.source")).exists());
            assert!(!fs::read_dir(&directory.0).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".rank-blob-")
            }));
        }
    }
    #[test]
    fn frame_size_is_bounded_before_allocation() {
        let (root, mut peer) = sockets();
        let mut receiver = Channel::new(state(&root, true)).unwrap();
        peer.write_all(&65537u32.to_le_bytes()).unwrap();
        assert!(
            receiver
                .receive("export-data")
                .err()
                .unwrap()
                .to_string()
                .contains("capacity")
        );
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn received_verified_companion_adopts_without_payload_readback_and_rejects_mismatch() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let directory = Directory::new();
        let payload: Vec<u8> = (0..(BLOCK_BYTES + 137) as u32)
            .map(|i| (i % 251) as u8)
            .collect();
        let (source, id) = cache_backed_output(&directory, &payload);
        let sent = directory
            .0
            .join("sender-cache")
            .join("blobs")
            .join(id.file_name());
        let cache = directory.0.join("verified-cache");
        fs::create_dir_all(cache.join("blobs")).unwrap();

        // Direct verified adoption links the synced partial inode into the cache
        // without reading payload bytes from disk.
        let partial = cache.join("blobs").join(".rank-blob-verified");
        fs::copy(&sent, &partial).unwrap();
        fs::set_permissions(&partial, fs::Permissions::from_mode(0o444)).unwrap();
        let partial_meta = fs::metadata(&partial).unwrap();
        let dev_ino = (partial_meta.dev(), partial_meta.ino());
        let (adopted, bytes_read) =
            helios_core::core::container::count_payload_reads(&[partial.clone()], || {
                install_verified_payload_object(&cache, id, payload.len() as u64, &partial, dev_ino)
            });
        let object = adopted.unwrap();
        assert_eq!(bytes_read, 0, "verified adoption re-read payload bytes");
        assert_eq!(
            (
                fs::metadata(object.path()).unwrap().dev(),
                fs::metadata(object.path()).unwrap().ino()
            ),
            dev_ino,
            "verified canonical wrapper was copied instead of adopted"
        );
        fs::remove_file(&partial).unwrap();
        assert_eq!(object.slice().digest().unwrap(), id);
        drop(object);

        // Multi-frame transfer across BLOCK_BYTES adopts into a fresh receiver cache.
        let multi = Directory::new();
        let received = roundtrip_with_cache(&multi, "rank.json", &source, false).unwrap();
        assert_eq!(received.leases.0.len(), 1);
        assert_eq!(received.leases.0[0].id(), id);
        assert_eq!(received.leases.0[0].slice().digest().unwrap(), id);
        drop(received);

        // A companion whose transport sha256 matches the manifest but whose payload
        // bytes disagree with the declared PayloadId must be rejected before cache entry.
        let bad_dir = Directory::new();
        let declared_bytes = vec![0x3a; 8192];
        let wrong_bytes = vec![0x3b; 8192];
        let declared_id = helios_core::core::PayloadId::of(&declared_bytes);
        let mut wrong_wrapper =
            helios_core::core::cache::companion_prefix(wrong_bytes.len() as u64).unwrap();
        wrong_wrapper.extend_from_slice(&wrong_bytes);
        let transport_sha = sha256_hex(&wrong_wrapper);
        let staged_res = bad_dir.0.join("staged-res");
        fs::write(&staged_res, b"res").unwrap();
        let res_sha = sha256_hex(b"res");
        let fake_companion = bad_dir.0.join("fake-companion.safetensors");
        let mut rank_out = output(staged_res.clone(), 3);
        rank_out.weight_payload = Some(crate::export::WeightPayloadSources {
            cached: Vec::new(),
            engram: None,
            groups: vec![crate::export::WeightPayloadGroup {
                name: "layer-000".into(),
                storage: "stable".into(),
                nbytes: declared_bytes.len() as u64,
                spans: Vec::new(),
                payload_id: declared_id.as_bytes().to_vec(),
            }],
            initial_state: crate::export::WeightPayloadGroup::default(),
            views: Vec::new(),
        });
        let mut files = BTreeMap::new();
        files.insert(
            staged_res.clone(),
            Source {
                size: 3,
                sha256: res_sha,
                blob: false,
                payload: None,
            },
        );
        files.insert(
            fake_companion.clone(),
            Source {
                size: wrong_wrapper.len() as u64,
                sha256: transport_sha,
                blob: false,
                payload: Some(declared_id.to_hex()),
            },
        );
        let (control_root, control_peer) = sockets();
        let (root, peer) = sockets();
        let receiver_cache = bad_dir.0.join("cache");
        let error = std::thread::scope(|scope| {
            let sender = scope.spawn(move || {
                let mut control = Socket::borrow(control_peer.as_raw_fd()).unwrap();
                send_control(
                    &mut control,
                    &Manifest {
                        version: PROTOCOL,
                        output: rank_out,
                        files,
                        inventory: Value::Null,
                    },
                )
                .unwrap();
                let req: Request = receive_control(&mut control).unwrap();
                let mut channel = Channel::new(state(&peer, false)).unwrap();
                for requested_path in &req.files {
                    if requested_path == &staged_res {
                        channel
                            .send(
                                json!({"kind":"export-data","name":staged_res,"offset":0}),
                                b"res",
                            )
                            .unwrap();
                    } else {
                        channel
                            .send(
                                json!({"kind":"export-data","name":fake_companion,"offset":0}),
                                &wrong_wrapper,
                            )
                            .unwrap();
                    }
                }
            });
            let result = receive_ranks(
                vec![ReceiveRequest {
                    control_fd: control_root.as_raw_fd(),
                    channels: vec![state(&root, true)],
                    destination: bad_dir.0.join("rank.json"),
                }],
                &receiver_cache,
                false,
            );
            sender.join().unwrap();
            result.err().expect("mismatched payload digest must fail")
        });
        assert!(
            error
                .to_string()
                .contains("received payload digest does not match its declared identity"),
            "{error}"
        );
        let store = helios_core::core::cache::Root::open(&receiver_cache).unwrap();
        assert!(store.payload(declared_id).unwrap().is_none());
        assert!(
            store
                .payload(helios_core::core::PayloadId::of(&wrong_bytes))
                .unwrap()
                .is_none()
        );
        assert!(
            !fs::read_dir(receiver_cache.join("blobs"))
                .unwrap()
                .any(|entry| entry.unwrap().path().is_file())
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn failed_receive_rolls_back_retained_descriptors_and_aborts_zero_window_peers_promptly() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;
        use std::time::Duration;

        // 1. Retained-descriptor rollback: while another FileTransaction keeps reserved > 0,
        // a failed receive_rank that leased cached payloads must restore its descriptor count
        // when its partial PayloadObject list drops.
        {
            let directory = Directory::new();
            let payload_bytes = vec![0x51; 4096];
            let (mut source, id) = cache_backed_output(&directory, &payload_bytes);
            // Seed receiver cache so `sources.cached` opens a 2-FD PayloadObject lease,
            // then force receive_rank to fail on an absent `--no-weights` second group.
            roundtrip_with_cache(&directory, "seed.json", &source, false).unwrap();
            let missing_id = helios_core::core::PayloadId::of(b"missing-second-group");
            let missing_blob = directory.0.join("missing-companion.safetensors");
            let mut missing_wrapper = helios_core::core::cache::companion_prefix(20).unwrap();
            missing_wrapper.extend_from_slice(b"missing-second-group");
            fs::write(&missing_blob, &missing_wrapper).unwrap();
            let missing_lease = {
                let extra_cache = directory.0.join("extra-sender-cache");
                let extra_root = helios_core::core::cache::Root::open(&extra_cache).unwrap();
                extra_root
                    .insert_payload(missing_id, 20, &b"missing-second-group"[..])
                    .unwrap()
            };
            {
                let weight = source.weight_payload.as_mut().unwrap();
                weight.cached = vec![helios_core::core::schema::wire::PayloadDescriptor {
                    id: id.as_bytes().to_vec(),
                    nbytes: payload_bytes.len() as u64,
                }];
                weight.groups.push(crate::export::WeightPayloadGroup {
                    name: "layer-001".into(),
                    storage: "stable".into(),
                    nbytes: 20,
                    spans: Vec::new(),
                    payload_id: missing_id.as_bytes().to_vec(),
                });
            }
            source.payload_leases = crate::export::PayloadLeases(vec![missing_lease]);

            let _pin = FileTransaction::acquire(FileDirection::Send, 1).unwrap();
            let before_capacity = FILE_TRANSACTIONS.0.lock().unwrap().capacity;
            // Thirty-two failing attempts open and drop 64 cached-payload descriptors;
            // without rollback on failure, capacity drops by at least 64 even when
            // sibling tests run concurrently in the same process.
            for attempt in 0..32 {
                let dest = format!("fail-no-weights-{attempt}.json");
                assert!(roundtrip_with_cache(&directory, &dest, &source, true).is_err());
            }
            let after_capacity = FILE_TRANSACTIONS.0.lock().unwrap().capacity;
            assert!(
                before_capacity.saturating_sub(after_capacity) < 32,
                "failed receive_rank leaked retained-descriptor subtractions: before={before_capacity}, after={after_capacity}"
            );
        }

        // 2. Early scan failure in send_rank must abort borrowed sockets so a concurrent
        // receive_ranks does not block in receive_control until the 2-hour idle timeout.
        {
            let directory = Directory::new();
            let absent_source = directory.0.join("absent-source.bin");
            let (control_root, control_peer) = sockets();
            let (root, peer) = sockets();
            let control_root = Arc::new(control_root);
            let control_peer = Arc::new(control_peer);
            let root = Arc::new(root);
            let peer = Arc::new(peer);
            let (done_tx, done_rx) = mpsc::channel();
            let c_root = Arc::clone(&control_root);
            let c_peer = Arc::clone(&control_peer);
            let d_root = Arc::clone(&root);
            let d_peer = Arc::clone(&peer);
            let dir_path = directory.0.clone();
            std::thread::spawn(move || {
                let sender = std::thread::spawn(move || {
                    send_rank(
                        output(absent_source, 64),
                        Value::Null,
                        c_peer.as_raw_fd(),
                        vec![state(&d_peer, false)],
                    )
                });
                let receiver = receive_ranks(
                    vec![ReceiveRequest {
                        control_fd: c_root.as_raw_fd(),
                        channels: vec![state(&d_root, true)],
                        destination: dir_path.join("rank.json"),
                    }],
                    &dir_path.join("cache"),
                    false,
                );
                let sent = sender.join().unwrap();
                let _ = done_tx.send((sent.is_err(), receiver.is_err()));
            });
            let (sent_err, recv_err) = done_rx
                .recv_timeout(Duration::from_secs(3))
                .expect("receive_ranks hung after peer send_rank failed during source scan");
            assert!(sent_err && recv_err);
        }

        // 3. Multi-peer TCP zero-window abort: Peer 1 fails mid-receive on Rank 0 while
        // Peer 2 is blocked inside Socket::write sending multi-megabyte frames across a
        // zero-window TCP buffer while socket owners remain open.
        {
            let directory = Directory::new();
            let small_socket_pair = || -> (Arc<TcpStream>, Arc<TcpStream>) {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                unsafe {
                    let rcvbuf: libc::c_int = 32_768;
                    libc::setsockopt(
                        listener.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_RCVBUF,
                        (&rcvbuf as *const libc::c_int).cast(),
                        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    );
                }
                let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
                unsafe {
                    let sndbuf: libc::c_int = 32_768;
                    libc::setsockopt(
                        client.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_SNDBUF,
                        (&sndbuf as *const libc::c_int).cast(),
                        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    );
                }
                let (accepted, _) = listener.accept().unwrap();
                (Arc::new(accepted), Arc::new(client))
            };

            let peer1_data = vec![0x33u8; BLOCK_BYTES * 2];
            let peer2_data = vec![0x44u8; BLOCK_BYTES * 4];
            let peer1_src = directory.0.join("peer1.bin");
            let peer2_src = directory.0.join("peer2.bin");
            fs::write(&peer1_src, &peer1_data).unwrap();
            fs::write(&peer2_src, &peer2_data).unwrap();

            // Peer 1's destination parent is read-only so creating `.rank-blob-*` fails on
            // Rank 0 right as Peer 1's first frame arrives, while Peer 2 is streaming 16 MiB.
            let peer1_ro_dir = directory.0.join("ro-stage");
            fs::create_dir_all(&peer1_ro_dir).unwrap();

            let (c1_root, c1_peer) = small_socket_pair();
            let (d1_root, d1_peer) = small_socket_pair();
            let (c2_root, c2_peer) = small_socket_pair();
            let (d2_root, d2_peer) = small_socket_pair();

            let retained = vec![
                Arc::clone(&c1_root),
                Arc::clone(&d1_root),
                Arc::clone(&c2_root),
                Arc::clone(&d2_root),
                Arc::clone(&c1_peer),
                Arc::clone(&d1_peer),
                Arc::clone(&c2_peer),
                Arc::clone(&d2_peer),
            ];
            let cache_dir = directory.0.join("cache");
            let peer2_dst = directory.0.join("peer2-rank.json");
            let peer1_dst = peer1_ro_dir.join("peer1-rank.json");
            fs::set_permissions(&peer1_ro_dir, fs::Permissions::from_mode(0o555)).unwrap();

            let (done_tx, done_rx) = mpsc::channel();
            let ro_cleanup = peer1_ro_dir.clone();
            std::thread::spawn(move || {
                let s1 = std::thread::spawn(move || {
                    send_rank(
                        output(peer1_src, peer1_data.len() as u64),
                        json!({"rank": 1}),
                        c1_peer.as_raw_fd(),
                        vec![state(&d1_peer, false)],
                    )
                });
                let s2 = std::thread::spawn(move || {
                    send_rank(
                        output(peer2_src, peer2_data.len() as u64),
                        json!({"rank": 2}),
                        c2_peer.as_raw_fd(),
                        vec![state(&d2_peer, false)],
                    )
                });
                let recv_res = receive_ranks(
                    vec![
                        ReceiveRequest {
                            control_fd: c1_root.as_raw_fd(),
                            channels: vec![state(&d1_root, true)],
                            destination: peer1_dst,
                        },
                        ReceiveRequest {
                            control_fd: c2_root.as_raw_fd(),
                            channels: vec![state(&d2_root, true)],
                            destination: peer2_dst,
                        },
                    ],
                    &cache_dir,
                    false,
                );
                let r1 = s1.join().unwrap();
                let r2 = s2.join().unwrap();
                drop(retained);
                let _ = done_tx.send((recv_res.is_err(), r1.is_err(), r2_err_flag(&r2)));
            });

            let outcome = done_rx.recv_timeout(Duration::from_secs(5));
            let _ = fs::set_permissions(&ro_cleanup, fs::Permissions::from_mode(0o755));
            let (recv_err, r1_err, r2_err) =
                outcome.expect("peer send_rank or receive_ranks hung after mid-transfer abort");
            assert!(recv_err, "receive_ranks must report failure");
            assert!(r1_err, "peer 1 send_rank must terminate with error");
            assert!(
                r2_err,
                "peer 2 blocked in zero-window send must terminate with error"
            );
        }
    }

    fn r2_err_flag<T, E>(res: &std::result::Result<T, E>) -> bool {
        res.is_err()
    }
}
