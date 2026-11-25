/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::{
    Key, Result,
    arena::{Arena, Extent, ExtentSlice},
    disk::{self, DiskExtent},
    invalid,
    local::Local,
    proto,
    record::{self, Descriptor, NamespaceId},
};
use flare_selene::shared::index::Index as Tree;
use flare_selene::shared::region::Region;
use flare_selene::{
    CaptureRole, JobId,
    shared::{
        disk::{DiskThread, Job as DiskJob, Span},
        record::Header,
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

// Stay above NO_PROGRESS (3 s) so a stalled blocking put is cleaned up before its waiter expires.
const PUT_INFLIGHT_WAIT: Duration = Duration::from_secs(5);

pub const OPEN_GETS: usize = 4;
pub const MAX_CHAIN: usize = flare_selene::shared::MAX_CHAIN_NODES as usize;
/// Records in a chain at which a put starts a new root instead of linking. Lookups, claims
/// and Gets accept a chain of at most `MAX_CHAIN` records and a linked put adds one record,
/// so the requirement is only that no stored chain exceeds `MAX_CHAIN`. The other half is
/// headroom that no check depends on. The depth also bounds the metadata blocks a restore
/// reads before its first window to this many records.
pub const REROOT_DEPTH: usize = MAX_CHAIN / 2;
pub use flare_selene::shared::PROGRESS_TIMEOUT as NO_PROGRESS;

/// Whether an operation that has been idle for `idle` has stalled. A put whose client has
/// delivered everything (`delivered`) waits on this manager's own disk, whose jobs queue behind
/// each other, so it has stalled only if the disk made no progress either (`disk_idle`, `None`
/// when no job is outstanding). A get with reads in flight waits the same way on its own
/// namespace's reads thread, whose idle time is `disk_idle`. Otherwise the idle time alone
/// decides.
fn stalled(idle: Duration, delivered: bool, disk_idle: Option<Duration>) -> bool {
    idle >= NO_PROGRESS && !(delivered && disk_idle.is_some_and(|disk| disk < NO_PROGRESS))
}
pub use flare_selene::shared::MAX_WINDOW_BYTES as DEFAULT_MAX_WINDOW;

/// The staging of a store built in code (an embedding or a test). The command line always sets it
/// from `--staging`, whose default is larger.
pub const EMBEDDED_STAGING: usize = 16 * 1024 * 1024;

pub use crate::config::Config;

#[derive(Clone)]
pub struct Location {
    pub header: Header,
    pub memory: Option<Arc<Extent>>,
    pub disk: Option<Arc<DiskExtent>>,
    pub durable: bool,
    pub writing: bool,
    /// The disk job currently transforming these bytes, so a stale completion is ignored.
    pub job: Option<u64>,
    pub retry: Instant,
    pub admitted: Instant,
    pub bytes: u64,
}

/// What a finished Put should do to the tree entry for its key.
enum PublishTarget {
    /// No entry exists; index a new record.
    Insert,
    /// A page-only record is present; take its place.
    Replace,
}

impl Location {
    /// Whether a disk job is writing or truncating this record's file.
    ///
    /// Removing the record then would strand the job's completion, so removal
    /// cascades leave such a record in the tree.
    pub fn disk_busy(&self) -> bool {
        self.writing || self.job.is_some()
    }

    pub fn reading(&self) -> bool {
        self.memory
            .as_ref()
            .is_some_and(|extent| Arc::strong_count(extent) > 1)
    }
}

pub struct Namespace {
    pub directory: Option<PathBuf>,
    pub descriptor: Descriptor,
    pub tree: Tree<Location>,
    pub disk: Option<DiskThread>,
    reads: Option<DiskThread>,
    /// Reads submitted to `reads` and not yet completed.
    reads_pending: usize,
    /// The later of the last read completion and the moment `reads` last went from idle
    /// to busy. A reads thread that is busy and has not moved for `NO_PROGRESS` is hung.
    read_progress: Instant,
    pub inflight: HashSet<Key>,
    /// Records that are not durable yet, so durability and scheduling visit only these.
    /// Invariant: when the namespace has a disk, every record that is not durable is in this
    /// set, so code that inserts into `tree` must call `refresh_durability` for the new key.
    /// A removed record may linger here until the next refresh drops it.
    undurable: HashSet<Key>,
}

impl Namespace {
    fn path(&self, key: Key) -> Option<PathBuf> {
        self.directory
            .as_ref()
            .map(|directory| directory.join(record::file_name(&key)))
    }
    fn temporary(&self, id: u64) -> Option<Arc<DiskExtent>> {
        let directory = self.directory.as_ref()?;
        let extent = Arc::new(DiskExtent::new(directory.join(format!(".write-{id}.tmp"))));
        extent.retire();
        Some(extent)
    }
    fn parent_durable(&self, parent: Option<Key>) -> bool {
        parent.is_none_or(|key| self.tree[&key].location.durable)
    }
}

#[derive(Clone, Copy)]
struct PlannedGrant {
    record: u32,
    record_offset: u64,
    length: u64,
    pages: u32,
    row: bool,
    page_first: u64,
}

impl PlannedGrant {
    fn read_prefix(self) -> usize {
        (self.record_offset % 4096) as usize
    }
    fn read_length(self) -> usize {
        usize::try_from(self.length)
            .unwrap_or(usize::MAX)
            .saturating_add(self.read_prefix())
    }

    fn window(
        self,
        transfer: u64,
        index: u32,
        region: Region,
        offset: u64,
        hold: Arc<Extent>,
        header: &Header,
    ) -> WindowGrant {
        let packed = header.packed.as_ref().map(|p| {
            let first = self.page_first.saturating_sub(p.offer().first_page) as usize;
            flare_selene::shared::wire::PackedWindow {
                units: p
                    .units(first, self.pages as usize, self.row)
                    .expect("planned range")
                    .to_vec(),
            }
        });
        WindowGrant {
            packed,
            transfer,
            index,
            record: self.record,
            record_offset: self.record_offset,
            length: self.length,
            region: region_wire(region),
            offset,
            hold,
            page_first: self.page_first,
            page_count: self.pages,
            row: self.row,
        }
    }
}

struct FilledWindows {
    granted: HashMap<u32, PlannedGrant>,
    next: usize,
    pages: Vec<Option<u64>>,
    row: Option<u64>,
    pages_filled: u32,
    row_filled: bool,
}

/// Physical resources a Put occupies, for the shape (arena or streamed) it
/// will actually get. The arena shape is one buddy-rounded record extent;
/// the streaming shape is the disk record plus one fixed
/// staging hold of two of its largest windows. `unit` keeps the largest
/// indivisible window separate so an oversized window is never mistaken for a
/// whole-record requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PutRequirement {
    streaming: bool,
    /// Charge against the overall budget (`charge_total`).
    charge: u64,
    /// Portion retained until the Put finishes (arena extent or disk record).
    reservation: u64,
    /// Region bytes the shape needs in total.
    allocation: u64,
    /// Largest indivisible region allocation (one window, or the extent).
    unit: u64,
}

impl PutRequirement {
    fn arena(extent: u64) -> Self {
        Self {
            streaming: false,
            charge: extent,
            reservation: extent,
            allocation: extent,
            unit: extent,
        }
    }

    /// A streamed Put reserves one fixed hold of two of its largest windows.
    /// Live windows are slices of that hold, so the staging headroom the
    /// transfer will need cannot be taken by a concurrent operation.
    fn streaming(disk: u64, unit: u64) -> Self {
        let allocation = unit.saturating_mul(2);
        Self {
            streaming: true,
            charge: disk.saturating_add(allocation),
            reservation: disk,
            allocation,
            unit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HoldSlot {
    index: u32,
    /// Chunk write still reading this slot; the slot is reusable once it
    /// completes.
    job: Option<u64>,
}

fn set_slot_job(put: &mut Put, index: u32, job: u64) {
    for slot in put.hold_slots.iter_mut().flatten() {
        if slot.index == index {
            slot.job = Some(job);
        }
    }
}

fn release_slot_job(put: &mut Put, job: u64) {
    for slot in &mut put.hold_slots {
        if slot.is_some_and(|slot| slot.job == Some(job)) {
            *slot = None;
        }
    }
}

enum RootChoice {
    Store(Extent),
    Wait,
}

pub struct Put {
    row: bool,
    stored_role: CaptureRole,
    namespace: NamespaceId,
    header: Header,
    first_page: u32,
    grants: Vec<PlannedGrant>,
    requested: usize,
    windows: FilledWindows,
    memory: Option<Arc<Extent>>,
    temporary: Option<Arc<DiskExtent>>,
    streaming: bool,
    commit_bytes: Vec<u8>,
    metadata: Option<Arc<Extent>>,
    committed: bool,
    finalizing: bool,
    cancelled: bool,
    terminal: bool,
    finished: bool,
    pending: usize,
    blocked: VecDeque<(u64, u64, ExtentSlice)>,
    pins: VecDeque<Key>,
    reservation: u64,
    /// Fixed staging reservation for a streamed Put. Live windows are
    /// slices of it, so concurrent operations cannot consume the headroom
    /// the transfer still needs.
    staging_hold: Option<Arc<Extent>>,
    /// Slot size inside `staging_hold`.
    hold_unit: usize,
    /// Window occupying each hold slot, and the write draining it.
    hold_slots: [Option<HoldSlot>; 2],
    events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
    outbox: VecDeque<Result<PutAnswer>>,
}

impl Put {
    fn owns_payload(&self) -> bool {
        self.windows.pages_filled as usize == self.header.tail_pages as usize
            && (self.windows.row_filled || self.header.row_record_bytes() == 0)
    }
}

struct GetRecord {
    header: Header,
    memory: Option<Arc<Extent>>,
    file: Option<Arc<File>>,
    meta: Option<Vec<u8>>,
    meta_sent: bool,
}

pub struct Get {
    namespace: NamespaceId,
    pins: VecDeque<Key>,
    records: Vec<GetRecord>,
    grants: Vec<PlannedGrant>,
    requested: usize,
    next_grant: usize,
    /// The first record whose metadata read has not been started. A record's metadata read
    /// waits for staging, so this advances in record order as staging allows.
    next_meta_read: usize,
    next_meta: usize,
    window_reads: usize,
    pending_reads: usize,
    cancelled: bool,
    terminal: bool,
    finished: bool,
    events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
    outbox: VecDeque<Result<GetAnswer>>,
}

pub enum Operation {
    Put(Box<Put>),
    Get(Box<Get>),
}

pub struct Open {
    pub operation: Operation,
    pub progress: Instant,
}

impl Open {
    /// Already failed or cancelled: failing it again would only count and refuse it again.
    fn is_over(&self) -> bool {
        match &self.operation {
            Operation::Put(put) => put.terminal || put.cancelled,
            Operation::Get(get) => get.terminal || get.cancelled,
        }
    }

    /// An operation that waits on this manager's own disk: a put that owes nothing more to its
    /// client because it is committed, or has asked for windows, holds none, and both staging
    /// slots are still draining or chunks are blocked on a full disk queue; or a get with reads
    /// in flight, which waits on its namespace's reads thread.
    fn waits_on_disk(&self) -> bool {
        match &self.operation {
            Operation::Put(put) => {
                put.committed
                    || (put.requested > 0
                        && put.windows.granted.is_empty()
                        && (put.hold_slots.iter().all(Option::is_some) || !put.blocked.is_empty()))
            }
            Operation::Get(get) => get.pending_reads > 0,
        }
    }

    /// A put whose next window the pool has not yet asked for, with no granted window for the
    /// client to fill, is queued behind other transfers' window credits. That wait is not a
    /// client stall: `progress` counts from the grant.
    fn waits_for_grant(&self) -> bool {
        matches!(&self.operation, Operation::Put(put) if !put.committed
            && put.requested == 0
            && put.windows.granted.is_empty()
            && put.windows.next < put.grants.len())
    }
}

pub struct Session {
    pub namespace: NamespaceId,
    pub ops: HashMap<u64, Open>,
    pub pid: u32,
    pub closed: bool,
}

#[derive(Clone, Copy)]
enum CatalogKind {
    Added,
    Changed,
    Removed,
}

enum ReadKind {
    Meta { length: usize },
    Window { grant: u32 },
}

enum Job {
    Compact {
        namespace: NamespaceId,
        key: Key,
    },
    Write {
        namespace: NamespaceId,
        key: Key,
        reason: &'static str,
        _memory: Arc<Extent>,
        _temporary: Arc<DiskExtent>,
    },
    Chunk {
        connection: u64,
        transfer: u64,
        _data: ExtentSlice,
        _temporary: Arc<DiskExtent>,
    },
    Finish {
        connection: u64,
        transfer: u64,
        final_extent: Arc<DiskExtent>,
        _metadata: Arc<Extent>,
        _temporary: Arc<DiskExtent>,
    },
    GetRead {
        namespace: NamespaceId,
        connection: u64,
        transfer: u64,
        record: u32,
        kind: ReadKind,
        buffer: Extent,
    },
}

fn finish_record(
    put: &Put,
    job: u64,
    connection: u64,
    transfer: u64,
    path: PathBuf,
) -> Option<(DiskJob, Job)> {
    let metadata = put.metadata.clone()?;
    let temporary = put.temporary.clone()?;
    let final_extent = Arc::new(DiskExtent::new(path));
    let task = DiskJob::write_record(
        JobId(job),
        temporary.path.clone(),
        0,
        metadata.bytes(),
        Some((final_extent.path.clone(), put.header.length(put.row))),
    );
    Some((
        task,
        Job::Finish {
            connection,
            transfer,
            final_extent,
            _metadata: metadata,
            _temporary: temporary,
        },
    ))
}

#[derive(Default)]
pub struct Metrics {
    pub dropped: u64,
    /// Operations stopped by cancel before becoming terminal, excluding committed puts.
    pub cancels: u64,
    pub puts_ok: u64,
    pub puts_error: u64,
    /// Page-only puts that cannot link to a held ancestor, refused `INVALID`.
    pub puts_unlinked: u64,
    pub put_links_linked: u64,
    pub put_links_root: u64,
    pub put_page_bytes: u64,
    pub put_row_bytes: u64,
    pub put_shapes_row: u64,
    pub put_shapes_page_only: u64,
    pub put_shapes_empty: u64,
    pub present: u64,
    pub retry_queue: u64,
    pub retry_budget: u64,
    put_writeback_waits: u64,
    put_inflight_waits: u64,
    put_inflight_expired: u64,
    put_inflight_wait_max: Duration,
    shutdown_unflushed: Option<(u64, u64)>,
    pub retry_inflight: u64,
    pub oversize: u64,
    pub gets_ok: u64,
    pub gets_error: u64,
    pub evictions: u64,
    pub expired: [u64; 2],
    pub compactions: u64,
    record_actions: HashMap<(CaptureRole, &'static str), (u64, u64)>,
    /// Durable bytes written; restore reads count in `nvme_read_bytes`.
    pub nvme_bytes: u64,
    pub nvme_read_bytes: u64,
    pub nvme_seconds: f64,
    pub nvme_errors: u64,
    pub hits: HashMap<(String, &'static str), u64>,
}

#[derive(Clone, Debug)]
pub struct EntryData {
    pub packed: bool,
    pub required_profiles: Vec<Vec<u8>>,
    pub key: Key,
    pub parent: Option<Key>,
    pub bucket: [u8; 16],
    pub tail_pages: u32,
    pub row: bool,
    /// 0 = DRAM, 1 = NVMe.
    pub tier: u8,
    pub durable: bool,
    pub bytes: u64,
}

struct PendingPut {
    inflight_since: Option<Instant>,
    longest_inflight_wait: Duration,
    connection: u64,
    transfer: u64,
    request: PutRequest,
    events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
}

struct PendingGet {
    connection: u64,
    transfer: u64,
    request: GetRequest,
    events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
}

pub struct PutRequest {
    pub namespace: NamespaceId,
    pub key: Key,
    pub published_parent: Option<Key>,
    pub ancestor_keys: Vec<Key>,
    pub bucket: [u8; 16],
    pub chain_pages: u32,
    pub published_tail_pages: u32,
    pub capture_role: CaptureRole,
    pub child_paths: u32,
    pub descendant_row: bool,
    pub next_child: Option<Key>,
    pub row: bool,
    pub packed: Option<flare_selene::shared::wire::PackedOffer>,
}

pub struct GetRequest {
    pub namespace: NamespaceId,
    pub keys: Vec<Key>,
    pub skip_pages: u32,
    pub expected_pages: u32,
    pub row: bool,
}

const WRITEBACK_WORKERS: usize = 4;

pub struct Store {
    config: Config,
    local: Local,
    memory: Arena,
    staging: Arena,
    // Engines drain before the root lock and job buffers are released.
    namespaces: HashMap<NamespaceId, Namespace>,
    writers: Vec<DiskThread>,
    _lock: Option<disk::RootLock>,
    sessions: HashMap<u64, Session>,
    jobs: HashMap<u64, Job>,
    job_counter: u64,
    /// The last time any disk job finished, or the last tick with none outstanding.
    disk_progress: Instant,
    clock: u64,
    metrics: Metrics,
    charged: u64,
    /// The highest `charge_total` seen at an admission or a tick.
    reserved_peak: u64,
    writeback: HashSet<(NamespaceId, Key)>,
    pending_former_tips: HashSet<(NamespaceId, Key)>,
    retired_records: Vec<(NamespaceId, Key, Location)>,
    pending_puts: HashMap<u64, VecDeque<PendingPut>>,
    pending_gets: HashMap<u64, VecDeque<PendingGet>>,
    put_order: VecDeque<u64>,
    get_order: VecDeque<u64>,
    catalogs: HashMap<NamespaceId, tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>>,
    read_backlog: VecDeque<(NamespaceId, u64, DiskJob)>,
    _name_lock: File,
}

/// Names the startup step and path an error came from.
fn step_error(step: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{step} at {}: {error}", path.display()),
    )
}

fn put_mut(
    sessions: &mut HashMap<u64, Session>,
    connection: u64,
    transfer: u64,
) -> Option<(&mut Put, &mut Instant)> {
    let open = sessions.get_mut(&connection)?.ops.get_mut(&transfer)?;
    match &mut open.operation {
        Operation::Put(put) => Some((put, &mut open.progress)),
        _ => None,
    }
}

fn get_mut(
    sessions: &mut HashMap<u64, Session>,
    connection: u64,
    transfer: u64,
) -> Option<(&mut Get, &mut Instant)> {
    let open = sessions.get_mut(&connection)?.ops.get_mut(&transfer)?;
    match &mut open.operation {
        Operation::Get(get) => Some((get, &mut open.progress)),
        _ => None,
    }
}

fn get_ref(sessions: &HashMap<u64, Session>, connection: u64, transfer: u64) -> Option<&Get> {
    match &sessions.get(&connection)?.ops.get(&transfer)?.operation {
        Operation::Get(get) => Some(get),
        _ => None,
    }
}

fn put_ref(sessions: &HashMap<u64, Session>, connection: u64, transfer: u64) -> Option<&Put> {
    match &sessions.get(&connection)?.ops.get(&transfer)?.operation {
        Operation::Put(put) => Some(put),
        _ => None,
    }
}

impl Store {
    pub fn open(config: Config) -> Result<Self> {
        config.validate()?;
        let memory = usize::try_from(config.memory)
            .map_err(|_| invalid("--memory is too large for this host"))?;
        let staging = usize::try_from(config.staging)
            .map_err(|_| invalid("--staging is too large for this host"))?;
        let name_lock = crate::local::lock_pool(&config.name, &config.socket)?;
        let mut writers = Vec::new();
        if config.root.is_some() {
            for _ in 0..WRITEBACK_WORKERS {
                writers.push(Self::disk_thread(&config, "manager-writeback").map_err(|error| {
                    invalid(format!(
                        "the store cannot start its disk threads: {error}. The manager needs \
                         io_uring: allow io_uring_setup, io_uring_enter and io_uring_register in \
                         the container's seccomp profile"
                    ))
                })?);
            }
        }
        let lock = config
            .root
            .as_ref()
            .map(|root| {
                disk::RootLock::acquire(root)
                    .map_err(|error| step_error("lock the root", root, invalid(error.to_string())))
            })
            .transpose()?;
        let recovered = match (&config.root, &lock) {
            (Some(root), Some(_)) => disk::recover(root, &config.name)
                .map_err(|error| step_error("recover", root, error))?,
            _ => Vec::new(),
        };
        let local = Local::open(&config.region_name()?, memory, staging).map_err(|error| {
            step_error(
                "open shared memory in",
                Path::new(crate::local::SHM_ROOT),
                error,
            )
        })?;
        let memory = local.arena();
        let staging = local.staging();
        let marker = local.directory().join("last-shutdown");
        let shutdown_unflushed = match fs::read_to_string(&marker) {
            Ok(counts) => {
                fs::remove_file(&marker)?;
                counts
                    .trim()
                    .split_once(' ')
                    .and_then(|(records, bytes)| Some((records.parse().ok()?, bytes.parse().ok()?)))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        let mut store = Self {
            _name_lock: name_lock,
            config: config.clone(),
            local,
            memory,
            staging,
            writers,
            _lock: lock,
            namespaces: HashMap::new(),
            sessions: HashMap::new(),
            jobs: HashMap::new(),
            job_counter: 0,
            disk_progress: Instant::now(),
            clock: 0,
            metrics: Metrics {
                shutdown_unflushed,
                ..Metrics::default()
            },
            charged: 0,
            reserved_peak: 0,
            writeback: HashSet::new(),
            pending_former_tips: HashSet::new(),
            retired_records: Vec::new(),
            pending_puts: HashMap::new(),
            pending_gets: HashMap::new(),
            put_order: VecDeque::new(),
            get_order: VecDeque::new(),
            catalogs: HashMap::new(),
            read_backlog: VecDeque::new(),
        };
        let mut recovered_age = Vec::new();
        for recovered in recovered {
            let mut namespace = store.new_namespace(recovered.descriptor)?;
            let id = namespace.descriptor.namespace_id();
            for (header, row, extent) in recovered.records {
                let modified = fs::metadata(&extent.path)?
                    .modified()
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                recovered_age.push((modified, id, header.key));
                let bytes = header.length(row);
                namespace
                    .tree
                    .insert(
                        header.key,
                        header.parent,
                        header.bucket,
                        header.tail_pages as usize,
                        row,
                        true,
                        Location {
                            header,
                            memory: None,
                            disk: Some(extent),
                            durable: false,
                            writing: false,
                            job: None,
                            retry: Instant::now(),
                            admitted: Instant::now(),
                            bytes,
                        },
                    )
                    .map_err(|e| invalid(e.to_string()))?;
                store.charged += bytes;
            }
            store.namespaces.insert(id, namespace);
            store.refresh_durability(id, None);
        }
        recovered_age.sort_unstable();
        for (modified, id, key) in recovered_age {
            store.clock += 1;
            let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
            tree.access_at(&key, store.clock);
            let node = tree.node_mut(&key).unwrap();
            node.last_used = modified;
        }
        Ok(store)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    /// What the pool service needs from the store: the shared region files, a handle on the
    /// staging region.
    pub fn pool_view(&self) -> (crate::proto::v2::SharedMemory, crate::arena::Arena) {
        (
            crate::proto::v2::SharedMemory {
                arena_path: self.local.arena_path().display().to_string(),
                arena_bytes: self.local.arena_size(),
                staging_path: self.local.staging_path().display().to_string(),
                staging_bytes: self.local.staging_size(),
                regions_generation: self.local.regions_generation(),
            },
            self.local.staging(),
        )
    }

    /// True while the shared regions are held for the previous run's processes.
    pub fn held_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.local.held_flag()
    }

    pub(crate) fn check_regions(&self) -> Result<()> {
        self.local.check_regions()
    }

    pub fn shm_directory(&self) -> PathBuf {
        self.local.directory().to_path_buf()
    }
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn disk_thread(_config: &Config, name: &str) -> Result<DiskThread> {
        #[cfg(any(test, feature = "test-support"))]
        let worker = DiskThread::start_with_faults(name, _config.faults.clone());
        #[cfg(not(any(test, feature = "test-support")))]
        let worker = DiskThread::start(name);
        worker.map_err(|e| invalid(e.to_string()))
    }

    fn new_namespace(&self, descriptor: Descriptor) -> Result<Namespace> {
        let directory = self
            .config
            .root
            .as_ref()
            .map(|root| root.join(descriptor.name()));
        let (disk, reads) = if directory.is_some() {
            (
                Some(Self::disk_thread(&self.config, "manager-write")?),
                Some(Self::disk_thread(&self.config, "manager-read")?),
            )
        } else {
            (None, None)
        };
        let mut tree = Tree::new(descriptor.page_tokens);
        tree.max_age = self.config.max_age;
        Ok(Namespace {
            directory,
            tree,
            descriptor,
            disk,
            reads,
            reads_pending: 0,
            read_progress: Instant::now(),
            inflight: HashSet::new(),
            undurable: HashSet::new(),
        })
    }

    pub fn register_namespace(&mut self, descriptor: Descriptor) -> Result<NamespaceId> {
        descriptor.validate()?;
        let id = descriptor.namespace_id();
        if let Some(stored) = self.namespaces.get(&id) {
            if stored.descriptor != descriptor {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "namespace descriptor mismatch",
                ));
            }
            return Ok(id);
        }
        if let Some(root) = &self.config.root {
            fs::create_dir_all(root.join(descriptor.name()))?;
        }
        let namespace = self.new_namespace(descriptor)?;
        self.namespaces.insert(id, namespace);
        Ok(id)
    }

    pub fn namespace_id(&self) -> Option<NamespaceId> {
        self.namespaces.keys().next().copied()
    }

    pub fn session(&self, connection: u64) -> Option<&Session> {
        self.sessions.get(&connection)
    }
    pub fn hello(
        &mut self,
        connection: u64,
        pid: u32,
        process_nonce: Option<[u8; 16]>,
        descriptor: Descriptor,
    ) -> Result<()> {
        let id = self.register_namespace(descriptor)?;
        self.local.accept(connection, pid, process_nonce)?;
        self.sessions.insert(
            connection,
            Session {
                namespace: id,
                ops: HashMap::new(),
                pid,
                closed: false,
            },
        );
        Ok(())
    }

    /// The arena and staging mappings, for the data channels.
    pub fn regions(&self) -> crate::regions::LocalRegions {
        crate::regions::LocalRegions::new(self.memory.clone(), self.staging.clone())
    }

    /// Remove leaves through the shared tree, accounting bytes and files.
    fn evict_accounted(&mut self, id: NamespaceId, key: Key) -> u64 {
        let eligible = self
            .namespaces
            .get(&id)
            .and_then(|namespace| namespace.tree.node(&key))
            .is_some_and(|node| {
                node.ready && node.children() == 0 && !node.pinned() && !node.location.disk_busy()
            });
        if !eligible {
            return 0;
        }
        let removed = self.namespaces.get_mut(&id).and_then(|namespace| {
            namespace
                .tree
                .remove(&key, |node| node.location.disk_busy())
                .ok()
        });
        let Some(nodes) = removed else {
            return 0;
        };
        let count = nodes.len() as u64;
        for node in nodes {
            self.namespaces
                .get_mut(&id)
                .unwrap()
                .undurable
                .remove(&node.key);
            self.writeback.remove(&(id, node.key));
            if node.location.memory.is_some() || node.location.disk.is_some() {
                self.charged = self.charged.saturating_sub(node.location.bytes);
            }
            if let Some(disk) = &node.location.disk {
                disk.retire();
            } else if let Some(prior_disk) =
                self.retired_records.iter().find_map(|(scope, key, prior)| {
                    (*scope == id && *key == node.key)
                        .then_some(prior.disk.as_ref())
                        .flatten()
                })
            {
                // A dirty replacement may still have its predecessor's file at this name.
                if let Err(error) = fs::remove_file(&prior_disk.path)
                    && error.kind() != io::ErrorKind::NotFound
                {
                    self.metrics.nvme_errors += 1;
                }
            }
            self.delta(id, CatalogKind::Removed, node.key);
        }
        count
    }

    fn reused(&mut self, id: NamespaceId, key: Key) {
        self.clock += 1;
        let namespace = self.namespaces.get_mut(&id).unwrap();
        namespace.tree.access_at(&key, self.clock);
        namespace.tree.touch_ancestors(&key);
    }

    pub fn snapshot(&self, id: NamespaceId) -> Option<Vec<EntryData>> {
        let namespace = self.namespaces.get(&id)?;
        // The only consumer upserts into a hash index, so the order is unspecified.
        Some(
            namespace
                .tree
                .nodes()
                .map(|node| Self::entry(namespace, node.key))
                .collect(),
        )
    }

    fn entry(namespace: &Namespace, key: Key) -> EntryData {
        let node = &namespace.tree[&key];
        let header = &node.location.header;
        EntryData {
            packed: header.packed.is_some(),
            required_profiles: header
                .packed
                .as_ref()
                .map_or_else(Vec::new, |p| p.required_profiles(node.row)),
            key,
            parent: node.parent,
            bucket: node.bucket,
            tail_pages: node.tail_pages as u32,
            row: node.row,
            tier: u8::from(node.location.memory.is_none()),
            durable: node.location.durable,
            bytes: node.location.bytes,
        }
    }

    fn delta(&mut self, id: NamespaceId, kind: CatalogKind, key: Key) {
        let Some(sender) = self.catalogs.get(&id) else {
            return;
        };
        let namespace = &self.namespaces[&id];
        let event = match kind {
            CatalogKind::Added => proto::CatalogEvent::Added(Self::entry(namespace, key)),
            CatalogKind::Changed => proto::CatalogEvent::Changed(Self::entry(namespace, key)),
            CatalogKind::Removed => proto::CatalogEvent::Removed(key),
        };
        if sender.send(event).is_err() {
            self.catalogs.remove(&id);
        }
    }

    /// Recomputes chain-closed durability parents-first, emitting `changed` on flips. With a
    /// `touched` record it visits that record, and also the not-yet-durable set when the
    /// record has children and has just become durable (only that can make another record
    /// durable, and a parent exists before its children; a durable record only becomes
    /// non-durable when its own record is replaced, which passes `None`). With `None` it
    /// visits every record and rebuilds the set.
    fn refresh_durability(&mut self, id: NamespaceId, touched: Option<Key>) {
        let namespace = self.namespaces.get_mut(&id).unwrap();
        if namespace.disk.is_none() {
            return; // nothing is durable without a disk
        }
        let mut keys: Vec<Key> = match touched {
            Some(key) => {
                let became = namespace.tree.node(&key).is_some_and(|node| {
                    node.children() > 0
                        && !node.location.durable
                        && node.location.disk.is_some()
                        && namespace.parent_durable(node.parent)
                });
                if became {
                    namespace.undurable.iter().copied().chain([key]).collect()
                } else {
                    namespace.undurable.insert(key);
                    vec![key]
                }
            }
            None => {
                namespace.undurable.clear();
                namespace.tree.nodes().map(|node| node.key).collect()
            }
        };
        keys.sort_unstable_by_key(|key| key.consumed_tokens);
        let mut flipped = Vec::new();
        for key in keys {
            let Some(node) = namespace.tree.node(&key) else {
                namespace.undurable.remove(&key);
                continue;
            };
            let parent_ok = node
                .parent
                .is_none_or(|parent| namespace.tree[&parent].location.durable);
            let durable = node.location.disk.is_some() && parent_ok;
            if node.location.durable != durable {
                namespace.tree[&key].location.durable = durable;
                flipped.push(key);
            }
            if durable {
                namespace.undurable.remove(&key);
            } else {
                namespace.undurable.insert(key);
            }
        }
        for key in flipped {
            self.delta(id, CatalogKind::Changed, key);
        }
    }

    /// The NVMe limit covers retained records and arena extents, unfinished disk
    /// records, and streamed puts' staging holds. Restore and export windows are
    /// bounded separately by `--staging`.
    fn charge_total(&self) -> u64 {
        self.charged.saturating_add(self.put_staging_held())
    }

    /// Staging held until each streamed Put and its disk writes finish.
    fn put_staging_held(&self) -> u64 {
        fn held(hold: &Option<Arc<Extent>>) -> u64 {
            hold.as_ref().map_or(0, |hold| hold.capacity() as u64)
        }
        let mut total = 0u64;
        for session in self.sessions.values() {
            for open in session.ops.values() {
                if let Operation::Put(put) = &open.operation {
                    total = total.saturating_add(held(&put.staging_hold));
                }
            }
        }
        total
    }

    /// Remember the most the limit's measure has been.
    fn note_reserved(&mut self) {
        let now = self.charge_total();
        self.reserved_peak = self.reserved_peak.max(now);
    }

    pub fn submit_put(
        &mut self,
        connection: u64,
        transfer: u64,
        request: PutRequest,
        events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
    ) {
        self.queue_put(PendingPut {
            inflight_since: None,
            longest_inflight_wait: Duration::ZERO,
            connection,
            transfer,
            request,
            events,
        });
    }

    fn queue_put(&mut self, pending: PendingPut) {
        let connection = pending.connection;
        let queue = self.pending_puts.entry(connection).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(pending);
        if was_empty {
            self.put_order.push_back(connection);
        }
    }

    pub fn submit_get(
        &mut self,
        connection: u64,
        transfer: u64,
        request: GetRequest,
        events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
    ) {
        let open = self
            .sessions
            .get(&connection)
            .map(Self::get_count)
            .unwrap_or(0);
        let queued = self
            .pending_gets
            .get(&connection)
            .map_or(0, |queue| queue.len());
        if open + queued >= OPEN_GETS {
            self.metrics.gets_error += 1;
            let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                reason: proto::Reason::Invalid as u8,
                detail: "open Get limit".into(),
            })));
            return;
        }
        let queue = self.pending_gets.entry(connection).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(PendingGet {
            connection,
            transfer,
            request,
            events,
        });
        if was_empty {
            self.get_order.push_back(connection);
        }
    }

    fn get_count(session: &Session) -> usize {
        session
            .ops
            .values()
            .filter(|open| matches!(open.operation, Operation::Get(_)))
            .count()
    }

    fn pop_get(&mut self) -> Option<PendingGet> {
        while let Some(connection) = self.get_order.pop_front() {
            let Some(queue) = self.pending_gets.get_mut(&connection) else {
                continue;
            };
            let Some(pending) = queue.pop_front() else {
                self.pending_gets.remove(&connection);
                continue;
            };
            if queue.is_empty() {
                self.pending_gets.remove(&connection);
            } else {
                self.get_order.push_back(connection);
            }
            return Some(pending);
        }
        None
    }

    fn pop_put(&mut self) -> Option<PendingPut> {
        while let Some(connection) = self.put_order.pop_front() {
            let Some(queue) = self.pending_puts.get_mut(&connection) else {
                continue;
            };
            let Some(pending) = queue.pop_front() else {
                self.pending_puts.remove(&connection);
                continue;
            };
            if queue.is_empty() {
                self.pending_puts.remove(&connection);
            } else {
                self.put_order.push_back(connection);
            }
            return Some(pending);
        }
        None
    }

    #[cfg(test)]
    fn dispatch_trace(&mut self) -> Vec<&'static str> {
        let mut trace = Vec::new();
        let mut gets = 0;
        for _ in 0..4 {
            let Some(pending) = self.pop_get() else {
                break;
            };
            gets += 1;
            self.begin_get(pending);
        }
        trace.extend(std::iter::repeat_n("get", gets));
        if let Some(pending) = self.pop_put() {
            trace.push("put");
            self.begin_put(pending);
        }
        trace
    }

    fn dispatch(&mut self) {
        for _ in 0..4 {
            let Some(pending) = self.pop_get() else {
                break;
            };
            self.begin_get(pending);
        }
        if let Some(pending) = self.pop_put() {
            self.begin_put(pending);
        }
    }

    // ---------------------------------------------------------------- Put begin

    fn reject_put(&mut self, events: &tokio::sync::mpsc::Sender<Result<PutAnswer>>, reason: u8) {
        self.reject_put_detail(events, reason, &format!("reason {reason}"));
    }

    fn reject_put_detail(
        &mut self,
        events: &tokio::sync::mpsc::Sender<Result<PutAnswer>>,
        reason: u8,
        detail: &str,
    ) {
        let _ = events.try_send(Ok(PutAnswer::Refused(Refused {
            reason,
            detail: detail.to_string(),
        })));
    }

    /// A transfer id is unusable while any operation, queued request or drain with that
    /// (connection, transfer) is still alive, in either direction.
    fn transfer_in_use(&self, connection: u64, transfer: u64) -> bool {
        if self
            .sessions
            .get(&connection)
            .is_some_and(|session| session.ops.contains_key(&transfer))
        {
            return true;
        }
        if self
            .pending_puts
            .get(&connection)
            .is_some_and(|queue| queue.iter().any(|pending| pending.transfer == transfer))
            || self
                .pending_gets
                .get(&connection)
                .is_some_and(|queue| queue.iter().any(|pending| pending.transfer == transfer))
        {
            return true;
        }
        false
    }

    fn wait_for_inflight(&mut self, mut pending: PendingPut) {
        let key = &pending.request.key;
        // This writer may wait for admission before releasing its own lease or put.
        let session = &self.sessions[&pending.connection];
        let owns_pin = session.ops.values().any(|open| match &open.operation {
            Operation::Get(get) => get.pins.contains(key),
            Operation::Put(put) => put.pins.contains(key),
        });
        let expired = !owns_pin
            && pending
                .inflight_since
                .is_some_and(|start| start.elapsed() >= PUT_INFLIGHT_WAIT);
        if owns_pin || expired {
            if expired {
                self.metrics.put_inflight_expired += 1;
            }
            self.metrics.retry_inflight += 1;
            self.reject_put(&pending.events, proto::Reason::Inflight as u8);
            return;
        }
        if pending.inflight_since.is_none() {
            self.metrics.put_inflight_waits += 1;
            pending.inflight_since = Some(Instant::now());
        }
        self.queue_put(pending);
    }

    fn put_present(&mut self, pending: &PendingPut) {
        self.metrics.present += 1;
        self.metrics.put_inflight_wait_max = self
            .metrics
            .put_inflight_wait_max
            .max(pending.longest_inflight_wait);
        self.reject_put(&pending.events, proto::Reason::Present as u8);
    }

    fn begin_put(&mut self, mut pending: PendingPut) {
        let PendingPut {
            connection,
            transfer,
            ref request,
            ..
        } = pending;
        let Some(id) = self.sessions.get(&connection).map(|_| request.namespace) else {
            return;
        };
        let Some(descriptor) = self
            .namespaces
            .get(&id)
            .map(|namespace| namespace.descriptor.clone())
        else {
            self.reject_put(&pending.events, proto::Reason::Invalid as u8);
            return;
        };
        let invalid_geometry = request.key.consumed_tokens == 0
            || request.key.consumed_tokens / descriptor.page_tokens
                != u64::from(request.chain_pages)
            || request.published_tail_pages > request.chain_pages
            || (request.published_parent.is_none()
                && request.published_tail_pages != request.chain_pages)
            || request.published_parent.is_some_and(|parent| {
                parent.consumed_tokens >= request.key.consumed_tokens
                    || request.key.consumed_tokens / descriptor.page_tokens
                        - parent.consumed_tokens / descriptor.page_tokens
                        != u64::from(request.published_tail_pages)
            })
            || request
                .ancestor_keys
                .first()
                .is_some_and(|ancestor| ancestor.consumed_tokens >= request.key.consumed_tokens)
            || request
                .ancestor_keys
                .windows(2)
                .any(|keys| keys[0].consumed_tokens <= keys[1].consumed_tokens);
        if invalid_geometry {
            self.metrics.puts_error += 1;
            self.reject_put(&pending.events, proto::Reason::Invalid as u8);
            return;
        }
        if self.namespaces[&id].inflight.contains(&request.key) {
            self.wait_for_inflight(pending);
            return;
        }
        let initial_old = self.namespaces[&id].tree.node(&request.key).cloned();
        let role = initial_old.as_ref().map_or(request.capture_role, |old| {
            old.location
                .header
                .capture_role()
                .merge(request.capture_role)
        });
        // A legacy peer can only commit a header with zero role bits.
        let wire_role = if request.capture_role == CaptureRole::Legacy {
            CaptureRole::Legacy
        } else {
            role
        };
        let children = initial_old
            .as_ref()
            .map_or(0, |node| {
                let additional = request.next_child.is_some_and(|child| {
                    self.namespaces[&id]
                        .tree
                        .node(&child)
                        .is_none_or(|n| n.parent != Some(node.key))
                });
                node.children().saturating_add(u32::from(additional))
            })
            .max(request.child_paths);
        let needs_row = |parent: Option<Key>| {
            parent.is_none()
                || wire_role != CaptureRole::Endpoint
                || children > 1
                || ((children == 0 || !request.descendant_row)
                    && !self.namespaces[&id].tree.covered_endpoint(
                        request.key,
                        parent,
                        role,
                        children,
                        |node| node.ready,
                    ))
        };
        let present = if let Some(node) = &initial_old {
            if node.bucket != request.bucket {
                self.reject_put(&pending.events, proto::Reason::Invalid as u8);
                return;
            }
            let covers_role = node.location.header.capture_role() == role;
            (node.location.memory.is_some() || node.location.disk.is_some())
                && (!request.row
                    || (node.row && covers_role)
                    || (!node.row && !needs_row(node.parent)))
        } else {
            false
        };
        if !present
            && initial_old
                .as_ref()
                .is_some_and(|node| node.pinned() || node.location.disk_busy())
        {
            self.wait_for_inflight(pending);
            return;
        }
        if let Some(start) = pending.inflight_since.take() {
            pending.longest_inflight_wait = pending.longest_inflight_wait.max(start.elapsed());
        }
        if present {
            self.namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .touch_ancestors(&request.key);
            self.put_present(&pending);
            return;
        }
        let initial_bucket = match &initial_old {
            Some(node) => {
                if node.bucket != request.bucket {
                    self.reject_put(&pending.events, proto::Reason::Invalid as u8);
                    return;
                }
                node.bucket
            }
            None => request.bucket,
        };
        let ancestors = if request.ancestor_keys.is_empty() {
            request.published_parent.as_slice()
        } else {
            request.ancestor_keys.as_slice()
        };
        let initial_parent = match &initial_old {
            Some(node) => node.parent,
            None => ancestors
                .iter()
                .filter_map(|ancestor| self.namespaces[&id].tree.node(ancestor))
                .find(|node| {
                    node.ready && (node.location.memory.is_some() || node.location.disk.is_some())
                })
                .filter(|node| node.chain_depth < REROOT_DEPTH)
                .map(|node| node.key),
        };
        // A page-only record can end no restore, so it is stored only linked to a held
        // ancestor. One that cannot link, because no ancestor is held or the deepest one
        // ends a chain of `REROOT_DEPTH` records and a new root would need a row, is
        // refused. That is normal operation, not an error.
        if initial_parent.is_none() && !request.row {
            self.metrics.puts_unlinked += 1;
            self.reject_put_detail(&pending.events, proto::Reason::Invalid as u8, PUT_UNLINKED);
            return;
        }
        let mut row = request.row && needs_row(initial_parent);
        let initial_first =
            initial_parent.map_or(0, |parent| parent.consumed_tokens / descriptor.page_tokens);
        let Some(initial_tail) = request
            .chain_pages
            .checked_sub(u32::try_from(initial_first).unwrap_or(u32::MAX))
        else {
            self.reject_put(&pending.events, proto::Reason::Invalid as u8);
            return;
        };
        if initial_tail == 0 && !row {
            // No pages and no resume row: the held parent already covers it.
            if let Some(parent) = initial_parent {
                self.namespaces
                    .get_mut(&id)
                    .unwrap()
                    .tree
                    .touch_ancestors(&parent);
            }
            self.metrics.put_shapes_empty += 1;
            if request.row {
                let action = self
                    .metrics
                    .record_actions
                    .entry((role, "row_omitted"))
                    .or_default();
                action.0 += 1;
                action.1 += descriptor.row_bytes();
            }
            self.put_present(&pending);
            return;
        }
        let child_header = match crate::record::new_header(
            &descriptor,
            request.key,
            initial_parent,
            initial_bucket,
            initial_tail,
        )
        .and_then(|header| {
            let header = header.with_capture_role(wire_role);
            apply_packed_offer(
                header,
                request.packed.as_ref(),
                initial_first,
                initial_tail,
                row,
            )
        }) {
            Ok(header) => header,
            Err(_) => {
                self.reject_put(&pending.events, proto::Reason::Invalid as u8);
                return;
            }
        };
        let root_header = match crate::record::new_header(
            &descriptor,
            request.key,
            None,
            initial_bucket,
            request.chain_pages,
        )
        .and_then(|header| {
            let header = header.with_capture_role(wire_role);
            let root_packed = request.packed.as_ref().filter(|o| o.first_page == 0);
            apply_packed_offer(
                header,
                root_packed,
                0,
                request.chain_pages,
                request.row,
            )
        }) {
            Ok(header) => header,
            Err(_) => {
                self.reject_put(&pending.events, proto::Reason::Invalid as u8);
                return;
            }
        };
        let rootless = self.config.root.is_none();
        let replacement = initial_old.is_some();
        let arena_bytes = self.memory.capacity() as u64;
        // The arena rounds every record extent up to a power of two. Without a
        // root there is no disk to stream to, so a record whose own extent
        // cannot fit the arena can never be stored. Whether it fits beside
        // what else must stay resident is a capacity question.
        if rootless && child_header.length(row).max(4096).next_power_of_two() > arena_bytes {
            self.metrics.oversize += 1;
            self.reject_put(&pending.events, proto::Reason::Oversize as u8);
            return;
        }
        let selected_pins: Vec<Key> = initial_parent
            .into_iter()
            .chain(replacement.then_some(request.key))
            .collect();
        // Admission charges the durable budget with the shape's full
        // reservation: the rounded extent for a record the arena keeps, or
        // for a streamed record its file and
        // its mandatory two-window staging hold. Only the record's own charge
        // can make it OVERSIZE; a shortage caused by what else must stay
        // resident is a Budget refusal.
        let extent = child_header.length(row).max(4096).next_power_of_two();
        let charge = if extent <= arena_bytes {
            extent
        } else {
            self.put_requirement(&child_header, 0, row).charge
        };
        if !rootless && charge > self.config.nvme {
            self.metrics.oversize += 1;
            self.reject_put(&pending.events, proto::Reason::Oversize as u8);
            return;
        }
        // Pick the shape by asking the arena allocator. A failed allocation
        // starts write-back for a rooted manager. A rootless manager tries to evict
        // the parent and store the root. `memory` holds the extent the shape
        // was admitted with.
        let selected = self.put_requirement(&child_header, arena_bytes, row);
        let mut memory: Option<Arc<Extent>> = None;
        let (header, parent, pins, requirement): (Header, Option<Key>, Vec<Key>, PutRequirement) =
            if !selected.streaming {
                match self.memory_room_excluding(
                    child_header.length(row) as usize,
                    !rootless,
                    &selected_pins,
                ) {
                    Some(extent) => {
                        let requirement = PutRequirement::arena(extent.capacity() as u64);
                        memory = Some(Arc::new(extent));
                        (child_header, initial_parent, selected_pins, requirement)
                    }
                    None if rootless => {
                        if !request.row {
                            self.metrics.retry_budget += 1;
                            self.reject_put(&pending.events, proto::Reason::Budget as u8);
                            return;
                        }
                        let root_pins: Vec<Key> =
                            replacement.then_some(request.key).into_iter().collect();
                        match self.rootless_fallback(
                            &root_header,
                            root_pins,
                            arena_bytes,
                            request.row,
                        ) {
                            RootChoice::Store(extent) => {
                                row = request.row;
                                let requirement = PutRequirement::arena(extent.capacity() as u64);
                                memory = Some(Arc::new(extent));
                                let root_pins: Vec<Key> =
                                    replacement.then_some(request.key).into_iter().collect();
                                (root_header, None, root_pins, requirement)
                            }
                            RootChoice::Wait => {
                                self.metrics.retry_budget += 1;
                                self.reject_put(&pending.events, proto::Reason::Budget as u8);
                                return;
                            }
                        }
                    }
                    None if self
                        .jobs
                        .values()
                        .any(|job| matches!(job, Job::Write { .. })) =>
                    {
                        self.metrics.put_writeback_waits += 1;
                        self.queue_put(pending);
                        return;
                    }
                    None => {
                        self.metrics.retry_budget += 1;
                        self.reject_put(&pending.events, proto::Reason::Budget as u8);
                        return;
                    }
                }
            } else {
                let requirement = self.put_requirement(&child_header, 0, row);
                (child_header, initial_parent, selected_pins, requirement)
            };
        let length = header.length(row);
        // A streamed record reserves two windows in staging. When that hold
        // cannot fit the whole region it can never be admitted: if the record
        // has no arena shape either, refuse it as OVERSIZE. A record that still
        // fits an arena extent can be tried again after space is freed.
        if requirement.streaming && requirement.allocation > self.staging.capacity() as u64 {
            if selected.streaming {
                self.metrics.oversize += 1;
                self.reject_put(&pending.events, proto::Reason::Oversize as u8);
                return;
            }
            self.metrics.retry_budget += 1;
            self.reject_put(&pending.events, proto::Reason::Budget as u8);
            return;
        }
        for key in &pins {
            if self.namespaces.get_mut(&id).unwrap().tree.pin(key).is_err() {
                for key in &pins {
                    let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                }
                self.reject_put(&pending.events, proto::Reason::Invalid as u8);
                return;
            }
        }
        if !self.make_room_with(Some(id), requirement.charge, &pins) {
            drop(memory);
            for key in &pins {
                let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
            }
            if self
                .jobs
                .values()
                .any(|job| matches!(job, Job::Write { .. }))
            {
                self.metrics.put_writeback_waits += 1;
                self.queue_put(pending);
            } else {
                self.metrics.retry_budget += 1;
                self.reject_put(&pending.events, proto::Reason::Budget as u8);
            }
            return;
        }
        let staging_hold = if requirement.streaming {
            match self.staging.allocate(requirement.allocation as usize) {
                Some(extent) => Some(Arc::new(extent)),
                None => {
                    drop(memory);
                    for key in &pins {
                        let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                    }
                    self.metrics.retry_budget += 1;
                    self.reject_put(&pending.events, proto::Reason::Budget as u8);
                    return;
                }
            }
        } else {
            None
        };
        let temporary = if requirement.streaming {
            self.job_counter += 1;
            let job = self.job_counter;
            match self.namespaces[&id].temporary(job) {
                Some(extent) => Some(extent),
                None => {
                    drop(memory);
                    drop(staging_hold);
                    for key in &pins {
                        let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                    }
                    self.reject_put(&pending.events, proto::Reason::Io as u8);
                    return;
                }
            }
        } else {
            None
        };
        self.charged += requirement.reservation;
        // The hold is allocated, and counts from here.
        let held_now = staging_hold
            .as_ref()
            .map_or(0, |hold| hold.capacity() as u64);
        self.reserved_peak = self
            .reserved_peak
            .max(self.charge_total().saturating_add(held_now));
        let first =
            parent.map_or(0, |parent| parent.consumed_tokens / descriptor.page_tokens) as u32;
        let grants = record_grants(&header, first, 0, row, DEFAULT_MAX_WINDOW, 0);
        let pages = vec![None; header.tail_pages as usize];
        let mut put = Put {
            row,
            stored_role: role,
            namespace: id,
            header: header.clone(),
            first_page: first,
            grants,
            requested: 0,
            windows: FilledWindows {
                granted: HashMap::new(),
                next: 0,
                pages,
                row: None,
                pages_filled: 0,
                row_filled: !row || header.row_record_bytes() == 0,
            },
            memory,
            temporary,
            streaming: requirement.streaming,
            commit_bytes: Vec::new(),
            metadata: None,
            committed: false,
            finalizing: false,
            cancelled: false,
            terminal: false,
            finished: false,
            pending: 0,
            blocked: VecDeque::new(),
            pins: pins.into(),
            reservation: requirement.reservation,
            staging_hold,
            hold_unit: requirement.unit as usize,
            hold_slots: [None, None],
            events: pending.events,
            outbox: VecDeque::new(),
        };
        let namespace = self.namespaces.get_mut(&id).unwrap();
        namespace.inflight.insert(request.key);
        if parent.is_some() {
            self.metrics.put_links_linked += 1;
        } else {
            self.metrics.put_links_root += 1;
        }
        self.metrics.put_page_bytes += u64::from(header.tail_pages) * header.page_record_bytes();
        self.metrics.put_row_bytes += if row { header.row_record_bytes() } else { 0 };
        if row {
            self.metrics.put_shapes_row += 1;
        } else {
            self.metrics.put_shapes_page_only += 1;
            if request.row {
                let action = self
                    .metrics
                    .record_actions
                    .entry((role, "row_omitted"))
                    .or_default();
                action.0 += 1;
                action.1 += descriptor.row_bytes();
            }
        }
        self.metrics.put_inflight_wait_max = self
            .metrics
            .put_inflight_wait_max
            .max(pending.longest_inflight_wait);
        put.outbox.push_back(Ok(PutAnswer::Accept(PutAccept {
            capture_role: wire_role,
            transfer,
            first_page: first,
            parent,
            record_bytes: length,
            windows: put.grants.len(),
            packed: header.packed.is_some(),
        })));
        self.sessions.get_mut(&connection).unwrap().ops.insert(
            transfer,
            Open {
                progress: Instant::now(),
                operation: Operation::Put(Box::new(put)),
            },
        );
    }

    fn oldest(
        &self,
        eligible: impl Fn(NamespaceId, &flare_selene::shared::index::Node<Location>) -> bool,
    ) -> Option<(NamespaceId, Key)> {
        self.namespaces
            .iter()
            .filter_map(|(id, namespace)| {
                namespace
                    .tree
                    .ordered()
                    .find(|node| {
                        !node.pinned()
                            && !node.location.disk_busy()
                            && !node.location.reading()
                            && eligible(*id, node)
                    })
                    .map(move |node| (node.accessed(), node.key, *id))
            })
            .min_by_key(|(accessed, key, _)| (*accessed, key.consumed_tokens))
            .map(|(_, key, id)| (id, key))
    }

    /// Attempt the arena allocation, reclaiming eligible copies one at a time
    /// until it succeeds. Asking the allocator rather than summing free bytes
    /// handles buddy fragmentation and chains whose parent only becomes a
    /// leaf after its child is evicted. Excluded records keep their arena copy
    /// unless a durable disk copy preserves them.
    fn memory_room_excluding(
        &mut self,
        bytes: usize,
        require_durable: bool,
        excluded: &[Key],
    ) -> Option<Extent> {
        if bytes.max(4096).checked_next_power_of_two()? > self.memory.capacity() {
            return None;
        }
        loop {
            if let Some(extent) = self.memory.allocate(bytes) {
                return Some(extent);
            }
            let Some((id, key)) = self.oldest(|_, node| {
                Self::memory_evictable(node, require_durable)
                    && (require_durable || !excluded.contains(&node.key))
            }) else {
                if require_durable {
                    let idle = self.writers.iter().filter(|w| w.is_idle()).count();
                    for _ in 0..idle {
                        let Some((id, key)) = self.oldest(|id, node| {
                            node.location.memory.is_some()
                                && !node.location.durable
                                && !self.writeback.contains(&(id, node.key))
                                && self.writable_ancestor(id, node.key, false).is_some()
                        }) else { break };
                        self.request_writeback(id, key);
                    }
                }
                return None;
            };
            if !self.evict_memory(id, key) {
                return None;
            }
        }
    }

    /// Shared eligibility for reclaiming an arena extent. A rooted manager may
    /// only drop records that have a disk copy; a rootless one may drop true
    /// leaves outright.
    fn memory_evictable(
        node: &flare_selene::shared::index::Node<Location>,
        require_durable: bool,
    ) -> bool {
        node.location.memory.is_some()
            && if require_durable {
                node.location.durable
            } else {
                node.children() == 0
            }
    }

    /// Try the whole chain when the arena cannot hold the child beside its parent.
    fn rootless_fallback(
        &mut self,
        root: &Header,
        root_pins: Vec<Key>,
        arena_bytes: u64,
        row: bool,
    ) -> RootChoice {
        let requirement = self.put_requirement(root, arena_bytes, row);
        if requirement.streaming {
            return RootChoice::Wait;
        }
        match self.memory_room_excluding(root.length(row) as usize, false, &root_pins) {
            Some(extent) => RootChoice::Store(extent),
            None => RootChoice::Wait,
        }
    }

    /// Returns whether the candidate was actually evicted. A node rejected by
    /// `Tree::remove` (because it gained children) is left untouched.
    fn evict_memory(&mut self, id: NamespaceId, key: Key) -> bool {
        let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
        if node.location.disk_busy() {
            return false;
        }
        if node.location.disk.is_some() {
            self.release_arena(id, key);
            return true;
        }
        let removed = self.evict_accounted(id, key);
        self.metrics.evictions += removed;
        removed != 0
    }

    fn release_arena(&mut self, id: NamespaceId, key: Key) {
        let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
        let bytes = node.location.header.length(node.row);
        node.location.memory = None;
        self.charged = self.charged - node.location.bytes + bytes;
        node.location.bytes = bytes;
        self.delta(id, CatalogKind::Changed, key);
    }

    /// The physical requirement for this Put shape. A record whose
    /// buddy-rounded extent fits `arena_bytes` is stored in one arena extent;
    /// otherwise it streams to disk behind a fixed staging hold of two of its
    /// largest windows. `arena_bytes` is the arena capacity for the shape
    /// decision; admission confirms it by allocating.
    fn put_requirement(&self, header: &Header, arena_bytes: u64, row: bool) -> PutRequirement {
        let length = header.length(row);
        let extent = length.max(4096).next_power_of_two();
        if extent <= arena_bytes {
            return PutRequirement::arena(extent);
        }
        let first = header
            .parent
            .map_or(0, |parent| parent.consumed_tokens / header.page_tokens)
            as u32;
        let unit = record_grants(header, first, 0, row, DEFAULT_MAX_WINDOW, 0)
            .iter()
            .map(|grant| grant.length.max(4096).next_power_of_two())
            .max()
            .unwrap_or(4096);
        PutRequirement::streaming(length, unit)
    }

    /// Reserve budget space while the shape's protected keys stay put, so an
    /// eviction cannot invalidate the shape the reservation was computed for.
    fn make_room_with(
        &mut self,
        namespace: Option<NamespaceId>,
        bytes: u64,
        protected: &[Key],
    ) -> bool {
        if self.config.root.is_none() {
            return true;
        }
        let mut pinned = Vec::new();
        if let Some(id) = namespace {
            for key in protected {
                if let Some(namespace) = self.namespaces.get_mut(&id) {
                    if namespace.tree.pin(key).is_ok() {
                        pinned.push(*key);
                    }
                }
            }
        }
        let result = self.make_room_evict(bytes);
        if let Some(id) = namespace {
            for key in pinned {
                let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(&key);
            }
        }
        result
    }

    fn make_room_evict(&mut self, bytes: u64) -> bool {
        if self.config.root.is_none() {
            return true;
        }
        let mut compact = true;
        let mut row_children: Option<HashMap<(NamespaceId, Key), u32>> = None;
        while self.charge_total().saturating_add(bytes) > self.config.nvme {
            let counts = row_children.get_or_insert_with(|| {
                let mut counts = HashMap::new();
                for (&id, namespace) in &self.namespaces {
                    for node in namespace.tree.nodes() {
                        if node.row
                            && node.location.durable
                            && let Some(parent) = node.parent
                        {
                            *counts.entry((id, parent)).or_default() += 1;
                        }
                    }
                }
                counts
            });
            let candidate = self.oldest(|id, node| {
                node.location.durable
                    && (node.children() == 0
                        || (compact
                            && node.row
                            && node.parent.is_some()
                            && node.children() == 1
                            && node.location.header.capture_role() == CaptureRole::Endpoint
                            && counts.get(&(id, node.key)).is_some_and(|&n| n > 0)))
            });
            let Some((id, key)) = candidate else {
                if let Some((id, key)) = self.oldest(|id, node| {
                    !node.location.durable
                        && node.children() == 0
                        && self.writable_ancestor(id, node.key, false).is_some()
                }) {
                    self.request_writeback(id, key);
                }
                return false;
            };
            let node = &self.namespaces[&id].tree[&key];
            let row_parent = node.parent.filter(|_| node.row);
            if node.children() == 0 {
                // Cascading removals after this leaf contain only page-only ancestors.
                let removed = self.evict_accounted(id, key);
                if removed == 0 {
                    return false;
                }
                self.metrics.evictions += removed;
            } else {
                let namespace = self.namespaces.get_mut(&id).unwrap();
                let node = &namespace.tree[&key];
                let disk = node
                    .location
                    .disk
                    .as_ref()
                    .expect("durable record has a file");
                let job = self.job_counter + 1;
                let task = DiskJob::Truncate {
                    id: JobId(job),
                    path: disk.path.clone(),
                    length: node.location.header.row_segment,
                };
                if namespace.disk.as_ref().unwrap().submit(task).is_err() {
                    compact = false;
                    continue;
                }
                self.job_counter = job;
                self.jobs.insert(job, Job::Compact { namespace: id, key });
                namespace.tree.pin(&key).unwrap();
                namespace.tree.drop_row(&key);
                let location = &mut namespace.tree[&key].location;
                location.writing = true;
                location.job = Some(job);
                self.metrics.compactions += 1;
                self.delta(id, CatalogKind::Changed, key);
            }
            if let Some(n) = row_parent.and_then(|p| counts.get_mut(&(id, p))) {
                *n -= 1;
            }
        }
        true
    }

    // ----------------------------------------------------------- Window grants

    fn pump_grants(&mut self) {
        let mut ops: Vec<_> = self
            .sessions
            .iter()
            .flat_map(|(connection, session)| {
                session.ops.iter().map(move |(transfer, open)| {
                    (
                        matches!(open.operation, Operation::Put(_)),
                        *connection,
                        *transfer,
                    )
                })
            })
            .collect();
        ops.sort_unstable();
        for (put, connection, transfer) in ops {
            if put {
                while self.send_put_grant(connection, transfer) {}
            } else {
                if let Some(namespace) =
                    get_ref(&self.sessions, connection, transfer).map(|get| get.namespace)
                {
                    self.prepare_get_records(connection, transfer, namespace);
                }
                self.send_get_meta(connection, transfer);
                while self.send_get_grant(connection, transfer) {}
            }
        }
    }

    fn send_put_grant(&mut self, connection: u64, transfer: u64) -> bool {
        let prepared = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return false;
            };
            if put.cancelled
                || put.terminal
                || put.requested == 0
                || put.windows.next >= put.grants.len()
            {
                return false;
            }
            let index = put.windows.next as u32;
            let plan = put.grants[put.windows.next];
            let memory = put.memory.clone();
            let hold = put.staging_hold.clone();
            let hold_unit = put.hold_unit;
            let slot = if memory.is_some() {
                None
            } else {
                put.hold_slots.iter().position(Option::is_none)
            };
            let packed = put.header.packed.is_some();
            (index, plan, memory, hold, hold_unit, slot, packed)
        };
        let (index, plan, memory, hold, hold_unit, slot, packed) = prepared;
        if memory.is_none() && slot.is_none() {
            // Both hold slots are still live; retry once one is released.
            return false;
        }
        let (region, extent, within) = match &memory {
            Some(memory) => (Region::Arena, memory.clone(), plan.record_offset as usize),
            None => {
                let Some(hold) = hold else {
                    self.fail_put(
                        connection,
                        transfer,
                        proto::Reason::Invalid as u8,
                        invalid("missing staging hold"),
                    );
                    return false;
                };
                (Region::Staging, hold, slot.unwrap() * hold_unit)
            }
        };
        let result = self.local.offset(region, &extent, within, plan.length, packed);
        match result {
            Ok(offset) => {
                if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer) {
                    put.windows.next += 1;
                    put.requested -= 1;
                    if let Some(slot) = slot {
                        put.hold_slots[slot] = Some(HoldSlot { index, job: None });
                    }
                    put.windows.granted.insert(index, plan);
                    put.outbox.push_back(Ok(PutAnswer::Grant(plan.window(
                        transfer,
                        index,
                        region,
                        offset,
                        extent,
                        &put.header,
                    ))));
                    *progress = Instant::now();
                }
                true
            }
            Err(error) => {
                self.fail_put(connection, transfer, proto::Reason::Invalid as u8, error);
                false
            }
        }
    }

    fn fail_put(&mut self, connection: u64, transfer: u64, reason: u8, error: io::Error) {
        self.metrics.puts_error += 1;
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
            put.terminal = true;
            // Ownership is ending; never submit unsent chunks afterwards.
            put.blocked.clear();
            put.outbox.push_back(Ok(PutAnswer::Refused(Refused {
                reason,
                detail: error.to_string(),
            })));
            if !put.finished && put.pending == 0 {
                put.finished = true;
                let id = put.namespace;
                let pins = std::mem::take(&mut put.pins);
                let key = put.header.key;
                let reservation = put.reservation;
                let namespace = self.namespaces.get_mut(&id).unwrap();
                for key in &pins {
                    let _ = namespace.tree.unpin(key);
                }
                namespace.inflight.remove(&key);
                self.charged = self.charged.saturating_sub(reservation);
            }
        }
    }

    fn send_get_meta(&mut self, connection: u64, transfer: u64) -> bool {
        let mut progressed = false;
        loop {
            let Some((get, progress)) = get_mut(&mut self.sessions, connection, transfer) else {
                return progressed;
            };
            if get.cancelled || get.terminal || get.next_meta >= get.records.len() {
                return progressed;
            }
            let record = get.next_meta as u32;
            if get.records[get.next_meta].meta_sent {
                get.next_meta += 1;
                continue;
            }
            let Some(metadata) = get.records[get.next_meta].meta.clone() else {
                return progressed;
            };
            let source = if get.records[get.next_meta].memory.is_some() {
                flare_selene::shared::Tier::Dram as i32
            } else {
                flare_selene::shared::Tier::Nvme as i32
            };
            let header = match Header::decode(&metadata) {
                Ok(header) => header,
                Err(error) => {
                    self.fail_get(
                        connection,
                        transfer,
                        proto::Reason::Io as u8,
                        invalid(error.to_string()),
                    );
                    return progressed;
                }
            };
            let sum_count = if header.packed.is_some() {
                0
            } else {
                header.tail_pages as usize
            };
            let Some(table) = metadata.get(4096..4096 + sum_count * 8) else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("incomplete checksum table"),
                );
                return progressed;
            };
            let pages = table
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            get.outbox.push_back(Ok(GetAnswer::Meta {
                record,
                pages,
                row: header.row_checksum,
                source,
            }));
            get.records[get.next_meta].meta_sent = true;
            get.next_meta += 1;
            *progress = Instant::now();
            progressed = true;
        }
    }

    fn send_get_grant(&mut self, connection: u64, transfer: u64) -> bool {
        let Some(ready) = get_ref(&self.sessions, connection, transfer).map(|get| {
            (
                get.namespace,
                get.next_grant,
                get.grants.get(get.next_grant).copied(),
            )
        }) else {
            return false;
        };
        let (namespace, position, Some(plan)) = ready else {
            return false;
        };
        enum Next {
            Stop,
            Fail(proto::Reason, io::Error),
            Lease { memory: Arc<Extent>, packed: bool },
            Read { file: Arc<File> },
        }
        let action = {
            let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                return false;
            };
            if get.cancelled
                || get.terminal
                || get.requested == 0
                || !get.records[plan.record as usize].meta_sent
            {
                Next::Stop
            } else {
                let record = &get.records[plan.record as usize];
                if let Some(memory) = &record.memory {
                    if get.window_reads > 0 {
                        // Windows are answered in grant order. A streamed window is
                        // answered when its read completes, and the namespace's one
                        // reads thread completes reads in the order they were
                        // submitted, so a resident window, which needs no read, waits
                        // until every streamed window granted before it is answered.
                        Next::Stop
                    } else {
                        Next::Lease {
                            memory: memory.clone(),
                            packed: record.header.packed.is_some(),
                        }
                    }
                } else if let Some(file) = &record.file {
                    Next::Read { file: file.clone() }
                } else {
                    Next::Fail(proto::Reason::Io, invalid("record file closed"))
                }
            }
        };
        let schedule = match action {
            Next::Stop => return false,
            Next::Fail(reason, error) => {
                self.fail_get(connection, transfer, reason as u8, error);
                return false;
            }
            Next::Lease { memory, packed } => {
                let offset = match self.local.offset(
                    Region::Arena,
                    &memory,
                    plan.record_offset as usize,
                    plan.length,
                    packed,
                ) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.fail_get(connection, transfer, proto::Reason::Invalid as u8, error);
                        return false;
                    }
                };
                let Some((get, progress)) = get_mut(&mut self.sessions, connection, transfer)
                else {
                    return false;
                };
                get.next_grant += 1;
                get.requested -= 1;
                get.outbox.push_back(Ok(GetAnswer::Ready(plan.window(
                    transfer,
                    position as u32,
                    Region::Arena,
                    offset,
                    memory,
                    &get.records[plan.record as usize].header,
                ))));
                *progress = Instant::now();
                return true;
            }
            Next::Read { file } => {
                let length = plan.read_length();
                // Restore windows are transient staging traffic: reads do not
                // grow the durable set, so they are bounded by the staging
                // region alone and never by the NVMe budget.
                let Some(extent) = self.staging.allocate(length) else {
                    // A preceding window can occupy the size class this one
                    // needs. Retry after release without consuming its credit.
                    // `begin_get` refuses a Get with a window that staging can
                    // never hold, so only a shortage that a release ends gets here.
                    debug_assert!(
                        self.staging_can_ever_hold(length),
                        "begin_get admitted a window larger than staging"
                    );
                    return false;
                };
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.next_grant += 1;
                    get.requested -= 1;
                    get.window_reads += 1;
                    get.pending_reads += 1;
                }
                Some((namespace, file, extent))
            }
        };
        let Some((namespace, file, extent)) = schedule else {
            self.fail_get(
                connection,
                transfer,
                proto::Reason::Io as u8,
                invalid("record file closed"),
            );
            return false;
        };
        self.job_counter += 1;
        let job = self.job_counter;
        let task = DiskJob::ReadRecord {
            id: JobId(job),
            file,
            span: Span {
                address: extent.bytes().as_ptr() as usize,
                offset: plan.record_offset - plan.read_prefix() as u64,
                length: extent.bytes().len(),
            },
        };
        self.note_read_submitted(namespace);
        self.jobs.insert(
            job,
            Job::GetRead {
                namespace,
                connection,
                transfer,
                record: plan.record,
                kind: ReadKind::Window {
                    grant: position as u32,
                },
                buffer: extent,
            },
        );
        self.submit_read(namespace, job, task);
        true
    }

    fn fail_get(&mut self, connection: u64, transfer: u64, reason: u8, error: io::Error) {
        self.metrics.gets_error += 1;
        if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
            get.terminal = true;
            get.cancelled = true;
            get.outbox.push_back(Ok(GetAnswer::Refused(Refused {
                reason,
                detail: error.to_string(),
            })));
        }
    }

    /// Count a read for `namespace`'s reads thread. A read that finds the thread idle starts
    /// its progress clock, so a long-idle thread is not mistaken for a hung one.
    fn note_read_submitted(&mut self, namespace: NamespaceId) {
        let namespace = self.namespaces.get_mut(&namespace).unwrap();
        if namespace.reads_pending == 0 {
            namespace.read_progress = Instant::now();
        }
        namespace.reads_pending += 1;
    }

    fn submit_read(&mut self, namespace: NamespaceId, job: u64, task: DiskJob) {
        use std::sync::mpsc::TrySendError::{Disconnected, Full};
        // A namespace has one reads thread, which runs its jobs one at a time in the order
        // they were submitted and reports them in that order. Streamed windows are answered
        // when their reads complete, so submitting a namespace's reads in grant order is
        // what keeps a restore's windows in order. A read therefore queues behind any
        // backlogged read of its namespace instead of taking a free queue slot ahead of it.
        if self.read_backlog.iter().any(|(id, ..)| *id == namespace) {
            self.read_backlog.push_back((namespace, job, task));
            return;
        }
        let result = match &self.namespaces[&namespace].reads {
            Some(worker) => worker.try_submit(task),
            None => Err(Disconnected(task)),
        };
        match result {
            Ok(()) => {}
            Err(Full(task)) => self.read_backlog.push_back((namespace, job, task)),
            Err(Disconnected(_)) => {
                if let Some(Job::GetRead {
                    namespace,
                    connection,
                    transfer,
                    ..
                }) = self.jobs.remove(&job)
                {
                    let namespace = self.namespaces.get_mut(&namespace).unwrap();
                    namespace.reads_pending = namespace.reads_pending.saturating_sub(1);
                    self.read_failed(connection, transfer);
                }
            }
        }
    }

    // --------------------------------------------------------------- Put frames

    fn window_filled(
        &mut self,
        connection: u64,
        transfer: u64,
        index: u32,
        page_xxh3: Vec<u64>,
        row_xxh3: Option<u64>,
    ) -> Result<()> {
        struct Prepared {
            plan: PlannedGrant,
            memory: Option<Arc<Extent>>,
            hold: Option<Arc<Extent>>,
            within: usize,
            header: Header,
            first_page: u32,
            temporary: Option<Arc<DiskExtent>>,
            streaming: bool,
        }
        let prepared = {
            let Some(open) = self
                .sessions
                .get(&connection)
                .and_then(|session| session.ops.get(&transfer))
            else {
                return Err(invalid("unknown Put transfer"));
            };
            let Operation::Put(put) = &open.operation else {
                return Err(invalid("transfer is not a Put"));
            };
            if put.cancelled || put.terminal {
                return Ok(());
            }
            let slot = put
                .hold_slots
                .iter()
                .position(|slot| slot.is_some_and(|slot| slot.index == index));
            put.windows
                .granted
                .get(&index)
                .copied()
                .map(|plan| Prepared {
                    plan,
                    memory: put.memory.clone(),
                    hold: put.staging_hold.clone(),
                    within: slot.map_or(0, |slot| slot * put.hold_unit),
                    header: put.header.clone(),
                    first_page: put.first_page,
                    temporary: put.temporary.clone(),
                    streaming: put.streaming,
                })
        };
        let Some(prepared) = prepared else {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("filled for an ungranted window"),
            );
            return Ok(());
        };
        let plan = prepared.plan;
        let is_packed = prepared.header.packed.is_some();
        let bad_shape = if is_packed {
            !page_xxh3.is_empty() || row_xxh3.is_some() || plan.pages > 64
        } else {
            page_xxh3.len() != plan.pages as usize
                || (plan.row && row_xxh3.is_none())
                || (!plan.row && row_xxh3.is_some())
                || plan.pages > 64
        };
        if bad_shape {
            self.abort_put_window(connection, transfer, index);
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("window hash shape"),
            );
            return Ok(());
        }
        // The window bytes stay in the arena extent or staging hold; a chunk
        // keeps this slice alive until its write completes, so no copy is
        // buffered in the manager.
        let data = if let Some(memory) = &prepared.memory {
            ExtentSlice::new(
                memory.clone(),
                plan.record_offset as usize,
                plan.length as usize,
            )
        } else if let Some(hold) = &prepared.hold {
            ExtentSlice::new(hold.clone(), prepared.within, plan.length as usize)
        } else {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("window vanished"),
            );
            return Ok(());
        };
        let bytes = data.bytes();
        let mut verified = Vec::with_capacity(page_xxh3.len());
        if let Some(packed) = &prepared.header.packed {
            let first = if plan.pages == 0 {
                packed.page_count()
            } else {
                plan.page_first.saturating_sub(packed.offer().first_page) as usize
            };
            let ok = packed
                .units(first, plan.pages as usize, plan.row)
                .is_some_and(|units| flare_selene::shared::packed::verify_units(units, bytes).is_ok());
            if !ok {
                self.abort_put_window(connection, transfer, index);
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("packed window checksum"),
                );
                return Ok(());
            }
        } else {
            let page_bytes = prepared.header.page_record_bytes() as usize;
            for (position, hash) in page_xxh3.iter().enumerate() {
                let start = position * page_bytes;
                if flare_selene::shared::record::checksum(&bytes[start..start + page_bytes]) != *hash
                {
                    self.abort_put_window(connection, transfer, index);
                    self.fail_put(
                        connection,
                        transfer,
                        proto::Reason::Invalid as u8,
                        invalid("page checksum"),
                    );
                    return Ok(());
                }
                verified.push(*hash);
            }
            if plan.row {
                let row_bytes = prepared.header.row_record_bytes() as usize;
                let start = bytes.len() - row_bytes;
                if flare_selene::shared::record::checksum(&bytes[start..]) != row_xxh3.unwrap() {
                    self.abort_put_window(connection, transfer, index);
                    self.fail_put(
                        connection,
                        transfer,
                        proto::Reason::Invalid as u8,
                        invalid("row checksum"),
                    );
                    return Ok(());
                }
            }
        }
        {
            let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer) else {
                return Ok(());
            };
            put.windows.granted.remove(&index);
            // The slot stays occupied until the chunk write drains.
            if is_packed {
                put.windows.pages_filled += plan.pages;
                if plan.row {
                    put.windows.row_filled = true;
                }
            } else {
                for (position, hash) in verified.iter().enumerate() {
                    let relative =
                        (plan.page_first - u64::from(prepared.first_page)) as usize + position;
                    put.windows.pages[relative] = Some(*hash);
                    put.windows.pages_filled += 1;
                }
                if plan.row {
                    put.windows.row = row_xxh3;
                    put.windows.row_filled = true;
                }
            }
            *progress = Instant::now();
        }
        if prepared.streaming {
            let Some(temporary) = prepared.temporary else {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("missing temporary file"),
                );
                return Ok(());
            };
            let Some(namespace) =
                put_ref(&self.sessions, connection, transfer).map(|put| put.namespace)
            else {
                return Ok(());
            };
            self.job_counter += 1;
            let job = self.job_counter;
            if self.namespaces[&namespace].disk.is_none() {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("no disk"),
                );
                return Ok(());
            }
            match self.write_chunk(
                namespace,
                job,
                connection,
                transfer,
                temporary,
                plan.record_offset,
                data,
            ) {
                Ok(()) => {
                    if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
                        put.pending += 1;
                        set_slot_job(put, index, job);
                    }
                    #[cfg(any(test, feature = "test-support"))]
                    self.config.faults.checkpoint("data");
                }
                Err(data) => {
                    if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
                        put.blocked.push_back((job, plan.record_offset, data));
                        set_slot_job(put, index, job);
                    }
                }
            }
        }
        Ok(())
    }

    fn abort_put_window(&mut self, connection: u64, transfer: u64, index: u32) {
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
            if let Some(slot) = put
                .hold_slots
                .iter_mut()
                .find(|slot| slot.is_some_and(|slot| slot.index == index))
            {
                *slot = None;
            }
        }
    }

    fn put_commit(
        &mut self,
        connection: u64,
        transfer: u64,
        metadata_offset: u64,
        metadata: Vec<u8>,
    ) -> Result<()> {
        enum Action {
            None,
            Fail(proto::Reason, io::Error),
            Publish,
        }
        let namespace = put_ref(&self.sessions, connection, transfer).map(|put| put.namespace);
        let Some(namespace) = namespace else {
            return Err(invalid("unknown Put transfer"));
        };
        let descriptor = self.namespaces[&namespace].descriptor.clone();
        let action = {
            let Some(open) = self
                .sessions
                .get_mut(&connection)
                .and_then(|session| session.ops.get_mut(&transfer))
            else {
                return Err(invalid("unknown Put transfer"));
            };
            let Operation::Put(put) = &mut open.operation else {
                return Err(invalid("transfer is not a Put"));
            };
            if put.cancelled || put.terminal || put.committed {
                return Ok(());
            }
            if !put.owns_payload() {
                Action::Fail(
                    proto::Reason::Invalid,
                    invalid("commit before every payload range"),
                )
            } else {
                let header_page_segment = put.header.page_segment as usize;
                let end = put.commit_bytes.len().saturating_add(metadata.len());
                let align_ok = metadata.len().is_multiple_of(4096)
                    || (put.header.packed.is_some() && end == header_page_segment);
                if metadata_offset as usize != put.commit_bytes.len()
                    || metadata.is_empty()
                    || !align_ok
                    || end > header_page_segment
                    || !(metadata_offset as usize).is_multiple_of(4096)
                {
                    Action::Fail(proto::Reason::Invalid, invalid("commit offset or length"))
                } else {
                    put.commit_bytes.extend_from_slice(&metadata);
                    if put.commit_bytes.len() < header_page_segment {
                        Action::None
                    } else if let Err(error) = validate_commit(put, &descriptor) {
                        Action::Fail(proto::Reason::Invalid, error)
                    } else {
                        if let Some(memory) = &mut put.memory {
                            let bytes = Arc::get_mut(memory).unwrap().bytes_mut();
                            bytes[..header_page_segment]
                                .copy_from_slice(&put.commit_bytes[..header_page_segment]);
                            put.committed = true;
                            Action::Publish
                        } else {
                            match Arena::new(header_page_segment.next_power_of_two()) {
                                Ok(arena) => {
                                    let mut metadata = arena.allocate(header_page_segment).unwrap();
                                    metadata.bytes_mut()[..header_page_segment]
                                        .copy_from_slice(&put.commit_bytes[..header_page_segment]);
                                    put.metadata = Some(Arc::new(metadata));
                                    put.commit_bytes = Vec::new();
                                    put.committed = true;
                                    Action::None
                                }
                                Err(error) => Action::Fail(proto::Reason::Io, error),
                            }
                        }
                    }
                }
            }
        };
        match action {
            Action::None => {}
            Action::Fail(reason, error) => {
                self.fail_put(connection, transfer, reason as u8, error);
            }
            Action::Publish => {
                self.publish_put(connection, transfer, None);
            }
        }
        if let Some(open) = self
            .sessions
            .get_mut(&connection)
            .and_then(|session| session.ops.get_mut(&transfer))
        {
            open.progress = Instant::now();
        }
        Ok(())
    }

    fn publish_target(&self, id: NamespaceId, header: &Header) -> PublishTarget {
        let namespace = &self.namespaces[&id];
        let Some(_) = namespace.tree.node(&header.key) else {
            return PublishTarget::Insert;
        };
        PublishTarget::Replace
    }

    fn publish_put(
        &mut self,
        connection: u64,
        transfer: u64,
        disk: Option<Arc<DiskExtent>>,
    ) {
        let Some((id, header)) = put_ref(&self.sessions, connection, transfer)
            .filter(|put| !put.finished)
            .map(|put| (put.namespace, put.header.clone()))
        else {
            return;
        };
        let target = self.publish_target(id, &header);
        let prepared = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return;
            };
            if put.finished {
                return;
            }
            let header = put.header.clone();
            let memory = put.memory.clone();
            let bytes = if let Some(memory) = &memory {
                memory.capacity() as u64
            } else {
                header.length(put.row)
            };
            let reservation = put.reservation;
            let pins = std::mem::take(&mut put.pins);
            let record_bytes = header.length(put.row);
            let key = header.key;
            let already_terminal = put.terminal;
            put.terminal = true;
            (
                header,
                put.row,
                memory,
                bytes,
                reservation,
                pins,
                key,
                already_terminal,
                record_bytes,
            )
        };
        let (header, row, memory, bytes, reservation, pins, key, already_terminal, record_bytes) =
            prepared;
        let replace = matches!(target, PublishTarget::Replace);
        if disk.is_some() {
            let action = self
                .metrics
                .record_actions
                .entry((header.capture_role(), "streamed"))
                .or_default();
            action.0 += 1;
            action.1 += record_bytes;
        }
        let location = Location {
            header: header.clone(),
            memory,
            disk,
            durable: false,
            writing: false,
            job: None,
            retry: Instant::now(),
            admitted: Instant::now(),
            bytes,
        };
        let mut rescan = false;
        let inserted = {
            let namespace = self.namespaces.get_mut(&id).unwrap();
            let inserted = if replace {
                let node = namespace.tree.node_mut(&key).unwrap();
                let old = std::mem::replace(&mut node.location, location);
                rescan = old.durable;
                node.row = row;
                self.retired_records.push((id, key, old));
                true
            } else {
                namespace
                    .tree
                    .insert(
                        key,
                        header.parent,
                        header.bucket,
                        header.tail_pages as usize,
                        row,
                        true,
                        location,
                    )
                    .is_ok()
            };
            for key in &pins {
                let _ = namespace.tree.unpin(key);
            }
            namespace.inflight.remove(&key);
            inserted
        };
        if !inserted {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("publish insert"),
            );
            return;
        }
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer)
            && !put.finished
        {
            put.finished = true;
            put.pending = 0;
            if !already_terminal {
                put.outbox.push_back(Ok(PutAnswer::Stored {
                    bytes: record_bytes,
                }));
            }
        }
        self.reused(id, key);
        self.charged = self.charged.saturating_sub(reservation) + bytes;
        self.refresh_durability(id, (!rescan).then_some(key));
        let mut ancestor = header.parent;
        while let Some(key) = ancestor {
            let node = &self.namespaces[&id].tree[&key];
            if node.children() != 1 {
                self.pending_former_tips.remove(&(id, key));
                break;
            }
            if node.row {
                if row {
                    self.prune_former_tip(id, key);
                }
                break;
            }
            ancestor = node.parent;
        }
        self.metrics.puts_ok += 1;
        self.delta(
            id,
            if replace {
                CatalogKind::Changed
            } else {
                CatalogKind::Added
            },
            key,
        );
    }

    /// Whether a block for `length` bytes fits in the whole staging region, which
    /// hands out power-of-two blocks. A block that does not fit is never available,
    /// however many windows are released.
    fn staging_can_ever_hold(&self, length: usize) -> bool {
        length
            .max(4096)
            .checked_next_power_of_two()
            .is_some_and(|bytes| bytes <= self.staging.capacity())
    }

    fn prune_former_tip(&mut self, id: NamespaceId, parent: Key) {
        self.pending_former_tips.remove(&(id, parent));
        let namespace = self.namespaces.get_mut(&id).unwrap();
        let Some(node) = namespace.tree.node(&parent) else {
            return;
        };
        if node.parent.is_none()
            || !node.row
            || node.location.header.capture_role() != CaptureRole::Endpoint
            || node.location.disk.is_some()
            || !namespace.tree.has_row_below(&parent, |n| n.ready && n.row)
        {
            return;
        }
        let node = &mut namespace.tree[&parent];
        // A claim keeps the row shape and the extent until its last window drains.
        if node.pinned()
            || node.location.reading()
            || node.location.disk_busy()
            || namespace.inflight.contains(&parent)
        {
            self.pending_former_tips.insert((id, parent));
            return;
        }
        if let Some(extent) = node.location.memory.as_mut().and_then(Arc::get_mut) {
            extent.truncate(node.location.header.row_segment as usize);
            let bytes = extent.capacity() as u64;
            self.charged -= node.location.bytes - bytes;
            node.location.bytes = bytes;
        }
        node.row = false;
        let action = self
            .metrics
            .record_actions
            .entry((CaptureRole::Endpoint, "former_tip"))
            .or_default();
        action.0 += 1;
        action.1 += node.location.header.row_record_bytes();
        self.metrics.compactions += 1;
        self.delta(id, CatalogKind::Changed, parent);
    }

    fn begin_get(&mut self, pending: PendingGet) {
        let PendingGet {
            connection,
            transfer,
            request,
            events,
        } = pending;
        let Some(id) = self.sessions.get(&connection).map(|_| request.namespace) else {
            return;
        };
        let parsed = self.plan_get(id, &request);
        let (records, grants) = match parsed {
            Ok(value) => value,
            Err(error) => {
                self.metrics.gets_error += 1;
                let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                    reason: proto::Reason::Invalid as u8,
                    detail: error.to_string(),
                })));
                return;
            }
        };
        // The metadata and the windows of a record that is not resident are read into
        // staging. A block larger than staging can never be read, in practice the
        // endpoint's row window, so the Get is refused before it reads anything. The
        // reason is BUDGET, which the pool reports to the client as NO_WINDOWS
        // (`pool/claims.rs`). Every other shortage is temporary: the read waits for a
        // release, so no refusal can follow once the Get has started.
        let unservable = records.iter().any(|record| {
            record.memory.is_none()
                && !self.staging_can_ever_hold(
                    usize::try_from(record.header.page_segment).unwrap_or(usize::MAX),
                )
        }) || grants.iter().any(|grant| {
            records[grant.record as usize].memory.is_none()
                && !self.staging_can_ever_hold(grant.read_length())
        });
        if unservable {
            self.metrics.gets_error += 1;
            let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                reason: proto::Reason::Budget as u8,
                detail: "staging budget".into(),
            })));
            return;
        }
        let pages: Vec<Key> = request.keys.clone();
        {
            let namespace = self.namespaces.get_mut(&id).unwrap();
            for key in &pages {
                if namespace.tree.pin(key).is_err() {
                    let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                        reason: proto::Reason::Invalid as u8,
                        detail: "pin".into(),
                    })));
                    return;
                }
            }
        }
        for key in pages.iter().rev().skip(1).rev() {
            self.clock += 1;
            self.namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .access_at(key, self.clock);
        }
        if let Some(key) = pages.last() {
            self.reused(id, *key);
        }
        let get = Get {
            namespace: id,
            pins: pages.into(),
            records,
            grants,
            requested: 0,
            next_grant: 0,
            next_meta_read: 0,
            next_meta: 0,
            window_reads: 0,
            pending_reads: 0,
            cancelled: false,
            terminal: false,
            finished: false,
            events,
            outbox: VecDeque::new(),
        };
        let mut get = get;
        get.outbox.push_back(Ok(GetAnswer::Leased {
            packed: get
                .records
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    r.header.packed.as_ref().map(|p| {
                        let skip = if i == 0 {
                            request.skip_pages as usize
                        } else {
                            0
                        };
                        let row = i + 1 == get.records.len() && request.row;
                        flare_selene::shared::wire::PackedOffer {
                            first_page: p.offer().first_page + skip as u64,
                            units: p.units(skip, p.page_count() - skip, row).unwrap().to_vec(),
                        }
                    })
                })
                .collect(),
            transfer,
            windows: get.grants.len(),
        }));
        self.sessions.get_mut(&connection).unwrap().ops.insert(
            transfer,
            Open {
                progress: Instant::now(),
                operation: Operation::Get(Box::new(get)),
            },
        );
        self.prepare_get_records(connection, transfer, id);
    }

    fn plan_get(
        &mut self,
        id: NamespaceId,
        request: &GetRequest,
    ) -> Result<(Vec<GetRecord>, Vec<PlannedGrant>)> {
        if request.keys.is_empty() || request.keys.len() > MAX_CHAIN || !request.row {
            return Err(invalid("Get node count or row"));
        }
        let namespace = self
            .namespaces
            .get(&id)
            .ok_or_else(|| invalid("Get namespace is not registered"))?;
        let mut total = 0u64;
        let mut records = Vec::new();
        let mut grants = Vec::new();
        for (index, key) in request.keys.iter().enumerate() {
            let node = namespace
                .tree
                .node(key)
                .ok_or_else(|| invalid("Get key missing"))?;
            if index > 0 {
                let previous = request.keys[index - 1];
                let mut parent = node.parent;
                while parent != Some(previous) {
                    let skipped = parent
                        .and_then(|key| namespace.tree.node(&key))
                        .filter(|node| node.tail_pages == 0)
                        .ok_or_else(|| invalid("Get chain"))?;
                    parent = skipped.parent;
                }
            }
            if index + 1 == request.keys.len() && !node.row {
                return Err(invalid("Get chain"));
            }
            if index == 0 && u64::from(request.skip_pages) > node.tail_pages as u64 {
                return Err(invalid("Get skip"));
            }
            total += node.tail_pages as u64;
            let header = node.location.header.clone();
            let skip = if index == 0 { request.skip_pages } else { 0 };
            let first_page = node.parent.map_or(0, |parent| {
                parent.consumed_tokens / namespace.descriptor.page_tokens
            }) as u32;
            grants.extend(record_grants(
                &header,
                first_page,
                skip,
                index + 1 == request.keys.len(),
                DEFAULT_MAX_WINDOW,
                index as u32,
            ));
            records.push(GetRecord {
                memory: node.location.memory.clone(),
                file: node
                    .location
                    .disk
                    .as_ref()
                    .map(|extent| extent.reader())
                    .transpose()?,
                header,
                meta: None,
                meta_sent: false,
            });
        }
        if total - u64::from(request.skip_pages) != u64::from(request.expected_pages) {
            return Err(invalid("Get page count"));
        }
        for record in &records {
            *self
                .metrics
                .hits
                .entry((
                    flare_selene::shared::record::hex(&id),
                    if record.memory.is_some() {
                        "memory"
                    } else {
                        "disk"
                    },
                ))
                .or_default() += 1;
        }
        Ok((records, grants))
    }

    /// Starts the metadata reads of a Get's records, in record order. A record whose
    /// metadata block does not fit staging now keeps its place: the reads after it wait,
    /// because the namespace's reads thread runs reads in submission order and a record's
    /// windows need its metadata, and `pump_grants` resumes here once staging is released.
    fn prepare_get_records(&mut self, connection: u64, transfer: u64, id: NamespaceId) {
        loop {
            let (index, memory, file, header) = {
                let Some(get) = get_ref(&self.sessions, connection, transfer) else {
                    return;
                };
                if get.cancelled || get.terminal {
                    return;
                }
                let Some(record) = get.records.get(get.next_meta_read) else {
                    return;
                };
                (
                    get.next_meta_read,
                    record.memory.clone(),
                    record.file.clone(),
                    record.header.clone(),
                )
            };
            if let Some(memory) = memory {
                let meta = memory.bytes()[..header.page_segment as usize].to_vec();
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.records[index].meta = Some(meta);
                    get.next_meta_read += 1;
                }
                continue;
            }
            let Some(file) = file else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("missing file"),
                );
                return;
            };
            // A record that is not resident is never read whole into the arena. The
            // namespace's one reads thread runs reads in submission order, so a
            // whole-record read here would hold back the first window for the time to
            // read the largest record, row included, and its copy could not overlap the
            // disk. Only the metadata is read now; the windows follow in grant order.
            // Metadata is staging traffic too: a full durable budget must not refuse a
            // restore.
            let meta_length = header.page_segment as usize;
            let Some(buffer) = self.staging.allocate(meta_length) else {
                // Windows of other transfers or of earlier records can occupy the size
                // class this block needs. `begin_get` refuses a Get with a block that
                // staging can never hold, so only a shortage that a release ends gets here.
                debug_assert!(
                    self.staging_can_ever_hold(meta_length),
                    "begin_get admitted metadata larger than staging"
                );
                return;
            };
            self.job_counter += 1;
            let job = self.job_counter;
            let task = DiskJob::ReadRecord {
                id: JobId(job),
                file,
                span: Span {
                    address: buffer.bytes().as_ptr() as usize,
                    offset: 0,
                    length: buffer.bytes().len(),
                },
            };
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads += 1;
                get.next_meta_read += 1;
            }
            self.note_read_submitted(id);
            self.jobs.insert(
                job,
                Job::GetRead {
                    namespace: id,
                    connection,
                    transfer,
                    record: index as u32,
                    kind: ReadKind::Meta {
                        length: meta_length,
                    },
                    buffer,
                },
            );
            self.submit_read(id, job, task);
        }
    }

    fn get_read_done(
        &mut self,
        connection: u64,
        transfer: u64,
        record: u32,
        kind: ReadKind,
        data: Option<Extent>,
    ) {
        let terminal = self
            .sessions
            .get(&connection)
            .and_then(|session| session.ops.get(&transfer))
            .is_none_or(|open| {
                matches!(&open.operation, Operation::Get(get) if get.terminal || get.finished)
            });
        if terminal {
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads = get.pending_reads.saturating_sub(1);
                if matches!(kind, ReadKind::Window { .. }) {
                    get.window_reads = get.window_reads.saturating_sub(1);
                }
            }
            return;
        }
        let Some(extent) = data else {
            self.read_failed(connection, transfer);
            return;
        };
        let Some(in_range) = get_ref(&self.sessions, connection, transfer)
            .map(|get| usize::try_from(record).is_ok_and(|index| index < get.records.len()))
        else {
            return;
        };
        if !in_range {
            self.read_failed(connection, transfer);
            return;
        }
        match kind {
            ReadKind::Meta { length } => {
                let meta = extent.bytes()[..length].to_vec();
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.records[record as usize].meta = Some(meta);
                    get.pending_reads -= 1;
                }
            }
            ReadKind::Window { grant } => {
                let (plan, packed) = {
                    let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                        return;
                    };
                    get.pending_reads = get.pending_reads.saturating_sub(1);
                    get.window_reads = get.window_reads.saturating_sub(1);
                    let Some(plan) = get.grants.get(grant as usize) else {
                        return;
                    };
                    (*plan, get.records[plan.record as usize].header.packed.is_some())
                };
                let extent = Arc::new(extent);
                let offset = match self.local.offset(
                    Region::Staging,
                    &extent,
                    plan.read_prefix(),
                    plan.length,
                    packed,
                ) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.fail_get(connection, transfer, proto::Reason::Invalid as u8, error);
                        return;
                    }
                };
                if let Some((get, progress)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.outbox.push_back(Ok(GetAnswer::Ready(plan.window(
                        transfer,
                        grant,
                        Region::Staging,
                        offset,
                        extent,
                        &get.records[plan.record as usize].header,
                    ))));
                    *progress = Instant::now();
                }
            }
        }
    }

    fn read_failed(&mut self, connection: u64, transfer: u64) {
        let terminal = self
            .sessions
            .get(&connection)
            .and_then(|session| session.ops.get(&transfer))
            .is_none_or(|open| {
                matches!(&open.operation, Operation::Get(get) if get.terminal || get.finished)
            });
        if terminal {
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads = get.pending_reads.saturating_sub(1);
                get.window_reads = get.window_reads.saturating_sub(1);
            }
            return;
        }
        if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
            get.pending_reads = get.pending_reads.saturating_sub(1);
            get.window_reads = get.window_reads.saturating_sub(1);
        }
        self.fail_get(
            connection,
            transfer,
            proto::Reason::Io as u8,
            invalid("disk read failed"),
        );
    }

    fn maybe_finish_get(&mut self, connection: u64, transfer: u64) {
        let done = {
            let Some(get) = get_ref(&self.sessions, connection, transfer) else {
                return;
            };
            !get.terminal
                && !get.cancelled
                && get.next_grant >= get.grants.len()
                && get.window_reads == 0
                && get.pending_reads == 0
                && get.next_meta >= get.records.len()
        };
        if done {
            let (namespace, pins) = {
                let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                    return;
                };
                get.terminal = true;
                get.finished = true;
                get.outbox.push_back(Ok(GetAnswer::Done));
                (get.namespace, std::mem::take(&mut get.pins))
            };
            self.metrics.gets_ok += 1;
            self.release_pins(namespace, pins);
        }
    }

    fn release_pins(&mut self, id: NamespaceId, pins: VecDeque<Key>) {
        let namespace = self.namespaces.get_mut(&id).unwrap();
        for key in pins {
            let _ = namespace.tree.unpin(&key);
        }
    }

    // ----------------------------------------------------------- Cancel, close

    fn cancel(&mut self, connection: u64, transfer: u64) {
        // A Begin that has not reached dispatch yet must be withdrawn too.
        let mut removed_pending = false;
        if let Some(queue) = self.pending_puts.get_mut(&connection) {
            if let Some(index) = queue.iter().position(|put| put.transfer == transfer) {
                queue.remove(index);
                removed_pending = true;
            }
            if queue.is_empty() {
                self.pending_puts.remove(&connection);
                self.put_order.retain(|id| *id != connection);
            }
        }
        if let Some(queue) = self.pending_gets.get_mut(&connection) {
            if let Some(index) = queue.iter().position(|get| get.transfer == transfer) {
                queue.remove(index);
                removed_pending = true;
            }
            if queue.is_empty() {
                self.pending_gets.remove(&connection);
                self.get_order.retain(|id| *id != connection);
            }
        }
        if removed_pending {
            self.metrics.cancels += 1;
            return;
        }
        let Some(open) = self
            .sessions
            .get_mut(&connection)
            .and_then(|session| session.ops.get_mut(&transfer))
        else {
            return;
        };
        match &mut open.operation {
            Operation::Put(put) => {
                if put.committed {
                    return;
                }
                if !put.terminal {
                    self.metrics.cancels += 1;
                }
                put.cancelled = true;
                put.terminal = true;
                put.blocked.clear();
                put.outbox.clear();
            }
            Operation::Get(get) => {
                if !get.terminal {
                    self.metrics.cancels += 1;
                }
                get.cancelled = true;
                get.terminal = true;
                get.outbox.clear();
            }
        }
    }

    fn connection_closed(&mut self, connection: u64) {
        self.local.close(connection);
        self.close_operations(connection);
    }

    fn close_operations(&mut self, connection: u64) {
        self.pending_gets.remove(&connection);
        self.pending_puts.remove(&connection);
        self.get_order.retain(|id| *id != connection);
        self.put_order.retain(|id| *id != connection);
        let Some(session) = self.sessions.get_mut(&connection) else {
            return;
        };
        session.closed = true;
        let transfers: Vec<_> = session.ops.keys().copied().collect();
        for transfer in transfers {
            self.cancel(connection, transfer);
        }
    }

    // ------------------------------------------------------------- Maintenance

    fn flush_outboxes(&mut self) {
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        let mut remove = Vec::new();
        for connection in connections {
            let Some(session) = self.sessions.get_mut(&connection) else {
                continue;
            };
            for (transfer, open) in &mut session.ops {
                match &mut open.operation {
                    Operation::Put(put) => {
                        if session.closed {
                            put.outbox.clear();
                        }
                        while let Some(answer) = put.outbox.pop_front() {
                            match put.events.try_send(answer) {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Full(answer)) => {
                                    put.outbox.push_front(answer);
                                    break;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                            }
                        }
                        // Keep failures until the pool fences the writer and cancels the put.
                        if put.terminal
                            && (put.cancelled || put.committed || session.closed)
                            && put.outbox.is_empty()
                            && put.pending == 0
                        {
                            remove.push((connection, *transfer));
                        }
                    }
                    Operation::Get(get) => {
                        if session.closed {
                            get.outbox.clear();
                        }
                        while let Some(answer) = get.outbox.pop_front() {
                            match get.events.try_send(answer) {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Full(answer)) => {
                                    get.outbox.push_front(answer);
                                    break;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                            }
                        }
                        if get.terminal
                            && get.finished
                            && get.pending_reads == 0
                            && get.outbox.is_empty()
                        {
                            remove.push((connection, *transfer));
                        }
                    }
                }
            }
        }
        for (connection, transfer) in remove {
            if let Some(session) = self.sessions.get_mut(&connection) {
                session.ops.remove(&transfer);
            }
        }
        self.sessions
            .retain(|_, session| !session.closed || !session.ops.is_empty());
    }

    fn settle_terminal_puts(&mut self) {
        let mut ops = Vec::new();
        for (connection, session) in &self.sessions {
            for (transfer, open) in &session.ops {
                if matches!(&open.operation, Operation::Put(put) if put.terminal && !put.finished && put.pending == 0)
                {
                    ops.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in ops {
            self.settle_put(connection, transfer);
        }
    }

    fn cleanup_turn(&mut self) {
        let mut pins = Vec::new();
        for session in self.sessions.values_mut() {
            for open in session.ops.values_mut() {
                if let Operation::Get(get) = &mut open.operation {
                    if get.terminal && !get.finished && get.pending_reads == 0 {
                        get.finished = true;
                        pins.push((get.namespace, std::mem::take(&mut get.pins)));
                    }
                }
            }
        }
        for (namespace, pins) in pins {
            self.release_pins(namespace, pins);
        }
        self.retired_records.retain(|(id, key, location)| {
            let replaced = location.disk.is_none()
                || self.namespaces[id]
                    .tree
                    .node(key)
                    .is_none_or(|node| node.location.durable);
            if !replaced || location.reading() {
                true
            } else {
                self.charged = self.charged.saturating_sub(location.bytes);
                false
            }
        });
    }

    fn disk_events(&mut self) {
        let mut completions = Vec::new();
        for worker in &self.writers {
            worker.poll(&mut completions);
        }
        for namespace in self.namespaces.values() {
            for worker in [&namespace.reads, &namespace.disk].into_iter().flatten() {
                worker.poll(&mut completions);
            }
        }
        if !completions.is_empty() {
            self.disk_progress = Instant::now();
        }
        for completion in completions {
            let failed = completion.result.is_err();
            if failed {
                self.metrics.nvme_errors += 1;
            }
            let Some(job) = self.jobs.remove(&completion.id.0) else {
                continue;
            };
            if matches!(job, Job::GetRead { .. }) {
                self.metrics.nvme_read_bytes += completion.bytes;
            } else {
                self.metrics.nvme_bytes += completion.bytes;
            }
            if completion
                .result
                .as_ref()
                .is_err_and(|error| error.no_space())
                && !matches!(job, Job::GetRead { .. })
                && let Some((id, key)) =
                    self.oldest(|_, node| node.location.durable && node.children() == 0)
            {
                let removed = self.evict_accounted(id, key);
                self.metrics.evictions += removed;
            }
            self.metrics.nvme_seconds += completion.elapsed.as_secs_f64();
            match job {
                Job::Write {
                    namespace,
                    key,
                    reason,
                    ..
                } => {
                    let path = self.namespaces[&namespace].path(key);
                    // Removal keeps a record whose write is in flight (the keep predicate),
                    // so the record is still in the tree when its completion arrives.
                    let node = self
                        .namespaces
                        .get_mut(&namespace)
                        .unwrap()
                        .tree
                        .node_mut(&key)
                        .expect("a record with a disk write in flight is never removed");
                    if node.location.job != Some(completion.id.0) {
                        // The record this write belonged to was replaced. The replacement owns
                        // the key's file now, so the stale completion must not touch either the
                        // node or that path.
                        continue;
                    }
                    node.location.job = None;
                    node.location.writing = false;
                    if !failed {
                        let action = self
                            .metrics
                            .record_actions
                            .entry((node.location.header.capture_role(), reason))
                            .or_default();
                        action.0 += 1;
                        action.1 += completion.bytes;
                        if let Some(path) = path {
                            node.location.disk = Some(Arc::new(DiskExtent::new(path)));
                        }
                        node.location.retry = Instant::now();
                    } else {
                        node.location.retry = Instant::now() + Duration::from_millis(100);
                    }
                    let _ = self
                        .namespaces
                        .get_mut(&namespace)
                        .unwrap()
                        .tree
                        .unpin(&key);
                    self.refresh_durability(namespace, Some(key));
                }
                Job::Chunk {
                    connection,
                    transfer,
                    ..
                } => {
                    if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer)
                    {
                        put.pending = put.pending.saturating_sub(1);
                        release_slot_job(put, completion.id.0);
                        *progress = Instant::now();
                    }
                    if failed {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("streaming disk write failed"),
                        );
                    }
                    self.settle_put(connection, transfer);
                }
                Job::Finish {
                    connection,
                    transfer,
                    final_extent,
                    ..
                } => {
                    if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer)
                    {
                        put.pending = put.pending.saturating_sub(1);
                        *progress = Instant::now();
                    }
                    if failed {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("streaming disk write failed"),
                        );
                        self.settle_put(connection, transfer);
                    } else {
                        final_extent.keep();
                        self.publish_put(connection, transfer, Some(final_extent));
                    }
                }
                Job::Compact { namespace, key } => {
                    let Some(node) = self
                        .namespaces
                        .get_mut(&namespace)
                        .and_then(|namespace| namespace.tree.node_mut(&key))
                    else {
                        continue;
                    };
                    if node.location.job != Some(completion.id.0) {
                        // The record was replaced; its successor owns this path.
                        continue;
                    }
                    node.location.job = None;
                    node.location.writing = false;
                    if !failed {
                        self.release_arena(namespace, key);
                    } else {
                        node.row = true;
                        self.delta(namespace, CatalogKind::Changed, key);
                    }
                    let _ = self
                        .namespaces
                        .get_mut(&namespace)
                        .unwrap()
                        .tree
                        .unpin(&key);
                    self.refresh_durability(namespace, Some(key));
                }
                Job::GetRead {
                    namespace,
                    connection,
                    transfer,
                    record,
                    kind,
                    buffer,
                } => {
                    let namespace = self.namespaces.get_mut(&namespace).unwrap();
                    namespace.reads_pending = namespace.reads_pending.saturating_sub(1);
                    namespace.read_progress = Instant::now();
                    if failed {
                        self.read_failed(connection, transfer);
                    } else {
                        self.get_read_done(connection, transfer, record, kind, Some(buffer));
                    }
                }
            }
        }
    }

    fn settle_put(&mut self, connection: u64, transfer: u64) {
        let released = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return;
            };
            if !put.terminal || put.finished || put.pending > 0 {
                return;
            }
            put.finished = true;
            put.blocked.clear();
            Some((
                put.namespace,
                std::mem::take(&mut put.pins),
                put.header.key,
                put.reservation,
            ))
        };
        let Some((id, pins, key, reservation)) = released else {
            return;
        };
        let namespace = self.namespaces.get_mut(&id).unwrap();
        for key in &pins {
            let _ = namespace.tree.unpin(key);
        }
        namespace.inflight.remove(&key);
        self.charged = self.charged.saturating_sub(reservation);
    }

    fn pump_reads(&mut self) {
        for (namespace, job, task) in std::mem::take(&mut self.read_backlog) {
            self.submit_read(namespace, job, task);
        }
    }

    fn write_chunk(
        &mut self,
        namespace: NamespaceId,
        job: u64,
        connection: u64,
        transfer: u64,
        temporary: Arc<DiskExtent>,
        offset: u64,
        data: ExtentSlice,
    ) -> std::result::Result<(), ExtentSlice> {
        let task = DiskJob::write_record(
            JobId(job),
            temporary.path.clone(),
            offset,
            data.bytes(),
            None,
        );
        let worker = self.namespaces[&namespace].disk.as_ref().unwrap();
        if let Err(error) = worker.try_submit(task) {
            if matches!(error, std::sync::mpsc::TrySendError::Disconnected(_)) {
                let error = invalid("disk worker stopped");
                self.fail_put(connection, transfer, proto::Reason::Io as u8, error);
            }
            return Err(data);
        }
        self.jobs.insert(
            job,
            Job::Chunk {
                connection,
                transfer,
                _data: data,
                _temporary: temporary,
            },
        );
        Ok(())
    }

    fn pump_chunks(&mut self) {
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        for connection in connections {
            let transfers: Vec<u64> = self.sessions[&connection]
                .ops
                .iter()
                .filter(|(_, open)| matches!(open.operation, Operation::Put(_)))
                .map(|(transfer, _)| *transfer)
                .collect();
            for transfer in transfers {
                loop {
                    let next = {
                        let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer)
                        else {
                            break;
                        };
                        if put.cancelled || put.terminal {
                            break;
                        }
                        let Some((job, offset, data)) = put.blocked.pop_front() else {
                            break;
                        };
                        put.temporary
                            .clone()
                            .map(|temporary| (job, offset, data, temporary))
                    };
                    let Some((job, offset, data, temporary)) = next else {
                        break;
                    };
                    let namespace =
                        put_ref(&self.sessions, connection, transfer).map(|put| put.namespace);
                    let Some(namespace) = namespace else {
                        break;
                    };
                    if self.namespaces[&namespace].disk.is_none() {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("no disk"),
                        );
                        break;
                    }
                    match self.write_chunk(
                        namespace, job, connection, transfer, temporary, offset, data,
                    ) {
                        Ok(()) => {
                            if let Some((put, _)) =
                                put_mut(&mut self.sessions, connection, transfer)
                            {
                                put.pending += 1;
                            }
                        }
                        Err(data) => {
                            if let Some((put, _)) =
                                put_mut(&mut self.sessions, connection, transfer)
                            {
                                put.blocked.push_front((job, offset, data));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    fn request_writeback(&mut self, id: NamespaceId, key: Key) {
        self.writeback.insert((id, key));
        self.write_ancestor(id, key, "eviction");
    }

    fn writable_ancestor(
        &self,
        id: NamespaceId,
        target: Key,
        allow_pending_prune: bool,
    ) -> Option<Key> {
        let namespace = &self.namespaces[&id];
        let mut node = namespace.tree.node(&target)?;
        while let Some(parent) = node.parent.filter(|p| !namespace.tree[p].location.durable) {
            node = &namespace.tree[&parent];
        }
        let key = node.key;
        if namespace.inflight.contains(&key)
            || (!allow_pending_prune && self.pending_former_tips.contains(&(id, key)))
            || node.location.disk_busy()
            || node.location.retry > Instant::now()
        {
            return None;
        }
        Some(key)
    }

    fn write_ancestor(&mut self, id: NamespaceId, target: Key, reason: &'static str) {
        let namespace = &self.namespaces[&id];
        let Some(node) = namespace.tree.node(&target) else {
            self.writeback.remove(&(id, target));
            return;
        };
        if node.location.durable {
            if node.location.memory.is_none()
                || (!node.pinned() && !node.location.reading() && self.evict_memory(id, target))
            {
                self.writeback.remove(&(id, target));
            }
            return;
        }
        let Some(key) = self.writable_ancestor(id, target, reason != "eviction") else {
            return;
        };
        let node = &namespace.tree[&key];
        let Some(writer) = self.writers.iter().find(|w| w.is_idle()) else {
            return;
        };
        let Some(memory) = node.location.memory.clone() else {
            return;
        };
        let job = self.job_counter + 1;
        let Some(temporary) = namespace.temporary(job) else {
            return;
        };
        let length = node.location.header.length(node.row);
        let path = namespace.path(key).unwrap();
        let task = DiskJob::write_record(
            JobId(job),
            temporary.path.clone(),
            0,
            &memory.bytes()[..length as usize],
            Some((path, length)),
        );
        if writer.submit(task).is_err() {
            return;
        }
        self.job_counter = job;
        self.jobs.insert(
            job,
            Job::Write {
                namespace: id,
                key,
                reason: if key == target { reason } else { "dependency" },
                _memory: memory,
                _temporary: temporary,
            },
        );
        let tree = &mut self.namespaces.get_mut(&id).unwrap().tree;
        tree.pin(&key).expect("writeback holds a resident record");
        let location = &mut tree[&key].location;
        location.writing = true;
        location.job = Some(job);
    }

    fn schedule(&mut self) {
        for (id, key) in self.writeback.iter().copied().collect::<Vec<_>>() {
            self.write_ancestor(id, key, "eviction");
        }
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        for connection in connections {
            let ops: Vec<(u64, NamespaceId)> = self.sessions[&connection]
                .ops
                .iter()
                .filter_map(|(transfer, open)| match &open.operation {
                    Operation::Put(put)
                        if put.committed
                            && put.metadata.is_some()
                            && put.pending == 0
                            && put.blocked.is_empty()
                            && put.temporary.is_some()
                            && !put.terminal
                            && !put.finalizing =>
                    {
                        Some((*transfer, put.namespace))
                    }
                    _ => None,
                })
                .collect();
            for (transfer, id) in ops {
                self.schedule_finish(connection, transfer, id);
            }
        }
    }

    fn schedule_finish(&mut self, connection: u64, transfer: u64, id: NamespaceId) {
        let Some((parent, terminal)) = put_ref(&self.sessions, connection, transfer)
            .map(|put| (put.header.parent, put.terminal))
        else {
            return;
        };
        if terminal {
            return;
        }
        if !self.namespaces[&id].parent_durable(parent) {
            self.write_ancestor(id, parent.unwrap(), "dependency");
            return;
        }
        let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
            return;
        };
        self.job_counter += 1;
        let job = self.job_counter;
        let Some(path) = self.namespaces[&id].path(put.header.key) else {
            return;
        };
        let Some((task, completion)) = finish_record(put, job, connection, transfer, path) else {
            return;
        };
        let Some(worker) = &self.namespaces[&id].disk else {
            return;
        };
        if worker.submit(task).is_ok() {
            put.finalizing = true;
            put.pending += 1;
            self.jobs.insert(job, completion);
        }
    }

    fn tick(&mut self) {
        self.note_reserved();
        let now = Instant::now();
        if self.jobs.is_empty() {
            self.disk_progress = now;
        }
        self.timeouts();
        if self.config.max_age.is_zero() {
            return;
        }
        let ids: Vec<_> = self.namespaces.keys().copied().collect();
        for id in ids {
            let keys = self.namespaces.get_mut(&id).unwrap().tree.expired(|node| {
                !node.pinned() && !node.location.disk_busy() && !node.location.reading()
            });
            for key in keys {
                let node = &self.namespaces[&id].tree[&key];
                let tier = usize::from(node.location.disk.is_some());
                let removed = self.evict_accounted(id, key);
                self.metrics.expired[tier] += removed;
            }
        }
    }

    fn timeouts(&mut self) {
        let mut expired = Vec::new();
        let disk_idle = (!self.jobs.is_empty()).then(|| self.disk_progress.elapsed());
        // A restore waits on its own namespace's reads thread, not on the disk as a whole:
        // other namespaces' jobs completing say nothing about a thread that is hung.
        let read_idle: HashMap<NamespaceId, Duration> = self
            .namespaces
            .iter()
            .filter(|(_, namespace)| namespace.reads_pending > 0)
            .map(|(id, namespace)| (*id, namespace.read_progress.elapsed()))
            .collect();
        for (connection, session) in &mut self.sessions {
            if session.closed {
                continue;
            }
            for (transfer, open) in &mut session.ops {
                if open.is_over() {
                    continue;
                }
                if open.waits_for_grant() {
                    open.progress = Instant::now();
                    continue;
                }
                let delivered = open.waits_on_disk();
                let idle = match &open.operation {
                    Operation::Put(_) => disk_idle,
                    Operation::Get(get) => read_idle.get(&get.namespace).copied(),
                };
                if stalled(open.progress.elapsed(), delivered, idle) {
                    expired.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in expired {
            let is_put = self
                .sessions
                .get(&connection)
                .and_then(|session| session.ops.get(&transfer))
                .is_some_and(|open| matches!(open.operation, Operation::Put(_)));
            if is_put {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("no progress"),
                );
            } else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("no progress"),
                );
            }
        }
    }

    pub(crate) fn drain_until(&mut self, deadline: Instant) -> bool {
        // The pool releases proven-safe holders. A forced stop must preserve the others.
        for connection in self.sessions.keys().copied().collect::<Vec<_>>() {
            self.close_operations(connection);
        }
        if self.config.root.is_none() {
            return true;
        }
        self.writeback.clear();
        let mut targets = self.shutdown_targets();
        let mut complete = false;
        while Instant::now() < deadline {
            self.pump();
            if targets.iter().all(VecDeque::is_empty) && self.jobs.is_empty() {
                targets = self.shutdown_targets();
                if targets.iter().all(VecDeque::is_empty)
                    && self
                        .sessions
                        .values()
                        .all(|s| s.ops.values().all(Open::is_over))
                {
                    complete = true;
                    break;
                }
            }
            for queue in &mut targets {
                for _ in 0..queue.len() {
                    if !self.writers.iter().any(|w| w.is_idle()) {
                        break;
                    }
                    let (id, key) = queue.pop_front().unwrap();
                    if !self.namespaces[&id]
                        .tree
                        .node(&key)
                        .is_some_and(|n| n.row && !n.location.durable)
                    {
                        continue;
                    }
                    self.write_ancestor(id, key, "shutdown");
                    if !self.namespaces[&id].tree[&key].location.writing {
                        queue.push_back((id, key));
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.record_shutdown_inventory();
        complete
    }

    fn shutdown_targets(&self) -> [VecDeque<(NamespaceId, Key)>; 2] {
        let mut targets = [VecDeque::new(), VecDeque::new()];
        for (&id, ns) in &self.namespaces {
            for key in &ns.undurable {
                // A page-only record is worth writing only as a dependency of a row.
                if let Some(node) = ns.tree.node(key).filter(|n| n.row) {
                    let interior_endpoint = node.location.header.capture_role()
                        == CaptureRole::Endpoint
                        && node.parent.is_some()
                        && node.children() == 1;
                    targets[usize::from(interior_endpoint)].push_back((id, *key));
                }
            }
        }
        targets
    }

    fn record_shutdown_inventory(&mut self) {
        let mut records = 0;
        let mut bytes = 0;
        for ns in self.namespaces.values() {
            let mut counted = HashSet::new();
            for key in ns
                .undurable
                .iter()
                .filter(|k| ns.tree.node(k).is_some_and(|n| n.row))
            {
                let mut next = Some(*key);
                while let Some(k) = next {
                    let node = &ns.tree[&k];
                    if node.location.durable || !counted.insert(k) {
                        break;
                    }
                    if node.location.disk.is_none() {
                        records += 1;
                        bytes += node.location.bytes;
                    }
                    next = node.parent;
                }
            }
        }
        self.metrics.shutdown_unflushed = Some((records, bytes));
        if records > 0 {
            eprintln!("selene: shutdown left {records} dirty records ({bytes} bytes)");
        }
        // The NVMe device may be the reason the flush timed out.
        let path = self.local.directory().join("last-shutdown");
        if let Err(e) = crate::local::write_atomically(&path, format!("{records} {bytes}\n").as_bytes()) {
            eprintln!("selene: could not save shutdown inventory: {e}");
        }
    }

    pub fn pump(&mut self) {
        self.pump_reads();
        self.pump_chunks();
        self.disk_events();
        self.dispatch();
        self.pump_grants();
        self.cleanup_turn();
        self.settle_terminal_puts();
        for (id, key) in self.pending_former_tips.iter().copied().collect::<Vec<_>>() {
            self.prune_former_tip(id, key);
        }
        self.schedule();
        self.tick();
        self.maybe_finish_gets();
        self.flush_outboxes();
        self.local.poll_prior();
    }

    fn maybe_finish_gets(&mut self) {
        let mut ops = Vec::new();
        for (connection, session) in &self.sessions {
            for (transfer, open) in &session.ops {
                if matches!(open.operation, Operation::Get(_)) {
                    ops.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in ops {
            self.maybe_finish_get(connection, transfer);
        }
    }

    pub fn catalog(
        &mut self,
        descriptor: Descriptor,
        sender: tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>,
    ) -> Result<Vec<EntryData>> {
        let namespace = self.register_namespace(descriptor)?;
        let entries = self
            .snapshot(namespace)
            .ok_or_else(|| invalid("namespace"))?;
        self.catalogs.insert(namespace, sender);
        Ok(entries)
    }

    pub fn metrics_text(&self) -> String {
        let mut text = String::new();
        if self.config.root.is_some() {
            let mut arena_only = 0u64;
            let mut oldest = Duration::ZERO;
            for ns in self.namespaces.values() {
                for node in ns
                    .tree
                    .nodes()
                    .filter(|n| n.location.memory.is_some() && n.location.disk.is_none())
                {
                    arena_only += node.location.bytes;
                    oldest = oldest.max(node.location.admitted.elapsed());
                }
            }
            text.push_str(&format!(
                "selene_arena_only_bytes {arena_only}\nselene_oldest_dirty_seconds {}\n",
                oldest.as_secs_f64()
            ));
            if let Some((records, bytes)) = self.metrics.shutdown_unflushed {
                text.push_str(&format!("selene_shutdown_unflushed_records {records}\nselene_shutdown_unflushed_bytes {bytes}\n"));
            }
        }
        for ((role, reason), (records, bytes)) in &self.metrics.record_actions {
            let role = role.as_str_name().to_ascii_lowercase();
            text.push_str(&format!("selene_policy_records_total{{role=\"{role}\",reason=\"{reason}\"}} {records}\nselene_policy_bytes_total{{role=\"{role}\",reason=\"{reason}\"}} {bytes}\n"));
        }
        for (i, tier) in ["dram", "nvme"].into_iter().enumerate() {
            text.push_str(&format!(
                "selene_expired_total{{tier=\"{tier}\"}} {}\n",
                self.metrics.expired[i]
            ));
        }
        for (name, value) in [
            ("unknown_frames_total", self.metrics.dropped),
            ("cancels_total", self.metrics.cancels),
            (
                "put_writeback_waits_total",
                self.metrics.put_writeback_waits,
            ),
            ("put_inflight_waits_total", self.metrics.put_inflight_waits),
            (
                "put_inflight_expired_total",
                self.metrics.put_inflight_expired,
            ),
            ("evictions_total", self.metrics.evictions),
            ("compactions_total", self.metrics.compactions),
            ("nvme_bytes_total", self.metrics.nvme_bytes),
            ("nvme_read_bytes_total", self.metrics.nvme_read_bytes),
            ("nvme_errors_total", self.metrics.nvme_errors),
            ("connections", self.sessions.len() as u64),
            (
                "leases",
                self.sessions
                    .values()
                    .flat_map(|session| session.ops.values())
                    .filter(|open| matches!(open.operation, Operation::Get(_)))
                    .count() as u64,
            ),
            (
                "open_ops",
                self.sessions
                    .values()
                    .map(|session| session.ops.len() as u64)
                    .sum(),
            ),
            ("memory_allocated_bytes", self.local.arena_used() as u64),
            ("staging_allocated_bytes", self.local.staging_used() as u64),
            ("nvme_reserved_bytes", self.charge_total()),
            (
                "nvme_reserved_peak_bytes",
                self.reserved_peak.max(self.charge_total()),
            ),
            (
                "cleanup_records",
                self.sessions
                    .values()
                    .flat_map(|session| session.ops.values())
                    .map(|open| match &open.operation {
                        Operation::Put(put) if put.terminal && put.pending > 0 => {
                            put.pins.len() as u64
                        }
                        Operation::Get(get) if get.terminal && get.pending_reads > 0 => {
                            get.pins.len() as u64
                        }
                        _ => 0,
                    })
                    .sum::<u64>(),
            ),
            (
                "pending_puts",
                self.pending_puts.values().map(|q| q.len() as u64).sum(),
            ),
            (
                "pending_gets",
                self.pending_gets.values().map(|q| q.len() as u64).sum(),
            ),
        ] {
            text.push_str(&format!("selene_{name} {value}\n"));
        }
        text.push_str(&format!(
            "selene_put_inflight_wait_max_seconds {}\n",
            self.metrics.put_inflight_wait_max.as_secs_f64()
        ));
        text.push_str(&format!(
            "selene_nvme_seconds_total {}\n",
            self.metrics.nvme_seconds
        ));
        for (part, count) in [
            ("pages", self.metrics.put_page_bytes),
            ("rows", self.metrics.put_row_bytes),
        ] {
            text.push_str(&format!(
                "selene_put_bytes_total{{part=\"{part}\"}} {count}\n"
            ));
        }
        for (parent, count) in [
            ("linked", self.metrics.put_links_linked),
            ("root", self.metrics.put_links_root),
        ] {
            text.push_str(&format!(
                "selene_put_links_total{{parent=\"{parent}\"}} {count}\n"
            ));
        }
        for (shape, count) in [
            ("row", self.metrics.put_shapes_row),
            ("page_only", self.metrics.put_shapes_page_only),
            ("empty", self.metrics.put_shapes_empty),
        ] {
            text.push_str(&format!(
                "selene_put_shapes_total{{shape=\"{shape}\"}} {count}\n"
            ));
        }
        for (outcome, count) in [
            ("ok", self.metrics.puts_ok),
            ("error", self.metrics.puts_error),
            ("unlinked", self.metrics.puts_unlinked),
            ("present", self.metrics.present),
            ("queue", self.metrics.retry_queue),
            ("budget", self.metrics.retry_budget),
            ("inflight", self.metrics.retry_inflight),
            ("oversize", self.metrics.oversize),
        ] {
            text.push_str(&format!(
                "selene_puts_total{{outcome=\"{outcome}\"}} {count}\n"
            ));
        }
        for (outcome, count) in [
            ("ok", self.metrics.gets_ok),
            ("error", self.metrics.gets_error),
        ] {
            text.push_str(&format!(
                "selene_gets_total{{outcome=\"{outcome}\"}} {count}\n"
            ));
        }
        let mut ids: Vec<&NamespaceId> = self.namespaces.keys().collect();
        ids.sort_unstable();
        for id in ids {
            let name = flare_selene::shared::record::hex(id);
            let namespace = &self.namespaces[id];
            for tier in ["memory", "disk"] {
                let mut count = 0;
                let mut bytes = 0;
                for node in namespace.tree.nodes() {
                    let retained = if tier == "memory" {
                        node.location.memory.is_some()
                    } else {
                        node.location.durable
                    };
                    if retained {
                        count += 1;
                        bytes += node.location.header.length(node.row);
                    }
                }
                text.push_str(&format!("selene_records{{namespace=\"{name}\",tier=\"{tier}\"}} {count}\nselene_bytes{{namespace=\"{name}\",tier=\"{tier}\"}} {bytes}\n"));
                let hits = self
                    .metrics
                    .hits
                    .get(&(name.clone(), tier))
                    .copied()
                    .unwrap_or(0);
                text.push_str(&format!(
                    "selene_hits_total{{namespace=\"{name}\",tier=\"{tier}\"}} {hits}\n"
                ));
            }
        }
        text
    }
}

pub type Reply<T> = tokio::sync::oneshot::Sender<T>;

pub struct PutAccept {
    pub capture_role: CaptureRole,
    pub transfer: u64,
    pub first_page: u32,
    pub parent: Option<Key>,
    pub record_bytes: u64,
    pub windows: usize,
    pub packed: bool,
}

pub struct WindowGrant {
    pub hold: Arc<Extent>,
    pub transfer: u64,
    pub index: u32,
    pub record: u32,
    pub packed: Option<flare_selene::shared::wire::PackedWindow>,
    pub record_offset: u64,
    pub length: u64,
    pub region: i32,
    pub offset: u64,
    pub page_first: u64,
    pub page_count: u32,
    pub row: bool,
}

/// The detail of the `INVALID` refusal of a page-only put that cannot link to a held ancestor,
/// which the pool counts and logs in aggregate because it is routine. No ancestor is held, or
/// the deepest one ends a chain of `REROOT_DEPTH` records, so the put would start a new root,
/// and a root needs a row.
pub use flare_selene::shared::PUT_UNLINKED;

pub struct Refused {
    pub reason: u8,
    pub detail: String,
}

pub enum PutAnswer {
    Accept(PutAccept),
    Grant(WindowGrant),
    Stored { bytes: u64 },
    Refused(Refused),
}

pub enum GetAnswer {
    Leased {
        packed: Vec<Option<flare_selene::shared::wire::PackedOffer>>,
        transfer: u64,
        windows: usize,
    },
    Meta {
        record: u32,
        pages: Vec<u64>,
        row: u64,
        source: i32,
    },
    Ready(WindowGrant),
    Done,
    Refused(Refused),
}

pub enum StoreCommand {
    Metrics {
        reply: std::sync::mpsc::Sender<String>,
    },
    Hello {
        connection: u64,
        pid: u32,
        process_nonce: Option<[u8; 16]>,
        descriptor: Descriptor,
        reply: Reply<Result<()>>,
    },
    Catalog {
        descriptor: Descriptor,
        events: tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>,
        reply: Reply<Result<Vec<EntryData>>>,
    },
    PutBegin {
        connection: u64,
        transfer: u64,
        namespace: NamespaceId,
        descriptor: Descriptor,
        key: Key,
        published_parent: Option<Key>,
        ancestor_keys: Vec<Key>,
        bucket: [u8; 16],
        chain_pages: u32,
        published_tail_pages: u32,
        capture_role: CaptureRole,
        child_paths: u32,
        descendant_row: bool,
        next_child: Option<Key>,
        row: bool,
        packed: Option<flare_selene::shared::wire::PackedOffer>,
        events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
        reply: Reply<Result<()>>,
    },
    WindowFilled {
        connection: u64,
        transfer: u64,
        index: u32,
        page_xxh3: Vec<u64>,
        row_xxh3: Option<u64>,
        reply: Reply<Result<()>>,
    },
    PutCommit {
        connection: u64,
        transfer: u64,
        metadata_offset: u64,
        metadata: Vec<u8>,
        reply: Reply<Result<()>>,
    },
    GetBegin {
        connection: u64,
        transfer: u64,
        namespace: NamespaceId,
        chain: Vec<Key>,
        skip_pages: u32,
        expected_pages: u32,
        row: bool,
        events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
        reply: Reply<Result<()>>,
    },
    RequestWindow {
        connection: u64,
        transfer: u64,
        reply: Reply<Result<()>>,
    },
    Cancel {
        connection: u64,
        transfer: u64,
    },
    ConnectionClosed {
        connection: u64,
    },
    Tick,
    Shutdown {
        deadline: Instant,
    },
}

fn region_wire(region: Region) -> i32 {
    match region {
        Region::Arena => proto::Region::Arena as i32,
        Region::Staging => proto::Region::Staging as i32,
    }
}

fn record_grants(
    header: &Header,
    first_page: u32,
    skip: u32,
    include_row: bool,
    max_window: u64,
    record: u32,
) -> Vec<PlannedGrant> {
    if let Some(packed) = &header.packed {
        return packed
            .windows(skip, include_row, max_window)
            .into_iter()
            .map(|w| PlannedGrant {
                record,
                record_offset: header.page_segment + w.offset,
                length: w.length,
                pages: w.pages,
                row: w.row,
                page_first: if w.pages == 0 {
                    0
                } else {
                    u64::from(first_page) + u64::from(w.first)
                },
            })
            .collect();
    }
    use crate::pool::windows::{self, RecordShape};
    let shape = RecordShape {
        first_page: skip,
        pages: header.tail_pages.saturating_sub(skip),
        row: include_row,
    };
    if shape.pages == 0 && !shape.row {
        return Vec::new();
    }
    windows::plan(
        &[shape],
        header.page_record_bytes(),
        header.row_record_bytes(),
        max_window,
    )
    .expect("stored records have validated geometry")
    .into_iter()
    .map(|w| PlannedGrant {
        record,
        record_offset: if w.page_count == 0 {
            header.row_segment
        } else {
            header.page_segment + u64::from(w.first_page) * header.page_record_bytes()
        },
        length: w.length,
        pages: w.page_count,
        row: w.row,
        page_first: if w.page_count == 0 {
            0
        } else {
            u64::from(first_page) + u64::from(w.first_page)
        },
    })
    .collect()
}

impl Store {
    pub fn handle(&mut self, command: StoreCommand) {
        match command {
            StoreCommand::Hello {
                connection,
                pid,
                process_nonce,
                descriptor,
                reply,
            } => {
                let result = self.hello(connection, pid, process_nonce, descriptor);
                let _ = reply.send(result);
            }
            StoreCommand::Catalog {
                descriptor,
                events,
                reply,
            } => {
                let _ = reply.send(self.catalog(descriptor, events));
            }
            StoreCommand::PutBegin {
                connection,
                transfer,
                namespace,
                descriptor,
                key,
                published_parent,
                ancestor_keys,
                bucket,
                chain_pages,
                published_tail_pages,
                capture_role,
                child_paths,
                descendant_row,
                next_child,
                row,
                packed,
                events,
                reply,
            } => {
                let session_ok = self
                    .sessions
                    .get(&connection)
                    .is_some_and(|session| !session.closed && session.namespace == namespace);
                let valid = session_ok
                    && descriptor.namespace_id() == namespace
                    && self.register_namespace(descriptor).is_ok();
                if !valid {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "PutBegin namespace or descriptor",
                    )));
                    return;
                }
                if self.transfer_in_use(connection, transfer) {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "Put transfer id is already open on this connection",
                    )));
                    return;
                }
                self.submit_put(
                    connection,
                    transfer,
                    PutRequest {
                        namespace,
                        key,
                        published_parent,
                        ancestor_keys,
                        bucket,
                        chain_pages,
                        published_tail_pages,
                        capture_role,
                        child_paths,
                        descendant_row,
                        next_child,
                        row,
                        packed,
                    },
                    events,
                );
                let _ = reply.send(Ok(()));
            }
            StoreCommand::WindowFilled {
                connection,
                transfer,
                index,
                page_xxh3,
                row_xxh3,
                reply,
            } => {
                let result = self.window_filled(connection, transfer, index, page_xxh3, row_xxh3);
                let _ = reply.send(result);
            }
            StoreCommand::PutCommit {
                connection,
                transfer,
                metadata_offset,
                metadata,
                reply,
            } => {
                let result = self.put_commit(connection, transfer, metadata_offset, metadata);
                let _ = reply.send(result);
            }
            StoreCommand::GetBegin {
                connection,
                transfer,
                namespace,
                chain,
                skip_pages,
                expected_pages,
                row,
                events,
                reply,
            } => {
                let allowed = self
                    .sessions
                    .get(&connection)
                    .is_some_and(|session| !session.closed && session.namespace == namespace);
                if !allowed {
                    // a Get for a namespace the session has not registered is a
                    // terminal operation refusal.
                    let _ = reply.send(Ok(()));
                    let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                        reason: proto::Reason::Invalid as u8,
                        detail: "Get namespace is not registered".into(),
                    })));
                    return;
                }
                if self.transfer_in_use(connection, transfer) {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "Get transfer id is already open on this connection",
                    )));
                    return;
                }
                self.submit_get(
                    connection,
                    transfer,
                    GetRequest {
                        namespace,
                        keys: chain,
                        skip_pages,
                        expected_pages,
                        row,
                    },
                    events,
                );
                let _ = reply.send(Ok(()));
            }
            StoreCommand::RequestWindow {
                connection,
                transfer,
                reply,
            } => {
                let result = self
                    .sessions
                    .get_mut(&connection)
                    .and_then(|session| session.ops.get_mut(&transfer))
                    .ok_or_else(|| invalid("unknown operation"))
                    .map(|open| match &mut open.operation {
                        Operation::Put(put) => put.requested += 1,
                        Operation::Get(get) => get.requested += 1,
                    });
                let _ = reply.send(result);
            }
            StoreCommand::Cancel {
                connection,
                transfer,
            } => {
                self.cancel(connection, transfer);
            }
            StoreCommand::ConnectionClosed { connection } => self.connection_closed(connection),
            StoreCommand::Metrics { reply } => {
                let _ = reply.send(self.metrics_text());
            }
            StoreCommand::Tick => {}
            StoreCommand::Shutdown { .. } => {}
        }
    }
}

fn apply_packed_offer(
    header: Header,
    offer: Option<&flare_selene::shared::wire::PackedOffer>,
    first_page: u64,
    tail_pages: u32,
    include_row: bool,
) -> Result<Header> {
    let Some(offer) = offer else {
        return Ok(header);
    };
    let end_page = first_page
        .checked_add(u64::from(tail_pages))
        .ok_or_else(|| invalid("packed put page range overflow"))?;
    let units: Vec<_> = offer
        .units
        .iter()
        .filter(|u| match flare_selene::shared::wire::packed_unit::Kind::try_from(u.kind) {
            Ok(flare_selene::shared::wire::packed_unit::Kind::Page) => {
                u.logical_index >= first_page && u.logical_index < end_page
            }
            Ok(flare_selene::shared::wire::packed_unit::Kind::Row) => include_row,
            Err(_) => false,
        })
        .cloned()
        .collect();
    // If every unit in the accepted suffix is a raw literal, fall back to a standard v3 record.
    if !units.iter().any(|u| {
        u.representation == flare_selene::shared::wire::packed_unit::Representation::Packet as i32
    }) {
        return Ok(header);
    }
    let sliced = flare_selene::shared::wire::PackedOffer { first_page, units };
    header
        .with_packed(sliced)
        .map_err(|e| invalid(e.to_string()))
}

fn validate_commit(put: &mut Put, descriptor: &Descriptor) -> Result<()> {
    let planned = &put.header;
    let page_segment = planned.page_segment as usize;
    let header = Header::decode(if planned.packed.is_some() {
        &put.commit_bytes[..page_segment]
    } else {
        &put.commit_bytes[..4096]
    })
    .map_err(|e| invalid(e.to_string()))?;
    let row_ok = if planned.packed.is_some() {
        header.row_checksum == 0 && header.packed == planned.packed
    } else {
        match put.windows.row {
            Some(hash) => header.row_checksum == hash,
            None => header.row_checksum == planned.row_checksum,
        }
    };
    if header.key != planned.key
        || header.parent != planned.parent
        || header.bucket != planned.bucket
        || header.tail_pages != planned.tail_pages
        || header.page_segment != planned.page_segment
        || header.row_segment != planned.row_segment
        || header.flags != planned.flags
        || !row_ok
        || !descriptor.matches(&header)
    {
        return Err(invalid("commit header disagrees with accept"));
    }
    if planned.packed.is_none() {
        let checksum_segment = &put.commit_bytes[4096..page_segment];
        for (index, hash) in put.windows.pages.iter().enumerate() {
            let Some(hash) = hash else {
                return Err(invalid("missing page hash"));
            };
            if checksum_segment[index * 8..index * 8 + 8] != hash.to_le_bytes() {
                return Err(invalid("commit checksum table"));
            }
        }
    }
    if header.capture_role() == put.stored_role {
        put.header = header;
    } else {
        put.header = header.with_capture_role(put.stored_role);
        put.commit_bytes[..4096]
            .copy_from_slice(&put.header.encode().map_err(|e| invalid(e.to_string()))?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_delivered_put_waits_while_the_disk_makes_progress() {
        use super::{NO_PROGRESS, stalled};
        use std::time::Duration;
        let long = NO_PROGRESS + Duration::from_secs(1);
        let short = Duration::from_millis(10);
        // Waiting on the client for too long is a stall.
        assert!(stalled(long, false, Some(short)));
        assert!(!stalled(short, false, None));
        // Everything delivered and the disk busy but moving: not a stall, however long.
        assert!(!stalled(long, true, Some(short)));
        // The disk has made no progress either, or nothing is queued for it.
        assert!(stalled(long, true, Some(long)));
        assert!(stalled(long, true, None));
    }

    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn descriptor() -> Descriptor {
        Descriptor {
            identity: [7; 32],
            fingerprint: [3; 16],
            page_tokens: 64,
            ranks: vec![flare_selene::shared::record::RankSizes {
                planes: vec![4000],
                row_bytes: 1000,
            }],
            pool_name: String::new(),
        }
    }

    fn store(label: &str) -> (Store, PathBuf) {
        dram_store(label, 1024 * 1024)
    }

    fn dram_store(label: &str, memory: usize) -> (Store, PathBuf) {
        let name = format!(
            "selene-unit-{}-{label}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let mut config =
            Config::dram_only(std::env::temp_dir().join(format!("{name}.sock")), memory);
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        (store, directory)
    }

    fn store_with_root(label: &str, memory: usize) -> (Store, PathBuf, PathBuf) {
        let name = format!(
            "selene-unit-{}-{label}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            memory,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        (store, base, directory)
    }

    #[test]
    fn a_restart_after_the_io_uring_probe_reuses_the_regions() {
        let name = format!(
            "selene-unit-{}-eintr-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let config = || {
            let mut config = Config::new(
                base.join("records"),
                base.join("manager.sock"),
                1 << 20,
                64 << 20,
            );
            config.name = name.clone();
            config.staging = 8 << 20;
            config
        };
        let first = Store::open(config()).unwrap();
        let generation = first.local.regions_generation();
        let directory = first.shm_directory();
        drop(first);
        // Opening a store with a root sets up io_uring on this thread just before the regions
        // are checked, which used to interrupt the reservation and recreate them.
        for _ in 0..30 {
            let again = Store::open(config()).unwrap();
            assert_eq!(again.local.regions_generation(), generation);
            drop(again);
        }
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_file(Path::new("/dev/shm/selene").join(format!("{name}.lock")));
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn admission_classifies_shapes_and_retained_capacity() {
        use proto::Reason::{Budget, Oversize};
        const K: u64 = 1024;
        const M: u64 = 1024 * K;
        // label, arena, disk budget, staging, planes, row, ancestors, held arena,
        // requested pages, expected refusal (None means accepted).
        #[rustfmt::skip]
        let cases = [
            ("small-arena", 16 * K, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 1, None),
            ("stream-charge", 4096, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 2, Some(Oversize)),
            ("row-only", 4096, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 0, None),
            ("two-windows-tight", 4096, Some(1100 * K), 8 * M, vec![4000], 1000, vec![], 0, 128, Some(Oversize)),
            ("two-windows-fit", 4096, Some(2 * M), 8 * M, vec![4000], 1000, vec![], 0, 128, None),
            ("trailing-window-tight", 4096, Some(1400 * K), 8 * M, vec![4000], 1000, vec![], 0, 192, Some(Oversize)),
            ("trailing-window-fit", 4096, Some(2 * M), 8 * M, vec![4000], 1000, vec![], 0, 192, None),
            ("oversized-page", 16 * M, Some(64 * M), 8 * M, vec![16 * M], 1000, vec![], 0, 1, Some(Oversize)),
            ("retained-parent", 32 * K, None, 8 * M, vec![4000], 1000, vec![2], 0, 6, Some(Budget)),
            ("exact-arena", 16 * M, None, 8 * M, vec![65_539, 4100], 1_000_003, vec![], 0, 140, None),
            ("over-arena", 16 * M, None, 8 * M, vec![65_539, 4100], 1_000_003, vec![], 0, 220, Some(Oversize)),
            ("rounded-over-arena", 12 * M, None, 8 * M, vec![4000], 1000, vec![], 0, 2048, Some(Oversize)),
            ("rounded-within-arena", 12 * M, None, 8 * M, vec![4000], 1000, vec![], 0, 1024, None),
            ("staging-hold-tight", M, Some(64 * M), 8 * M, vec![4000], 8 * M, vec![], 0, 1, Some(Oversize)),
            ("staging-hold-fit", M, Some(64 * M), 16 * M, vec![4000], 8 * M, vec![], 0, 1, None),
            ("temporary-pressure", 48 * K, None, 8 * M, vec![4000], 1000, vec![1], 32 * K, 6, Some(Budget)),
            ("pressure-released", 48 * K, None, 8 * M, vec![4000], 1000, vec![1], 0, 6, None),
            ("rounded-charge-tight", 2 * M, Some(1_500_000), 8 * M, vec![4000], 1000, vec![], 0, 300, Some(Oversize)),
            ("rounded-charge-fit", 2 * M, Some(2_300_000), 8 * M, vec![4000], 1000, vec![], 0, 300, None),
            ("retained-chain", 64 * K, None, 8 * M, vec![4000], 1000, vec![5, 10], 0, 14, Some(Budget)),
            ("parent-disk-charge", 64 * K, Some(56 * K), 8 * M, vec![4000], 1000, vec![5], 0, 10, Some(Budget)),
            ("four-open-children", 8 * M, Some(24 * M), 8 * M, vec![4000], 4096 * 1023, vec![1], 0, 2, None),
        ];
        for (label, arena, disk, staging, planes, row, ancestors, held, pages, refusal) in cases {
            let name = format!(
                "selene-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let base = std::env::temp_dir().join(&name);
            let mut config = Config::dram_only(base.join("manager.sock"), arena as usize);
            config.name = name;
            config.root = disk.map(|_| base.join("records"));
            config.nvme = disk.unwrap_or(0);
            config.staging = staging;
            let mut store = Store::open(config).unwrap();
            let mut value = descriptor();
            value.ranks[0].planes = planes;
            value.ranks[0].row_bytes = row;
            store.hello(1, std::process::id(), None, value.clone()).unwrap();
            let id = value.namespace_id();
            let key = |pages: u32| Key {
                consumed_tokens: u64::from(pages) * 64 + 3,
                digest: (pages as u128).to_le_bytes(),
            };
            let mut parent = None;
            let mut first = 0;
            for ancestor in ancestors {
                let header = crate::record::new_header(
                    &value,
                    key(ancestor),
                    parent,
                    [0; 16],
                    ancestor - first,
                )
                .unwrap();
                let extent = store
                    .memory
                    .allocate(header.length_with_row() as usize)
                    .unwrap();
                insert_memory_node(
                    &mut store,
                    id,
                    &value,
                    key(ancestor),
                    parent,
                    ancestor - first,
                    extent,
                );
                parent = Some(key(ancestor));
                first = ancestor;
            }
            let pressure = (held > 0).then(|| store.memory.allocate(held as usize).unwrap());
            let (events, mut answers) = tokio::sync::mpsc::channel(16);
            let count = if label == "four-open-children" { 4 } else { 1 };
            let mut accepted = 0;
            for index in 0..count {
                store.begin_put(PendingPut {
                    inflight_since: None,
                    longest_inflight_wait: Duration::ZERO,
                    connection: 1,
                    transfer: index + 1,
                    request: PutRequest {
                        ancestor_keys: Vec::new(),
                        namespace: id,
                        key: key(pages + index as u32),
                        published_parent: parent,
                        bucket: [0; 16],
                        chain_pages: pages + index as u32,
                        published_tail_pages: pages + index as u32 - first,
                        capture_role: CaptureRole::Legacy,
                        child_paths: 0,
                        descendant_row: false,
                        next_child: None,
                        row: true,
                packed: None,
                    },
                    events: events.clone(),
                });
                store.flush_outboxes();
                let deadline = Instant::now() + Duration::from_secs(5);
                let answer = loop {
                    if let Ok(answer) = answers.try_recv() {
                        break answer.unwrap();
                    }
                    assert!(Instant::now() < deadline, "{label}: admission timed out");
                    store.pump();
                    std::thread::yield_now();
                };
                match answer {
                    PutAnswer::Accept(_) => {
                        assert!(refusal.is_none(), "{label}: accepted");
                        accepted += 1;
                    }
                    PutAnswer::Refused(reason) => {
                        let expected = if count > 1 {
                            Budget
                        } else {
                            refusal.expect(label)
                        };
                        assert_eq!(reason.reason, expected as u8, "{label}");
                    }
                    _ => panic!("{label}: expected an admission verdict"),
                }
            }
            if refusal.is_none() {
                assert!(accepted > 0, "{label}: no put admitted");
            }
            drop(pressure);
            let directory = store.shm_directory();
            drop(store);
            let _ = fs::remove_dir_all(base);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn recovery_counts_compacted_files_without_their_removed_rows() {
        let (store, base, directory) = store_with_root("recovery-bytes", 64 * 1024);
        let config = store.config.clone();
        drop(store);
        let value = descriptor();
        let records = config.root.as_ref().unwrap().join(value.name());
        fs::create_dir_all(&records).unwrap();
        let mut parent = None;
        let mut expected = 0;
        for (pages, row) in [(1u32, false), (2, true)] {
            let key = Key {
                consumed_tokens: u64::from(pages) * 64,
                digest: [pages as u8; 16],
            };
            let mut header = crate::record::new_header(&value, key, parent, [0; 16], 1).unwrap();
            let mut bytes = vec![0; header.length(row) as usize];
            let page_hash = flare_selene::shared::record::checksum(
                &bytes[header.page_segment as usize..header.row_segment as usize],
            );
            if row {
                header.row_checksum =
                    flare_selene::shared::record::checksum(&bytes[header.row_segment as usize..]);
            }
            bytes[..4096].copy_from_slice(&header.encode().unwrap());
            bytes[4096..4104].copy_from_slice(&page_hash.to_le_bytes());
            expected += bytes.len() as u64;
            fs::write(records.join(record::file_name(&key)), bytes).unwrap();
            parent = Some(key);
        }
        let mut store = Store::open(config.clone()).unwrap();
        assert_eq!(store.charge_total(), expected);
        let id = store.namespace_id().unwrap();
        assert_eq!(store.namespaces[&id].tree.nodes().count(), 2);
        let name = flare_selene::shared::record::hex(&id);
        assert!(store.metrics_text().contains(&format!(
            "selene_bytes{{namespace=\"{name}\",tier=\"disk\"}} {expected}\n"
        )));
        let old = SystemTime::now() - Duration::from_secs(10800);
        let key = parent.unwrap();
        let path = records.join(record::file_name(&key));
        let bytes = fs::read(&path).unwrap();
        let last_used = store.namespaces[&id].tree[&key].last_used;
        store.reused(id, key);
        assert!(store.jobs.is_empty());
        assert!(!store.namespaces[&id].tree[&key].pinned());
        assert_eq!(store.namespaces[&id].tree[&key].last_used, last_used);
        store.config.max_age = Duration::from_secs(7200);
        store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
        store.reused(id, key);
        assert!(store.namespaces[&id].tree[&key].last_used > last_used);
        store.reused(id, key);
        assert!(store.jobs.is_empty());
        assert!(!store.namespaces[&id].tree[&key].pinned());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), last_used);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        drop(store);
        let mut store = Store::open(config).unwrap();
        assert_eq!(store.namespaces[&id].tree[&key].last_used, old);
        store.config.max_age = Duration::from_secs(7200);
        store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
        store.tick();
        assert!(!store.namespaces[&id].tree.contains(&key));
        assert_eq!(store.metrics.expired, [0, 2]);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn put_requirement_matches_the_shape_the_put_gets() {
        let (store, directory) = store("requirement");
        let value = descriptor();
        let key = |tokens| Key {
            consumed_tokens: tokens,
            digest: [0; 16],
        };
        let one_page = crate::record::new_header(&value, key(64), None, [0; 16], 1).unwrap();
        // The arena shape charges its 16 KiB extent.
        let arena = store.put_requirement(&one_page, 16 * 1024, true);
        assert!(!arena.streaming);
        assert_eq!(arena.charge, 16 * 1024);
        assert_eq!(arena.reservation, 16 * 1024);
        assert_eq!(arena.allocation, 16 * 1024);
        assert_eq!(arena.unit, 16 * 1024);
        // 128 pages and a 4 KiB row: 524 KiB of disk record plus a fixed
        // hold of two 512 KiB windows.
        let many = crate::record::new_header(&value, key(128 * 64), None, [0; 16], 128).unwrap();
        let stream = store.put_requirement(&many, 0, true);
        assert!(stream.streaming);
        assert_eq!(stream.reservation, 524 * 1024);
        assert_eq!(stream.allocation, 2 * 512 * 1024);
        assert_eq!(stream.unit, 512 * 1024);
        assert_eq!(stream.charge, (524 + 2 * 512) * 1024);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    /// Physical charge: the arena allocator size class or the padded disk record.
    fn record_charge(header: &Header, in_memory: bool) -> u64 {
        let length = header.length_with_row();
        if in_memory {
            length.max(4096).next_power_of_two()
        } else {
            length
        }
    }

    fn insert_memory_node(
        store: &mut Store,
        id: NamespaceId,
        value: &Descriptor,
        key: Key,
        parent: Option<Key>,
        tail: u32,
        extent: Extent,
    ) {
        let header = crate::record::new_header(value, key, parent, [0; 16], tail).unwrap();
        let bytes = record_charge(&header, true);
        store.charged += bytes;
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                parent,
                [0; 16],
                tail as usize,
                true,
                true,
                Location {
                    header,
                    memory: Some(Arc::new(extent)),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    admitted: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
    }

    /// A Get of records that are all on disk: the metadata reads it submits, then, once the
    /// metadata has arrived and credit is given, its window reads.
    #[test]
    fn a_get_streams_records_that_are_not_resident_and_reads_the_first_window_first() {
        let (mut store, base, directory) = store_with_root("get-streams", 64 << 20);
        // A 4 MiB row is larger than a window can share with a page, so the endpoint's row
        // is a window of its own.
        let mut value = descriptor();
        value.ranks[0].row_bytes = 4 << 20;
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let path = base.join("chain.ckpt");
        fs::write(&path, vec![0; 16 << 20]).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let count = 4;
        let records: Vec<_> = (1..=count)
            .map(|pages| GetRecord {
                header: crate::record::new_header(
                    &value,
                    page_key(pages),
                    (pages > 1).then(|| page_key(pages - 1)),
                    [0; 16],
                    1,
                )
                .unwrap(),
                memory: None,
                file: Some(file.clone()),
                meta: None,
                meta_sent: false,
            })
            .collect();
        let grants: Vec<_> = records
            .iter()
            .enumerate()
            .flat_map(|(index, record)| {
                record_grants(
                    &record.header,
                    index as u32,
                    0,
                    index + 1 == records.len(),
                    DEFAULT_MAX_WINDOW,
                    index as u32,
                )
            })
            .collect();
        // One page window per record, then the endpoint's row.
        assert_eq!(grants.len(), count as usize + 1);
        let row_bytes = records[0].header.row_record_bytes();
        assert_eq!(
            grants.last().map(|grant| (grant.row, grant.length)),
            Some((true, row_bytes))
        );
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.sessions.get_mut(&1).unwrap().ops.insert(
            1,
            Open {
                operation: Operation::Get(Box::new(Get {
                    namespace: id,
                    pins: VecDeque::new(),
                    records,
                    grants: grants.clone(),
                    requested: 0,
                    next_grant: 0,
                    next_meta_read: 0,
                    next_meta: 0,
                    window_reads: 0,
                    pending_reads: 0,
                    cancelled: false,
                    terminal: false,
                    finished: false,
                    events,
                    outbox: VecDeque::new(),
                })),
                progress: Instant::now(),
            },
        );
        let arena_used = store.memory.used();

        store.prepare_get_records(1, 1, id);
        // Only metadata is read up front, one read per record and no more than the page
        // segment, which leaves the arena untouched and the endpoint's row unread.
        let mut jobs: Vec<_> = store.jobs.keys().copied().collect();
        jobs.sort_unstable();
        assert_eq!(jobs.len(), count as usize);
        // The reads thread fills each metadata buffer from the zero-filled file. A header
        // written before that read lands would be overwritten, so wait for every read.
        let mut completed = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while completed < jobs.len() {
            assert!(
                Instant::now() < deadline,
                "{completed} of {} metadata reads completed",
                jobs.len()
            );
            let mut replies = Vec::new();
            store.namespaces[&id]
                .reads
                .as_ref()
                .unwrap()
                .poll(&mut replies);
            assert!(replies.iter().all(|reply| reply.result.is_ok()));
            completed += replies.len();
            std::thread::sleep(Duration::from_millis(1));
        }
        for (record, job) in jobs.into_iter().enumerate() {
            let Some(Job::GetRead {
                record: read,
                kind: ReadKind::Meta { length },
                mut buffer,
                ..
            }) = store.jobs.remove(&job)
            else {
                panic!("job {job} is not a metadata read");
            };
            assert_eq!(read as usize, record);
            let header = get_in(&mut store).records[record].header.clone();
            assert_eq!(length as u64, header.page_segment);
            buffer.bytes_mut()[..4096].copy_from_slice(&header.encode().unwrap());
            store.get_read_done(1, 1, read, ReadKind::Meta { length }, Some(buffer));
        }
        assert_eq!(store.memory.used(), arena_used);

        // With credit for every window, windows are read in grant order and the row after
        // every page window.
        get_in(&mut store).requested = grants.len();
        store.pump_grants();
        let mut reads: Vec<_> = store
            .jobs
            .iter()
            .map(|(job, read)| {
                let Job::GetRead { kind, buffer, .. } = read else {
                    panic!("job {job} is not a Get read");
                };
                let ReadKind::Window { grant } = kind else {
                    panic!("job {job} is not a window read");
                };
                (*job, *grant, buffer.bytes().len() as u64)
            })
            .collect();
        reads.sort_unstable();
        assert_eq!(
            reads.iter().map(|read| read.1).collect::<Vec<_>>(),
            (0..grants.len() as u32).collect::<Vec<_>>(),
            "windows are submitted in grant order"
        );
        for (job, grant, bytes) in &reads {
            assert_eq!(*bytes, grants[*grant as usize].length, "job {job}");
        }
        let (row_job, _, row_length) = reads[grants.len() - 1];
        let (first_job, ..) = reads[0];
        assert!(first_job < row_job);
        assert_eq!(row_length, row_bytes);
        // No record gained an arena copy.
        assert_eq!(store.memory.used(), arena_used);
        assert!(
            get_in(&mut store)
                .records
                .iter()
                .all(|record| record.memory.is_none())
        );
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    /// Makes a resident record disk-only: its arena copy is dropped and its file, which
    /// holds the record's header followed by zeros, is the only copy.
    fn move_to_disk(store: &mut Store, id: NamespaceId, key: Key, path: PathBuf) {
        let node = store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .node_mut(&key)
            .unwrap();
        let header = node.location.header.clone();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        // The length is set, not written, so a record of many pages is a sparse file.
        let mut file = File::create(&path).unwrap();
        io::Write::write_all(&mut file, &header.encode().unwrap()).unwrap();
        file.set_len(header.length_with_row()).unwrap();
        node.location.memory = None;
        node.location.disk = Some(Arc::new(DiskExtent::new(path)));
        node.location.durable = true;
    }

    #[test]
    fn a_get_whose_row_window_cannot_fit_in_staging_is_refused_before_any_read() {
        let name = format!(
            "selene-unit-{}-staging-row-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            1 << 20,
            64 << 20,
        );
        config.name = name.clone();
        // The endpoint's row is a 16 MiB window of its own, which is larger than staging.
        config.staging = 8 << 20;
        let mut store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        let mut value = descriptor();
        value.ranks[0].row_bytes = 16 << 20;
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let extent = store.memory.allocate(4096).unwrap();
        insert_memory_node(&mut store, id, &value, page_key(1), None, 1, extent);
        move_to_disk(
            &mut store,
            id,
            page_key(1),
            base.join("records").join("row.ckpt"),
        );
        let (events, mut receiver) = tokio::sync::mpsc::channel(8);
        store.submit_get(
            1,
            1,
            GetRequest {
                namespace: id,
                keys: vec![page_key(1)],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            events,
        );
        store.dispatch();
        store.flush_outboxes();
        match receiver.try_recv() {
            Ok(Ok(GetAnswer::Refused(refused))) => {
                assert_eq!(refused.reason, proto::Reason::Budget as u8);
            }
            Ok(Ok(GetAnswer::Leased { .. })) => panic!("the Get started"),
            _ => panic!("expected the Get to be refused"),
        }
        assert!(receiver.try_recv().is_err());
        assert!(store.jobs.is_empty(), "the refused Get submitted a read");
        assert!(!store.sessions[&1].ops.contains_key(&1));
        assert!(!store.namespaces[&id].tree[&page_key(1)].pinned());
        drop(store);
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(base);
    }

    /// The metadata block of a record that is not resident is read into staging before any
    /// window, so a block larger than staging is refused when the Get starts even though every
    /// window of the record fits. The block holds a 4096-byte header and 8 bytes of checksum
    /// per page, so a record of a million pages is needed to exceed the smallest staging.
    #[test]
    fn a_get_whose_metadata_block_cannot_fit_in_staging_is_refused_before_any_read() {
        let name = format!(
            "selene-unit-{}-staging-metadata-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            1 << 20,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 << 20;
        let mut store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        let value = descriptor();
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let pages = (8u64 << 20) / 8 + 1;
        let record = page_key(pages);
        let extent = store.memory.allocate(4096).unwrap();
        insert_memory_node(&mut store, id, &value, record, None, pages as u32, extent);
        move_to_disk(
            &mut store,
            id,
            record,
            base.join("records").join("metadata.ckpt"),
        );
        let header = store.namespaces[&id].tree[&record].location.header.clone();
        let staging = store.staging.capacity() as u64;
        assert!(header.page_segment > staging);
        for grant in record_grants(&header, 0, 0, true, DEFAULT_MAX_WINDOW, 0) {
            assert!(grant.length <= staging, "a window must fit for this test");
        }
        let (events, mut receiver) = tokio::sync::mpsc::channel(8);
        store.submit_get(
            1,
            1,
            GetRequest {
                namespace: id,
                keys: vec![record],
                skip_pages: 0,
                expected_pages: pages as u32,
                row: true,
            },
            events,
        );
        store.dispatch();
        store.flush_outboxes();
        match receiver.try_recv() {
            Ok(Ok(GetAnswer::Refused(refused))) => {
                assert_eq!(refused.reason, proto::Reason::Budget as u8);
            }
            Ok(Ok(GetAnswer::Leased { .. })) => panic!("the Get started"),
            _ => panic!("expected the Get to be refused"),
        }
        assert!(receiver.try_recv().is_err());
        assert!(store.jobs.is_empty(), "the refused Get submitted a read");
        assert!(!store.sessions[&1].ops.contains_key(&1));
        assert!(!store.namespaces[&id].tree[&record].pinned());
        drop(store);
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(base);
    }

    /// Staging that other transfers hold is a temporary shortage. The Get has been told it is
    /// leased, so it waits for the release instead of being refused.
    #[test]
    fn a_get_whose_metadata_read_finds_staging_full_waits_for_a_release() {
        let name = format!(
            "selene-unit-{}-staging-meta-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            1 << 20,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 << 20;
        let mut store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        let value = descriptor();
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let extent = store.memory.allocate(4096).unwrap();
        insert_memory_node(&mut store, id, &value, page_key(1), None, 1, extent);
        move_to_disk(
            &mut store,
            id,
            page_key(1),
            base.join("records").join("record.ckpt"),
        );
        let held = store.staging.allocate(store.staging.capacity()).unwrap();
        let (events, mut receiver) = tokio::sync::mpsc::channel(8);
        store.submit_get(
            1,
            1,
            GetRequest {
                namespace: id,
                keys: vec![page_key(1)],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            events,
        );
        let mut answers = Vec::new();
        for _ in 0..5 {
            store.pump();
            while let Ok(answer) = receiver.try_recv() {
                answers.push(answer);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            matches!(answers.as_slice(), [Ok(GetAnswer::Leased { .. })]),
            "a Get that only waits for staging was refused or answered early"
        );
        assert!(store.jobs.is_empty());
        drop(held);
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut metadata = false;
        while !metadata {
            assert!(Instant::now() < deadline, "the metadata never arrived");
            store.pump();
            while let Ok(answer) = receiver.try_recv() {
                match answer {
                    Ok(GetAnswer::Meta { record, .. }) => {
                        assert_eq!(record, 0);
                        metadata = true;
                    }
                    Ok(GetAnswer::Refused(refused)) => {
                        panic!("the Get was refused: {}", refused.detail)
                    }
                    _ => {}
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(store);
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(base);
    }

    /// Windows are answered in grant order. A resident window granted after a streamed one
    /// must not be answered while the streamed window's read is still in flight.
    #[test]
    fn a_resident_window_after_a_streamed_one_waits_for_it() {
        let name = format!(
            "selene-unit-{}-resident-after-streamed-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            1 << 20,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 << 20;
        let (gate, entered, release) = crate::disk::IoGate::new();
        config.faults.read_gate = Some(gate.clone());
        let mut store = Store::open(config).unwrap();
        // Dropping the store joins the reads thread, so a failed assertion must release the
        // gate first or the test would hang instead of failing. `release` is declared after
        // `store`, so it drops first.
        struct Release(std::sync::mpsc::SyncSender<()>);
        impl Drop for Release {
            fn drop(&mut self) {
                let _ = self.0.try_send(());
            }
        }
        let release = Release(release);
        let directory = store.shm_directory();
        let value = descriptor();
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        // A parent recovered from disk and a child still in the arena.
        let extent = store.memory.allocate(4096).unwrap();
        insert_memory_node(&mut store, id, &value, page_key(1), None, 1, extent);
        move_to_disk(
            &mut store,
            id,
            page_key(1),
            base.join("records").join("parent.ckpt"),
        );
        let header =
            crate::record::new_header(&value, page_key(2), Some(page_key(1)), [0; 16], 1).unwrap();
        let mut extent = store
            .memory
            .allocate(header.length_with_row() as usize)
            .unwrap();
        extent.bytes_mut()[..4096].copy_from_slice(&header.encode().unwrap());
        insert_memory_node(
            &mut store,
            id,
            &value,
            page_key(2),
            Some(page_key(1)),
            1,
            extent,
        );
        let (events, mut receiver) = tokio::sync::mpsc::channel(32);
        store.submit_get(
            1,
            1,
            GetRequest {
                namespace: id,
                keys: vec![page_key(1), page_key(2)],
                skip_pages: 0,
                expected_pages: 2,
                row: true,
            },
            events,
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        // The gate holds the parent's metadata read; releasing it lets the metadata arrive.
        store.pump();
        entered
            .recv_timeout(Duration::from_secs(10))
            .expect("the metadata read never reached the disk thread");
        release.0.send(()).unwrap();
        loop {
            while let Ok(answer) = receiver.try_recv() {
                if let Ok(GetAnswer::Refused(refused)) = answer {
                    panic!("the Get was refused: {}", refused.detail);
                }
            }
            if get_in(&mut store)
                .records
                .iter()
                .all(|record| record.meta_sent)
            {
                break;
            }
            assert!(Instant::now() < deadline, "the metadata never arrived");
            store.pump();
            std::thread::sleep(Duration::from_millis(1));
        }
        let grants = get_in(&mut store).grants.clone();
        assert!(grants.len() >= 2);
        assert_eq!(
            grants[0].record, 0,
            "the first window is the streamed parent's"
        );
        assert!(
            grants[1..].iter().all(|grant| grant.record == 1),
            "the later windows are the resident child's"
        );
        // Hold the parent's window read on the disk thread and give credit for every window.
        gate.arm();
        get_in(&mut store).requested = grants.len();
        store.pump();
        entered
            .recv_timeout(Duration::from_secs(10))
            .expect("the window read never reached the disk thread");
        for _ in 0..5 {
            store.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(get_in(&mut store).window_reads, 1);
        assert_eq!(
            get_in(&mut store).next_grant,
            1,
            "a resident window was answered ahead of the streamed window before it"
        );
        release.0.send(()).unwrap();
        let mut ready = Vec::new();
        while ready.len() < grants.len() {
            assert!(Instant::now() < deadline, "answered windows {ready:?}");
            store.pump();
            while let Ok(answer) = receiver.try_recv() {
                if let Ok(GetAnswer::Ready(window)) = answer {
                    ready.push(window.index);
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(ready, (0..grants.len() as u32).collect::<Vec<_>>());
        drop(store);
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn closed_get_keeps_one_operation_until_its_reads_drain() {
        let (mut store, directory) = store("get-drain");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), None, value)
            .unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        let get = Get {
            namespace: id,
            pins: VecDeque::new(),
            records: Vec::new(),
            grants: Vec::new(),
            requested: 0,
            next_grant: 0,
            next_meta_read: 0,
            next_meta: 0,
            window_reads: 0,
            pending_reads: 2,
            cancelled: false,
            terminal: true,
            finished: false,
            events,
            outbox: VecDeque::new(),
        };
        store.sessions.get_mut(&1).unwrap().ops.insert(
            1,
            Open {
                operation: Operation::Get(Box::new(get)),
                progress: Instant::now(),
            },
        );
        store.connection_closed(1);
        assert!(store.sessions[&1].closed);
        for expected in [1, 0] {
            let extent = store.staging.allocate(4096).unwrap();
            store.get_read_done(1, 1, 0, ReadKind::Meta { length: 0 }, Some(extent));
            store.flush_outboxes();
            let Operation::Get(get) = &store.sessions[&1].ops[&1].operation else {
                panic!("expected Get");
            };
            assert_eq!(get.pending_reads, expected);
        }
        store.cleanup_turn();
        store.flush_outboxes();
        assert!(!store.sessions.contains_key(&1));
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn reads_behind_a_backlogged_read_are_submitted_in_order() {
        use std::time::{Duration, Instant};
        let name = format!(
            "selene-unit-{}-read-order-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            1 << 20,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 << 20;
        let (gate, entered, release) = crate::disk::IoGate::new();
        config.faults.read_gate = Some(gate);
        let mut store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        let id = store.register_namespace(descriptor()).unwrap();
        // The disk thread answers jobs in the order it receives them. A span that is not
        // block aligned fails at once without touching the file, so a reply only records
        // when its job ran.
        let file = Arc::new(File::open("/dev/null").unwrap());
        let read = |job: u64| DiskJob::ReadRecord {
            id: JobId(job),
            file: file.clone(),
            span: Span {
                address: 1,
                offset: 0,
                length: 1,
            },
        };
        let mut order = Vec::new();
        let mut collect = |store: &Store, count: usize| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while order.len() < count {
                assert!(
                    Instant::now() < deadline,
                    "disk replies stopped at {order:?}"
                );
                let mut replies = Vec::new();
                store.namespaces[&id]
                    .reads
                    .as_ref()
                    .unwrap()
                    .poll(&mut replies);
                order.extend(replies.iter().map(|reply| reply.id.0));
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        // The gate holds the first job on the thread, so the next QUEUE_DEPTH jobs fill its
        // queue and the one after them waits in the backlog.
        store.submit_read(id, 0, read(0));
        entered
            .recv_timeout(Duration::from_secs(10))
            .expect("the read never reached the disk thread");
        let queued = flare_selene::shared::disk::QUEUE_DEPTH as u64;
        for job in 1..=queued + 1 {
            store.submit_read(id, job, read(job));
        }
        assert_eq!(store.read_backlog.len(), 1);
        // Every job but the backlogged one finishes, which leaves the queue empty.
        release.send(()).unwrap();
        collect(&store, queued as usize + 1);
        assert_eq!(store.read_backlog.len(), 1);
        // With room in the queue a later read must still wait behind the backlogged one,
        // or it would run first.
        store.submit_read(id, queued + 2, read(queued + 2));
        store.pump_reads();
        collect(&store, queued as usize + 3);
        assert_eq!(order, (0..queued + 3).collect::<Vec<_>>());
        assert!(store.read_backlog.is_empty());
        drop(store);
        let _ = fs::remove_dir_all(&directory);
        let _ = fs::remove_dir_all(base);
    }

    /// A store holding a Get on connection 1, transfer 1, whose two streamed windows were
    /// planned and whose reads are both in flight.
    fn get_with_two_windows(label: &str) -> (Store, PathBuf) {
        let (mut store, directory) = store(label);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.namespaces.get_mut(&id).unwrap().reads_pending = 2;
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let record = GetRecord {
            header: crate::record::new_header(&value, page_key(1), None, [0; 16], 1).unwrap(),
            memory: None,
            file: None,
            meta: None,
            meta_sent: false,
        };
        let window = |page_first| PlannedGrant {
            record: 0,
            record_offset: page_first * 4096,
            length: 4096,
            pages: 1,
            row: false,
            page_first,
        };
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        let get = Get {
            namespace: id,
            pins: VecDeque::new(),
            records: vec![record],
            grants: vec![window(0), window(1)],
            requested: 0,
            next_grant: 2,
            // Its one record's reads are already in flight.
            next_meta_read: 1,
            next_meta: 0,
            window_reads: 2,
            pending_reads: 2,
            cancelled: false,
            terminal: false,
            finished: false,
            events,
            outbox: VecDeque::new(),
        };
        store.sessions.get_mut(&1).unwrap().ops.insert(
            1,
            Open {
                operation: Operation::Get(Box::new(get)),
                progress: Instant::now(),
            },
        );
        (store, directory)
    }

    fn get_in(store: &mut Store) -> &mut Get {
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Get(get) = &mut open.operation else {
            panic!("expected Get");
        };
        get
    }

    fn window_read_done(store: &mut Store, grant: u32) {
        let extent = store.staging.allocate(4096).unwrap();
        store.get_read_done(1, 1, 0, ReadKind::Window { grant }, Some(extent));
    }

    #[test]
    fn a_get_waiting_on_its_reads_is_not_a_client_stall() {
        let (mut store, directory) = get_with_two_windows("get-waits-on-disk");
        assert!(store.sessions[&1].ops[&1].waits_on_disk());
        get_in(&mut store).pending_reads = 0;
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        get_in(&mut store).pending_reads = 2;
        // Its namespace's reads thread moved just now, so a Get that has heard nothing
        // for longer than the limit is not expired.
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        open.progress = Instant::now() - NO_PROGRESS * 2;
        store.timeouts();
        assert!(!get_in(&mut store).terminal);
        // Releasing a window is progress for the Get itself.
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        open.progress = Instant::now() - NO_PROGRESS * 2;
        window_read_done(&mut store, 1);
        assert!(store.sessions[&1].ops[&1].progress.elapsed() < NO_PROGRESS);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_get_whose_namespace_reads_made_no_progress_expires_while_the_disk_moves() {
        let (mut store, directory) = get_with_two_windows("hung-reads");
        let id = get_in(&mut store).namespace;
        // Reads for this namespace were submitted long ago and none has completed.
        store.namespaces.get_mut(&id).unwrap().read_progress = Instant::now() - NO_PROGRESS * 2;
        // Other jobs keep completing, so the disk as a whole is moving.
        let buffer = store.staging.allocate(4096).unwrap();
        store.jobs.insert(
            u64::MAX,
            Job::GetRead {
                namespace: id,
                connection: 1,
                transfer: 1,
                record: 0,
                kind: ReadKind::Meta { length: 0 },
                buffer,
            },
        );
        store.disk_progress = Instant::now();
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        open.progress = Instant::now() - NO_PROGRESS * 2;
        store.timeouts();
        assert!(
            get_in(&mut store).terminal,
            "a Get behind a hung reads thread was not expired"
        );
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn namespace_registration_is_first_sight_and_rejects_mismatch() {
        let (mut store, directory) = store("register");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        assert_eq!(id, value.namespace_id());
        assert!(store.namespaces.contains_key(&id));
        assert_eq!(store.register_namespace(value.clone()).unwrap(), id);
        let mut mismatch = value;
        mismatch.ranks[0].row_bytes += 1;
        assert_eq!(mismatch.namespace_id(), id);
        let error = store.register_namespace(mismatch).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    fn insert_node(
        store: &mut Store,
        id: NamespaceId,
        value: &Descriptor,
        key: Key,
        parent: Option<Key>,
        tail: u32,
    ) {
        let header = crate::record::new_header(value, key, parent, [0; 16], tail).unwrap();
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                parent,
                [0; 16],
                tail as usize,
                true,
                true,
                Location {
                    header,
                    memory: None,
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    admitted: Instant::now(),
                    bytes: 0,
                },
            )
            .unwrap();
    }

    /// The durability refresh and scheduling after a publish must not grow with the number of
    /// records held, even with a backlog of records that are not durable. (It calls them
    /// directly rather than publishing.) Release builds only: `cargo test --release --features test-support -- --ignored`.
    #[test]
    #[ignore = "scale test; release builds only"]
    fn publishing_into_a_large_store_takes_under_a_millisecond() {
        let (mut store, base, _directory) = store_with_root("scale", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let extent = || Some(Arc::new(DiskExtent::new(base.join("record"))));
        let insert = |store: &mut Store, key: Key, parent: Option<Key>, on_disk: bool| {
            let header = crate::record::new_header(&value, key, parent, [0; 16], 1).unwrap();
            let location = Location {
                header,
                memory: None,
                disk: if on_disk { extent() } else { None },
                durable: false,
                writing: false,
                job: None,
                retry: Instant::now(),
                admitted: Instant::now(),
                bytes: 0,
            };
            let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
            tree.insert(key, parent, [0; 16], 1, true, true, location)
                .unwrap();
        };
        let mut last = None;
        for chain in 0..6_250u64 {
            let mut parent = None;
            for depth in 0..16u64 {
                let mut digest = [0; 16];
                digest[..8].copy_from_slice(&chain.to_le_bytes());
                digest[8] = depth as u8;
                let key = Key {
                    consumed_tokens: (depth + 1) * 64,
                    digest,
                };
                insert(&mut store, key, parent, true);
                parent = Some(key);
            }
            last = parent;
        }
        store.refresh_durability(id, None);
        assert!(store.namespaces[&id].tree.nodes().all(|n| n.location.durable));
        // A backlog of records that are not on disk yet.
        for i in 0..10_000u64 {
            let mut digest = [7; 16];
            digest[..8].copy_from_slice(&i.to_le_bytes());
            let key = Key {
                consumed_tokens: 17 * 64,
                digest,
            };
            insert(&mut store, key, last, false);
            store.refresh_durability(id, Some(key));
        }
        assert_eq!(store.namespaces[&id].undurable.len(), 10_000);
        let leaf = Key {
            consumed_tokens: 17 * 64,
            digest: [255; 16],
        };
        // A put published from memory is not on disk yet, so the backlog is not scanned.
        insert(&mut store, leaf, last, false);
        let started = Instant::now();
        store.refresh_durability(id, Some(leaf));
        let elapsed = started.elapsed();
        store.schedule();
        assert!(store.namespaces[&id].undurable.contains(&leaf));
        assert!(elapsed < Duration::from_millis(1), "took {elapsed:?}");
        // A streamed publish arrives with its disk extent and a durable parent.
        let streamed = Key {
            consumed_tokens: 17 * 64,
            digest: [254; 16],
        };
        insert(&mut store, streamed, last, true);
        let started = Instant::now();
        store.refresh_durability(id, Some(streamed));
        let elapsed = started.elapsed();
        assert!(store.namespaces[&id].tree[&streamed].location.durable);
        assert!(elapsed < Duration::from_millis(1), "streamed took {elapsed:?}");
        // The incremental refreshes agree with a full rescan.
        let flags = |store: &Store| {
            let mut all: Vec<_> = store.namespaces[&id]
                .tree
                .nodes()
                .map(|node| (node.key, node.location.durable))
                .collect();
            all.sort_unstable();
            all
        };
        let incremental = flags(&store);
        store.refresh_durability(id, None);
        assert!(incremental == flags(&store), "a full rescan changed durability");
        assert_eq!(store.namespaces[&id].undurable.len(), 10_001);
        drop(store);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn catalog_snapshot_precedes_ordered_unbounded_deltas() {
        let (mut store, directory) = store("catalog");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        insert_node(&mut store, id, &value, key, None, 1);
        store.delta(id, CatalogKind::Added, key);
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let snapshot = store.catalog(value, sender).unwrap();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].key, key);
        assert!(receiver.try_recv().is_err());
        for _ in 0..70_000 {
            store.delta(id, CatalogKind::Changed, key);
        }
        store.delta(id, CatalogKind::Removed, key);
        for _ in 0..70_000 {
            assert!(
                matches!(receiver.try_recv().unwrap(), proto::CatalogEvent::Changed(entry) if entry.key == key)
            );
        }
        assert!(
            matches!(receiver.try_recv().unwrap(), proto::CatalogEvent::Removed(removed) if removed == key)
        );
        assert!(receiver.try_recv().is_err());
        drop(receiver);
        store.delta(id, CatalogKind::Removed, key);
        assert!(store.catalogs.is_empty());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_pid_zero_put_with_an_absent_parent_falls_back_to_a_root() {
        let (mut store, directory) = store("parent");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        let parent = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let key = Key {
            consumed_tokens: 128,
            digest: [9; 16],
        };
        let (events, mut receiver) = tokio::sync::mpsc::channel(4);
        let (reply, _answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::PutBegin {
            ancestor_keys: Vec::new(),
            connection: 1,
            transfer: 1,
            namespace: id,
            descriptor: value.clone(),
            key,
            published_parent: Some(parent),
            bucket: [0; 16],
            chain_pages: 2,
            published_tail_pages: 1,
            capture_role: CaptureRole::Legacy,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
                packed: None,
            events,
            reply,
        });
        store.pump();
        match receiver.try_recv().unwrap().unwrap() {
            PutAnswer::Accept(accept) => {
                assert!(accept.parent.is_none());
                assert_eq!(accept.first_page, 0);
                let header = crate::record::new_header(&value, key, None, [0; 16], 2).unwrap();
                assert_eq!(accept.record_bytes, header.length_with_row());
            }
            PutAnswer::Refused(refused) => panic!("expected a root accept: {}", refused.detail),
            _ => panic!("expected an accept"),
        }
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_local_put_with_an_unusable_parent_falls_back_to_a_root() {
        let (mut store, directory) = store("local-parent");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), None, value.clone())
            .unwrap();
        let parent = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let key = Key {
            consumed_tokens: 128,
            digest: [9; 16],
        };
        let (events, mut receiver) = tokio::sync::mpsc::channel(4);
        let (reply, _answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::PutBegin {
            ancestor_keys: Vec::new(),
            connection: 1,
            transfer: 1,
            namespace: id,
            descriptor: value.clone(),
            key,
            published_parent: Some(parent),
            bucket: [0; 16],
            chain_pages: 2,
            published_tail_pages: 1,
            capture_role: CaptureRole::Legacy,
            child_paths: 0,
            descendant_row: false,
            next_child: None,
            row: true,
                packed: None,
            events,
            reply,
        });
        store.pump();
        match receiver.try_recv().unwrap().unwrap() {
            PutAnswer::Accept(accept) => assert!(
                accept.parent.is_none(),
                "a local Put whose parent hint is unusable must be accepted as a root"
            ),
            PutAnswer::Refused(refused) => panic!("expected an accept: {}", refused.detail),
            _ => panic!("expected an accept"),
        }
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    /// The record holding `pages` pages of a chain, as the tests below name it.
    fn page_key(pages: u64) -> Key {
        Key {
            consumed_tokens: pages * 64,
            digest: [pages as u8; 16],
        }
    }

    /// Records at the given page counts, each linked to the one before and resident.
    fn insert_chain(store: &mut Store, id: NamespaceId, value: &Descriptor, pages: &[u64]) {
        let mut parent = None;
        let mut first = 0;
        for &end in pages {
            let tail = (end - first) as u32;
            let header =
                crate::record::new_header(value, page_key(end), parent, [0; 16], tail).unwrap();
            let extent = store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap();
            insert_memory_node(store, id, value, page_key(end), parent, tail, extent);
            parent = Some(page_key(end));
            first = end;
        }
    }

    /// A store with a connected session holding a chain of resident records, of
    /// which those ending at `page_only` have lost their rows.
    fn held_chain(
        label: &str,
        memory: usize,
        pages: &[u64],
        page_only: &[u64],
    ) -> (Store, PathBuf, NamespaceId) {
        let (mut store, directory) = dram_store(label, memory);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        insert_chain(&mut store, id, &value, pages);
        for &end in page_only {
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .drop_row(&page_key(end));
        }
        (store, directory, id)
    }

    /// A put of `key` whose published parent is `parent`, answered by the first reply.
    fn first_put_answer(
        store: &mut Store,
        id: NamespaceId,
        key: Key,
        parent: Option<Key>,
        ancestor_keys: Vec<Key>,
        row: bool,
    ) -> PutAnswer {
        let chain_pages = key.consumed_tokens / 64;
        let (events, mut answers) = tokio::sync::mpsc::channel(16);
        store.begin_put(PendingPut {
            inflight_since: None,
            longest_inflight_wait: Duration::ZERO,
            connection: 1,
            transfer: 1,
            request: PutRequest {
                namespace: id,
                key,
                published_parent: parent,
                ancestor_keys,
                bucket: [0; 16],
                chain_pages: chain_pages as u32,
                published_tail_pages: (chain_pages
                    - parent.map_or(0, |parent| parent.consumed_tokens / 64))
                    as u32,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row,
                packed: None,
            },
            events,
        });
        store.flush_outboxes();
        answers.try_recv().unwrap().unwrap()
    }

    #[test]
    fn a_page_only_put_without_a_held_ancestor_is_unlinked_and_not_an_error() {
        for parent_held in [false, true] {
            let (mut store, directory, id) = held_chain(
                "page-only-root",
                16 << 10,
                if parent_held { &[1] } else { &[] },
                &[],
            );
            let parent = page_key(1);
            let answer = first_put_answer(
                &mut store,
                id,
                page_key(2),
                parent_held.then_some(parent),
                Vec::new(),
                false,
            );
            // Only a root could hold it when no ancestor is held, and a root without a
            // row ends no restore. With a held parent the arena is the limit.
            let reason = if parent_held {
                proto::Reason::Budget
            } else {
                proto::Reason::Invalid
            };
            assert!(
                matches!(answer, PutAnswer::Refused(refusal) if refusal.reason == reason as u8)
            );
            assert!(store.sessions[&1].ops.is_empty());
            assert_eq!(store.metrics.puts_error, 0);
            assert_eq!(store.metrics.puts_unlinked, u64::from(!parent_held));
            assert!(store.metrics_text().contains(&format!(
                "selene_puts_total{{outcome=\"unlinked\"}} {}\n",
                u8::from(!parent_held)
            )));
            if parent_held {
                assert!(
                    store.namespaces[&id].tree[&parent]
                        .location
                        .memory
                        .is_some()
                );
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    /// A held ancestor in a chain of `REROOT_DEPTH` records does not count as one to link
    /// to: the put would start a new root, and a root needs a row.
    #[test]
    fn a_page_only_put_below_a_chain_at_the_reroot_depth_is_unlinked() {
        for (depth, row, linked) in [
            (REROOT_DEPTH - 1, false, true),
            (REROOT_DEPTH, false, false),
            (REROOT_DEPTH, true, true),
        ] {
            let (mut store, directory, id) = held_chain("page-only-reroot", 64 << 20, &[1], &[]);
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .node_mut(&page_key(1))
                .unwrap()
                .chain_depth = depth;
            let answer = first_put_answer(
                &mut store,
                id,
                page_key(2),
                Some(page_key(1)),
                Vec::new(),
                row,
            );
            match answer {
                PutAnswer::Accept(accept) => {
                    assert!(linked, "depth {depth}, row {row}");
                    // A put that cannot link starts a root, with every page of the chain.
                    let expected = if depth == REROOT_DEPTH {
                        (None, 0)
                    } else {
                        (Some(page_key(1)), 1)
                    };
                    assert_eq!((accept.parent, accept.first_page), expected);
                    assert_eq!(store.metrics.puts_unlinked, 0);
                }
                PutAnswer::Refused(refusal) => {
                    assert!(!linked, "depth {depth}, row {row}");
                    assert_eq!(refusal.reason, proto::Reason::Invalid as u8);
                    assert_eq!(refusal.detail, PUT_UNLINKED);
                    assert_eq!(store.metrics.puts_unlinked, 1);
                    assert_eq!(store.metrics.puts_error, 0);
                }
                _ => panic!("unexpected answer"),
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn puts_keep_the_row_they_send_and_pages_already_held_are_present() {
        // The chain holds a row-bearing record of one page and a page-only record of two.
        let empty = Key {
            consumed_tokens: 130,
            digest: [130; 16],
        };
        // (key, its parent in the chain, row, counted shape)
        for (key, parent, row, shape) in [
            (page_key(3), page_key(2), true, Some("row")),
            (page_key(3), page_key(2), false, Some("page_only")),
            (empty, page_key(2), false, Some("empty")),
            (empty, page_key(2), true, Some("row")),
            // A held page-only record is rewritten only to gain its row.
            (page_key(2), page_key(1), true, Some("row")),
            (page_key(2), page_key(1), false, None),
        ] {
            let (mut store, directory, id) = held_chain("row-kept-put", 64 << 20, &[1, 2], &[2]);
            let accepted = matches!(shape, Some("row" | "page_only"));
            match first_put_answer(&mut store, id, key, Some(parent), Vec::new(), row) {
                PutAnswer::Accept(accept) => {
                    assert!(accepted, "{key:?}, row={row}");
                    let put = put_ref(&store.sessions, 1, 1).unwrap();
                    assert_eq!(accept.record_bytes, put.header.length(row));
                    assert_eq!(put.grants.iter().any(|grant| grant.row), row);
                    assert_eq!(
                        store.metrics.put_row_bytes,
                        if row {
                            put.header.row_record_bytes()
                        } else {
                            0
                        }
                    );
                    assert_eq!(store.metrics.present, 0);
                }
                PutAnswer::Refused(refused) => {
                    assert!(!accepted, "{key:?}, row={row}");
                    assert_eq!(refused.reason, proto::Reason::Present as u8);
                    assert_eq!(store.metrics.put_page_bytes, 0);
                    assert_eq!(store.metrics.present, 1);
                }
                _ => panic!("unexpected answer"),
            }
            let metrics = store.metrics_text();
            for name in ["row", "page_only", "empty"] {
                let count = u8::from(shape == Some(name));
                assert!(
                    metrics.contains(&format!(
                        "selene_put_shapes_total{{shape=\"{name}\"}} {count}\n"
                    )),
                    "{key:?}, row={row}, shape {name}"
                );
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn put_rows_follow_capture_role_and_topology() {
        for (parent, tokens, role, children, expected_row) in [
            (None, 64, CaptureRole::Endpoint, 1, true),
            (Some(128), 192, CaptureRole::Endpoint, 1, false),
            (Some(128), 192, CaptureRole::Endpoint, 0, true),
            (Some(128), 192, CaptureRole::Endpoint, 2, true),
            (Some(128), 192, CaptureRole::Message, 1, true),
            (Some(128), 192, CaptureRole::Anchor, 1, true),
            (Some(128), 192, CaptureRole::Legacy, 1, true),
        ] {
            let (mut store, directory) = dram_store("anchor-put", 64 << 20);
            let value = descriptor();
            let id = store.register_namespace(value.clone()).unwrap();
            store.hello(1, 0, None, value.clone()).unwrap();
            let key = |tokens| Key {
                consumed_tokens: tokens,
                digest: [tokens as u8; 16],
            };
            for (tokens, parent) in [(64, None), (128, Some(key(64)))] {
                let header =
                    crate::record::new_header(&value, key(tokens), parent, [0; 16], 1).unwrap();
                let extent = store
                    .memory
                    .allocate(header.length_with_row() as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key(tokens), parent, 1, extent);
            }
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .drop_row(&key(128));
            let (events, mut answers) = tokio::sync::mpsc::channel(16);
            // Use a distinct digest for a root at the same token count.
            let endpoint = Key {
                consumed_tokens: tokens,
                digest: [99; 16],
            };
            store.begin_put(PendingPut {
                inflight_since: None,
                longest_inflight_wait: Duration::ZERO,
                connection: 1,
                transfer: 1,
                request: PutRequest {
                    namespace: id,
                    key: endpoint,
                    published_parent: parent.map(key),
                    ancestor_keys: Vec::new(),
                    bucket: [0; 16],
                    chain_pages: (tokens / 64) as u32,
                    published_tail_pages: (tokens / 64 - parent.unwrap_or(0) / 64) as u32,
                    capture_role: role,
                    child_paths: children,
                    descendant_row: children == 1,
                    next_child: None,
                    row: true,
                packed: None,
                },
                events,
            });
            store.flush_outboxes();
            let PutAnswer::Accept(accept) = answers.try_recv().unwrap().unwrap() else {
                panic!("expected accept");
            };
            assert_eq!(accept.capture_role, role);
            let put = put_ref(&store.sessions, 1, 1).unwrap();
            assert_eq!(accept.record_bytes, put.header.length(expected_row));
            assert_eq!(put.grants.iter().any(|grant| grant.row), expected_row);
            assert_eq!(
                store.metrics.put_row_bytes,
                if expected_row {
                    put.header.row_record_bytes()
                } else {
                    0
                }
            );
            let shape = if expected_row { "row" } else { "page_only" };
            assert!(
                store
                    .metrics_text()
                    .contains(&format!("selene_put_shapes_total{{shape=\"{shape}\"}} 1\n"))
            );
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn a_held_page_only_record_is_rewritten_only_to_become_an_anchor() {
        for (role, children, rewritten) in [
            (CaptureRole::Endpoint, 1, false),
            (CaptureRole::Message, 1, true),
            (CaptureRole::Endpoint, 2, true),
        ] {
            let (mut store, directory) = dram_store("page-only-reput", 64 << 20);
            let value = descriptor();
            let id = store.register_namespace(value.clone()).unwrap();
            store.hello(1, 0, None, value.clone()).unwrap();
            let key = |tokens| Key {
                consumed_tokens: tokens,
                digest: [tokens as u8; 16],
            };
            for (tokens, parent) in [(64, None), (128, Some(key(64)))] {
                let header =
                    crate::record::new_header(&value, key(tokens), parent, [0; 16], 1).unwrap();
                let extent = store
                    .memory
                    .allocate(header.length_with_row() as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key(tokens), parent, 1, extent);
            }
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .drop_row(&key(128));
            let loc = &mut store.namespaces.get_mut(&id).unwrap().tree[&key(128)].location;
            loc.header = loc.header.clone().with_capture_role(CaptureRole::Endpoint);
            let (events, mut answers) = tokio::sync::mpsc::channel(16);
            store.begin_put(PendingPut {
                inflight_since: None,
                longest_inflight_wait: Duration::ZERO,
                connection: 1,
                transfer: 1,
                request: PutRequest {
                    namespace: id,
                    key: key(128),
                    published_parent: Some(key(64)),
                    ancestor_keys: Vec::new(),
                    bucket: [0; 16],
                    chain_pages: 2,
                    published_tail_pages: 1,
                    capture_role: role,
                    child_paths: children,
                    descendant_row: children == 1,
                    next_child: None,
                    row: true,
                packed: None,
                },
                events,
            });
            store.flush_outboxes();
            match answers.try_recv().unwrap().unwrap() {
                PutAnswer::Accept(accept) => {
                    assert!(rewritten, "role {role:?}: a page-only record was rewritten");
                    assert_eq!(accept.capture_role, role);
                    let put = put_ref(&store.sessions, 1, 1).unwrap();
                    assert!(put.grants.iter().any(|grant| grant.row));
                }
                PutAnswer::Refused(refused) => {
                    assert!(!rewritten, "role {role:?}: the anchor row was refused");
                    assert_eq!(refused.reason, proto::Reason::Present as u8);
                    assert_eq!(store.metrics.put_page_bytes, 0);
                }
                _ => panic!("unexpected answer"),
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn put_selects_the_deepest_held_ancestor_including_page_only_records() {
        for (candidates, published, unready, expected) in [
            (vec![5, 3, 1], None, None, Some(3)),
            (vec![5, 4, 3, 1], None, None, Some(4)),
            (vec![5, 1], None, None, Some(1)),
            (vec![5], None, None, None),
            (vec![], Some(3), None, Some(3)),
            (vec![5, 4, 3, 1], None, Some(4), Some(3)),
            (vec![], Some(4), Some(4), None),
        ] {
            let (mut store, directory, id) = held_chain("ancestor-put", 64 << 20, &[1, 3, 4], &[3]);
            if let Some(unready) = unready {
                store
                    .namespaces
                    .get_mut(&id)
                    .unwrap()
                    .tree
                    .set_ready(&page_key(unready), false)
                    .unwrap();
            }
            let pages = candidates.first().copied().unwrap_or(5).max(5) + 1;
            let answer = first_put_answer(
                &mut store,
                id,
                page_key(pages),
                published.map(page_key),
                candidates.into_iter().map(page_key).collect(),
                true,
            );
            let PutAnswer::Accept(accept) = answer else {
                panic!("expected an accept");
            };
            assert_eq!(accept.parent, expected.map(page_key));
            assert_eq!(accept.first_page, expected.unwrap_or(0) as u32);
            let header = crate::record::new_header(
                &descriptor(),
                page_key(pages),
                expected.map(page_key),
                [0; 16],
                pages as u32 - accept.first_page,
            )
            .unwrap();
            assert_eq!(accept.record_bytes, header.length_with_row());
            let metrics = store.metrics_text();
            for line in [
                format!(
                    "selene_put_links_total{{parent=\"linked\"}} {}\n",
                    u8::from(expected.is_some())
                ),
                format!(
                    "selene_put_links_total{{parent=\"root\"}} {}\n",
                    u8::from(expected.is_none())
                ),
                format!(
                    "selene_put_bytes_total{{part=\"pages\"}} {}\n",
                    u64::from(header.tail_pages) * header.page_record_bytes()
                ),
                format!(
                    "selene_put_bytes_total{{part=\"rows\"}} {}\n",
                    header.row_record_bytes()
                ),
            ] {
                assert!(metrics.contains(&line), "missing metric {line}");
            }
            if let Some(parent) = accept.parent {
                assert!(store.namespaces[&id].tree[&parent].pinned());
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn an_ancestor_list_must_be_strictly_decreasing_and_end_before_the_record() {
        // The record ends at page 6, and pages 1, 3 and 4 are held.
        for (candidates, accepted) in [
            (vec![5, 4, 3], true),
            (vec![4, 1], true),
            (vec![6], false),
            (vec![7, 3], false),
            (vec![4, 4], false),
            (vec![3, 4], false),
            (vec![5, 3, 3], false),
            (vec![5, 1, 3], false),
        ] {
            let (mut store, directory, id) =
                held_chain("ancestor-order", 64 << 20, &[1, 3, 4], &[]);
            let errors = store.metrics.puts_error;
            let answer = first_put_answer(
                &mut store,
                id,
                page_key(6),
                None,
                candidates.iter().copied().map(page_key).collect(),
                true,
            );
            match answer {
                PutAnswer::Accept(_) => assert!(accepted, "{candidates:?} was accepted"),
                PutAnswer::Refused(refused) => {
                    assert!(!accepted, "{candidates:?} was refused");
                    assert_eq!(refused.reason, proto::Reason::Invalid as u8);
                    assert_eq!(store.metrics.puts_error, errors + 1);
                    assert!(!store.namespaces[&id].inflight.contains(&page_key(6)));
                }
                _ => panic!("unexpected answer for {candidates:?}"),
            }
            drop(store);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn a_page_only_record_is_replaced_by_a_complete_put() {
        let (mut store, directory) = store("publish-target");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [6; 16],
        };
        let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
        let extent = Arc::new(
            store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap(),
        );
        let bytes = record_charge(&header, true);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                None,
                [0; 16],
                1,
                true,
                true,
                Location {
                    header: header.clone(),
                    memory: Some(extent),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    admitted: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
        // Compaction drops the row, keeping the pages as a replacement target.
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .node_mut(&key)
            .unwrap()
            .row = false;
        assert!(
            matches!(store.publish_target(id, &header), PublishTarget::Replace),
            "a page-only record must be replaced, not treated as present"
        );
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn replacing_a_durable_record_makes_its_children_wait_for_the_rewrite() {
        let (mut store, base, _root) = store_with_root("replace-durable", 1 << 20);
        let value = descriptor();
        store.hello(1, std::process::id(), None, value.clone()).unwrap();
        let id = value.namespace_id();
        let parent = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let child = Key {
            consumed_tokens: 128,
            digest: [2; 16],
        };
        for (key, above, tail) in [(parent, None, 1), (child, Some(parent), 1)] {
            let header = crate::record::new_header(&value, key, above, [0; 16], tail).unwrap();
            let bytes = header.length_with_row();
            store.charged += bytes;
            let location = Location {
                header,
                memory: None,
                disk: Some(Arc::new(DiskExtent::new(base.join(format!("old-{}", key.digest[0]))))),
                durable: false,
                writing: false,
                job: None,
                retry: Instant::now(),
                admitted: Instant::now(),
                bytes,
            };
            let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
            tree.insert(key, above, [0; 16], tail as usize, true, true, location)
                .unwrap();
        }
        store.refresh_durability(id, None);
        assert!(store.namespaces[&id].tree.nodes().all(|n| n.location.durable));
        // Compaction left the parent page-only, so a complete put replaces it.
        store.namespaces.get_mut(&id).unwrap().tree[&parent].row = false;
        let (events, _answers) = tokio::sync::mpsc::channel(8);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key: parent,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 1,
                published_tail_pages: 1,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch();
        let header = put_of(&mut store).header.clone();
        put_of(&mut store).memory = Some(Arc::new(
            store.memory.allocate(header.length_with_row() as usize).unwrap(),
        ));
        store.publish_put(1, 1, None);
        let namespace = &store.namespaces[&id];
        assert!(!namespace.tree[&parent].location.durable);
        assert!(
            !namespace.tree[&child].location.durable,
            "a child cannot stay durable under a replaced parent"
        );
        assert!(namespace.undurable.contains(&child));
        store.config.nvme = store.charge_total();
        assert!(!store.make_room_evict(1));
        assert!(store.namespaces[&id].tree[&parent].location.writing);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !store.namespaces[&id].tree[&child].location.durable {
            assert!(Instant::now() < deadline, "the rewrite never completed");
            store.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(store.namespaces[&id].tree[&parent].location.durable);
        assert!(store.namespaces[&id].undurable.is_empty());
        assert!(store.make_room_evict(1));
        drop(store);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn a_get_skips_only_zero_page_ancestors() {
        let (mut store, directory) = store("get-zero-pages");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = |tokens| Key {
            consumed_tokens: tokens,
            digest: [7; 16],
        };
        insert_node(&mut store, id, &value, key(64), None, 1);
        insert_node(&mut store, id, &value, key(65), Some(key(64)), 0);
        insert_node(&mut store, id, &value, key(66), Some(key(65)), 0);
        insert_node(&mut store, id, &value, key(128), Some(key(66)), 1);
        insert_node(&mut store, id, &value, key(192), Some(key(128)), 1);
        let (records, grants) = store
            .plan_get(
                id,
                &GetRequest {
                    namespace: id,
                    keys: vec![key(64), key(128)],
                    skip_pages: 1,
                    expected_pages: 1,
                    row: true,
                },
            )
            .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(grants.len(), 1);
        assert_eq!(
            (grants[0].page_first, grants[0].pages, grants[0].row),
            (1, 1, true)
        );
        for keys in [vec![key(64), key(192)], vec![key(128), key(64)]] {
            let error = store
                .plan_get(
                    id,
                    &GetRequest {
                        namespace: id,
                        keys,
                        skip_pages: 0,
                        expected_pages: 2,
                        row: true,
                    },
                )
                .err()
                .expect("a missing page or unrelated key must be refused");
            assert_eq!(error.to_string(), "Get chain");
        }
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_get_without_a_row_is_refused() {
        let (mut store, directory) = store("get-row");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [7; 16],
        };
        insert_node(&mut store, id, &value, key, None, 1);
        let refused = store.plan_get(
            id,
            &GetRequest {
                namespace: id,
                keys: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: false,
            },
        );
        assert!(
            refused.is_err(),
            "the Get contract requires the endpoint row"
        );
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn idle_arena_records_are_lost_on_an_unclean_stop() {
        let (mut store, root, shm) = store_with_root("idle-arena", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [4; 16],
        };
        let extent = store.memory.allocate(16 << 10).unwrap();
        insert_memory_node(&mut store, id, &value, key, None, 1, extent);
        store.refresh_durability(id, Some(key));
        store.pump();
        assert!(
            store.jobs.is_empty(),
            "idle pump wrote an arena-only checkpoint"
        );
        let config = store.config.clone();
        assert!(!store.drain_until(Instant::now()));
        assert!(
            store
                .metrics_text()
                .contains("selene_shutdown_unflushed_records 1\n")
        );
        assert!(store.namespaces[&id].tree[&key].location.memory.is_some());
        drop(store);
        let store = Store::open(config.clone()).unwrap();
        assert!(store.namespaces.values().all(|ns| !ns.tree.contains(&key)));
        assert!(
            store
                .metrics_text()
                .contains("selene_shutdown_unflushed_bytes 16384\n")
        );
        drop(store); // This instance exits without recording a shutdown.
        let store = Store::open(config).unwrap();
        assert!(!store.metrics_text().contains("selene_shutdown_unflushed_"));
        drop(store);
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(shm);
    }

    #[test]
    fn drain_returns_with_a_dirty_record_when_disk_work_outlasts_its_deadline() {
        let (store, base, shm) = store_with_root("drain-deadline", 1 << 20);
        let mut config = store.config.clone();
        drop(store);
        config.faults.delay_ms = 500;
        let mut store = Store::open(config).unwrap();
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let extent = store.memory.allocate(16 << 10).unwrap();
        insert_memory_node(&mut store, id, &value, key, None, 1, extent);
        store.refresh_durability(id, Some(key));
        let started = Instant::now();
        assert!(!store.drain_until(started + Duration::from_millis(30)));
        assert!(started.elapsed() < Duration::from_millis(400));
        assert!(store.namespaces[&id].tree[&key].location.memory.is_some());
        assert!(store.drain_until(Instant::now() + Duration::from_secs(5)));
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(shm);
    }

    #[test]
    fn a_stale_disk_write_does_not_attach_to_a_replacement() {
        let (mut store, directory, _root) = store_with_root("stale-write", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [12; 16],
        };
        let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
        let extent = Arc::new(
            store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap(),
        );
        let bytes = record_charge(&header, true);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                None,
                [0; 16],
                1,
                true,
                true,
                Location {
                    header,
                    memory: Some(extent),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    admitted: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
        // Pressure starts the write before a replacement acquires the same key.
        store.refresh_durability(id, Some(key));
        store.request_writeback(id, key);
        store.pump();
        let job = store.namespaces[&id]
            .tree
            .node(&key)
            .unwrap()
            .location
            .job
            .expect("the write-back was submitted");
        // Pretend the record these bytes belong to was replaced while the write drained.
        let replacement = job + 1000;
        {
            let location = &mut store.namespaces.get_mut(&id).unwrap().tree[&key].location;
            location.job = Some(replacement);
            location.writing = true;
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while store.jobs.contains_key(&job) {
            assert!(Instant::now() < deadline, "the disk write never completed");
            store.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        let location = &store.namespaces[&id].tree[&key].location;
        assert!(
            location.disk.is_none(),
            "a stale write attached its file to the replacement record"
        );
        assert_eq!(location.job, Some(replacement));
        assert!(location.writing, "a stale completion cleared the new write");
        assert!(store.namespaces[&id].tree[&key].pinned());
        drop(store);
        let _ = fs::remove_dir_all(directory);
        // `_root` is the instance's shared-memory directory.
        let _ = fs::remove_dir_all(_root);
    }

    #[test]
    fn a_draining_or_opposite_transfer_id_is_fenced() {
        let (mut store, directory) = store("fence");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [6; 16],
        };
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        let get = |store: &mut Store, transfer: u64| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::GetBegin {
                connection: 1,
                transfer,
                namespace: id,
                chain: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
                events: get_events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        let put = |store: &mut Store, transfer: u64| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::PutBegin {
                ancestor_keys: Vec::new(),
                connection: 1,
                transfer,
                namespace: id,
                descriptor: value.clone(),
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 1,
                published_tail_pages: 1,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
                events: put_events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        assert!(get(&mut store, 7).is_ok());
        // The opposite direction is fenced while the Get is pending.
        assert!(put(&mut store, 7).is_err());
        // A draining Get keeps its id reserved until the disk copy drains.
        store.sessions.get_mut(&1).unwrap().ops.insert(
            8,
            Open {
                operation: Operation::Get(Box::new(Get {
                    namespace: id,
                    pins: VecDeque::new(),
                    records: Vec::new(),
                    grants: Vec::new(),
                    requested: 0,
                    next_grant: 0,
                    next_meta_read: 0,
                    next_meta: 0,
                    window_reads: 0,
                    pending_reads: 1,
                    cancelled: true,
                    terminal: true,
                    finished: false,
                    events: get_events.clone(),
                    outbox: VecDeque::new(),
                })),
                progress: Instant::now(),
            },
        );
        assert!(get(&mut store, 8).is_err());
        assert!(put(&mut store, 8).is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn catalog_feeds_keep_namespaces_separate() {
        let (mut store, directory) = store("multi-sub");
        let first = descriptor();
        let mut second = descriptor();
        second.pool_name = "tenant-b".to_string();
        let (sender_a, mut receiver_a) = tokio::sync::mpsc::unbounded_channel();
        let (sender_b, mut receiver_b) = tokio::sync::mpsc::unbounded_channel();
        assert!(store.catalog(first.clone(), sender_a).unwrap().is_empty());
        assert!(store.catalog(second, sender_b).unwrap().is_empty());
        let key = Key {
            consumed_tokens: 64,
            digest: [5; 16],
        };
        insert_node(&mut store, first.namespace_id(), &first, key, None, 1);
        store.delta(first.namespace_id(), CatalogKind::Added, key);
        assert!(receiver_a.try_recv().is_ok());
        assert!(receiver_b.try_recv().is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_reused_get_transfer_id_is_refused() {
        let (mut store, directory) = store("dup");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [4; 16],
        };
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        let begin = |store: &mut Store| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::GetBegin {
                connection: 1,
                transfer: 7,
                namespace: id,
                chain: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
                events: events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        assert!(begin(&mut store).is_ok());
        assert!(begin(&mut store).is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn rootless_eviction_only_removes_leaves_and_accounts_them() {
        let (mut store, directory) = store("evict");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let root_key = Key {
            consumed_tokens: 64,
            digest: [2; 16],
        };
        let child_key = Key {
            consumed_tokens: 128,
            digest: [3; 16],
        };
        let root_header = crate::record::new_header(&value, root_key, None, [0; 16], 1).unwrap();
        let child_header = crate::record::new_header(&value, child_key, Some(root_key), [0; 16], 1).unwrap();
        for header in [&root_header, &child_header] {
            let extent = Arc::new(store.memory.allocate(4096).unwrap());
            let bytes = record_charge(header, true);
            store.charged += bytes;
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .insert(
                    header.key,
                    header.parent,
                    [0; 16],
                    1,
                    true,
                    true,
                    Location {
                        header: header.clone(),
                        memory: Some(extent),
                        disk: None,
                        durable: false,
                        writing: false,
                        job: None,
                        retry: Instant::now(),
                        admitted: Instant::now(),
                        bytes,
                    },
                )
                .unwrap();
        }
        let charged = store.charged;
        assert!(!store.evict_memory(id, root_key));
        assert_eq!(store.charged, charged);
        assert!(store.namespaces[&id].tree.node(&root_key).is_some());
        assert!(store.evict_memory(id, child_key));
        assert!(store.namespaces[&id].tree.node(&child_key).is_none());
        assert!(store.evict_memory(id, root_key));
        assert_eq!(store.charged, 0);
        assert!(store.namespaces[&id].tree.node(&root_key).is_none());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_chain_admits_the_root_by_successive_leaf_eviction() {
        // A full 64 KiB rootless arena: 32 -> 16 -> 16 KiB. The new record
        // needs the arena only after three successive leaf evictions.
        let (mut store, directory) = dram_store("chain", 64 * 1024);
        let value = descriptor();
        store.hello(1, std::process::id(), None, value.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        let root = store.memory.allocate(32 * 1024).unwrap();
        let child = store.memory.allocate(16 * 1024).unwrap();
        let leaf = store.memory.allocate(16 * 1024).unwrap();
        let (r, c, l) = (
            Key {
                consumed_tokens: 128,
                digest: [1; 16],
            },
            Key {
                consumed_tokens: 192,
                digest: [2; 16],
            },
            Key {
                consumed_tokens: 256,
                digest: [3; 16],
            },
        );
        insert_memory_node(&mut store, id, &value, r, None, 2, root);
        insert_memory_node(&mut store, id, &value, c, Some(r), 1, child);
        insert_memory_node(&mut store, id, &value, l, Some(c), 1, leaf);
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key: Key {
                    consumed_tokens: 384,
                    digest: [4; 16],
                },
                published_parent: Some(l),
                bucket: [0; 16],
                chain_pages: 6,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        let open = &store.sessions[&1].ops[&1];
        let Operation::Put(put) = &open.operation else {
            panic!("expected Put");
        };
        assert!(!put.streaming);
        assert!(put.memory.is_some());
        assert_eq!(put.header.parent, None);
        assert_eq!(put.header.tail_pages, 6);
        let tree = &store.namespaces[&id].tree;
        assert!(tree.node(&r).is_none());
        assert!(tree.node(&c).is_none());
        assert!(tree.node(&l).is_none());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_put_queued_for_credits_is_not_stalled_but_an_idle_granted_one_is() {
        let (mut store, directory) = store("queued-put");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), None, value).unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(8);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        let stale = Instant::now() - NO_PROGRESS * 2;
        let live = |store: &Store| match store.sessions[&1].ops.get(&1) {
            Some(Open {
                operation: Operation::Put(put),
                ..
            }) => !put.terminal,
            _ => false,
        };
        // Waiting for the pool to ask for a window, however long: not a stall.
        store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap().progress =
            stale;
        store.tick();
        assert!(live(&store));
        // After the grant the client owes the data; idle for too long, the put is aborted.
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put");
        };
        put.requested = 1;
        assert!(store.send_put_grant(1, 1));
        store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap().progress =
            stale;
        store.tick();
        assert!(!live(&store));
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    fn idle_put(label: &str) -> (Store, PathBuf) {
        let (mut store, directory) = store(label);
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), None, value).unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(8);
        let namespace = store.namespace_id().unwrap();
        let key = Key {
            consumed_tokens: 128,
            digest: [8; 16],
        };
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace,
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        (store, directory)
    }

    fn put_of(store: &mut Store) -> &mut Put {
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put");
        };
        put
    }

    #[test]
    fn a_failed_put_is_failed_once_however_many_ticks_pass() {
        let (mut store, directory) = idle_put("fail-once");
        put_of(&mut store).requested = 1;
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        open.progress = Instant::now() - NO_PROGRESS * 2;
        for _ in 0..20 {
            store.tick();
        }
        assert_eq!(store.metrics.puts_error, 1);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_put_waiting_on_our_own_disk_is_not_a_client_stall() {
        let (mut store, directory) = idle_put("waits-on-disk");
        let put = put_of(&mut store);
        put.requested = 1;
        put.windows.granted.clear();
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        let slot = |index, job| Some(HoldSlot { index, job: Some(job) });
        put_of(&mut store).hold_slots = [slot(0, 1), slot(1, 2)];
        assert!(store.sessions[&1].ops[&1].waits_on_disk());
        put_of(&mut store).hold_slots = [None, None];
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        let arena = Arena::new(4096).unwrap();
        let extent = Arc::new(arena.allocate(64).unwrap());
        let chunk = ExtentSlice::new(extent, 0, 64);
        put_of(&mut store).blocked.push_back((1, 0, chunk));
        assert!(store.sessions[&1].ops[&1].waits_on_disk());
        // A granted window outstanding is the client's to fill.
        let grant = PlannedGrant {
            record: 0,
            record_offset: 0,
            length: 64,
            pages: 1,
            row: false,
            page_first: 0,
        };
        put_of(&mut store).windows.granted.insert(0, grant);
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn cancel_removes_a_queued_begin() {
        let (mut store, directory) = store("cancel-queued");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), None, value)
            .unwrap();
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            put_events,
        );
        store.cancel(1, 1);
        assert!(store.dispatch_trace().is_empty());
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_get(
            1,
            2,
            GetRequest {
                namespace: store.namespace_id().unwrap(),
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        store.cancel(1, 2);
        assert!(store.dispatch_trace().is_empty());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn cancels_count_only_the_operations_they_stop() {
        let (mut refused_store, directory) = idle_put("cancel-refused");
        refused_store.window_filled(1, 1, 0, Vec::new(), None).unwrap();
        let put = put_of(&mut refused_store);
        assert!(put.terminal);
        assert!(!put.cancelled);
        assert!(matches!(put.outbox.back(), Some(Ok(PutAnswer::Refused(_)))));
        refused_store.flush_outboxes();
        assert!(refused_store.sessions[&1].ops.contains_key(&1));
        assert_eq!(refused_store.metrics.cancels, 0);
        // The pool cancels a refused put to finish its cleanup.
        refused_store.cancel(1, 1);
        assert_eq!(refused_store.metrics.cancels, 0);
        drop(refused_store);
        let _ = fs::remove_dir_all(directory);

        let (mut store, directory) = store("cancel-count");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), None, value).unwrap();
        // Cleanup of an operation that is not there: not a cancellation.
        store.cancel(1, 99);
        assert_eq!(store.metrics.cancels, 0);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            put_events,
        );
        store.cancel(1, 1);
        assert_eq!(store.metrics.cancels, 1);
        // The pool's cleanup Cancel for the same, already withdrawn put.
        store.cancel(1, 1);
        assert_eq!(store.metrics.cancels, 1);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn committed_streamed_put_finishes_on_its_closed_connection() {
        let (mut store, base, directory) = store_with_root("closed-commit", 4096);
        let value = descriptor();
        store.hello(1, std::process::id(), None, value.clone()).unwrap();
        let id = value.namespace_id();
        let key = Key {
            consumed_tokens: 128,
            digest: [11; 16],
        };
        let (events, mut answers) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch();
        let (reply, answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::RequestWindow {
            connection: 1,
            transfer: 1,
            reply,
        });
        answer.blocking_recv().unwrap().unwrap();
        store.pump();
        assert!(matches!(
            answers.try_recv().unwrap().unwrap(),
            PutAnswer::Accept(_)
        ));
        let PutAnswer::Grant(grant) = answers.try_recv().unwrap().unwrap() else {
            panic!("expected grant")
        };
        let payload = vec![0x5a; grant.length as usize];
        // The test writer exclusively owns this granted window until WindowFilled.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                grant
                    .hold
                    .bytes()
                    .as_ptr()
                    .cast_mut()
                    .add((grant.offset as usize) - grant.hold.offset()),
                payload.len(),
            );
        }
        let checksum = flare_selene::shared::record::checksum(&payload[..4096]);
        store
            .window_filled(1, 1, grant.index, vec![checksum; 2], Some(checksum))
            .unwrap();
        drop(grant);
        let Operation::Put(put) = &store.sessions[&1].ops[&1].operation else {
            panic!("expected Put")
        };
        assert!(put.pending > 0);
        let mut header = put.header.clone();
        header.row_checksum = checksum;
        let mut metadata = vec![0; header.page_segment as usize];
        metadata[..4096].copy_from_slice(&header.encode().unwrap());
        for page in 0..2 {
            metadata[4096 + page * 8..4104 + page * 8].copy_from_slice(&checksum.to_le_bytes());
        }
        store.put_commit(1, 1, 0, metadata.clone()).unwrap();
        store.cancel(1, 1);
        store.connection_closed(1);
        assert!(store.sessions[&1].closed);
        let deadline = Instant::now() + Duration::from_secs(5);
        while store.sessions.contains_key(&1) {
            store.pump();
            assert!(Instant::now() < deadline, "committed Put did not drain");
            std::thread::sleep(Duration::from_millis(1));
        }
        let node = &store.namespaces[&id].tree[&key];
        assert!(node.location.durable);
        let bytes = fs::read(&node.location.disk.as_ref().unwrap().path).unwrap();
        metadata.extend_from_slice(&payload);
        assert_eq!(bytes, metadata);
        record::verify_record(&bytes, &header, true).unwrap();
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn cancelled_closed_puts_do_not_submit_blocked_chunks() {
        let (mut store, base, directory) = store_with_root("detached-blocked", 4096);
        let value = descriptor();
        store.hello(1, std::process::id(), None, value).unwrap();
        let id = store.sessions[&1].namespace;
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key: Key {
                    consumed_tokens: 128,
                    digest: [10; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put")
        };
        let slice = ExtentSlice::new(Arc::new(store.staging.allocate(4096).unwrap()), 0, 4096);
        put.blocked.push_back((777, 0, slice));
        store.connection_closed(1);
        store.pump_chunks();
        assert!(!store.jobs.contains_key(&777));
        let Operation::Put(put) = &store.sessions[&1].ops[&1].operation else {
            panic!("expected Put")
        };
        assert!(put.blocked.is_empty());
        assert!(put.cancelled);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn dispatch_prioritises_gets() {
        let (mut store, directory) = store("dispatch");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), None, value)
            .unwrap();
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        for index in 0..4 {
            store.submit_get(
                1,
                100 + index,
                GetRequest {
                    namespace: id,
                    keys: vec![Key {
                        consumed_tokens: 64,
                        digest: [4; 16],
                    }],
                    skip_pages: 0,
                    expected_pages: 1,
                    row: true,
                },
                get_events.clone(),
            );
        }
        for index in 0..4 {
            store.submit_put(
                1,
                200 + index,
                PutRequest {
                    ancestor_keys: Vec::new(),
                    namespace: id,
                    key: Key {
                        consumed_tokens: 128 + index * 64,
                        digest: [5; 16],
                    },
                    published_parent: None,
                    bucket: [0; 16],
                    chain_pages: (128 + index * 64) as u32 / 64,
                    published_tail_pages: (128 + index * 64) as u32 / 64,
                    capture_role: CaptureRole::Legacy,
                    child_paths: 0,
                    descendant_row: false,
                    next_child: None,
                    row: true,
                packed: None,
                },
                put_events.clone(),
            );
        }
        assert_eq!(
            store.dispatch_trace(),
            vec!["get", "get", "get", "get", "put"]
        );
        // One more round: the fifth get still precedes the queued Put.
        store.submit_get(
            1,
            300,
            GetRequest {
                namespace: id,
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        assert_eq!(store.dispatch_trace(), vec!["get", "put"]);
        let _ = id;
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn failed_puts_drop_unsent_chunks() {
        let (mut store, base, directory) = store_with_root("failed-chunks", 4096);
        let value = descriptor();
        store
            .hello(1, std::process::id(), None, value)
            .unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [9; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        {
            let operation = &mut store
                .sessions
                .get_mut(&1)
                .unwrap()
                .ops
                .get_mut(&1)
                .unwrap()
                .operation;
            let Operation::Put(put) = operation else {
                panic!("expected Put");
            };
            let slice = ExtentSlice::new(Arc::new(store.staging.allocate(4096).unwrap()), 0, 4096);
            put.blocked.push_back((999, 0, slice));
        }
        store.fail_put(1, 1, proto::Reason::Io as u8, invalid("test"));
        let operation = &store.sessions[&1].ops[&1].operation;
        let Operation::Put(put) = operation else {
            panic!("expected Put");
        };
        assert!(put.blocked.is_empty());
        store.pump_chunks();
        assert!(store.jobs.keys().all(|job| *job != 999));
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn an_impossible_allocation_preserves_warm_records() {
        let (mut store, base, directory) = store_with_root("oversize-allocation", 96 * 1024);
        let descriptor = descriptor();
        store
            .hello(1, std::process::id(), None, descriptor.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let key = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let extent = store.memory.allocate(16 * 1024).unwrap();
        insert_memory_node(&mut store, id, &descriptor, key, None, 1, extent);
        let node = store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .node_mut(&key)
            .unwrap();
        node.location.disk = Some(Arc::new(DiskExtent::new(base.join("warm.ckpt"))));
        node.location.durable = true;
        let used = store.memory.used();

        assert!(store.memory_room_excluding(80 * 1024, true, &[]).is_none());
        assert_eq!(store.memory.used(), used);
        assert!(store.namespaces[&id].tree[&key].location.memory.is_some());

        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn disk_pressure_compacts_only_an_endpoint_with_a_durable_row_child() {
        use CaptureRole::{Anchor, Endpoint, Message};
        let (mut store, base, directory) = store_with_root("durable-rows", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1, 2, 3, 4, 5]);
        for (pages, role) in [
            (1, Endpoint),
            (2, Message),
            (3, Anchor),
            (4, Endpoint),
            (5, Endpoint),
        ] {
            let key = page_key(pages);
            let ns = store.namespaces.get_mut(&id).unwrap();
            let path = ns.path(key).unwrap();
            let loc = &mut ns.tree[&key].location;
            loc.header = loc.header.clone().with_capture_role(role);
            fs::File::create(&path)
                .unwrap()
                .set_len(loc.header.length(true))
                .unwrap();
            loc.disk = Some(Arc::new(DiskExtent::new(path)));
            loc.durable = pages != 5;
        }
        let tip = page_key(5);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .pin(&tip)
            .unwrap();
        store.config.nvme = store.charge_total();
        assert!(!store.make_room_evict(1));
        assert!(store.jobs.is_empty());
        store.namespaces.get_mut(&id).unwrap().tree[&tip]
            .location
            .durable = true;
        assert!(!store.make_room_evict(1));
        assert_eq!(store.metrics.compactions, 1);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !store.jobs.is_empty() {
            assert!(Instant::now() < deadline);
            store.disk_events();
            std::thread::yield_now();
        }
        assert!(store.make_room_evict(1));
        for pages in 1..=5 {
            let node = &store.namespaces[&id].tree[&page_key(pages)];
            let row = pages != 4;
            assert_eq!(node.row, row);
            assert_eq!(
                fs::metadata(&node.location.disk.as_ref().unwrap().path)
                    .unwrap()
                    .len(),
                node.location.header.length(row)
            );
        }
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn disk_pressure_keeps_the_only_row_above_a_page_only_child() {
        let (mut store, base, directory) = store_with_root("anchor-compaction", 64 * 1024);
        let descriptor = descriptor();
        store
            .hello(1, std::process::id(), None, descriptor.clone())
            .unwrap();
        let id = store.sessions[&1].namespace;
        let root = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let tail = Key {
            consumed_tokens: 128,
            digest: [2; 16],
        };
        for (key, parent) in [(root, None), (tail, Some(root))] {
            let extent = store.memory.allocate(16 * 1024).unwrap();
            insert_memory_node(&mut store, id, &descriptor, key, parent, 1, extent);
            let node = &mut store.namespaces.get_mut(&id).unwrap().tree[&key];
            node.location.durable = true;
        }
        let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
        tree[&tail].row = false;
        tree.pin(&tail).unwrap();
        store.config.nvme = 0;

        // The pinned page-only tail resumes from the root's row, so the root
        // must not be compacted to make room.
        assert!(!store.make_room_evict(1));
        assert!(store.namespaces[&id].tree[&root].row);
        assert_eq!(store.metrics.compactions, 0);

        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    /// Durable, resident records of one namespace linked as `links` says, each of one page,
    /// whose keys are named by `(pages, digest)`.
    fn durable_records(
        store: &mut Store,
        id: NamespaceId,
        value: &Descriptor,
        links: &[((u64, u8), Option<(u64, u8)>)],
        page_only: Option<(u64, u8)>,
    ) {
        let key = |(pages, digest): (u64, u8)| Key {
            consumed_tokens: pages * 64,
            digest: [digest; 16],
        };
        for &(record, parent) in links {
            let header = crate::record::new_header(value, key(record), parent.map(key), [0; 16], 1)
                .unwrap()
                .with_capture_role(CaptureRole::Endpoint);
            let extent = store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap();
            insert_memory_node(store, id, value, key(record), parent.map(key), 1, extent);
            let namespace = store.namespaces.get_mut(&id).unwrap();
            let path = namespace.path(key(record)).unwrap();
            let row = Some(record) != page_only;
            File::create(&path)
                .unwrap()
                .set_len(header.length(row))
                .unwrap();
            if !row {
                namespace.tree.drop_row(&key(record));
            }
            let location = &mut namespace.tree[&key(record)].location;
            location.header = header;
            location.disk = Some(Arc::new(DiskExtent::new(path)));
            location.durable = true;
        }
    }

    #[test]
    fn evicting_a_leaf_with_a_row_updates_the_row_counts_that_guard_compaction() {
        let (mut store, base, directory) = store_with_root("row-counts-leaf", 64 * 1024);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let (root, fork, with_row, page_only) = ((1, 1), (2, 2), (3, 3), (3, 4));
        durable_records(
            &mut store,
            id,
            &value,
            &[
                (root, None),
                (fork, Some(root)),
                (with_row, Some(fork)),
                (page_only, Some(fork)),
            ],
            Some(page_only),
        );
        let key = |(pages, digest): (u64, u8)| Key {
            consumed_tokens: pages * 64,
            digest: [digest; 16],
        };
        let charge = store.namespaces[&id].tree[&key(fork)].location.bytes;
        // Room is made by evicting the two leaves, oldest first. Once the leaf with the
        // row is gone, the fork's only remaining child has no row, so it keeps its row.
        store.config.nvme = 2 * charge;
        assert!(store.make_room_evict(0));
        let tree = &store.namespaces[&id].tree;
        assert!(tree.node(&key(with_row)).is_none() && tree.node(&key(page_only)).is_none());
        assert!(
            tree[&key(fork)].row,
            "the parent of a page-only child lost its row"
        );
        assert_eq!(store.metrics.compactions, 0);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn compacting_a_record_updates_the_row_counts_that_guard_its_parents_compaction() {
        let (mut store, base, directory) = store_with_root("row-counts-compact", 64 * 1024);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let (root, top, middle, leaf) = ((1, 1), (2, 2), (3, 3), (4, 4));
        durable_records(
            &mut store,
            id,
            &value,
            &[
                (root, None),
                (top, Some(root)),
                (middle, Some(top)),
                (leaf, Some(middle)),
            ],
            None,
        );
        let key = |(pages, digest): (u64, u8)| Key {
            consumed_tokens: pages * 64,
            digest: [digest; 16],
        };
        let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
        // The middle record is the oldest, so it is compacted first; the top record is
        // next in age, and the leaf is newest.
        tree.access_at(&key(middle), 1);
        tree.access_at(&key(top), 2);
        tree.access_at(&key(leaf), 3);
        let charge = tree[&key(top)].location.bytes;
        store.config.nvme = 3 * charge;
        assert!(store.make_room_evict(0));
        finish_jobs(&mut store);
        // After the middle record lost its row, the top record's only child has none, so
        // the top record keeps its row and the leaf is evicted instead.
        let tree = &store.namespaces[&id].tree;
        assert!(!tree[&key(middle)].row);
        assert!(tree[&key(top)].row, "the top record was compacted as well");
        assert!(tree.node(&key(leaf)).is_none());
        assert_eq!(store.metrics.compactions, 1);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn fragmented_arena_writes_a_dirty_record_before_admission() {
        // 64 KiB arena with 16 KiB live at 0 and 32 KiB; the freed buddies
        // leave 32 KiB free as two separated 16 KiB holes.
        let (mut store, base, directory) = store_with_root("fragmented", 64 * 1024);
        let value = descriptor();
        store.hello(1, std::process::id(), None, value.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        let first = store.memory.allocate(16 * 1024).unwrap();
        let second = store.memory.allocate(16 * 1024).unwrap();
        let third = store.memory.allocate(16 * 1024).unwrap();
        let fourth = store.memory.allocate(16 * 1024).unwrap();
        assert_eq!(first.offset(), 0);
        assert_eq!(second.offset(), 16 * 1024);
        assert_eq!(third.offset(), 32 * 1024);
        assert_eq!(fourth.offset(), 48 * 1024);
        drop(second);
        drop(fourth);
        insert_memory_node(
            &mut store,
            id,
            &value,
            Key {
                consumed_tokens: 64,
                digest: [1; 16],
            },
            None,
            1,
            first,
        );
        insert_memory_node(
            &mut store,
            id,
            &value,
            Key {
                consumed_tokens: 128,
                digest: [2; 16],
            },
            Some(Key {
                consumed_tokens: 64,
                digest: [1; 16],
            }),
            1,
            third,
        );
        // The allocation needs a contiguous extent, so admission waits for write-back.
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key: Key {
                    consumed_tokens: 192,
                    digest: [3; 16],
                },
                published_parent: Some(Key {
                    consumed_tokens: 64,
                    digest: [1; 16],
                }),
                bucket: [0; 16],
                chain_pages: 3,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        assert!(!store.writeback.is_empty());
        assert!(
            store
                .metrics_text()
                .contains("selene_put_writeback_waits_total 1\n")
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !store.sessions[&1].ops.contains_key(&1) {
            assert!(Instant::now() < deadline);
            store.pump();
            std::thread::yield_now();
        }
        let put = put_ref(&store.sessions, 1, 1).unwrap();
        assert!(put.memory.is_some());
        assert!(!put.streaming);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn queued_get_overtakes_an_earlier_put() {
        let (mut store, directory) = store("batch");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), None, value)
            .unwrap();
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [5; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            put_events,
        );
        store.submit_get(
            1,
            2,
            GetRequest {
                namespace: store.namespace_id().unwrap(),
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        assert_eq!(store.dispatch_trace(), vec!["get", "put"]);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn the_nvme_measure_counts_put_holds_and_reservations_not_windows_leased_for_claims() {
        let (store, base, directory) = store_with_root("nvme-measure", 4096);
        assert_eq!(store.charge_total(), 0);
        // Windows the pool leases from the staging region for claims and exports are not
        // records and never reach the disk: they do not use the limit.
        let leased = store
            .local
            .staging()
            .allocate(1 << 20)
            .expect("a window of staging");
        assert_eq!(store.local.staging_used(), 1 << 20);
        assert_eq!(store.charge_total(), 0);
        assert!(
            store
                .metrics_text()
                .contains("selene_nvme_reserved_bytes 0\n")
        );
        drop(leased);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn streaming_admission_uses_the_hold() {
        let name = format!(
            "selene-unit-{}-budget-staging-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            4096,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let mut store = Store::open(config).unwrap();
        let value = descriptor();
        store.hello(1, std::process::id(), None, value).unwrap();
        let id = store.sessions[&1].namespace;
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [7; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events.clone(),
        );
        store.dispatch_trace();
        // The hold is two 16 KiB windows (32 KiB), not two negotiated 4 MiB
        // maxima, so both Puts stay in budget.
        assert!(store.sessions[&1].ops.contains_key(&1));
        store.submit_put(
            1,
            2,
            PutRequest {
                ancestor_keys: Vec::new(),
                namespace: id,
                key: Key {
                    consumed_tokens: 192,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 3,
                published_tail_pages: 3,
                capture_role: CaptureRole::Legacy,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch_trace();
        assert!(store.sessions[&1].ops.contains_key(&2));
        assert_eq!(store.charge_total(), (20480 + 32768) + (24576 + 32768));
        store.cancel(1, 1);
        store.cancel(1, 2);
        store.pump();
        assert_eq!(store.charged, 0);
        assert_eq!(store.charge_total(), 0);
        let directory = store.shm_directory();
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }
    #[test]
    fn oldest_uses_each_namespaces_first_eligible_record() {
        let (mut store, directory) = store("oldest");
        let mut value = descriptor();
        let mut entries = Vec::new();
        for n in 1..=2 {
            value.identity = [n; 32];
            let id = store.register_namespace(value.clone()).unwrap();
            for pages in 1..=3 {
                let key = Key {
                    consumed_tokens: pages * 64,
                    digest: [n; 16],
                };
                let header =
                    crate::record::new_header(&value, key, None, [0; 16], pages as u32).unwrap();
                let extent = store
                    .memory
                    .allocate(record_charge(&header, true) as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key, None, pages as u32, extent);
                store.reused(id, key);
                entries.push((id, key));
            }
        }
        let (id, key) = entries[0];
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .pin(&key)
            .unwrap();
        let (id, key) = entries[1];
        store.namespaces.get_mut(&id).unwrap().tree[&key]
            .location
            .writing = true;
        assert_eq!(store.oldest(|_, _| true), Some(entries[2]));
        let (id, key) = entries[2];
        store.reused(id, key);
        assert_eq!(store.oldest(|_, _| true), Some(entries[3]));
        assert_eq!(
            store.oldest(|_, node| node.key.consumed_tokens == 128),
            Some(entries[4])
        );
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    /// A page-only parent whose write-back is in flight, with one resident child.
    /// The write is submitted but its completion is not polled until the next pump.
    fn page_only_parent_writing(label: &str) -> (Store, PathBuf, PathBuf, NamespaceId) {
        let (mut store, base, directory) = store_with_root(label, 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1, 2]);
        let namespace = store.namespaces.get_mut(&id).unwrap();
        namespace.tree.drop_row(&page_key(1));
        store.refresh_durability(id, Some(page_key(1)));
        store.request_writeback(id, page_key(1));
        assert!(
            store.namespaces[&id].tree[&page_key(1)].location.writing,
            "the parent's write-back was not submitted"
        );
        (store, base, directory, id)
    }

    fn finish_jobs(store: &mut Store) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !store.jobs.is_empty() {
            assert!(Instant::now() < deadline, "the disk jobs never completed");
            store.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn offer_root_upgrade(
        store: &mut Store,
        connection: u64,
    ) -> tokio::sync::mpsc::Receiver<Result<PutAnswer>> {
        let (events, answers) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            connection,
            2,
            PutRequest {
                namespace: store.sessions[&connection].namespace,
                key: page_key(1),
                published_parent: None,
                ancestor_keys: vec![],
                bucket: [0; 16],
                chain_pages: 1,
                published_tail_pages: 1,
                capture_role: CaptureRole::Message,
                child_paths: 0,
                descendant_row: false,
                next_child: None,
                row: true,
                packed: None,
            },
            events,
        );
        store.dispatch();
        store.flush_outboxes();
        answers
    }

    fn put_inflight_wait_max_seconds(store: &Store) -> f64 {
        store
            .metrics_text()
            .lines()
            .find_map(|line| line.strip_prefix("selene_put_inflight_wait_max_seconds "))
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn a_put_waits_for_its_records_writeback() {
        let (mut store, base, shm, id) = page_only_parent_writing("put-waits-writeback");
        store.hello(1, 0, None, descriptor()).unwrap();
        let start = Instant::now();
        let mut answers = offer_root_upgrade(&mut store, 1);
        finish_jobs(&mut store);
        assert!(store.namespaces[&id].tree[&page_key(1)].location.durable);
        match answers.try_recv().unwrap().unwrap() {
            PutAnswer::Accept(accept) => assert_eq!(accept.capture_role, CaptureRole::Message),
            PutAnswer::Refused(r) => panic!("write-back refused the upgrade: {}", r.detail),
            _ => panic!("expected an admission answer"),
        }
        let observed = start.elapsed().as_secs_f64();
        let max_wait = put_inflight_wait_max_seconds(&store);
        assert!(
            max_wait > 0.0 && max_wait <= observed,
            "gauge {max_wait}s, observed {observed}s"
        );
        assert!(
            store
                .metrics_text()
                .contains("selene_put_inflight_waits_total 1\n")
        );
        assert_eq!(store.metrics.retry_inflight, 0);
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn a_put_starts_a_fresh_wait_after_writeback_requeue() {
        let key = page_key(1);
        let value = descriptor();
        let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
        let extent = record_charge(&header, true) as usize;
        let (mut store, base, shm) = store_with_root("put-waits-twice", 4 * extent);
        let held = store.memory.allocate(2 * extent).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        insert_chain(&mut store, id, &value, &[1, 2]);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .pin(&key)
            .unwrap();
        let mut answers = offer_root_upgrade(&mut store, 1);
        assert_eq!(store.metrics.put_inflight_waits, 1);

        // The arena is full. Writing back the child first requires writing its dirty parent.
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .unpin(&key)
            .unwrap();
        store.dispatch();
        assert_eq!(store.metrics.put_writeback_waits, 1);
        assert!(store.namespaces[&id].tree[&key].location.writing);
        assert_eq!(put_inflight_wait_max_seconds(&store), 0.0);

        // Advance any clock still running during write-back, without polling its completion.
        let pending = store.pending_puts.get_mut(&1).unwrap().front_mut().unwrap();
        if let Some(start) = &mut pending.inflight_since {
            *start -= PUT_INFLIGHT_WAIT;
        }
        store.dispatch();
        match answers.try_recv() {
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
            Ok(Ok(PutAnswer::Refused(r))) => {
                panic!("second conflict refused the upgrade: {}", r.reason)
            }
            _ => panic!("expected the upgrade to stay queued"),
        }
        assert_eq!(store.metrics.put_inflight_waits, 2);
        drop(held);
        finish_jobs(&mut store);
        assert!(matches!(
            answers.try_recv().unwrap().unwrap(),
            PutAnswer::Accept(_)
        ));
        assert!(put_inflight_wait_max_seconds(&store) > 0.0);
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn a_put_refuses_to_wait_for_its_own_sessions_pin() {
        for read_lease in [false, true] {
            let (mut store, shm, id) = held_chain("own-put-pin", 1 << 20, &[1], &[]);
            if read_lease {
                let (events, _answers) = tokio::sync::mpsc::channel(4);
                store.submit_get(
                    1,
                    1,
                    GetRequest {
                        namespace: id,
                        keys: vec![page_key(1)],
                        skip_pages: 0,
                        expected_pages: 1,
                        row: true,
                    },
                    events,
                );
                store.dispatch();
            } else {
                assert!(matches!(
                    first_put_answer(&mut store, id, page_key(2), Some(page_key(1)), vec![], true),
                    PutAnswer::Accept(_)
                ));
            }
            assert!(store.namespaces[&id].tree[&page_key(1)].pinned());
            let mut answers = offer_root_upgrade(&mut store, 1);
            assert!(
                matches!(answers.try_recv().unwrap().unwrap(), PutAnswer::Refused(r) if r.reason == proto::Reason::Inflight as u8)
            );
            assert_eq!(store.metrics.put_inflight_waits, 0);
            assert_eq!(store.metrics.retry_inflight, 1);
            assert_eq!(store.metrics.put_inflight_expired, 0);
            store.hello(2, 0, None, descriptor()).unwrap();
            let mut other = offer_root_upgrade(&mut store, 2);
            assert_eq!(store.metrics.put_inflight_waits, 1);
            store.cancel(1, 1);
            store.pump();
            store.pump();
            assert!(matches!(
                other.try_recv().unwrap().unwrap(),
                PutAnswer::Accept(_)
            ));
            drop(store);
            fs::remove_dir_all(shm).unwrap();
        }
    }

    #[test]
    fn a_put_waiting_for_another_sessions_pin_expires() {
        let (mut store, shm, id) = held_chain("put-pin-deadline", 1 << 20, &[1], &[]);
        store.hello(2, 0, None, descriptor()).unwrap();
        // A child put pins its parent while it waits for window credit.
        assert!(matches!(
            first_put_answer(&mut store, id, page_key(2), Some(page_key(1)), vec![], true),
            PutAnswer::Accept(_)
        ));
        let mut upgrade = offer_root_upgrade(&mut store, 2);
        store.cancel(1, 1);
        store.pump();
        store.pump();
        assert!(matches!(
            upgrade.try_recv().unwrap().unwrap(),
            PutAnswer::Accept(_)
        ));
        let successful = put_inflight_wait_max_seconds(&store);
        assert!(successful > 0.0);

        store.cancel(2, 2);
        store.pump();
        assert!(matches!(
            first_put_answer(&mut store, id, page_key(2), Some(page_key(1)), vec![], true),
            PutAnswer::Accept(_)
        ));
        let mut answers = offer_root_upgrade(&mut store, 2);
        let mut pending = store.pop_put().unwrap();
        pending.inflight_since = Some(Instant::now() - PUT_INFLIGHT_WAIT);
        store.begin_put(pending);
        assert!(
            matches!(answers.try_recv().unwrap().unwrap(), PutAnswer::Refused(r) if r.reason == proto::Reason::Inflight as u8)
        );
        assert!(store.namespaces[&id].tree[&page_key(1)].pinned());
        assert_eq!(store.metrics.put_inflight_waits, 2);
        assert_eq!(store.metrics.retry_inflight, 1);
        assert!(
            store
                .metrics_text()
                .contains("selene_put_inflight_expired_total 1\n")
        );
        assert_eq!(put_inflight_wait_max_seconds(&store), successful);
        drop(store);
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn expiring_a_child_keeps_a_page_only_parent_whose_write_is_in_flight() {
        let (mut store, base, directory, id) = page_only_parent_writing("expire-writing-parent");
        store.config.max_age = Duration::from_secs(7200);
        let namespace = store.namespaces.get_mut(&id).unwrap();
        namespace.tree.max_age = store.config.max_age;
        namespace.tree.node_mut(&page_key(2)).unwrap().last_used = SystemTime::UNIX_EPOCH;
        // Exercise the disk-job guard independently of the write-back pin.
        namespace.tree.unpin(&page_key(1)).unwrap();
        store.tick();
        let tree = &store.namespaces[&id].tree;
        assert!(
            tree.node(&page_key(2)).is_none(),
            "the child did not expire"
        );
        assert!(
            tree.node(&page_key(1)).is_some(),
            "expiry removed a parent whose write is in flight"
        );
        assert_eq!(store.evict_accounted(id, page_key(1)), 0);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .pin(&page_key(1))
            .unwrap();
        finish_jobs(&mut store);
        let location = &store.namespaces[&id].tree[&page_key(1)].location;
        assert!(location.disk.is_some() && !location.writing);
        drop(store);
        let _ = fs::remove_dir_all(directory);
        let _ = fs::remove_dir_all(base);
    }

    fn mark_endpoint(store: &mut Store, id: NamespaceId, key: Key) {
        let loc = &mut store.namespaces.get_mut(&id).unwrap().tree[&key].location;
        loc.header = loc.header.clone().with_capture_role(CaptureRole::Endpoint);
    }

    #[test]
    fn a_finishing_stream_can_write_its_pinned_former_tip() {
        let (mut store, base, shm) = store_with_root("stream-prune", 64 << 10);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), None, value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1, 2]);
        store.refresh_durability(id, None);
        store.write_ancestor(id, page_key(1), "dependency");
        finish_jobs(&mut store);
        let parent = page_key(2);
        mark_endpoint(&mut store, id, parent);
        assert!(matches!(
            first_put_answer(&mut store, id, page_key(32), Some(parent), vec![parent], true),
            PutAnswer::Accept(_)
        ));
        assert!(put_ref(&store.sessions, 1, 1).unwrap().streaming);
        let sibling = store.memory.allocate(16 << 10).unwrap();
        insert_memory_node(&mut store, id, &value, page_key(3), Some(parent), 1, sibling);
        store.prune_former_tip(id, parent);
        assert!(store.pending_former_tips.contains(&(id, parent)));
        store.schedule_finish(1, 1, id);
        finish_jobs(&mut store);
        let node = &store.namespaces[&id].tree[&parent];
        assert!(node.location.durable && node.row && node.pinned());
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn a_page_only_child_preserves_its_parents_last_row() {
        let (mut store, dir, id) = held_chain("last-parent", 1 << 20, &[1, 2, 3], &[3]);
        mark_endpoint(&mut store, id, page_key(2));
        store.prune_former_tip(id, page_key(2));
        assert!(store.namespaces[&id].tree[&page_key(2)].row);
        assert!(matches!(
            first_put_answer(&mut store, id, page_key(4), Some(page_key(3)), vec![], true),
            PutAnswer::Accept(_)
        ));
        store.publish_put(1, 1, None);
        assert!(store.namespaces[&id].tree[&page_key(4)].row);
        assert!(!store.namespaces[&id].tree[&page_key(2)].row);
        drop(store);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn evicting_a_sibling_preserves_the_forks_row() {
        let (mut store, dir, id) = held_chain("fork-evict", 1 << 20, &[1, 2, 3], &[]);
        let parent = page_key(2);
        mark_endpoint(&mut store, id, parent);
        let sibling = Key {
            digest: [99; 16],
            ..page_key(3)
        };
        let extent = store.memory.allocate(16 << 10).unwrap();
        insert_memory_node(
            &mut store,
            id,
            &descriptor(),
            sibling,
            Some(parent),
            1,
            extent,
        );
        assert_eq!(store.evict_accounted(id, sibling), 1);
        assert!(store.namespaces[&id].tree[&parent].row);
        drop(store);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_disk_flush_stops_at_the_signal_deadline() {
        let (mut store, base, shm) = store_with_root("flush-budget", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1]);
        store.refresh_durability(id, None);
        let path = store.namespaces[&id].path(page_key(1)).unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let deadline = Instant::now() + Duration::from_millis(150);
        let owner = std::thread::spawn(move || {
            crate::control::run_store_owner(
                store,
                rx,
                Arc::new(std::sync::atomic::AtomicBool::new(true)),
                Arc::new(std::sync::Mutex::new(None)),
            )
        });
        let (reply, seen) = std::sync::mpsc::channel();
        tx.blocking_send(StoreCommand::Metrics { reply }).unwrap();
        seen.recv_timeout(Duration::from_secs(5)).unwrap();
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        tx.blocking_send(StoreCommand::Shutdown { deadline })
            .unwrap();
        owner.join().unwrap();
        assert!(
            !path.exists(),
            "the flush started a write after the shared deadline"
        );
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn drain_skips_page_only_leaves_and_writes_rows_first() {
        let (store, base, shm) = store_with_root("drain-priority", 1 << 20);
        let mut config = store.config.clone();
        drop(store);
        config.faults.delay_ms = 100;
        let mut store = Store::open(config).unwrap();
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1, 2, 3]);
        mark_endpoint(&mut store, id, page_key(2));
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .drop_row(&page_key(3));
        for n in 10..14 {
            let key = Key {
                digest: [n; 16],
                ..page_key(1)
            };
            let extent = store.memory.allocate(16 << 10).unwrap();
            insert_memory_node(&mut store, id, &value, key, None, 1, extent);
        }
        store.refresh_durability(id, None);
        assert!(!store.drain_until(Instant::now() + Duration::from_millis(30)));
        assert_eq!(
            store
                .jobs
                .values()
                .filter(|j| matches!(j, Job::Write { key, .. } if key.consumed_tokens == 64))
                .count(),
            4
        );
        assert!(store.drain_until(Instant::now() + Duration::from_secs(5)));
        let ns = &store.namespaces[&id];
        assert!(ns.tree[&page_key(2)].location.durable);
        assert!(ns.tree[&page_key(3)].location.disk.is_none());
        assert!(
            store
                .metrics_text()
                .contains("selene_shutdown_unflushed_records 0\n")
        );
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn arena_pressure_starts_bounded_independent_writes() {
        let (mut store, base, shm) = store_with_root("arena-parallel", 128 << 10);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        for n in 1..=8 {
            let key = Key {
                digest: [n; 16],
                ..page_key(1)
            };
            let extent = store.memory.allocate(16 << 10).unwrap();
            insert_memory_node(&mut store, id, &value, key, None, 1, extent);
        }
        store.refresh_durability(id, None);
        assert!(store.memory_room_excluding(16 << 10, true, &[]).is_none());
        assert_eq!(
            store
                .jobs
                .values()
                .filter(|j| matches!(j, Job::Write { .. }))
                .count(),
            4
        );
        // Readers can hold the first four extents after their writes finish.
        let writing: Vec<_> = store.namespaces[&id].tree.nodes()
            .filter(|n| n.location.writing).map(|n| n.key).collect();
        for key in writing {
            store.namespaces.get_mut(&id).unwrap().tree.pin(&key).unwrap();
        }
        finish_jobs(&mut store);
        assert!(store.memory_room_excluding(16 << 10, true, &[]).is_none());
        assert_eq!(store.jobs.len(), 4);
        finish_jobs(&mut store);
        assert!(store.memory_room_excluding(16 << 10, true, &[]).is_some());
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn independent_dirty_roots_flush_before_a_serial_writers_deadline() {
        let (store, base, shm) = store_with_root("parallel-drain", 1 << 20);
        let mut config = store.config.clone();
        drop(store);
        config.faults.delay_ms = 100;
        let mut store = Store::open(config).unwrap();
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        for n in 1..=16 {
            let key = Key {
                digest: [n; 16],
                ..page_key(1)
            };
            let extent = store.memory.allocate(16 << 10).unwrap();
            insert_memory_node(&mut store, id, &value, key, None, 1, extent);
        }
        store.refresh_durability(id, None);
        assert!(store.drain_until(Instant::now() + Duration::from_millis(1250)));
        assert!(
            store.namespaces[&id]
                .tree
                .nodes()
                .all(|n| n.location.durable)
        );
        drop(store);
        fs::remove_dir_all(base).unwrap();
        fs::remove_dir_all(shm).unwrap();
    }

    #[test]
    fn rootless_metrics_omit_dirty_inventory() {
        let (store, dir) = store("dirty-gauges");
        let text = store.metrics_text();
        assert!(
            !text.contains("selene_arena_only_bytes")
                && !text.contains("selene_oldest_dirty_seconds")
        );
        drop(store);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn writeback_waits_for_a_put_replacing_its_dirty_ancestor() {
        let (mut store, base, directory) = store_with_root("replace-dirty", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, None, value.clone()).unwrap();
        insert_chain(&mut store, id, &value, &[1, 2]);
        let parent = page_key(1);
        store.namespaces.get_mut(&id).unwrap().tree.drop_row(&parent);
        assert!(matches!(
            first_put_answer(&mut store, id, parent, None, Vec::new(), true),
            PutAnswer::Accept(_)
        ));

        store.request_writeback(id, page_key(2));
        assert!(store.jobs.is_empty());
        store.cancel(1, 1);
        store.pump();
        assert!(store.namespaces[&id].tree[&parent].location.disk_busy());
        finish_jobs(&mut store);
        assert!(store.namespaces[&id].tree[&page_key(2)].location.durable);
        drop(store);
        let _ = fs::remove_dir_all(directory);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn expiry_is_bounded_and_releases_manager_memory() {
        let (mut store, directory) = dram_store("expiry", 8 << 20);
        let mut value = descriptor();
        for identity in [1, 2] {
            value.identity = [identity; 32];
            let id = store.register_namespace(value.clone()).unwrap();
            store.config.max_age = Duration::from_secs(7200);
            store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
            for n in 1..=70u64 {
                let key = Key {
                    consumed_tokens: 64,
                    digest: (n as u128).to_le_bytes(),
                };
                let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
                let extent = store
                    .memory
                    .allocate(record_charge(&header, true) as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key, None, 1, extent);
                store
                    .namespaces
                    .get_mut(&id)
                    .unwrap()
                    .tree
                    .node_mut(&key)
                    .unwrap()
                    .last_used = SystemTime::UNIX_EPOCH;
            }
        }
        let before = store.charged;
        let evictions = store.metrics.evictions;
        store.tick();
        assert!(store.namespaces.values().all(|ns| ns.tree.len() == 6));
        assert_eq!(store.metrics.expired, [128, 0]);
        assert_eq!(store.metrics.evictions, evictions);
        assert!(
            store
                .metrics_text()
                .contains("selene_expired_total{tier=\"dram\"} 128\n")
        );
        assert!(store.charged < before);
        store.tick();
        assert!(store.namespaces.values().all(|ns| ns.tree.len() == 6));
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }
}
