# Operating selene-manager

This runbook covers the deployment model, sizing, refusal reasons, metrics, restarts and
upgrades for `selene`. The wire contract is [PROTOCOL.md](../PROTOCOL.md) and
[selene.proto](../../selene/proto/selene.proto); building, testing and containers are in
[README.md](../README.md); remote hosts are in [network-operations.md](network-operations.md).

## Deployment model

- **One manager per pool.** A pool is the set of Flare instances that share records. Run one
  `selene` for it, on a host with local NVMe if you want records to outlive the instances.
  Instances do not talk to each other; they talk to the manager, which brokers every copy.
- **The manager owns its NVMe directory exclusively.** `--root` is locked with an exclusive
  `flock` on the directory: a second manager on the same directory refuses to start. Do not put the
  directory on a network file system, share it between managers, or edit it while the manager runs.
- **Instances configure only their private tiers plus the pool address.** Each Flare instance
  keeps its own DRAM and NVMe settings for its private tiers and adds
  `--prefix-cache-pool <address>`; see
  [address forms](network-operations.md#addresses-and-startup-warnings). The manager returns its `--name` in `Hello`, and the instance
  derives its scope and instance ids from that name. Everything shared,
  the index, holds, windows, the manager's own store and the network, lives in the manager.
- **Never put the manager behind a load balancer.** A session is stateful: the index entries an
  instance announced, the claims in flight and the windows granted to it all live in the one
  manager process it is connected to. A balancer that spreads instances over several managers
  splits the pool, and one that moves a stream between them loses its claims. Give each pool one
  address.
- **Same host, same identity.** The manager and every instance that uses shared memory must
  run with the same effective uid and gid, share the socket directory and `/dev/shm/selene` at
  identical paths. On Linux 6.5+, the manager uses `SO_PEERPIDFD` to watch clients
  in separate PID namespaces. On older kernels, a client whose pid is invisible
  gets no shared memory: share a PID namespace, or configure `--listen` and connect
  over TCP to use the data channel. Without shared memory or a data channel, instance startup fails.

The manager retains rows at roots and captured Message or Anchor boundaries.
A fork keeps its row when a writer still holds it. Existing Legacy-role disk
records keep all their available rows until rewritten. An Endpoint tip also keeps its row unless an ancestor on its
path already has a row in the same page. A returning request can rewind
exactly to either of two captured Message boundaries in one page.
Arena pressure writes dirty records parent-first. For shutdown timing, crash
exposure and the write counters, see [Operating the Selene checkpoint manager](../../docs/operating.md).

Record expiry is off by default. `--max-age DURATION|off`
accepts positive durations in ms, s, m, h, or bare seconds; `off` disables expiry.
Puts, gets and child publication refresh the required chain in memory only.
Recovery uses file mtime from its last write. After a restart, a record reused
from disk but never rewritten may expire up to one max age early. The cost is a
recompute, never wrong output. With expiry off, hits do not refresh last use;
enabling it later may expire old files at once.
The store's 10 ms tick sweeps every namespace, with each index limited to once a
minute. It walks from the oldest entry, skipping interior nodes, pinned records
and active I/O. It stops at the first fresh leaf or 64 expired leaves, removing
page-only ancestors with their last child. LRU order uses a store-wide counter,
independently of timestamps.
`selene_expired_total{tier="dram"|"nvme"}` counts those removals.

## Sizing

The service is CPU-only and never touches GPU memory. Three budgets are configured
separately and shared by every scope: `--memory` (the DRAM arena), `--staging` (windows) and
`--nvme` (durable records).

`--memory` is the arena for the manager's own records served without touching NVMe. Only puts
place records in it; a restore streams a record that is not in the arena and does not add it.
Size it for the records written recently that are likely to be restored soon, not for everything.
Allocation uses power-of-two size classes, so a record
consumes its length rounded up and usable capacity can be materially below the mapping size.
A record whose rounded extent exceeds the whole arena is not dropped (with `--root` it streams
to NVMe) but it will never be served from DRAM; without `--root` it can never be admitted and is
refused `OVERSIZE`.

`--staging` holds the windows of copies between instances and the reads of NVMe records. A
**GPU export is accepted only after staging for the whole export has been reserved** (one
block, rounded up to the allocator's size class), so the source's GPU pin lasts only for its
device-to-host copy: with too little free staging the claim fails `NO_WINDOWS` at once and
nothing is pinned. Host-tier exports that do not fit stream with two windows at a time. Provision
staging for the largest export you expect to run concurrently times `export_slots` (default 2 per
instance), plus the quarantined bytes below. The minimum is 8 MiB and the default 4 GiB.

One export of a long context needs staging for all of its pages and its row. When staging is
smaller than one export, every claim for that export is refused `NO_WINDOWS` and the requester
recomputes. When the first session of a scope arrives and one
export in the configured staging would hold fewer than 262,144 tokens of context (Flare's usual
per-request limit; the manager does not know a model's real maximum), the manager logs one
warning for that scope. The warning states the token count one export can hold, the page and row
bytes, and the staging size that would hold 262,144 tokens; a manager whose staging
holds that much logs nothing.
One export holds (`--staging` minus the row bytes) divided by the page bytes, rounded down,
times the page tokens: about 143,000 tokens at 2 GiB and 294,000 at the 4 GiB default for a
model with 905,216-byte pages of 64 tokens and a 122 MiB row. Size `--staging` from that
formula or from the figure in the warning.

A restore from NVMe also reads through staging. A restore reads only the pages and the
endpoint row it needs, as windows, and never reads a whole record into the arena. A record that
is in the arena is served from there. A restore window that cannot get a staging block of its
size class, because earlier windows hold that class, waits until one is released; it is not
refused. The endpoint row is one window and is never split, and staging blocks are powers of two,
so staging must be at least the row's size rounded up to a power of two (a page, if larger):
256 MiB for a 160 MiB row. A restore of a record that is not in the arena, with a window whose
block, rounded up to a power of two, is larger than the whole staging region, is refused when it
starts, before any read. The client sees `NO_WINDOWS` and recomputes the prefix. The manager prints a
warning with the minimum `--staging` when a scope's geometry first arrives and staging is smaller.
When a restore begins it starts reading the metadata of every record in its chain that is not in
the arena, in chain order, so a chain near the 2048-record depth at which puts start a new root can
need up to about 16 MiB of staging at once. A metadata read that finds staging full waits for a
release, as a window does. A restore is refused at its start, with `NO_WINDOWS`, when the
block of one record's metadata or one window, rounded up to a power of two, is larger than the whole
staging region.

`--nvme` bounds record reservations: the rounded arena extent for an in-memory
record, or the padded file plus the two-window hold for a streamed put. It must exceed the
largest record an instance can put, or the put is refused `OVERSIZE` and the instance discards
its local record. Keep it above `--memory`: DRAM eviction needs durability, so a full NVMe
budget stops DRAM from making room.

The arena and staging are preallocated files in `/dev/shm/selene/<name>-<hash>`, where the hash is
the 32-hex-digit BLAKE2b digest of the absolute `--socket` path, so the shared-memory mount
must provide at least `--memory + --staging` for each manager. A container's default 64 MiB
`/dev/shm` is far too small; use `--shm-size` or the host IPC namespace.

The files are fixed (`arena`, `staging`, and a `generation`, `holders` and `lock` file beside them) and
persist across restarts. A restarted manager with the same `--memory` and `--staging` reopens the
same files at the same offsets, so instances that registered them with CUDA at startup keep
working; a killed manager never blocks a restart, and debris from a crash is reclaimed. If a size
changed or a file is missing, the wrong size or has no valid generation, the manager
recreates them under a new generation and logs that running instances must restart to use the
pool: until they do, they run on local tiers only. A file owned by another user stops the
start, as does an unexpected I/O error while checking them
(an interrupted call is retried); either stops the manager's start with an error and deletes nothing.
After a crash or a clean stop with unfenced processes, the manager keeps reused
regions out of service for recorded holders ([PROTOCOL.md 8.6](../PROTOCOL.md#86-persistent-shared-regions)).
Visible-pid holders release on reconnect or observed exit, with no time limit.
A pid-zero holder from a separate PID namespace cannot be watched again after a
manager restart. It releases when its nonce reconnects or 60 seconds after
manager startup, whichever comes first; expiry logs the released nonce prefixes.
A live client drains its copies when its session ends before reconnecting
(PROTOCOL.md 8.1). The bound allows that drain and prevents a dead client from
holding the regions indefinitely, assuming session-loss detection and drainage
finish within those 60 seconds. The files outlive a stopped manager and keep
their memory: stop the manager and instances before removing the
`/dev/shm/selene/<name>-<hash>` directory to give it back.

Protect these files from systemd-logind `RemoveIPC`, which can unlink a normal
user's shared memory when their last login session ends, including files used
by containers with that uid. Run the manager and local instances under a
system account, preferably through the supplied systemd service. If a normal
login uid is required, set `RemoveIPC=no` in `/etc/systemd/logind.conf` and
apply the logind configuration according to the host's service procedures.
The manager checks the region files once per second. If one disappears or is
replaced, it drains and exits with an error naming `RemoveIPC`. Fix the host
configuration, then restart the manager and instances so they map the recreated
regions; the manager never recreates regions underneath live copies.

Pool names isolate key namespaces. Instances of one pool map its shared arena and
staging files. Processes running under the same uid can read other pools' files.
Give each trust boundary its own uid (with distinct manager names) or its own host.

### The manager's network card is the pool's aggregate bandwidth

Every copy between hosts passes through this manager: a restore that crosses the network
leaves the source's host, is assembled in the manager's staging and reaches the requester over
the manager's data listener (a source or requester on the manager's own host uses shared
memory and does not touch the card). The manager's NIC therefore bounds the pool's total
cross-host copy rate, summed over every instance and every rank, in both directions. A
multi-process instance (tensor-parallel ranks that are separate processes) multiplies the
number of copies in flight by its rank count, but moves the same bytes: each rank sends or
receives only its own slice, so a record of N bytes crosses the card once whatever the number
of ranks. Size the card (or bond several) for the aggregate you expect, put the manager on the
host whose card has the most headroom, and watch `selene_pool_channel_bytes_total`. Hashing for
multi-process instances (verifying a source's slices and computing the stored checksums, or the
reverse) runs on up to eight threads of the manager's host, overlapped with the network.

### Instance DRAM rows

A pooled instance keeps a retiring record's local slots pinned until its put
finishes. A slow manager can therefore exhaust the instance's free slots and skip
new host publications. With a local memory or disk tier, the instance's host pool
is one set of page-sized slots: the warm checkpoint slots that
`--prefix-cache-memory` buys plus an automatic intake of
`2 × --max-concurrency × row slots + 128` slots, where a row occupies as many slots
as its bytes need. Flare prints one startup warning, `host prefix intake shrunk
to N slots to fit the host budget`, when the host memory budget cannot hold the
full intake; the warning states the bytes the full intake needs. Free host memory
or reduce concurrency; the intake size is a sizing rule, not a guarantee
that an arbitrarily long persistence backlog will fit.

## Multi-process instances

An instance whose tensor-parallel ranks are separate processes (possibly on separate hosts)
is one instance of the pool. Rank 0 opens the only session; every other rank runs an agent
that attaches to the manager with the `Rank` stream and moves only its own slice of each
window, in the manager's shared memory when it is on the manager's host and rank 0 is too, and
on its own data channel otherwise. Records at rest are the ordinary composite records: an
instance of this kind and a single-process instance of the same shape share them.

- Claims, exports and puts of such an instance are refused (`RANKS_DETACHED`, put
  `PUT_NOT_READY`) until every rank is attached.
- A rank that was replaced keeps its obligations: the manager reuses memory a rank could still
  touch only after that rank (identified by its own process, not by its number) attests it
  stopped, says goodbye quiesced, or exits. `selene_pool_quarantined_bytes` includes ranks.
- When rank 0 dies, the agents stay attached only to drain; they cannot be granted anything.
- Each peer has one persistent `RankLink` to rank 0 for control, jobs and completions,
  outside the manager. It is as trusted as the instance's execution rendezvous.

## Startup

See [network startup warnings and TLS rules](network-operations.md#addresses-and-startup-warnings).
A build without the `rdma` feature refuses `--rdma`. Configuration and TLS
validation run before the store or any listener opens.

Startup order is the pool-name lock (`/dev/shm/selene/<name>.lock`), the root lock, NVMe recovery,
shared-memory reclaim, the data, metrics and network listeners, then the Unix socket. A second
manager on the same root or the same `--name` exits at once; a stale socket is replaced only after
confirming nothing is listening.

## Start the manager before the instance

An instance attaches to the pool once, at startup, and never later. Registering the manager's
shared memory with the GPU consumes device memory, which a serving instance must not do, so Flare
registers the manager's `arena` and `staging` files before it admits requests and only then. An
instance that starts while the manager is down waits up to 30 seconds
(`flare_selene::pool::STARTUP_WAIT`), matching the readiness probe window below.
Registration time grows with the manager's `--memory` and `--staging` sizes and
does not count against the startup wait; rank 0 and single-process session opening
have a separate 40-second allowance, while peer agents wait without an opening
deadline as long as their rank link is alive. Failure to attach stops instance startup
with the cause and corrective action. This includes
an untrusted certificate, incompatible protocol, encryption or scope, a rejected
host/device memory budget, registration failure, and session-opening timeout.
The `Reach prefix cache pool` phase checks the connection and `Hello` before weight
loading. Registration and session opening still happen after weights load, so a
manager that becomes unavailable during loading can fail startup after that I/O.
Start the manager first if recovery takes longer; remove `--prefix-cache-pool` to run local-only.
An already running instance whose manager recreates its regions (a size changed)
keeps serving on its local tiers and prints one warning; restart it to use the pool.

Order the manager first. With compose, give the manager a health check and make the instance wait
for it:

```yaml
services:
  manager:
    image: ${MANAGER_IMAGE:-selene:local}
    # ... the options of deploy/compose.yaml ...
    healthcheck:
      test: ["CMD-SHELL", "test -S /run/selene/pool.sock"]
      interval: 1s
      retries: 30
  flare:
    image: ${FLARE_IMAGE}
    depends_on:
      manager:
        condition: service_healthy
```

With systemd, give the instance's unit `Requires=selene.service` and `After=selene.service`. Restarting
the manager alone is safe for running instances as long as its `--memory` and `--staging` are
unchanged; changing either means restarting the instances too.

## Readiness and restarts

Disk recovery completes before the manager binds any listener, so a manager that
accepts connections is `READY` and serves lookups immediately. Claims receive
`REGIONS_HELD` while recorded processes from the previous run hold the shared
regions. A crash can leave holders, as can a clean stop when a writer, peer rank
or claim requester still has an unfenced window. A requester can still hold its
window after its claim ends.
Visible-pid holders release on reconnect or observed exit; pid-zero holders
release on nonce reconnect or 60 seconds after startup (see [Sizing](#sizing)).
While the manager drains for shutdown, requests answer `NOT_READY`.
Sources reconnect and resnapshot independently, so their hints may temporarily miss.

- Records the manager held only in DRAM are lost with it; NVMe records are recovered and start in the
  disk tier when their parent chain is complete. At `Stored` the manager can
  serve the record and the writer may release it. A crash before write-back
  can lose an acknowledged record or an ancestor it needs. A returning request
  then uses a surviving copy or recomputes the missing suffix.
- On `SIGTERM` or `SIGINT` the manager refuses new sessions, lookups, claims and puts.
  It drains in-flight claims and puts and waits for pending put fences from connected
  writers and peer ranks, up to the claim deadline plus two seconds (32 seconds by
  default, capped at four minutes). A reported quarantine ends the wait for that
  fence. An unannounced disconnect also leaves shared windows quarantined.
  The manager then closes sessions with `SHUTDOWN` and flushes dirty records using the
  time left in the 270-second budget measured from the stop signal. Any unfenced
  holder keeps its registration across restart, even if the drain finishes early.

## Refusal reasons

Claims are refused with `FAILED` and a reason; puts with `PutRefused`. Instances treat every
refusal as a normal outcome and move to another candidate, another tier, or recompute.

| Reason | Meaning | Operator action |
|---|---|---|
| `NOT_READY` | The manager is draining for shutdown. | None; check `selene_pool_state`. |
| `REGIONS_HELD` | A recorded process still holds the shared regions after a crash or a clean stop with unfenced windows. | Visible-pid holders wait for reconnect or observed exit; pid-zero holders release within 60 s of startup. Check `selene_pool_regions_held`. |
| `CLAIM_LIMIT` | The requester already has its claim slots in flight (default 2). | None; limits do not scale with concurrency. |
| `SOURCE_LIMIT` / `SOURCE_BUSY` | The source already serves its export slots (default 2). | None. |
| `UNKNOWN_SOURCE` / `STALE_SOURCE` | The named source or entry is gone, or the source reconnected. | None; the index catches up. |
| `NO_WINDOWS` | Staging cannot hold the export, or too much of it is quarantined. | Raise `--staging`; look at `selene_pool_quarantined_bytes`. |
| `NOT_FOUND`, `CHAIN_GAP`, `NO_HEADROOM` | The source does not hold it, or has no room to pin it. | None. |
| `SOURCE_TIMEOUT` | The source did not answer within `pin_deadline_ms` (default 250). | Look at the source's load. |
| `SOURCE_LOST` / `REQUESTER_LOST` | The other side's stream ended. | Look at the instance that went. |
| `RANKS_DETACHED` / `RANK_LOST` | A rank of a multi-process instance is not attached, or its attachment ended during the transfer. | Look at that rank's process. |
| `COPY_ERROR` / `CHECKSUM` | A read, write or checksum failed, or the copy was not exact. | Alert on repeats; check the device. |
| `DEADLINE` / `NO_PROGRESS` | The claim ran past its deadline or stopped making progress (3s). | Look at copy throughput. |
| put `PRESENT` | The manager already holds the record or its parent covers the empty put. Nothing is transferred. | None. |
| put `BUDGET` | Arena, staging or disk pressure. Retry once after 50 ms. On exhaustion, retry an ancestor refusal as a leaf-only put or drop the refused leaf. | Add capacity if sustained. |
| put `INFLIGHT` | Another put or a protected read holds the key. Same retry and leaf-only fallback as `BUDGET`. | None. |
| put `OVERSIZE` | The record exceeds capacity. An ancestor refusal triggers a leaf-only put. A refused leaf is dropped. | Raise the budget or add `--root`. |
| put `QUEUE` | The writer has `max_open_puts` (4) open. | None. |
| put `INVALID` | An ancestor refusal triggers a leaf-only put. A refused leaf is dropped. A page-only put with no usable ancestor returns `PUT_UNLINKED`. This includes an ancestor at the 2048-record reroot limit, where a new root would need a row. The manager counts these refusals in `selene_puts_total{outcome="unlinked"}` and logs the cumulative count at powers of two. Other `INVALID` refusals indicate a client or protocol error. | Alert on `INVALID` refusals beyond those counted as `unlinked`. |
| put `IO` | A device error. | Check the NVMe and the file system. |
| put `PUT_NOT_READY` | Not ready (see Readiness). | None. |

A leaf-only attempt uses the deepest ancestor the manager still holds. A refusal of that attempt for `BUDGET`, `INFLIGHT`, `OVERSIZE` or `INVALID` drops the leaf in either tier.

## Quarantine

Memory is reused only when every process that could still touch it has said it stopped, or is
confirmed gone. If a process cannot say so (it reports `QUARANTINED`, or its stream ended while the
process lives and it has not reported on a new session), the windows it held stay out of service
until it exits, or until the manager does. `selene_pool_quarantined_bytes` shows
the total; a nonzero value that does not fall means an instance is hung, not a leak.

## Metrics

`GET /metrics` on `--metrics HOST:PORT` returns Prometheus text (`selene_` prefix, content type
`text/plain; version=0.0.4`); only that path is served. Store series are built when scraped, on the store thread; if it
does not answer within 1 s the previous store text is served.

| Series | Meaning | Alert on |
|---|---|---|
| `selene_pool_state`, `selene_pool_ready` | 3 ready, 4 draining; and 1 when ready. The endpoint binds after disk recovery, so no value reports recovery. | Draining that does not finish. |
| `selene_pool_regions_held` | 1 while the shared regions are held for processes of the previous run. | Staying at 1. |
| `selene_pool_sessions` | Live instance sessions. | Dropping to zero while instances serve. |
| `selene_pool_index_entries` | Entries announced to the index. | Unbounded growth. |
| `selene_pool_claims_in_flight`, `selene_pool_claim_records`, `selene_pool_lookups_queued` | Work in flight and records kept for fences. | Records that never fall. |
| `selene_pool_claims_total{state,reason}`, `selene_pool_claims_refused_total{reason}`, `selene_pool_lookups_total{status}` | Outcomes. Label values are the Rust variant names, such as `reason="CopyError"`. | `CopyError`, `Checksum`, `NoWindows` rising. |
| `selene_pool_puts_open` | Puts in flight. | |
| `selene_pool_quarantined_bytes` | Window bytes out of service. | Anything sustained. |
| `selene_puts_total{outcome}` | `ok`, `error`, `unlinked`, `present`, `queue`, `budget`, `inflight`, `oversize` for the store's puts. `unlinked` counts page-only puts refused `INVALID` because no held ancestor could be linked, which includes a deepest held ancestor whose chain has 2048 records, where only a put with a row can start a new root; the writer drops them and the refusal is normal. `present` includes puts with no new pages and no row. | Any `error`; `budget` growing faster than `ok`. |
| `selene_put_links_total{parent="linked"\|"root"}` | Accepted puts linked to a held ancestor or stored as complete roots. Includes puts that later fail or are cancelled; excludes refusals, including `PRESENT`. | A rising root fraction can indicate missing ancestors, periodic depth limits, or arena pressure. |
| `selene_put_shapes_total{shape="row"\|"page_only"\|"empty"}` | Accepted puts that carry a row, accepted puts that carry pages only, including later failures or cancellation, and puts with no new pages and no row, which are answered `PRESENT` and also count under `present`. | A rising `page_only` share shows writers sending pages without rows. |
| `selene_put_bytes_total{part="pages"\|"rows"}` | Padded page and row payload bytes accepted per put, excluding headers and checksum tables. Counts the accepted shape once, including puts that later fail or are cancelled. | Page bytes per accepted put rising without a similar increase in conversation length. |
| `selene_gets_total{outcome}` | `ok`, `error` for the store's reads. | Any `error`. |
| `selene_pool_codec_incompatible_total{action}` | Lookups (`action="lookup"`) and claims (`action="claim"`) skipped because the reader does not advertise the record's codec profile. | Any sustained increase; a reader not advertising the record's profile misses packed records. |
| `selene_nvme_errors_total` | NVMe read and write failures. | Any increase. |
| `selene_nvme_bytes_total`, `selene_nvme_read_bytes_total`, `selene_nvme_seconds_total` | Durable bytes written, bytes read for restores, and disk time for both directions. | Disk time growing faster than the sum of written and read bytes. |
| `selene_memory_allocated_bytes`, `selene_staging_allocated_bytes` | Arena and staging in use. | Near `--memory` / `--staging`. |
| `selene_nvme_reserved_bytes`, `selene_nvme_reserved_peak_bytes` | What `--nvme` bounds, as its present value and its high-water mark since start: every record's padded file or rounded arena extent, what each unfinished put will leave on disk, and the staging hold of each unfinished streamed put. Never above `--nvme`. Windows leased for claims and exports are not in it; `selene_staging_allocated_bytes` shows them with the holds, and `--staging` bounds them. | Near `--nvme`: puts are refused `BUDGET` and DRAM eviction stops. |
| `selene_evictions_total`, `selene_compactions_total` | DRAM evictions and row compactions. | Hot set not staying resident. |
| `selene_hits_total{namespace,tier}`, `selene_records{namespace,tier}`, `selene_bytes{namespace,tier}` | Hits and inventory of the manager's own records. | Hit rate collapse. |

## Recovery on disk

Recovery per scope removes leftover `.*.tmp` files, validates every `.ckpt` header, layout,
name and length, sorts parents before children, and deletes orphaned
descendants. A record that is corrupt in a detectable way is removed; one that cannot be read
because of an I/O error fails startup without deletion, so an operator can inspect or move it. If
the disk fills during a write the manager keeps the non-durable DRAM copies and retries every
100 ms; puts are refused `BUDGET` until space returns. Never free space by deleting files under a
running manager.

## Upgrades and clearing the cache

Deployments that ran degraded on local tiers will now fail to start until the pool
attaches. Grep logs for `using local tiers only` before upgrading.

Stop the manager, replace the binary and start it with the same `--root`, `--name` and `--socket`.
Durable records are validated and reused; instances reconnect, resync and refill the DRAM tier from
their traffic. The protocol version must match the instances: a manager that does not speak the
instance's version answers `Hello` with `VERSION_MISMATCH`. This fails instance startup;
after startup it disables the pool until the instance restarts. `--name`, `--socket`,
`--listen`, `--listen-data` and the TLS files are restart-only.

Upgrade managers before clients. A manager that predates page-only puts ignores `row = false`,
admits the put with full admission, which can evict, and grants a window for a row. The client
then cancels the put on the size mismatch and retries the record after its backoff, so a client
that sends page-only puts to such a manager costs a begin, an accept and a cancel on every
attempt.

The root holds one directory per model identity and layout fingerprint, named
`<16 hex>-<32 hex>` from the first 8 bytes of the model identity and the 16-byte layout
fingerprint. Remove one such
directory to drop that model's durable records, or all of them to clear the pool, only while the
manager is stopped. A changed model identity or layout fingerprint selects a different directory, so
the directory it leaves unused is safe to delete. The pool name is not part of the directory name or
the record header: a manager started with a different `--name` on the same root recovers the records
in that root under its own name.

## Troubleshooting

| Symptom | Check |
|---|---|
| Manager exits at startup | Another manager on the same root or `--name`; a root that cannot do `O_DIRECT`; `the store cannot start its disk threads` (io_uring blocked: run with `--security-opt seccomp=selene-manager/deploy/seccomp-io-uring.json`, Docker's default profile plus the three io_uring calls; disabling seccomp is not needed); a certificate and key that do not match; `--rdma` with TLS. |
| Instances fail to attach the configured pool at startup | The pool address, socket permissions, uid/gid equality, `/dev/shm/selene` mounted at the same path in both containers, `selene_pool_ready`. |
| `NO_WINDOWS` | `--staging`; `selene_pool_quarantined_bytes`; many large exports at once. |
| Quarantined bytes do not fall | A hung instance (it will clear when the process exits). |

### Duplicate puts and stale lookup hints

`Present` means the requested pages and, when requested, row are already committed.
Duplicate put logs report cumulative totals at 1, 2, 4, 8, and subsequent doublings;
every put receives its own reply.

A restore warning ending in `UnknownSource` means that its lookup hint names
a source record absent from the index. A source disconnect, manager restart,
resync, or eviction between lookup and claim can cause this race. Lookup does
not pin records. The manager refuses the claim before granting windows or
requesting an export, and the instance falls back to an equivalent local
copy or recomputes after draining. An occasional warning during restarts is
expected; repeated warnings on a quiet pool warrant checking source sessions
and record retention.

### Checkpoint roots with codec envelopes

A manager reads and serves raw v1–v3 and packed v4 files verbatim without
re-encoding them. Every manager sharing a record root with codec-enabled
writers must support v4 envelopes; a manager without v4 support deletes
unrecognized v4 files during recovery and can prune dependent entries. A Flare
instance with `--lossless-kv-compression cpu` or `gpu` advertises its receive
profiles (`ZKV1` and `GKV1`) to the manager over shared memory, TCP, or RDMA.

For a cold restore, staging must fit the largest indivisible stored row. The
file's physical length includes the header, directory, and stored payload.
Incompatible pool lookups and claims increment
`selene_pool_codec_incompatible_total`.
