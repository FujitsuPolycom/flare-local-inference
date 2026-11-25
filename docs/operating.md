# Operating the Selene checkpoint manager

A manager with a disk root acknowledges an arena put when it can serve the
checkpoint. Idle checkpoints can stay only in RAM. Arena pressure writes
dirty records and their missing predecessors to disk. Records larger than
the arena stream directly to disk. A completed disk write syncs the file,
renames it into place and syncs its directory.

The claim drain and disk flush share a 270-second budget from the stop signal.
Set the container's stop grace period above 270 seconds. A five-minute grace
period leaves 30 seconds for teardown. The manager refuses new puts and drains
outstanding claims for up to the claim deadline plus two seconds, capped at
four minutes. It uses the remaining time to flush dirty records. Independent
chains use up to four write-back workers, with parents first within each chain. It prioritizes
Message, Anchor, root, fork and tip rows and skips page-only leaves.
After a crash or a deadline expiry, recovery uses complete on-disk chains.
An instance can fetch missing state
from a surviving peer or prefill the missing suffix.

A manager with a disk root exports `selene_arena_only_bytes` (arena capacity
occupied by records without files) and `selene_oldest_dirty_seconds` (age
of the oldest such record). `selene_put_writeback_waits_total` counts
admission attempts requeued for write-back. A put can wait more than once.
`selene_shutdown_unflushed_records` and `selene_shutdown_unflushed_bytes`
report the missing writes and their arena capacity at the preceding shutdown.
A rooted manager reads and clears the saved counts at startup. The gauges
are absent on first start or after a crash. A clean flush reports zero.
The counts include ancestors required by retained rows and exclude page-only leaves.
`selene_nvme_bytes_total` counts completed write I/O, including streaming.
`selene_policy_records_total` and `selene_policy_bytes_total` label row
omission, rows released at former tips, and completed writes
by capture role and reason.
Compare write volumes over matched request windows and include the closing
drain or record the change in dirty inventory. Monitor budget refusals and
NVMe errors alongside the dirty gauges.

Captured Message and Anchor boundaries are eligible for an exact rewind
while their rows and pages remain available. A tip with the Endpoint role
omits its row if an ancestor on that path already has a row in the same page.
Roots retain their rows. A fork can retain or regain its row while a writer
still holds it. Existing Legacy-role disk records keep all their available
rows until rewritten. Flare retains single-child Endpoint rows in DRAM for
side requests until it needs space. Under pressure it thins the least
recently used parents first, provided a descendant on the unbranched path
has a row.

On Flare, `flare_host_cache_retirement_seconds` is the latency histogram for
completed DRAM retirements. Use its cumulative buckets for p95 and p99.
Its count also appears as `flare_host_cache_retirement_paths_total` and its
sum in nanoseconds as `flare_host_cache_retirement_nanoseconds_total`.

`flare_host_cache_pool_refusal_drops_total` counts discarded leaves and exhausted ancestor retries
by pool refusal reason. `flare_host_cache_reoffer_calls_total` counts store
calls for reoffered GPU publications. `flare_host_cache_reoffer_present_total`
counts those calls answered PRESENT.

When `--lossless-kv-compression` (or `[prefix_cache] compression`) is `cpu` or
`gpu`, Flare also exports:

- `flare_kv_codec_active{placement="cpu"|"gpu"}` (gauge, `1` while active)
- `flare_kv_codec_original_bytes_total{placement}` and
  `flare_kv_codec_stored_bytes_total{placement}` (uncompressed and compressed
  bytes processed across capture and restore)
- `flare_kv_codec_transform_nanoseconds_total{placement}` (cumulative codec
  transform duration)
- `flare_kv_codec_failed_windows_total{placement}` (transfer windows that
  failed encode or decode)
- `flare_kv_codec_verification_failures_total{stage="stored"|"original"}`
  (CRC-32/IEEE verification failures on compressed or decoded payloads)
