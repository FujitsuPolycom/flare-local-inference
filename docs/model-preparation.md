# Model preparation and serving

## Runtime cores

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime core** and publishes a directory entry in
`paths.core_cache` (default `~/.cache/flare/cores`). Its primary,
`model.safetensors`, contains the signed `flare.manifest`, unsigned
`flare.locations`, optional reconstruction hints, and frontend assets. Readably
named safetensors companions hold layer weights, tuning-dependent weights,
initial state, zstd-compressed cubins, and graph records. Cache companions hardlink
immutable payload objects whose identities hash the payload bytes.

The first canonical form packs the core into one `.safetensors` file; the
second is the directory form used by the cache. Export always publishes to that
cache. `ignition convert CACHE_ENTRY --output DIRECTORY` creates a standalone
directory; `ignition convert CACHE_ENTRY --form packed --output FILE` creates
one exact file path. `CACHE_ENTRY` is the path printed by export. Packing and
unpacking preserve signed payloads and require no signing key or GPU. Layer payloads are not grouped by a size threshold. Internal
logical-resource cores use envelope 6 with monolithic, numbered-shard, or
blob layouts. Legacy blob publication uses a fixed maximum of 64 GiB per file;
there is no configurable shard-size threshold.

`ignition export --no-weights` requires the exact stable and tuned weight
payloads in the cache. Initial state and execution resources may change with
the prepared capacities. Preparation and weight reads still occur. Successful
exports select their cache entry as the model's default. Recipes can declare
`export_repository` as the Hugging Face publishing target; prepared storage
uses the shared Flare cache. Multi-host export publishes the complete entry
in the coordinator's cache. See [core snapshots](cli-design.md#core-snapshots) for cache
management, conversion, and upload.

`ignition export --reuse-weights` uses the model's cached default core as
the weight source. It reuses that version's signed weight hashes and initializer
mappings, skips reading declared numeric weight storage during export, and
hardlinks the cached payloads into the resulting core without rehashing their
contents. Checkpoint loading
and model preparation still run. The checkpoint revision, TP size, architecture,
and declared weight layouts must match; missing or incompatible weights fail
export. Cached weight bytes are authoritative and are not compared with live
tensor values. Use ordinary export after changing weight values or packing.
Reuse includes Engram's file-backed embedding weights and scale planes, in
both shared and row-partitioned table layouts. Their geometry, quantization,
vocabulary map, and hashing parameters must match the cached version.
Graphs and initial state are always exported from the prepared model. See
[export flags](cli-design.md#31-ignition-command-line-flags) for the metadata requirements.

Each tensor-parallel deployment contains independently prepared rank metadata,
graphs, and kernels. Combining exported deployments can share or reconstruct
byte-identical weights; it cannot create an unexported TP size. Ignition creates
a per-user development signing key when `[publishing]` is absent. `flare run`
needs the corresponding publisher profile and first-use acceptance. See
[signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the core files without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Recipes generally use FP8 E4M3 KV storage. Kimi DSpark drafts retain BF16 keys and values,
and Qwen Flash retains BF16 selector keys;
GLM and DeepSeek retain their packed latent formats, including scale metadata and any BF16 rotary fields.
Recurrent state, compressor accumulators, and hidden activations retain their
model-specific precision. Cache writers and attention readers use matching
scales where required, and memory planning includes each cache's declared page
precision and partial-page snapshots.
A core's cache layouts are fixed when it is prepared.

## Environment setup

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare-cli/Cargo.toml
```

`ignition.runtime_patches` installs process-local fixes for PyTorch 2.14.1+cu132
and Triton 3.8.0. Preparation commands set dependency paths and GPU visibility
before installation. Spawned ranks, tuning replicas, b12x compiler workers, and
Inductor workers install the fixes they require before using the affected code.
Dependency files and shared wheel-cache files remain unchanged. Version and
source-context checks reject incompatible implementations; existing source-patched
installations are also accepted.

The fixes preserve cooperative CUDA launch attributes in serialized kernels,
match Triton's 150-character cache filenames, retain dependencies on mutated
inputs when replacing kernel templates, and reduce compiler bookkeeping costs.
Inductor retains zeroed graph workspaces through the final graph operation, so
its memory planner cannot reuse their storage while the graph still needs it.
Triton's helper cache key uses machine, operating system, and interpreter pointer
width without invoking `file`, avoiding subprocess probing under profiler injection.
Compiled-forward and Torch graph cache identities include the runtime patches.

Standalone Python tools that use Torch or Triton directly must install the fixes
in the same process before compilation or CUDA driver initialization:

```python
from ignition import runtime_patches

runtime_patches.apply()
```

`python scripts/setup_preparation.py` updates FlashInfer to the tip of its `main`
branch, syncs preparation and test dependencies, and rebuilds FlashInfer and
Heliopause. It updates `uv.lock` when the branch advances.

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

Windows native builds also require Perl for the bundled static OpenSSL build,
alongside the C/C++ toolchain. The resulting binaries do not require OpenSSL DLLs.

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI; core export also invokes `nvcc`
to compile the embedded GPU KV-cache codec module (`kv_codec_gpu.cu`) for cores
that declare a codec layout. Compiled cubins are cached under
`[paths] kernel_cache/kv_codec` when configured, falling back to a private
mode-`0700` user directory under `$TMPDIR` (or a fresh temporary directory) when
`kernel_cache` is unset or unwritable. Native build libraries discover the
system toolkit; `[paths] cuda` overrides that discovery with an explicit toolkit
directory.

The `prepare` extra installs `cuda-tile[tileiras]`, including the `tileiras`
compiler toolchain required by cuTile kernels during model preparation.
cuTile discovers its matching compiler tools in the Python environment
independently of the toolkit used for native builds.

The DeepSeek Rust crates share a pinned revision of
[`local-inference-lab/deepseek-recipe`](https://github.com/local-inference-lab/deepseek-recipe)
in `helios/Cargo.toml`. Its `flare` branch exposes normalized-message byte
boundaries, advisory checkpoint hints, and numeric V4.1 reasoning effort.
To update these crates, integrate upstream changes
on that branch, validate the encoder, and update all three dependency revisions
together. Refresh and commit every lockfile that records the crates:
`helios/Cargo.lock`, `flare/Cargo.lock`, `flare-cli/Cargo.lock`,
`heliopause/Cargo.lock`, and `flare-winui/bridge/Cargo.lock`. Run the Helios
frontend/protocol tests and build Flare and Heliopause against the pinned
revision. Compare rendered prompts and token
IDs with upstream when changing encoder behavior.

The preparation dependency tracks the `main` branch of
`local-inference-lab/flashinfer` in `pyproject.toml`; `uv.lock` records the
resolved commit. `.github/workflows/track-flashinfer-main.yml` checks
`local-inference-lab/flashinfer` `main` every 15 minutes and updates `uv.lock`
on `main` when it advances, so `uv sync --locked --extra prepare --extra test`
tracks `main` without manual pin edits. To pull an unmerged commit ahead of the
scheduled refresh, run `uv lock --upgrade-package flashinfer-python` or
`python scripts/setup_preparation.py`.
For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

## GPU selection

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared core; a TP2 core cannot be loaded as TP1. Qualified end to
end: Qwen at TP1 and TP2, and GLM at TP4. Not qualified: Qwen at TP4.

## Preparation resources

Core assembly retains file handles and cache leases for every rank's weight
payloads until publication finishes. For sixteen-rank Kimi exports on Linux, set
`ulimit -Sn 65536` in the export shell; its hard limit must allow that value.

Checkpoint reads use b12x's pinned-host reader. Each rank initializes only its
destination GPU for file transfers, avoiding CUDA contexts on its peers' GPUs.

`--autotune-effort off|default|audit` controls b12x candidate filtering.
`off` prepares heuristic choices without tuning races or tuned-selection cache
lookups. `default` races the candidates that pass each kernel family's
performance exclusions. `audit` lifts those exclusions and races every
candidate satisfying correctness, resource, dominance, and explicit
configuration constraints; it exists to check that the default search prunes
no winners and is meant for occasional sampled runs, not routine tuning. The
audit search includes every default candidate. Kernel families retain their
declared tuning axes.

`--lock-gpu-clocks` locks each participating GPU to its maximum supported
graphics frequency for the b12x tuning batch and actual Inductor autotuning
trials during compiled-forward preparation. Frozen warmup, capture, and serving
run without these locks. It applies to serving ranks and
extra tuning replicas, and does nothing with `--autotune-effort off`. The GPU's
UUID selects the device under `CUDA_VISIBLE_DEVICES`. Run with permission to
control clocks: Ignition uses `sudo -n nvidia-smi` when not root. Unsupported
clock control or missing permission fails preparation. Locks reset to driver
defaults after tuning or a Python exception; preexisting manual clock locks
are not restored. Memory clocks and tuning caches are unchanged.

Without extra tuning replicas, tensor-parallel ranks on identical GPUs divide
the candidates of a shared local kernel race and adopt its measured winner.
When the model catalogs differ, preparation groups together each local request,
its dependencies and its declaration aliases. A group can share a race when all
ranks declare the same requests, candidate factories and memory requirements.
The remaining local operations prepare on their owning ranks. Collective kernels
prepare on every participating rank. This applies within one host and across hosts.

Set `[preparation] autotune_effort = "audit"` for a persistent setting; an
explicit flag wins. The plan carries the setting to every preparation rank
and tuning replica. Effort separates cached selections and prepared cores;
identical kernel programs share compilation artifacts across efforts.

Tuning checkpoints each completed measurement batch in a SQLite database beside
the kernel selection cache. Tensor-parallel ranks save independently, including
completed local races waiting for winner exchange. Restarting the same workload
restores measured candidates and the carried winner, then measures unfinished
candidates against that winner. The progress display distinguishes restored,
newly measured, and rejected candidates. An interruption loses the unfinished
measurement batch on each rank; compilation artifacts remain separately cached.

Partial checkpoints require the same candidate program identities, effort,
tuning shard count, and measurement settings. A kernel or predicate change
invalidates the affected race's checkpoint. Completed selection caches remain
usable independently of the tuning shard count. Resource-dependent cooperative
launch rejections are retried after restart; fatal CUDA errors require
restarting preparation in fresh processes.

`--compile-workers N` sets a positive CPU compiler process budget per host for
b12x kernel compilation and complete-forward compilation. Available RAM and
device memory can reduce it; local preparation ranks share the budget. Set
`[preparation] compile_workers = N` for a persistent default; the CLI flag
overrides that setting. Omitting both retains automatic sizing, including
shared-memory limits on DGX Spark.

Forward compilation allocates workers across each host after checking cached
forwards. Ranks without a worker lend missing forwards to other
ranks on that host; every serving rank binds and captures its full catalog.
Concurrency uses a 32 GiB host-RAM estimate per worker, plus device storage on
integrated GPUs. When that estimate admits no worker, one worker per host may
start under the 2 GiB minimum host allowance plus integrated device storage.
These allowances estimate compiler demand; actual peaks depend on the forward.
Linux cgroup-v2 memory limits also constrain the available RAM used for admission.

On integrated GPUs, kernel tuning is divided into memory-bounded jobs. The
terminal dashboard shows the active job number and total job count, with a
separate live display for each job. Redirecting output records plain kernel
progress milestones instead of the terminal dashboard.

`--tuning-gpus LIST` adds local GPU replicas for benchmarking b12x candidates.
The list begins with `--gpus` and continues in complete TP groups: for example,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` prepares a TP1 core with
four tuning GPUs. Corresponding GPUs in each replica must match in model and
architecture and have enough free memory for the model and preparation storage.
Helpers exit and release their resources after tuning; serving TP size and capture
devices stay fixed. Serving ranks then compile and capture their forwards.
Both GPU lists use `CUDA_VISIBLE_DEVICES` ordinals. Kernel compilation runs in
CPU workers, independently of the number of tuning GPUs.

`--tuning-gpus` is local-only and cannot be combined with multi-host `--ranks`
preparation, whose ranks divide tuning candidates as described above. An explicit
`--compile-workers` budget applies separately on each host.

Four-row Kimi K3.1 verification on a sixteen-rank decode context group can
select KV records directly on their owning ranks. When FlashInfer provides an
eligible PCIe channel, preparation times this path against the measured global-ID
selection and its physical page mapping. Both use the same rank-major ordering of
FP32 score words and the same causal and page-table checks. Diagnostic calls that
request the global IDs continue to materialize them.

## Exporting runtime cores

Export publishes a directory entry in the shared model cache and prints its
path. In the examples below, replace `CACHE_ENTRY` with that path before running
`ignition convert`. Conversion requires an unused destination and does not append
a suffix. Do not delete an existing core merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the core, and exits.
Compilation is enabled by default; `--no-compile` is for development.
The examples omit default settings. Configuration-file overrides still apply;
inspect `ignition export MODEL --help` and the printed preparation plan for the
resolved capacities. Select idle GPUs explicitly. Native serving listens on
`127.0.0.1:8000` unless configuration or `--host` / `--port` overrides it.
Add `--verbose` for per-phase detail or `--lock-gpu-clocks` to lock clocks during
autotuning.

### Re-exporting with cached weights

After an ordinary export, use `--reuse-weights` when rebuilding graphs with
the same checkpoint, TP size, architecture, and compatible weight layouts.
For a DeepSeek V4.1 TP4 export:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 --reuse-weights
flare-cli/target/release/flare run CACHE_ENTRY --gpus 0,1,2,3
```

Replace `CACHE_ENTRY` with the directory printed by this export. The command
uses the model's cached default version as its weight source and publishes
the resulting core as the default. Checkpoint loading, preparation, and
capture still run; compatible tuning and compilation caches remain reusable.
Use ordinary export after changing weight values or packing, or when no
compatible cached default exists. `--no-weights` instead hashes the live
weights and requires their exact payloads to be cached; it cannot be combined
with `--reuse-weights`.

## Official model recipes

Official model recipes, per-model export commands, hardware profiles, and
qualification notes live in
[`local-inference-lab/hyperflare-recipes`](https://github.com/local-inference-lab/hyperflare-recipes/blob/main/docs/model-preparation.md):

- Nemotron 3.5 Bolt 74B-A9B 24GB (`nemotron-bolt-24gb`)
- Qwen3.8 27B (`qwen3.8-27b`, `qwen3.8-27b-qad`) and Qwen3.8 Flash (`qwen3.8-flash`)
- GLM 5.3 Flash (`glm-5.3-flash`, `glm-5.3-flash-nvfp4`)
- DeepSeek V4.1 Flash (`deepseek-v4.1-flash`) and DeepSeek V4 Flash (`deepseek-v4-flash`)
- MiMo V2.6 Pro MOPD (`mimo-v2.6-pro-mopd`) and MiMo V2.6 Flash RL (`mimo-v2.6-flash-rl`)
- Step-5 Preview (`step5-preview-nvfp4`)
- Kimi K3 (`kimi-k3`, `kimi-k3-mxfp4-csf`, `kimi-k3-dspark`) and Kimi K3.1 Pro (`kimi-k3.1-pro`)
- MiniMax M3.1 Flash (`minimax-m3-1-flash`)
- Inkling 2 Small NVFP4 (`inkling-2-small-nvfp4`) and Inkling 2 Mini (`inkling-2-mini`)
- Muse Spark Open (`muse-spark-open-nvfp4`)
- Gemma 4 (`gemma-4-e2b`, `gemma-4-e4b`, `gemma-4-12b`, `gemma-4-26b-a4b`, `gemma-4-31b`, and NVFP4 variants)

External recipe repositories do not need to merge into either repository;
depend on `ignition` from this repository and pass your `Recipe` tuple to
`ignition.cli.main(...)`.

## Storage and graph capacities

Gemma 4 and Muse Spark default to four concurrent requests. Gemma 4's speculative
target-token capacity is the larger of padded concurrency and draft count plus
one, giving six rows at its default settings. Muse Spark declares 32 packed
target-token rows for seven DFlash2 drafts at concurrency four. Other recipes
default to 32 concurrent requests, with speculative recipes declaring 128 target-token rows independently
of concurrency.
Explicit flags and configuration override these defaults; a core's capacities
are fixed at export. Memory planning rejects capacities that do not fit the
selected GPUs.

| Recipe | Draft tokens | Verification layout | Speculative graphs at default concurrency | Maximum drafts at batch 32 |
|---|---:|---|---:|---:|
| Qwen3.8 Flash, GLM 5.3 | 5 | Rectangular | 51 | 3 |
| Step-5 | 3 | Rectangular | 36 | 3 |
| DeepSeek V4, MiniMax M3.1 | 5 | Packed | 64 packed + 9 width-one | Up to 5, sharing 128 rows across requests |
| DeepSeek V4.1, MiMo V2.6 Pro, Kimi K3, Kimi K3.1 Pro, Nemotron Bolt, Qwen3.8 27B | 7 | Packed | 64 packed + 9 width-one | Up to 7, sharing 128 rows across requests |
| Gemma 4 | 5 | Rectangular | 10 at concurrency 4 | Not prepared at default concurrency |
| Muse Spark | 7 | Packed | 49 packed + 3 width-one at concurrency 4 | Not prepared above concurrency 4 |

The 128-row rectangular catalogs cover full draft depth through batch 16 (and
four drafts at batch 24 for five-draft recipes).
Gemma 4's six-row default covers five drafts at batch 1 and two at batch 2,
with width-one execution at batch 4. The counts include
width-one speculative graphs and exclude ordinary decode, prefill, maintenance,
and vision graphs. Capturing full rectangular depth at batch 32 requires 192 rows
for five drafts or 256 rows for seven drafts, set with
`--speculative-token-capacity`. Every packed recipe's round graph count is bounded
independently by `--speculative-graph-budget`.

Every model export prints a VRAM inventory for each rank after final publication,
including exports without `--verbose`. Heliopause measures device usage and core
allocator mappings; Ignition attributes retained ranges to checkpoint tensors,
graph temporaries, I/O and persistent state, communication buffers,
kernel globals, and allocator slack. Owner fields and defining source locations
identify the largest consumers in each category. Aliased ranges count once;
shared storage displays representative tensor owners.

Preparation-only storage is separate from fixed core storage. Cache figures
count physically backed pages, not their virtual serving capacity. Mapped host
storage and imported peer mappings are listed outside local VRAM totals. PyTorch
allocated and reserved bytes overlap the attributed storage. Cubin data sizes are
not resident code sizes: graph executable overhead, kernel code, other driver
allocations, and other processes remain in the unattributed device-usage remainder.
The returned `RuntimeCore.memory_inventory` contains every attributed source
group; console output shows the largest five per category. Source locations name
owning classes or closures, not allocation stack traces.

Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `core_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally budgets 1 GiB for warm-up and capture, plus 128 KiB per
effective prefill token above 4096, including mixed prompt/decode rows.
Preparation KV backing covers a prefill chunk plus boundary pages per captured request row, unless an
explicit preparation page count is supplied programmatically. Continuation
backing covers the null row, live request and verification rows, and snapshot
destinations used during preparation. Planned prefix-cache slots remain virtual;
they do not require physical backing during export. The memory report separates
the declared continuation extent from `preparation_state_rows` and
`preparation_state_bytes` where those extents differ. Both caches retain their
full virtual extents in the core. Helios
sizes its caches after uploading the selected graphs, keeping a separate 64 MiB
driver margin.
At core load time, `flare run --kv-cache-limit context` bounds the shared
KV pool by one prepared context; `concurrency` uses one context per selected
active request, and the default `memory` fills the available budget. Runtime
`--memory-limit` accepts an occupied size (`30G`), free reserve (`-30G`), or
fraction of total device memory (`0.8`). It replaces the exported serving
reserve and includes model resources as well as caches. Both flags apply at
core load time without another export. See
[runtime memory limits](cli-design.md#runtime-memory-limits) for rounding,
minimum capacities, and integrated-memory accounting. These flags do not
affect preparation's memory policy.

Before compilation and capture, preparation prints a persistent graph catalog
with counts, batch and prefill capacities, logical widths, physical target rows,
and draft depths, including when `--verbose` is omitted.
Ignition captures batch capacities from its preparation catalog
(`1, 2, 4, 6, 8, 12, 16, 24, 32, 64`) through the padded concurrency limit
and width one at every bucket; Helios enumerates whatever valid batch
capacities a loaded core declares rather than requiring a fixed bucket list. Rectangular speculative and verification shapes
require `batch * width <= speculative_token_capacity`; the preparation flag
`--speculative-token-capacity` defaults to 128 physical target rows except for
Gemma 4's derived capacity and Muse Spark's 32 rows. Packed recipes use the
shared capacity selection
described under DeepSeek V4.1 rather than rectangular `batch * width` shapes.
Flare can reduce concurrency at load time. Rectangular catalogs permit smaller
positive draft limits; packed catalogs require their captured positive depth or
zero. Prefill capacity stays prepared.
With `--mixed-prefill`, Qwen also captures mixed prompt/decode graphs: every
prompt bucket at width one and one balanced prompt bucket at each wider
verification depth, for each batch capacity of at least two. The ordinary and
draft-state-maintaining catalogs remain distinct. Mixed capture is disabled by
default and independent of the MTP flags.
See [mixed graph coverage](scheduling-runtime-design.md).

## Serving runtime cores

Load an exported `.safetensors` core directly with the native CLI:

```sh
./flare-cli/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1
curl --fail http://127.0.0.1:8000/health
```

Select a subset of a core's captured execution capacities:

```sh
./flare-cli/target/release/flare run /data/flare-runtime/qwen38-tp1-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Speculative exports contain
these ordinary graphs by default. Ignition's `--mtp-only` omits their construction,
tuning, compilation, and capture, reducing preparation resources and core graph
count. It requires positive draft depth and rejects `--no-mtp`; cores exported
with it reject native `--mtp-draft-tokens 0`. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width where captured,
with width-one execution available at larger batch capacities. Omitted flags
use the core's serving defaults. Limits cannot exceed captured capacity;
missing required batch/width combinations fail before GPU initialization.
Concurrency three requires the batch-four bucket. Only selected graphs and
required kernel functions are loaded. Weights, MTP weights, and transport buffers
retain their prepared layouts. Continuation rows and KV pages occupy separate
fixed virtual ranges with declared maximum extents. Helios backs those ranges
according to selected concurrency, verification width, and available GPU memory;
reducing execution state leaves more physical memory for KV pages.

The selected GPU count must match the core, and each GPU must match its
rank's architecture and memory requirements. Flare serves all ranks in one
process with one execution thread per rank. Its dashboard runs on an
interactive terminal. Use `--api-key-file PATH` to supply a serving API key;
exported cores do not contain the key. Stop Flare with Ctrl-C or SIGTERM
so native replay drains before CUDA resources are released.

## DGX Spark

DGX Spark's GB10 is SM121 and shares physical DRAM with the CPU. Prepare an
SM121 core on Spark; an SM120 core does not satisfy the architecture
check. GPU weights, KV, registered host buffers, resident embedding tables,
compiler workers, and the OS compete for that memory. Set `[memory] reserve`
for host activity as well as driver headroom; `reserve = "4G"` leaves four GiB
outside the model's cache budget. RAM embedding tables load before cache
sizing. Disk embedding tables use bounded staging and can retain their scale
planes in RAM with `--embedding-table-resident-scales`. The loader accounts for
resident scales and the optional row cache before sizing KV, for both PLE and
Engram cores. Disk tables default to a 1 GiB host-RAM row cache per Spark;
`--embedding-table-cache-size` overrides it, and `0` disables it.

Spark's NVML interface does not report VRAM capacity. Ignition identifies an
integrated device through CUDA and uses Linux `MemAvailable`, capped by CUDA's
device total, for planning. Qwen preparation and Helios use the same available
RAM budget when allocating weights and caches; swap is not GPU capacity.
CUDA free-memory queries exclude reclaimable page cache on GB10. Compiler worker
budgets account for both CPU and GPU allocations in shared RAM. Helios sizes
its caches independently after loading.
VMM allocations and registered host mappings remain distinct CUDA
memory types despite sharing DRAM; do not dereference a device allocation on
the CPU or replace system-scope synchronization with volatile accesses. See
[NVIDIA's Spark CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

A model too large for one Spark runs tensor-parallel across two, three, or four Sparks,
one rank per host, reducing through b12x RoCEnante over their ConnectX-7 ports:
`ignition export --ranks N --rendezvous ADDR:PORT` on rank zero with
`ignition join` on each peer writes one core on rank zero, and
`flare orbit run NODE:CORE --cluster NAME` serves it from any machine,
starting every rank through its host's `flare agent`. See
[serving across DGX Sparks](multi-host.md) for agent installation, pairing,
and failures.

## Core and development serving

Export a core, then launch the native server:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 0,1
.venv/bin/ignition convert CACHE_ENTRY --form packed --output qwen.safetensors
flare-cli/target/release/flare run qwen.safetensors --gpus 0,1
```

To test a serving-only Rust change, rebuild Flare and load an existing compatible
core. Re-export when changing prepared kernels, capacities, or the core
format. Export publishes a cache entry and takes no destination;
`ignition convert --output` destinations must not already exist.

Use `.venv/bin/ignition export --help` for preparation flags and
`flare-cli/target/release/flare run --help` for serving flags. See [the runtime core design](runtime-core-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For fault diagnosis, `ignition debug-serve` hands prepared graphs directly to
Helios without exporting a core. Its native path requires Heliopause's
`qualification` Cargo feature. All debug modes, including `--direct` and
`--eager`, are unsuitable for actual serving or apples-to-apples performance
comparisons against other runtimes; use an exported core with `flare run`.
Select idle devices explicitly; see [the debug launch example](cli-design.md#development-without-core-export)
for the extension build and diagnostic commands.
