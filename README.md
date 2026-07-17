# copper_bench

Benchmarking scripts for characterizing **Copper**, a cooperative read-caching
filesystem, against Lustre for deep-learning I/O workloads on **Aurora** (ALCF).

Workloads are measured with and without Copper, across node counts, each
**cold** (first read, cache empty) and **warm** (re-read, cache populated):

- **`torch/`** — PyTorch package import (`import torch`). Startup I/O: loading
  the torch package + dependencies through Copper vs. directly from Lustre.
- **`resnet/`** — ResNet-50 DDP training on Places365. Dataset-loading I/O:
  reading training images through Copper vs. Lustre.
- **`synth_read/`** — synthetic redundant reads of fixed file sets (controlled
  file size and count) to map where Copper's crossover falls as a function of
  total file requests and file size.

## Repository layout

```
copper_bench/
├── README.md
├── poster.pdf            # findings / results (coming soon)
├── torch/                # import-torch pbs scripts, file_size_histogram.sh
├── resnet/               # xpu_resnet_ddp.py + pbs scripts
├── synth_read/           # generators, reader, sweep pbs
├── figures/              # interactive HTML visualizers
└── results/              # result CSVs
```

Datasets, the custom pip env, and generated synthetic files are **not** in the
repo (multi-GB, live on Lustre) — recreate them via *Setup* below.

## Environment (Aurora)

- **Hardware:** Intel XPU, 6 GPUs × 2 tiles = **12 ranks/node**.
- **Scheduler:** PBS; launcher: PALS `mpiexec`/`mpirun`.
- **Filesystem:** Lustre at `/flare` (→ `/lus/flare/projects/`).
- **Distributed backend:** `xccl` (not `ccl`); rank var `PMIX_RANK`.
- **Copper:** stock master build at
  `/soft/daos/tools/copper_1_0_latest/copper/build/` (RPATH-fixed; no patch
  needed). Launched via `launch_copper_aurora.sh`, mounted at
  `/tmp/$USER/copper_mount`, stopped via `stop_copper_aurora.sh`.

Paths below are the ones used during this project (`/flare/hpc-spectacle/...`);
adjust them to your own project directory.

### Aurora-specific conventions (non-obvious)
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT` — required so `torch.xpu.device_count()`
  returns 12, not 6 (else ranks 6–11 fail). Set automatically by
  `module load frameworks`; set manually otherwise.
- `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `--pmi=pmix` on mpiexec.
- cpu-bind list: `4:56:9:61:14:66:19:71:20:74:25:79`.
- Always run Python with `-u` (unbuffered) — otherwise MPI job output can be
  lost to buffering at job end.
- Copper mount mirrors the *resolved* path: a fileset at `/flare/.../data`
  is read at `${COPPER_MOUNT}/lus/flare/projects/.../data`. Scripts use
  `readlink -f` to resolve this.

## torch/ — import-torch benchmark

- `copper_import_torch.pbs` — imports torch **through the Copper mount**.
- `nocopper_import_torch.pbs` — imports torch **directly from Lustre** (baseline).
- `file_size_histogram.sh` — characterizes the file-size distribution of a
  directory (e.g. the torch env); multiply per-bucket counts by total ppn to
  project aggregate metadata load at scale.

Both benchmark scripts use a custom pip environment (Copper caches the
*package*, so it must be a user-installed torch on Lustre, not the system
module):

- **`lus_custom_pip_env`** (`/flare/hpc-spectacle/lus_custom_pip_env`) —
  torch 2.10.0+xpu, torchvision 0.25.0+xpu, torchaudio 2.11.0+xpu, IPEX 2.10.10,
  mpi4py 4.1.1 (plus the full Intel oneAPI runtime). Routed onto `PYTHONPATH`
  — via the Copper mount for the copper run, raw Lustre for the baseline.
- The Copper run loads `module load copper` for its launcher + Python; the
  baseline loads `module load copper` for the Python interpreter only (Copper
  not launched).

**Timing:** `/usr/bin/time -f "%e"` on the whole mpirun → one cold and one warm
number. Registration and first-RPC times harvested from Copper logs
(min/median/max, µs→s). Results appended to a shared CSV.

## resnet/ — ResNet-50 dataset benchmark

- `xpu_resnet_ddp.py` — ResNet-50 DDP, CUDA→XPU port. Auto-detects dataset
  layout: standard `ImageFolder` (ImageNet/Imagenette) **or** hierarchical
  (Places365) via a `RecursiveImageFolder` that treats each leaf directory as a
  class. Builds a portable JSON index (`.dataset_index.json`, root-relative
  paths) so the same cache works via raw Lustre or the Copper mount.
- `copper_resnet_ddp_places365.pbs` — dataset read **through Copper**.
- `nocopper_resnet_ddp_places365.pbs` — dataset read **directly from Lustre**.

Unlike the torch benchmark, this uses **system frameworks torch**
(`module load frameworks`) — Copper caches the *dataset*, not the package. Only
the dataset path is routed through the Copper mount.

- **Dataset:** Places365 (`data_256`, ~1.8M images, 365 classes) at
  `/flare/hpc-spectacle/places365/data_256`. No `train/` wrapper — the script
  points at the dataset root directly.
- **Cold/warm:** run with `--epochs 2` and a non-shuffling sampler so epoch 2
  re-reads epoch 1's files (epoch 1 = cold, epoch 2 = warm).
- **Metric:** per-epoch throughput / first-batch time, printed by the script.
- Note: dataset must be large enough that each rank gets ≥ `steps+1` batches;
  batch size / steps / node count interact (see script comments).

## synth_read/ — synthetic file-size sweep

Controlled experiment isolating file granularity: dirs of uniform file size,
each totaling ~10 GB, so file *count* varies inversely with size. Every rank
reads all files (redundant reads — the access pattern Copper is designed for).

- `gen_synth_parallel.py` / `gen_synth_parallel.pbs` — parallel generator
  (splits shards across ranks; near-linear speedup). Preferred for large
  file-count dirs.
- `gen_synth_core.sh`, `gen_synth_small.pbs`, `gen_synth_4kb.pbs` — serial
  generator (core logic + wrappers). Fine for low-file-count dirs.
- `read_synthetic.py` — redundant reader; recursively globs the sharded layout.
- `copper_synth_read.pbs` / `nocopper_synth_read.pbs` — cold/warm read, with
  and without Copper. Set `FILESET` / `FILE_SIZE` / `FILE_COUNT` per dir
  (10GB ÷ size: 10M→1024, 4M→2560, 1M→10240, 256K→40960, 64K→163840,
  4K→2621440).

Files are sharded 1000/subdir to keep Lustre directory metadata healthy at high
file counts.

## figures/ — interactive visualizers

Self-contained HTML (data embedded); open in a browser.

- `import_torch_explorer.html` — cold/warm × Copper/no-Copper import time vs.
  nodes; scatter + median, log/linear, cold median-vs-max toggle.
- `places365_throughput.html` — dataset cold/warm throughput vs. nodes.

## results/ — data

Result CSVs produced by the scripts (small; included for reference so the
numbers are visible without re-running large jobs).

## Notes

- "Cold" is best-effort: Aurora nodes retain page cache between jobs, so rapid
  resubmission on the same nodes yields warm-contaminated cold samples. Space
  out cold runs; report median (or max — cold contamination is one-directional,
  only ever making runs faster).
- Copper logs can grow large at scale; scripts use log level 4 and delete each
  job's raw logs after harvesting reg/RPC.
- High-node copper scripts write cold/warm to the CSV *before* the log-harvest
  phase, so a walltime kill during harvest (common at large scale) doesn't lose
  the read numbers. A killed job leaves a `pending` row flagging offline reg/RPC
  recovery.

## Setup: external dependencies

Neither the custom pip environment nor the Places365 dataset is in the repo
(both are multi-GB and live on Lustre). Recreate them as follows.

### Custom PyTorch environment (`torch/` workload only)

The import-torch benchmark caches the torch *package*, so torch must be a
user-installed copy on Lustre (not the system module), built against the
frameworks stack so it matches Aurora's:

```bash
module load frameworks
pip install --target=/flare/hpc-spectacle/lus_custom_pip_env \
  torch==2.10.0+xpu torchvision==0.25.0+xpu torchaudio==2.11.0+xpu \
  intel-extension-for-pytorch==2.10.10 mpi4py==4.1.1
```

Uses the frameworks module's pip (configured for Aurora's XPU package index)
with `--target` so the package tree lives on Lustre. Versions above are the
ones installed for this project. The install resolves the full Intel oneAPI
runtime dependency tree (dpcpp, MKL, oneCCL, etc.) — **do not** use `--no-deps`.

### Places365 dataset (`resnet/` workload only)

Download Places365-Standard, small (256×256) images, from the MIT Places
project: http://places2.csail.mit.edu/download.html — the "Small images
(256×256) — Places365-Standard" training archive. Extract to
`/flare/hpc-spectacle/places365/data_256`.

Layout: `data_256/<a–z>/<category>/[<subcategory>/]<images>` (~1.8M images,
365 classes, no `train/` wrapper). The script reads this root directly and
builds a `.dataset_index.json` on first run.

### Synthetic file sets (`synth_read/` workload only)

Generated on Lustre by the `gen_synth_*` scripts (see `synth_read/`). Each dir
totals ~10 GB; the 4K dir (~2.6M files) is best generated with the parallel
generator. Clean up after collecting data — these are large inode counts on a
shared filesystem.

## References
- Copper — https://github.com/argonne-lcf/copper
- dl_scaling (ResNet-50 source) — https://github.com/argonne-lcf/dl_scaling/tree/main/resnet50
- oneCCL env vars — https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
