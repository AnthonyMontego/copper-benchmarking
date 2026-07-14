# copper\_bench

Benchmarking scripts for characterizing **Copper**, a cooperative read-caching
filesystem, against Lustre for deep-learning I/O workloads on **Aurora** (ALCF).

Two workloads are measured, each with and without Copper, across node counts:

- **`torch/`** — PyTorch package import (`import torch`). Measures startup I/O:
  loading the torch package + dependencies through Copper vs. directly from Lustre.
- **`resnet/`** — ResNet-50 DDP training on the Places365 dataset. Measures
  dataset-loading I/O: reading training images through Copper vs. Lustre.

Each workload is measured **cold** (first read, cache empty) and **warm**
(re-read, cache populated).

## Environment (Aurora)

- **Hardware:** Intel XPU, 6 GPUs × 2 tiles = **12 ranks/node**.
- **Scheduler:** PBS; launcher: PALS `mpiexec`/`mpirun`.
- **Filesystem:** Lustre at `/flare` (→ `/lus/flare/projects/`).
- **Distributed backend:** `xccl` (not `ccl`); rank var `PMIX_RANK`.
- **Copper:** stock master build at
  `/soft/daos/tools/copper_1_0_latest/copper/build/` (RPATH-fixed; no patch
  needed). Launched via `launch_copper_aurora.sh`, mounted at
  `/tmp/$USER/copper_mount`, stopped via `stop_copper_aurora.sh`.

### Aurora-specific conventions (non-obvious)
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT` — required so `torch.xpu.device_count()`
  returns 12, not 6 (else ranks 6–11 fail). Set automatically by
  `module load frameworks`; set manually otherwise.
- `CCL_PROCESS_LAUNCHER=pmix`, `CCL_ATL_TRANSPORT=mpi`, `--pmi=pmix` on mpiexec.
- cpu-bind list: `4:56:9:61:14:66:19:71:20:74:25:79`.
- Always run Python with `-u` (unbuffered) — otherwise MPI job output can be
  lost to buffering at job end.

## torch/ — import-torch benchmark

- `copper_import_torch.pbs` — imports torch **through the Copper mount**.
- `nocopper_import_torch.pbs` — imports torch **directly from Lustre** (baseline).

Both use a custom pip environment (Copper caches the *package*, so it must be a
user-installed torch on Lustre, not the system module):

- **`lus_custom_pip_env`** at `<VERIFY: /flare/hpc-spectacle/lus_custom_pip_env>`
  — contains torch 2.10.0+xpu, torchvision 0.25.0+xpu, IPEX, mpi4py, installed
  via the frameworks pip with `--target`. Routed onto `PYTHONPATH`
  (via the Copper mount for the copper run, raw Lustre for the baseline).
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
  `<VERIFY: /flare/hpc-spectacle/places365/data_256>`. No `train/` wrapper —
  the script points at the dataset root directly.
- **Cold/warm:** run with `--epochs 2` and a non-shuffling sampler so epoch 2
  re-reads epoch 1's files (epoch 1 = cold, epoch 2 = warm).
- **Metric:** per-epoch throughput / first-batch time, printed by the script.
- Note: dataset must be large enough that each rank gets ≥ `steps+1` batches;
  batch size / steps / node count interact (see script comments).

## Notes

- "Cold" is best-effort: Aurora nodes retain page cache between jobs, so rapid
  resubmission on the same nodes yields warm-contaminated cold samples. Space
  out cold runs; report median (or max — cold contamination is one-directional,
  only ever making runs faster).
- Copper logs can grow large at scale; scripts use log level 4 and delete each
  job's raw logs after harvesting reg/RPC.

## Setup: external dependencies

Neither the custom pip environment nor the Places365 dataset is included in
this repo (both are multi-GB and live on Lustre). Recreate them as follows.

### Custom PyTorch environment (`torch/` workload only)

The import-torch benchmark caches the torch *package*, so torch must be a
user-installed copy on Lustre (not the system module). Installed against the
frameworks stack so the build matches Aurora's:

```bash
module load frameworks
pip install --target=/flare/hpc-spectacle/lus_custom_pip_env --no-deps \
  <VERIFY exact versions/index — e.g. torch==2.10.0+xpu torchvision==0.25.0+xpu \
   intel-extension-for-pytorch mpi4py>
```

Contains: torch 2.10.0+xpu, torchvision 0.25.0+xpu, IPEX, mpi4py. The torch
build must match the frameworks torch git hash; installing via the frameworks
pip with `--target`/`--no-deps` ensures this.

### Places365 dataset (`resnet/` workload only)

```bash
mkdir -p /flare/hpc-spectacle/places365
cd /flare/hpc-spectacle/places365
wget <VERIFY Places365 data_256 URL, e.g. \
  http://data.csail.mit.edu/places/places365/train_256_places365standard.tar>
tar -xf <archive>
```

Yields `data_256/<a–z>/<category>/[<subcategory>/]<images>` (~1.8M images,
365 classes). The script reads this root directly (no `train/` wrapper) and
builds a `.dataset_index.json` on first run.

## References
- Copper — https://github.com/argonne-lcf/copper
- dl\_scaling (ResNet-50 source) — https://github.com/argonne-lcf/dl\_scaling/tree/main/resnet50
- oneCCL env vars — https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
