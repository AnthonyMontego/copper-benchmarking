import os, sys, time, socket

# Parallel synthetic file generator. Each rank creates a disjoint slice of the
# shards (shard_index % nranks == rank), so ranks never touch the same dir --
# no coordination needed, near-linear speedup. Resumable: skips files that
# already exist. Sharded 1000 files/subdir to match the reader's recursive glob.
#
# Usage (under mpiexec):
#   python3 gen_synth_parallel.py <file_size_bytes> <dest_dir>
# file_size in BYTES (e.g. 4096 for 4KB, 65536 for 64KB).

SHARD_SIZE = 1000
TOTAL_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB

size = int(sys.argv[1])
dest = sys.argv[2]
count = TOTAL_BYTES // size
nshards = (count + SHARD_SIZE - 1) // SHARD_SIZE

rank = int(os.environ.get("PMIX_RANK", "0"))
nranks = int(os.environ.get("PALS_LOCAL_SIZE", "0")) or int(os.environ.get("WORLD_SIZE", "1"))
# prefer a global size if available; fall back to MPI via mpi4py
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    nranks = MPI.COMM_WORLD.Get_size()
except Exception:
    pass

host = socket.gethostname()
if rank == 0:
    print(f"[gen] size={size}B count={count} shards={nshards} ranks={nranks} dest={dest}", flush=True)

# random payload reused for every file (content doesn't matter for read timing)
payload = os.urandom(size)

t0 = time.time()
made = 0
skipped = 0
# each rank owns shards where shard % nranks == rank
for shard in range(rank, nshards, nranks):
    shard_dir = os.path.join(dest, f"shard_{shard:05d}")
    os.makedirs(shard_dir, exist_ok=True)
    start = shard * SHARD_SIZE
    end = min(start + SHARD_SIZE, count)
    for i in range(start, end):
        fname = os.path.join(shard_dir, f"f_{i:08d}.bin")
        if os.path.exists(fname):
            skipped += 1
            continue
        with open(fname, "wb") as f:
            f.write(payload)
        made += 1
    if rank == 0 and (shard // nranks) % 50 == 0:
        elapsed = time.time() - t0
        print(f"[gen] rank0 at shard {shard}/{nshards}  made={made} skipped={skipped}  {elapsed:.0f}s", flush=True)

dt = time.time() - t0
print(f"[gen] rank={rank} host={host} made={made} skipped={skipped} time={dt:.1f}s", flush=True)
