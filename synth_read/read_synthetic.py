import os, sys, time, glob, socket

# Redundant read: every rank reads ALL files in the target dir.
# Prints one line per rank: READ rank=<r> host=<h> read_s=<s> bytes=<b> nfiles=<n>
target = sys.argv[1]
rank = os.environ.get("PMIX_RANK", "0")
host = socket.gethostname()

files = sorted(glob.glob(os.path.join(target, "**", "f_*.bin"), recursive=True))
buf = bytearray(1024 * 1024)

t0 = time.time()
total = 0
for path in files:
    with open(path, "rb") as fh:
        while True:
            n = fh.readinto(buf)
            if not n:
                break
            total += n
dt = time.time() - t0

print(f"READ rank={rank} host={host} read_s={dt:.4f} bytes={total} nfiles={len(files)}", flush=True)
