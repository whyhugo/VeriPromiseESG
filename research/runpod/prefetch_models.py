"""Pull model weights on a CPU pod, so GPU time is never spent on downloads.

The first attempt at the LLM sweep ran downloads inside the scoring process on a
GPU pod. Two things went wrong: the 4090 sat at 0% for as long as each download
took, and one transfer stalled -- a 3.2 GB `.incomplete` file untouched for 20
minutes while the process waited in `futex_wait_queue` on an HTTP connection that
never came back. Nothing detected it.

So this runs on a CPU pod against the shared network volume, and it is built to
notice a stall rather than wait forever:

  * a monitor thread prints bytes-on-disk every 30s, so a stall is visible in the
    log instead of looking like slow progress
  * each repo gets several attempts, and `snapshot_download` resumes rather than
    restarting, so a retry costs only the bytes actually lost
  * the caller is expected to wrap this in a watchdog that kills and relaunches
    if the total stops growing; retries inside one process cannot help when the
    process itself is wedged

Only weights and config are fetched. `HF_TOKEN` is read from the environment and
never written to disk here -- anonymous downloads are rate limited to around
10 MB/s, which is what made the first attempt so expensive.
"""
import os
import subprocess
import sys
import threading
import time

from huggingface_hub import snapshot_download

REPOS = [
    "unsloth/gemma-3-12b-it",
    "unsloth/gemma-3-27b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]
PATTERNS = ["*.safetensors", "*.safetensors.index.json", "*.json", "*.model",
            "*.txt"]


def cache_dir(repo):
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(home, "hub", "models--" + repo.replace("/", "--"))


def dir_bytes(path):
    if not os.path.isdir(path):
        return 0
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True,
                             timeout=60).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def monitor(stop, repo):
    d = cache_dir(repo)
    last, stalls = -1, 0
    while not stop.is_set():
        n = dir_bytes(d)
        tag = ""
        if n == last:
            stalls += 1
            tag = "   <-- no growth x%d" % stalls
        else:
            stalls, last = 0, n
        print("    [%s] %s %.2f GB%s"
              % (time.strftime("%H:%M:%S"), repo.split("/")[-1], n / 1e9, tag),
              flush=True)
        stop.wait(30)


def pull(repo, attempts=6):
    for k in range(attempts):
        stop = threading.Event()
        t = threading.Thread(target=monitor, args=(stop, repo), daemon=True)
        t.start()
        try:
            # A cheap CPU pod can have a 3.7 GB cgroup limit. Several workers
            # buffering multi-GB shards at once overruns it and the process is
            # OOM-killed with no traceback and nothing in the log -- it simply
            # vanishes mid-download. One worker keeps the footprint flat.
            snapshot_download(repo, allow_patterns=PATTERNS,
                              max_workers=int(os.environ.get("PREFETCH_WORKERS", 1)),
                              etag_timeout=60)
            stop.set()
            print("  DONE %-45s %.1f GB" % (repo, dir_bytes(cache_dir(repo)) / 1e9),
                  flush=True)
            return True
        except Exception as e:
            stop.set()
            print("  attempt %d/%d failed for %s: %s"
                  % (k + 1, attempts, repo, str(e)[:200]), flush=True)
            time.sleep(15)
    print("  GIVING UP on %s" % repo, flush=True)
    return False


def main():
    only = sys.argv[1:] or REPOS
    print("token present: %s" % bool(os.environ.get("HF_TOKEN")), flush=True)
    ok = True
    for r in only:
        print("=== %s  start %s ===" % (r, time.strftime("%H:%M:%S")), flush=True)
        ok &= pull(r)
    print("=== PREFETCH_%s %s ===" % ("OK" if ok else "INCOMPLETE",
                                      time.strftime("%H:%M:%S")), flush=True)


if __name__ == "__main__":
    main()
