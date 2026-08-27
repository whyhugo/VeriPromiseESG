#!/usr/bin/env bash
# Fetch model weights with curl instead of the huggingface_hub client.
#
# On a small CPU pod the hub client could not do this. Its shards are 4.85 GB
# against a 3.7 GB cgroup limit, so the process was OOM-killed roughly every 90
# seconds, and on restart it opened a *new* .incomplete file rather than
# resuming the old one -- 63 abandoned partials totalling 42 GB with not one
# complete file to show for it. Killing it also left 161 stale lock files that
# blocked every later attempt.
#
# curl streams to disk in constant memory and `-C -` resumes a partial byte
# range for real, so an interrupted transfer costs only the bytes in flight.
# Files land in a plain directory; from_pretrained() takes a path just as
# happily as a repo id.
#
# usage: fetch_curl.sh <repo> <dest_dir>
set -u
REPO="$1"
DEST="$2"
mkdir -p "$DEST"

# Pods differ in which interpreter has huggingface_hub, so pick one that does
PY_BIN=""
for C in python3.13 python3.12 python3 python; do
  command -v "$C" >/dev/null 2>&1 || continue
  "$C" -c "import huggingface_hub" 2>/dev/null && { PY_BIN="$C"; break; }
done
[ -z "$PY_BIN" ] && { echo "no interpreter with huggingface_hub"; exit 1; }

FILES=$($PY_BIN - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
keep = (".safetensors", ".json", ".model", ".txt")
for s in HfApi().model_info(sys.argv[1]).siblings:
    f = s.rfilename
    if f.endswith(keep) and "/" not in f:
        print(f)
PY
)

echo "=== $REPO -> $DEST"
for F in $FILES; do
  URL="https://huggingface.co/$REPO/resolve/main/$F"
  OUT="$DEST/$F"
  # -C - resumes; a completed file makes curl exit 33 or 416, both meaning
  # "nothing left to fetch", so treat them as success
  for try in 1 2 3 4 5 6 7 8 9 10; do
    SZ=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    curl -sSL -C - --retry 5 --retry-delay 5 \
         --connect-timeout 30 --speed-time 120 --speed-limit 10240 \
         -H "Authorization: Bearer ${HF_TOKEN}" \
         -o "$OUT" "$URL"
    RC=$?
    NEW=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [ "$RC" = "0" ] || [ "$RC" = "33" ] || [ "$RC" = "416" ]; then
      printf "  ok   %-42s %8.2f GB\n" "$F" "$(echo "$NEW" | awk '{print $1/1073741824}')"
      break
    fi
    printf "  retry %-41s rc=%-3s %8.2f GB (try %s)\n" "$F" "$RC" \
           "$(echo "$NEW" | awk '{print $1/1073741824}')" "$try"
    [ "$NEW" = "$SZ" ] && sleep 20
  done
done
echo "=== $REPO done: $(du -sh "$DEST" | cut -f1)"
