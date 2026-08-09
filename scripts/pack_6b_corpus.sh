#!/usr/bin/env bash
# Pack finemath-4plus + infiwebmath-4plus into one ~6B-token corpus.
#
# Both configs sit at the same 4+ quality threshold, so this grows the corpus
# without moving the quality bar -- the existing 2.05B corpus is a subset of
# the same distribution, which keeps loss comparable across runs.
#
# SPLIT_SEED and HOLDOUT_PERMILLE are inherited unchanged from
# prepare_2b_corpus.py. That matters: the split hashes document keys, so every
# document held out of the 2.05B corpus stays held out here. Changing the seed
# would leak the old runs' heldout set into this train set and silently
# invalidate every loss comparison against them.
set -uo pipefail

DEEPSEEK=/home/tyrel/projects/llm-deepseek-v4-experiment
PY=$DEEPSEEK/.venv/bin/python
RAW=$DEEPSEEK/data/raw/finemath-6b
FLAT=$DEEPSEEK/data/raw/finemath-6b-flat
OUT=$DEEPSEEK/data/finemath-6b

log() { echo "[$(date '+%F %T')] $*"; }

# --- wait for the downloader -----------------------------------------------
DL_PID="${1:-}"
if [ -n "$DL_PID" ]; then
  log "waiting for download (pid $DL_PID)"
  while kill -0 "$DL_PID" 2>/dev/null; do sleep 30; done
fi

N4=$(find "$RAW/finemath-4plus"     -name '*.parquet' 2>/dev/null | wc -l)
NI=$(find "$RAW/infiwebmath-4plus"  -name '*.parquet' 2>/dev/null | wc -l)
log "shards: finemath-4plus=$N4/64 infiwebmath-4plus=$NI/32"
if [ "$N4" -lt 64 ] || [ "$NI" -lt 32 ]; then
  log "ERROR: incomplete download; refusing to pack a partial corpus"
  exit 1
fi

# --- flatten ---------------------------------------------------------------
# prepare_2b_corpus.py globs train-*.parquet in ONE directory, and both configs
# use the same train-NNNNN-of-NNNNN names, so they collide. Symlink with a
# per-config prefix into a flat dir. Symlinks, not copies: 20GB either way.
log "flattening into $FLAT"
rm -rf "$FLAT"; mkdir -p "$FLAT"
for cfg in finemath-4plus infiwebmath-4plus; do
  i=0
  for f in "$RAW/$cfg"/*.parquet; do
    ln -s "$f" "$FLAT/train-$(printf '%s-%05d' "$cfg" "$i").parquet"
    i=$((i+1))
  done
done
log "flat shards: $(find "$FLAT" -name 'train-*.parquet' | wc -l)"

# --- pack ------------------------------------------------------------------
# train-tokens is a ceiling, not a target: pack whatever the shards yield up to
# 8B. The 2.05B default would silently truncate this back to the old size.
log "packing -> $OUT"
cd /home/tyrel/projects/llm-modern-arch-experiment
PYTHONPATH=src $PY scripts/prepare_2b_corpus.py \
  --parquet-dir "$FLAT" \
  --output "$OUT" \
  --train-tokens 8000000000 \
  --heldout-tokens 10000000
rc=$?
log "pack rc=$rc"
[ $rc -eq 0 ] || exit $rc

$PY - <<PY
import os
for n in ("train","heldout"):
    p = os.path.join("$OUT", n + ".bin")
    if os.path.exists(p):
        b = os.path.getsize(p)
        print(f"{n}.bin: {b/1e9:.2f} GB -> {b//2:,} tokens")
PY
log "DONE"
