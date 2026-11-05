#!/usr/bin/env bash
# Overnight assembly sweep: dump the dd (double-double) preconditioned KMSV matrix M for the
# [2,12,5] triangle group at N = 500,550,...,3000, so the GPU analyzer can trace where the
# Hauptmodul / null-space overfits (the tail-freedom onset below the truncation floor rho^N).
# Sequential (each assembly is rayon-parallel across all cores). Resumable: skips valid outputs.
set -u
SWEEP=/home/john/sweep_2_12_5
LOG=$SWEEP/assemble.log
cd /home/john/RustMath || exit 1

echo "=== sweep assembly start $(date -Is) ===" | tee -a "$LOG"
# build once so the per-N cargo invocations just run
cargo build -p rustmath-curves --release --tests 2>>"$LOG" >>"$LOG"

for N in $(seq 500 50 3000); do
    OUT=$SWEEP/m_N${N}.bin
    DIM=$((N+1))
    WANT=$((5 + 32*DIM*DIM))          # header(5) + dim^2 * (2 re + 2 im limbs) * 8 bytes
    if [ -f "$OUT" ]; then
        GOT=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
        if [ "$GOT" = "$WANT" ]; then
            echo "[$(date +%H:%M:%S)] N=$N  skip (have $GOT bytes)" | tee -a "$LOG"; continue
        fi
        echo "[$(date +%H:%M:%S)] N=$N  re-doing (size $GOT != $WANT)" | tee -a "$LOG"
    fi
    T0=$(date +%s)
    M_N=$N M_K=4 M_PREC=140 M_LIMBS=2 M_OUT="$OUT.tmp" \
        cargo test -p rustmath-curves --release dump_2_12_5_matrix_ext -- --ignored --nocapture \
        >>"$LOG" 2>&1
    GOT=$(stat -c%s "$OUT.tmp" 2>/dev/null || echo 0)
    if [ "$GOT" = "$WANT" ]; then
        mv "$OUT.tmp" "$OUT"
        echo "[$(date +%H:%M:%S)] N=$N  done in $(( $(date +%s)-T0 ))s  ($GOT bytes)" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] N=$N  FAILED (got $GOT want $WANT) -- see log" | tee -a "$LOG"
        rm -f "$OUT.tmp"
    fi
done
echo "=== sweep assembly complete $(date -Is) ===" | tee -a "$LOG"
