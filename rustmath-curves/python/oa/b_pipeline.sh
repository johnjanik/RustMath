#!/bin/bash
# Post-dump pipeline for B's 12-charts: wait for each dump, dd-refine, glue, calibrate
# kappa12s, build samples, run the degree scan. Logs to /home/john/sweep_2_12_5/b_pipeline.log.
set -u
cd /home/john/RustMath/rustmath-curves/python/oa
export PYTHONPATH=$PWD
SW=/home/john/sweep_2_12_5
LOG=$SW/b_pipeline.log
echo "=== b_pipeline start $(date) ===" >> $LOG

wait_done () {
  while ! grep -q "streamed done" "$1" 2>/dev/null; do sleep 60; done
}

process () {  # key logfile binfile
  key=$1; logf=$2; binf=$3
  wait_done "$logf"
  RHO=$(grep -oE 'ρ_full=[0-9.e-]+' "$logf" | head -1 | cut -d= -f2)
  echo "[$(date +%H:%M)] $key done, rho=$RHO — refining" >> $LOG
  python3 dd_span_refine.py "$binf" "$RHO" "${binf%.bin}_ddspan.npz" 1 3 >> $LOG 2>&1
  python3 b_glue.py "$key" >> $LOG 2>&1
  echo "[$(date +%H:%M)] $key glued" >> $LOG
}

process b  $SW/assembleB_b_N6900.log  $SW/mB_b_N6900.bin &
P1=$!
process b2 $SW/assembleB_b2_N6900.log $SW/mB_b2_N6900.bin &
P2=$!
wait $P1 $P2

echo "[$(date +%H:%M)] both 12-charts glued — calibrating + samples" >> $LOG
python3 b_samples.py >> $LOG 2>&1
echo "[$(date +%H:%M)] samples built — degree scan" >> $LOG
python3 b_scan.py >> $LOG 2>&1
echo "=== b_pipeline complete $(date) ===" >> $LOG
