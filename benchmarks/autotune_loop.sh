#!/usr/bin/env bash
# Restart-loop driver for grout_autotune.
#
# cutile-rs 0.3.0 keeps a process-global kernel cache with no eviction, so
# a large tuning space accumulates device-resident modules until even the
# in-process engine reload OOMs (panic, exit 101). Trials are journaled and
# the tuner resumes, so the robust driver is simply: rerun until clean exit.
# Each restart clears the global cache and loses at most the in-flight trial.
set -u
MAX_RESTARTS="${MAX_RESTARTS:-25}"
for i in $(seq 1 "$MAX_RESTARTS"); do
  target/release/grout_autotune "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "autotune_loop: clean exit after $i run(s)"
    exit 0
  fi
  echo "autotune_loop: run $i exited rc=$rc — restarting (resumes from trial logs)"
done
echo "autotune_loop: exceeded MAX_RESTARTS=$MAX_RESTARTS" >&2
exit 1
