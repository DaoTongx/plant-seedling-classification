# -*- coding: utf-8 -*-
"""
One-shot verification of the whole project.

Runs every check in the order that fails fastest, and prints a single summary at
the end.  Use this before starting a long training run.

    python tools/verify_all.py            # full check (trains 1 epoch, a few min)
    python tools/verify_all.py --fast     # skip the training step

Exit code 0 means every check passed.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def run(title, argv, label):
    print("\n" + "=" * 72)
    print(">>> %s" % title)
    print("=" * 72, flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + argv, cwd=PROJECT,
                          capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    # echo the tail so the user sees the actual failure, not just a verdict
    tail = "\n".join(out.strip().splitlines()[-25:])
    print(tail)
    ok = proc.returncode == 0
    print("\n[%s] %s  (%.1f s, exit %d)"
          % ("OK  " if ok else "FAIL", title, time.time() - t0, proc.returncode))
    return label, ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the end-to-end training check")
    args = ap.parse_args()

    results = []

    # 1. notebook / source consistency: pure file inspection, no torch needed
    results.append(run("notebook matches src/plant_classifier.py",
                       [os.path.join("tools", "validate_notebook.py")],
                       "notebook consistency"))

    # 2. unit tests: metrics, decision tuning, pseudo-label gate, OOF assembly
    results.append(run("unit tests", [os.path.join("tools", "test_pipeline.py")],
                       "unit tests"))

    # 3. end-to-end: cache -> 1 epoch -> TTA predict -> submission, incl. tuning
    if args.fast:
        print("\n[SKIP] end-to-end training check (--fast)")
    else:
        results.append(run("end-to-end (1 epoch, 1 fold)",
                           [os.path.join("tools", "verify_end2end.py")],
                           "end-to-end"))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for label, ok, _out in results:
        print("  %-24s %s" % (label, "PASS" if ok else "FAIL"))
    failed = [label for label, ok, _o in results if not ok]
    if failed:
        print("\n%d check(s) failed: %s" % (len(failed), ", ".join(failed)))
        print("Fix these before starting a long training run.")
        return 1
    print("\nall checks passed -- safe to start the full training run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
