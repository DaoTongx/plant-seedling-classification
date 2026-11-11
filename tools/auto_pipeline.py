# -*- coding: utf-8 -*-
"""
Wait for the running training jobs to finish, then predict + write the submission.

Polling is based on the checkpoints on disk, so this script is safe to start at
any time and safe to re-run: it only does work that is still missing.

Usage:
    python tools/auto_pipeline.py                       # default: 5 folds, all found backbones
    python tools/auto_pipeline.py --folds 5 --tta 8 --out submission.csv --name 20231234
    python tools/auto_pipeline.py --no-wait             # run immediately with what exists
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import plant_classifier as pc                                    # noqa: E402


def discovered_models():
    """Backbones that have at least one checkpoint, most-complete first."""
    names = set()
    for path in glob.glob(os.path.join(pc.OUT, "checkpoints", "*.pt")):
        base = os.path.basename(path)[:-3]
        names.add(base.rsplit("_fold", 1)[0])
    return sorted(names, key=lambda m: (-len(pc.available_folds(m)), m))


def busy_python_processes(exclude_pid=None):
    """Count python processes other than this one.

    A fold takes tens of minutes, so file-based timestamps cannot tell apart
    "between folds" from "stalled".  Process liveness can: while any training
    process is alive we must not start predicting (it would steal CPU).
    """
    if os.name != "nt":
        return 0                                     # not needed on Linux/Kaggle
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-Process python -ErrorAction SilentlyContinue).Count"],
            capture_output=True, text=True, timeout=60)
        count = int((out.stdout or "0").strip() or 0)
    except Exception:
        return 0
    return max(0, count - 1)                         # minus ourselves


def wait_for_folds(folds, poll=120, max_minutes=600):
    """Block until the training jobs have stopped.

    Stops as soon as either
      * every backbone has `folds` folds (the ideal case), or
      * no other python process is alive any more (training crashed/ended), or
      * a hard time limit is reached.
    """
    t0 = time.time()
    while True:
        models = discovered_models()
        counts = {m: len(pc.available_folds(m)) for m in models}
        other = busy_python_processes()
        elapsed = (time.time() - t0) / 60.0

        print("[auto] %5.1f min | other python procs: %d | %s"
              % (elapsed, other,
                 ", ".join("%s %d/%d" % (m, counts[m], folds) for m in models)
                 or "no checkpoints yet"), flush=True)

        if models and all(counts[m] >= folds for m in models):
            print("[auto] all backbones trained")
            return models
        if other == 0 and models:
            print("[auto] no training process left; proceeding with %s" % counts)
            return models
        if elapsed > max_minutes:
            print("[auto] timeout after %.0f min; proceeding with %s"
                  % (max_minutes, counts))
            return models
        time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(PROJECT, "submission.csv"))
    ap.add_argument("--name", default="")
    ap.add_argument("--template", default=None)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the OOF decision-weight correction")
    args = ap.parse_args(argv)

    if args.no_wait:
        models = discovered_models()
    else:
        models = wait_for_folds(args.folds, poll=args.poll,
                                max_minutes=args.max_minutes)

    if not models:
        print("[auto] nothing to predict -- no checkpoints found")
        return 1

    print("\n[auto] predicting with backbones: %s" % ", ".join(models))
    probs, _per_model, test_names, oof_acc, weights = pc.predict_test(
        models, folds=args.folds, tta=args.tta, tune=not args.no_tune)

    print("\n[auto] ensemble OOF accuracy: %.4f" % oof_acc)
    pc.write_submission(probs, test_names, args.out, participant=args.name,
                        template=args.template, weights=weights)

    # also refresh the report
    try:
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "report.py")], check=False)
    except Exception as exc:
        print("[auto] report generation skipped: %s" % exc)

    print("\n[auto] done -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
