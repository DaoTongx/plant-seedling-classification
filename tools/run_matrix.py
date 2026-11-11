# -*- coding: utf-8 -*-
"""
Run a (backbone x fold) training matrix, skipping work already done.

Checkpoints are written per (model, fold) by plant_classifier.train_one, so a
matrix can be interrupted at any point and resumed by re-running this script.

Usage
-----
    python tools/run_matrix.py --plan default
    python tools/run_matrix.py --plan fast --epochs 25
    python tools/run_matrix.py --models efficientnet_b0 resnet50 --folds 5
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import plant_classifier as pc                                    # noqa: E402

#: model -> (folds to train, epochs)  -- chosen from measured CPU throughput
PLANS = {
    # full 5-fold for the two strongest/fastest backbones, fewer folds for the
    # expensive ones (3 folds still give a useful OOF estimate and test ensemble)
    "default": {
        "efficientnet_b0": (5, 30),
        "resnet50":        (5, 30),
        "convnext_tiny":   (3, 25),
        "densenet121":     (3, 25),
    },
    "fast": {
        "efficientnet_b0": (5, 22),
        "resnet18":        (5, 22),
    },
    "single": {
        "efficientnet_b0": (5, 30),
    },
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, choices=sorted(PLANS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=None, help="fold count for every model")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data", default=None)
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args(argv)

    if args.models:
        plan = {m: (args.folds or 5, args.epochs or 30) for m in args.models}
    elif args.plan:
        plan = dict(PLANS[args.plan])
    else:
        plan = dict(PLANS["default"])
    if args.epochs:
        plan = {m: (n, args.epochs) for m, (n, _e) in plan.items()}

    # ---- status table ----
    print("%-20s %-18s %s" % ("model", "progress", "val accuracy per fold"))
    todo = []
    for model, (n_folds, epochs) in plan.items():
        marks = []
        for fold in range(n_folds):
            path = os.path.join(pc.OUT, "checkpoints", "%s_fold%d.pt" % (model, fold))
            if os.path.exists(path):
                try:
                    import torch
                    ck = torch.load(path, map_location="cpu", weights_only=False)
                    marks.append("%.3f" % ck.get("val_acc", float("nan")))
                except Exception:
                    marks.append("?")
            else:
                marks.append("--")
                todo.append((model, fold, epochs))
        print("%-20s %-18s %s" % (model, "%d folds" % n_folds, "  ".join(marks)))

    if args.list:
        return 0

    print("\n%d (model, fold) runs to do" % len(todo))
    done, t0 = 0, time.time()
    for model, fold, epochs in todo:
        print("\n" + "=" * 72)
        print(">>> %s fold %d  (%d epochs)" % (model, fold, epochs))
        print("=" * 72, flush=True)
        try:
            pc.train_one(model, fold, folds=plan[model][0], epochs=epochs,
                         seed=args.seed, data_root=args.data, verbose=True)
        except Exception as exc:                              # keep the matrix going
            print("!!! FAILED %s fold %d: %s: %s" % (model, fold, type(exc).__name__, exc))
            continue
        done += 1
        print("progress %d/%d, elapsed %.1f min"
              % (done, len(todo), (time.time() - t0) / 60.0), flush=True)

    print("\nmatrix complete: %d runs in %.1f min" % (done, (time.time() - t0) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
