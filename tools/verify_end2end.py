# -*- coding: utf-8 -*-
"""
End-to-end check of the exact code path the Kaggle notebook uses, with the
smallest possible amount of compute: 1 epoch, 1 fold, 1 backbone, 2 TTA views.

This proves that the notebook chain
    cache -> train_one -> predict_test -> write_submission
runs without error and produces a valid, correctly formatted submission.csv.

Usage:  python tools/verify_end2end.py
"""
import csv
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import plant_classifier as pc                                    # noqa: E402


class TemporaryOutput:
    """Point the pipeline at a scratch directory so real results are untouched."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.saved = pc.OUT
        if os.path.isdir(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)
        pc.OUT = self.path
        return self.path

    def __exit__(self, *exc):
        pc.OUT = self.saved
        return False


def main():
    tmp = os.path.join(PROJECT, "outputs", "_e2e")
    failures = []

    def check(name, ok, detail=""):
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                               ("  -- " + detail) if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 70)
    print("end-to-end verification (1 epoch, 1 fold, 2 TTA views)")
    print("=" * 70)

    with TemporaryOutput(tmp):
        # ---- 1. cache (reuse the real one by copying the npz if present) ----
        real_cache = os.path.join(PROJECT, "outputs", "cache_%d.npz" % pc.CACHE_SIZE)
        if os.path.exists(real_cache):
            shutil.copy2(real_cache, os.path.join(tmp, "cache_%d.npz" % pc.CACHE_SIZE))
            print("[e2e] reused the existing image cache")
        t0 = time.time()
        pc.build_cache(size=pc.CACHE_SIZE)
        print("[e2e] cache ready (%.1f s)" % (time.time() - t0))
        train_x, train_y, test_x, test_names = pc.load_cache()
        check("cache loads", len(train_x) == 500 and len(test_x) == 378,
              "train=%d test=%d" % (len(train_x), len(test_x)))

        # ---- 2. train a single epoch ----
        t0 = time.time()
        acc = pc.train_one("efficientnet_b0", fold=0, folds=5, epochs=1,
                           seed=1234, verbose=True)
        print("[e2e] train_one finished in %.1f s" % (time.time() - t0))
        check("a checkpoint was written",
              os.path.exists(os.path.join(tmp, "checkpoints", "efficientnet_b0_fold0.pt")))
        check("validation accuracy is a sane number", 0.0 <= acc <= 1.0, "acc=%.4f" % acc)

        # ---- 3. predict with TTA (decision tuning on, as in the notebook) ----
        t0 = time.time()
        probs, per_model, names, oof, weights = pc.predict_test(
            ["efficientnet_b0"], folds=1, tta=2)
        print("[e2e] predict_test finished in %.1f s" % (time.time() - t0))
        check("probabilities are (378, 5)", probs.shape == (378, pc.NUM_CLASSES),
              str(probs.shape))
        check("probabilities sum to 1 per row",
              abs(probs.sum(axis=1) - 1.0).max() < 1e-6,
              "max deviation %.2e" % abs(probs.sum(axis=1) - 1.0).max())
        check("test names preserved", len(names) == 378)
        check("ensemble npz written",
              os.path.exists(os.path.join(tmp, "test_ensemble.npz")))
        check("decision weights returned with the right shape",
              np.asarray(weights).shape == (pc.NUM_CLASSES,), str(np.shape(weights)))

        # ---- 4. submission (tuned, exactly as the notebook writes it) ----
        out = os.path.join(tmp, "submission.csv")
        pc.write_submission(probs, names, out, weights=weights)
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        check("submission header is ID,Category", rows[0] == ["ID", "Category"])
        check("submission has 378 data rows", len(rows) == 379, "rows=%d" % (len(rows)))
        check("every category is one of the 5 class names",
              all(r[1] in pc.CLASSES for r in rows[1:]))
        check("ids match the test directory",
              {r[0] for r in rows[1:]} == set(test_names))
        check("tuned submission still predicts more than one class",
              len({r[1] for r in rows[1:]}) > 1)

        # ---- 5. template-ordered submission (what Kaggle actually uploads) ----
        # the template lives next to the dataset; look in the usual places so
        # this check does not depend on where the project folder sits
        template = None
        for cand in (os.path.join(PROJECT, "data", "submission-for-task2.csv"),
                     os.path.join(os.path.dirname(PROJECT), "data",
                                  "submission-for-task2.csv")):
            if os.path.exists(cand):
                template = cand
                break
        if template:
            out2 = os.path.join(tmp, "submission_templated.csv")
            pc.write_submission(probs, names, out2, template=template)
            with open(template, newline="", encoding="utf-8") as fh:
                tmpl = list(csv.reader(fh))
            with open(out2, newline="", encoding="utf-8") as fh:
                sub = list(csv.reader(fh))
            check("templated submission has the template's row count",
                  len(sub) == len(tmpl), "%d vs %d" % (len(sub), len(tmpl)))
            check("templated submission keeps the template's ID order",
                  [r[0] for r in sub[1:]] == [r[0] for r in tmpl[1:]])
        else:
            print("  [SKIP] submission template not found at %s" % template)

    print("\n" + "=" * 70)
    if failures:
        print("FAILED (%d): %s" % (len(failures), ", ".join(failures)))
        return 1
    print("end-to-end verification passed -- the notebook chain works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
