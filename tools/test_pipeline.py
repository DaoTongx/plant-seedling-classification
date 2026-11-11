# -*- coding: utf-8 -*-
"""
Unit tests for the parts of the pipeline that are easy to get wrong:
the stratified split, the test-time-augmentation views, and the submission
writer (ID ordering and format).

Run:  python tools/test_pipeline.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import plant_classifier as pc                                    # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_stratified_folds():
    print("\nstratified_folds")
    labels = np.repeat(np.arange(pc.NUM_CLASSES), 100)
    folds = pc.stratified_folds(labels, n_splits=5, seed=2026)

    check("returns the requested number of folds", len(folds) == 5)

    all_val = np.concatenate([v for _t, v in folds])
    check("validation sets are disjoint and cover every image exactly once",
          len(all_val) == len(labels) and len(np.unique(all_val)) == len(labels))

    sizes = [len(v) for _t, v in folds]
    check("fold sizes differ by at most 1", max(sizes) - min(sizes) <= 1,
          "sizes=%s" % sizes)

    # class balance: every fold must hold each class (roughly) n_splits-th of the data
    balanced = True
    for _t, v in folds:
        counts = np.bincount(labels[v], minlength=pc.NUM_CLASSES)
        if counts.min() < 100 // 5 - 1 or counts.max() > 100 // 5 + 1:
            balanced = False
    check("each fold contains ~20 images of every class", balanced)

    _tr, va = folds[0]
    check("train and validation never overlap",
          len(np.intersect1d(_tr, va)) == 0)


def test_tta_views():
    print("\nTTA views")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:8, :8] = 255                      # a corner marker makes each view distinct
    views = []
    for v in range(8):
        ds = pc.TestDataset([img], size=32, view=v)
        x, idx = ds[0]
        views.append((x.numpy(), int(idx)))
        check("view %d yields shape (3,32,32)" % v, tuple(x.shape) == (3, 32, 32))
    check("view 0 and view 1 (h-flip) differ", not np.allclose(views[0][0], views[1][0]))
    check("view 0 and view 4 (rotate90) differ", not np.allclose(views[0][0], views[4][0]))
    check("datasets return the item index (needed to re-order TTA results)",
          all(idx == 0 for _x, idx in views))


def test_submission_writer():
    print("\nwrite_submission")
    names = ["b.png", "a.png", "c.png"]
    probs = np.array([[0.1, 0.7, 0.05, 0.1, 0.05],      # -> Common wheat
                      [0.8, 0.05, 0.05, 0.05, 0.05],     # -> Black-grass
                      [0.05, 0.05, 0.05, 0.05, 0.8]], dtype=np.float64)  # -> Sugar beet

    # a workspace-local directory: the platform temp dir is not writable here
    tmp = os.path.join(PROJECT, "outputs", "_test_tmp")
    os.makedirs(tmp, exist_ok=True)
    for stale in os.listdir(tmp):
        try:
            os.remove(os.path.join(tmp, stale))
        except OSError:
            pass

    out = os.path.join(tmp, "sub.csv")
    pc.write_submission(probs, names, out)

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    check("header is exactly ID,Category", rows[0] == ["ID", "Category"], str(rows[0]))
    check("one row per test image", len(rows) == 4, str(len(rows)))
    check("argmax picks the right class names",
          [r[1] for r in rows[1:]] == ["Common wheat", "Black-grass", "Sugar beet"],
          str([r[1] for r in rows[1:]]))

    # with a template, the output must follow the *template* order, not our order
    tpl = os.path.join(tmp, "template.csv")
    with open(tpl, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "Category"])
        for n in ("c.png", "b.png", "a.png"):
            w.writerow([n, "Sugar beet"])

    out2 = os.path.join(tmp, "sub2.csv")
    pc.write_submission(probs, names, out2, template=tpl)
    with open(out2, newline="", encoding="utf-8") as fh:
        rows2 = list(csv.reader(fh))
    check("template order is respected",
          [r[0] for r in rows2[1:]] == ["c.png", "b.png", "a.png"],
          str([r[0] for r in rows2[1:]]))
    check("predictions follow the ids through the reordering",
          [r[1] for r in rows2[1:]] == ["Sugar beet", "Common wheat", "Black-grass"],
          str([r[1] for r in rows2[1:]]))


def test_model_zoo_and_presets():
    print("\nmodel zoo / presets")
    if pc.torch is None:
        print("  [SKIP] torch unavailable")
        return
    for name, preset in pc.MODEL_PRESETS.items():
        try:
            net = pc.build_model(name, pretrained=False)
            n_params = sum(p.numel() for p in net.parameters())
            out = net(pc.torch.zeros(1, 3, preset["size"], preset["size"]))
            ok = tuple(out.shape) == (1, pc.NUM_CLASSES)
            check("%-20s builds and outputs 5 logits (%d M params)"
                  % (name, n_params // 1_000_000), ok, str(tuple(out.shape)))
        except Exception as exc:
            check("%-20s builds" % name, False, "%s: %s" % (type(exc).__name__, exc))


def test_augmentation_shapes():
    print("\nTrainDataset augmentation")
    rng = np.random.RandomState(0)
    imgs = (rng.rand(16, 384, 384, 3) * 255).astype(np.uint8)
    labels = rng.randint(0, pc.NUM_CLASSES, 16)
    ds = pc.TrainDataset(imgs, labels, size=128, train=True)
    shapes, vals = set(), []
    for i in range(16):
        x, y = ds[i]
        shapes.add(tuple(x.shape))
        vals.append((float(x.mean()), float(x.std())))
        check("label %d is returned as an int in range" % i,
              0 <= y < pc.NUM_CLASSES) if i < 1 else None
    check("all augmented samples are (3,128,128)", shapes == {(3, 128, 128)}, str(shapes))
    means = [m for m, _s in vals]
    check("augmentation produces varied outputs (not a constant tensor)",
          len({round(m, 2) for m in means}) > 1)
    check("normalised values are roughly zero-centred",
          abs(np.mean(means)) < 1.0, "mean of means=%.3f" % np.mean(means))


def test_metrics():
    """accuracy / balanced accuracy / macro-F1 against hand-computed values."""
    print("\nmetrics")
    # 4 samples, 3 classes; one class predicted perfectly, one never predicted
    probs = np.array([[0.8, 0.1, 0.1],
                      [0.7, 0.2, 0.1],
                      [0.1, 0.8, 0.1],
                      [0.1, 0.1, 0.8]])
    labels = np.array([0, 0, 1, 0])          # class 2 never occurs as truth
    check("accuracy = 3/4", abs(pc.accuracy(probs, labels) - 0.75) < 1e-12,
          "%.4f" % pc.accuracy(probs, labels))
    # recalls: class0 2/3, class1 1/1, class2 0/0 -> treated as 0.0
    expected_bal = (2 / 3 + 1.0 + 0.0) / 3
    check("balanced accuracy averages per-class recall",
          abs(pc.balanced_accuracy(probs, labels, 3) - expected_bal) < 1e-12,
          "%.4f vs %.4f" % (pc.balanced_accuracy(probs, labels, 3), expected_bal))
    check("macro F1 is lower than accuracy when a class is missed",
          pc.macro_f1(probs, labels, 3) < pc.accuracy(probs, labels),
          "f1=%.4f acc=%.4f" % (pc.macro_f1(probs, labels, 3),
                                pc.accuracy(probs, labels)))

    cm = pc.confusion_matrix(probs, labels, 3)
    check("confusion matrix rows are the true class",
          cm.shape == (3, 3) and cm[0].sum() == 3 and cm[1].sum() == 1, str(cm.tolist()))

    dist = pc.class_distribution(probs, 3)
    check("class_distribution sums to 1", abs(dist.sum() - 1.0) < 1e-12, str(dist))


def test_decision_tuning():
    """The tuner must recover a deliberately skewed decision rule."""
    print("\ndecision tuning (prior correction)")
    rng = np.random.RandomState(7)
    n_per, n_cls = 120, 3

    labels = np.repeat(np.arange(n_cls), n_per)
    # Build a probability field where class 0 systematically steals samples:
    #   * true score t = 0.30 for every sample's own class;
    #   * a fixed boost d = 0.40 added to column 0;
    #   * but for samples whose own class IS 0, the true score gets t + d.
    # So the margin between class 0 and the true class is (d - t) = +0.10 for
    # foreign samples (class 0 wins) and (t + d) - d = t = 0.30 for class 0's
    # own samples (the true class wins).  A boost that is constant in absolute
    # terms can never do this, which is the point of constructing it this way.
    t, d = 0.30, 0.40
    probs = np.full((len(labels), n_cls), 0.02)
    probs[np.arange(len(labels)), labels] = t
    probs[:, 0] += d
    probs[np.arange(len(labels))[labels == 0], 0] = t + d
    probs += rng.rand(*probs.shape) * 0.01
    probs /= probs.sum(axis=1, keepdims=True)

    before_bal = pc.balanced_accuracy(probs, labels, n_cls)
    before_f1 = pc.macro_f1(probs, labels, n_cls)
    before_dist = pc.class_distribution(probs, n_cls)
    check("the injected bias really skews the predictions",
          before_dist[0] > 0.9, str(np.round(before_dist, 3)))
    check("the injected bias really hurts balanced accuracy",
          before_bal < 0.5, "before=%.4f" % before_bal)

    # ---- the prior correction alone should recover most of the loss ----
    base_w = pc.prior_weights(probs, n_cls=n_cls)
    base_bal = pc.balanced_accuracy(probs * base_w[None, :], labels, n_cls)
    check("plain prior correction already fixes the skew (%.3f -> %.3f)"
          % (before_bal, base_bal), base_bal > before_bal + 0.3,
          "before=%.4f base=%.4f" % (before_bal, base_bal))

    # ---- coordinate ascent must not be worse than the prior correction ----
    tuned = pc.tune_decision_weights(probs, labels, n_cls)
    w = tuned["weights"] * tuned["base_weights"]
    after_bal = pc.balanced_accuracy(probs * w[None, :], labels, n_cls)
    after_f1 = pc.macro_f1(probs * w[None, :], labels, n_cls)
    after_dist = pc.class_distribution(probs * w[None, :], n_cls)

    check("tuning is at least as good as the prior correction alone",
          after_bal >= base_bal - 1e-9,
          "base=%.4f tuned=%.4f" % (base_bal, after_bal))
    check("tuning improves balanced accuracy (%.3f -> %.3f)" % (before_bal, after_bal),
          after_bal > before_bal, "before=%.4f after=%.4f" % (before_bal, after_bal))
    check("tuning does not hurt macro F1 (%.3f -> %.3f)" % (before_f1, after_f1),
          after_f1 >= before_f1 - 1e-9, "before=%.4f after=%.4f" % (before_f1, after_f1))
    check("the inflated class gets a smaller weight than the others",
          w[0] < float(np.mean(np.delete(w, 0))), str(np.round(w, 3)))
    check("weights are normalised to mean 1", abs(w.mean() - 1.0) < 1e-9,
          "%.6f" % w.mean())
    check("apply_weights is equivalent to the weighted argmax",
          np.array_equal(pc.apply_weights(probs, w), (probs * w[None, :]).argmax(1)))
    check("predicted distribution becomes balanced",
          abs(after_dist[0] - 1.0 / n_cls) <= abs(before_dist[0] - 1.0 / n_cls) + 1e-9,
          "before=%.3f after=%.3f ideal=%.3f" % (before_dist[0], after_dist[0], 1 / n_cls))

    # prior_weights on already-balanced probabilities must be a no-op
    uniform = pc.prior_weights(np.full((50, n_cls), 1.0 / n_cls))
    check("prior_weights is ~1 when the model already predicts uniformly",
          np.allclose(uniform, 1.0, atol=1e-6), str(np.round(uniform, 6)))

    # leave-one-fold-out validation must not crash and must report both metrics
    fold_id = np.tile(np.arange(5), len(labels) // 5 + 1)[:len(labels)]
    val = pc.validate_decision_weights(probs, labels, fold_id, n_cls, verbose=False)
    check("leave-one-fold-out validation covers every image",
          val["n"] == len(labels), "n=%d of %d" % (val["n"], len(labels)))
    check("validation reports macro F1 as well as balanced accuracy",
          "f1_before" in val and "f1_after" in val)
    check("validation reports a non-negative balanced-accuracy gain on this signal",
          val["delta_bal"] > 0, "delta_bal=%+.4f" % val["delta_bal"])


def test_pseudo_label_selection():
    """The confidence and agreement gates must both actually apply."""
    print("\npseudo-label selection")
    n_cls = pc.NUM_CLASSES
    n = 20
    probs = np.full((n, n_cls), 0.01)
    labels = np.arange(n) % n_cls
    probs[np.arange(n), labels] = 0.96          # all confident
    probs /= probs.sum(axis=1, keepdims=True)

    idx, pl, diag = pc.select_pseudo_labels(probs, min_conf=0.90)
    check("high-confidence samples are all selected", len(idx) == n,
          "%d of %d" % (len(idx), n))
    check("agreement gate reported as not applied when no backbones given",
          diag["agreement_applied"] is False)

    # one backbone disagrees on half of the samples -> those must be dropped
    other = probs.copy()
    disagree_rows = np.arange(0, n, 2)
    other[disagree_rows] = 0.01
    other[disagree_rows, (labels[disagree_rows] + 1) % n_cls] = 0.96
    other /= other.sum(axis=1, keepdims=True)

    idx2, _pl2, diag2 = pc.select_pseudo_labels(
        probs, min_conf=0.90, require_agreement=True, per_model_probs=[probs, other])
    check("agreement gate is applied when backbones are supplied",
          diag2["agreement_applied"] is True)
    check("disagreeing samples are excluded",
          len(idx2) == n - len(disagree_rows), "%d of %d" % (len(idx2), n))

    # a high threshold must exclude the confident-but-not-enough samples
    idx3, _pl3, _d3 = pc.select_pseudo_labels(probs, min_conf=0.99)
    check("raising the threshold above the confidence selects nothing",
          len(idx3) == 0, "%d" % len(idx3))

    # labels returned must match the argmax
    idx4, pl4, _d4 = pc.select_pseudo_labels(probs, min_conf=0.90)
    check("returned labels equal the argmax", np.array_equal(pl4, labels[idx4]))


def test_oof_aggregation():
    """aggregate_oof must reconstruct one honest prediction per training image."""
    print("\nOOF aggregation")
    import shutil
    n_train, n_cls = 40, pc.NUM_CLASSES
    # absolute path: pc.OUT is used by plant_classifier from whatever cwd the
    # caller happens to have, so a relative path would silently find nothing
    root = os.path.join(PROJECT, "outputs", "_ooftmp")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "oof"))
    saved = pc.OUT
    pc.OUT = root
    try:
        labels = np.arange(n_train) % n_cls
        # two folds, two "backbones"
        for model in ("m_a", "m_b"):
            for fold in range(2):
                idx = np.where(np.arange(n_train) % 2 == fold)[0]
                p = np.full((len(idx), n_cls), 0.02)
                p[np.arange(len(idx)), labels[idx]] = 0.9
                p /= p.sum(axis=1, keepdims=True)
                np.savez(os.path.join(root, "oof", "%s_fold%d.npz" % (model, fold)),
                         probs=p, idx=idx, labels=labels[idx])

        check("available_folds locates the saved folds",
              pc.available_folds("m_a", 2) == [0, 1],
              str(pc.available_folds("m_a", 2)))

        oof = pc.aggregate_oof(["m_a", "m_b"], folds=2)
        check("both backbones are picked up", len(oof["names"]) == 2,
              str(oof["names"]))
        check("coverage is complete for both backbones",
              bool(oof["coverage"].all()),
              "%d of %d" % (oof["coverage"].sum(), oof["coverage"].size))
        probs, lab, fold_id = pc.ensemble_oof_probs(oof)
        check("one OOF prediction per training image", probs.shape == (n_train, n_cls),
              str(probs.shape))
        check("reconstructed labels are correct", np.array_equal(lab, labels))
        check("OOF accuracy is high for this synthetic setup",
              pc.accuracy(probs, lab) == 1.0, "%.4f" % pc.accuracy(probs, lab))
        check("fold ids are recovered", set(fold_id.tolist()) == {0, 1},
              str(sorted(set(fold_id.tolist()))))

        # a backbone list with a name that has no files must be ignored cleanly
        oof2 = pc.aggregate_oof(["m_a", "does_not_exist"], folds=2)
        check("unknown backbones are skipped without breaking the aggregate",
              oof2["probs"].shape[0] == 1, str(oof2["probs"].shape))
    finally:
        pc.OUT = saved
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("=" * 68)
    print("pipeline unit tests")
    print("=" * 68)
    test_stratified_folds()
    test_tta_views()
    test_submission_writer()
    test_model_zoo_and_presets()
    test_augmentation_shapes()
    test_metrics()
    test_decision_tuning()
    test_pseudo_label_selection()
    test_oof_aggregation()

    print("\n" + "=" * 68)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
