#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plant seedling classification -- transfer-learning pipeline.

Task
----
5-class classification of young plants photographed on a gravel tray:

    0 Black-grass          1 Common wheat        2 Loose Silky-bent
    3 Scentless Mayweed    4 Sugar beet

train : 500 labelled images (100 per class)
test  : 378 unlabelled images
metric: accuracy (leaderboard = ranking, higher is better)

Design
------
* Transfer learning from ImageNet weights: only 500 training images, so a
  randomly initialised CNN cannot compete with a pretrained backbone.
* 5-fold stratified cross-validation.  Every image is validated exactly once,
  which yields an honest out-of-fold (OOF) accuracy, and the 5 fold-models are
  averaged at test time (a free 5-model ensemble).
* Images are pre-resized once into a uint8 cache, so augmentation and training
  never touch the disk again.
* Test-time augmentation (8 geometric views) on the test predictions.
* Pure numpy / torch / torchvision -- no sklearn, no pandas, no scipy.

Usage
-----
    python src/plant_classifier.py cache
    python src/plant_classifier.py train  --model efficientnet_b0 --folds 5
    python src/plant_classifier.py predict
    python src/plant_classifier.py submit --out submission.csv --name <student id>
    python src/plant_classifier.py all --model efficientnet_b0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zipfile

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Make torch importable.
#
# On Kaggle / Colab torch is already installed, so none of this matters there.
# On a machine without torch, a previously extracted wheel tree may sit next to
# the project folder or inside it; both are searched.
# --------------------------------------------------------------------------- #
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG_ROOT)
for _name in ("_libs2", "libs"):
    for _base in (_PKG_ROOT, _PARENT):
        _extra = os.path.join(_base, _name)
        if os.path.isdir(_extra) and _extra not in sys.path:
            sys.path.insert(0, _extra)

# Keep pretrained-weight downloads inside the project when a local cache exists.
for _base in (_PKG_ROOT, _PARENT):
    _LOCAL_TORCH_HOME = os.path.join(_base, "_torchhome")
    if os.path.isdir(_LOCAL_TORCH_HOME) and not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = _LOCAL_TORCH_HOME
        break

# --------------------------------------------------------------------------- #
# Optional torch import: allow the module to be imported for inspection even
# where torch is not installed.
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms
    _TORCH_ERR = None
except Exception as exc:                                    # pragma: no cover
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None
    nn = None
    F = None
    torchvision = None
    _TORCH_ERR = exc


# =========================================================================== #
#  Paths and constants
# =========================================================================== #
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE

# The dataset is looked up in a few plausible places so that the very same
# script runs unchanged on Kaggle and on a local machine.
DATA_CANDIDATES = [
    os.environ.get("PLANT_DATA", ""),
    os.path.join(PROJECT, "data", "dataset-for-task2", "dataset-for-task2"),
    os.path.join(PROJECT, "data", "dataset-for-task2"),
    os.path.join(PROJECT, "dataset-for-task2", "dataset-for-task2"),
    # the data folder may sit one level up from the project folder
    os.path.join(os.path.dirname(PROJECT), "data", "dataset-for-task2",
                 "dataset-for-task2"),
    "/kaggle/input/dataset-for-task2/dataset-for-task2",
    "/kaggle/input/2026task1/dataset-for-task2/dataset-for-task2",
]
OUT = os.path.join(PROJECT, "outputs")

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent",
           "Scentless Mayweed", "Sugar beet"]
NUM_CLASSES = len(CLASSES)

# ImageNet normalisation (every backbone below is pretrained with these stats)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE_SIZE = 384        # images are stored at this resolution in the cache
SEED = 2026


# =========================================================================== #
#  Small helpers
# =========================================================================== #
def find_data_root(explicit=None):
    """Locate the directory containing train/ and test/."""
    cands = ([explicit] if explicit else []) + DATA_CANDIDATES
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test")):
            return c
    raise FileNotFoundError(
        "could not find the dataset (need <root>/train and <root>/test). "
        "Searched:\n  " + "\n  ".join(repr(c) for c in cands if c)
    )


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def expand2square(pil_img, background=(255, 255, 255)):
    """Pad a non-square image to a square with a white border."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    size = max(width, height)
    result = Image.new(pil_img.mode, (size, size), background)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def write_csv(path, header, rows):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# =========================================================================== #
#  Stage 1 -- build the image cache
# =========================================================================== #
def build_cache(data_root=None, size=CACHE_SIZE, force=False):
    """Resize every image once into a compressed .npz cache.

    Storing uint8 arrays instead of JPEG/PNG files removes the image decoder
    from the training loop, which on a CPU-only machine is a large win.
    """
    root = find_data_root(data_root)
    ensure_dir(OUT)
    cache_path = os.path.join(OUT, "cache_%d.npz" % size)
    if os.path.exists(cache_path) and not force:
        print("[cache] already present: %s" % cache_path)
        return cache_path

    print("[cache] building from %s" % root)
    t0 = time.time()

    # ---- train ----
    train_imgs, train_labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(root, "train", cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError("missing class directory: %s" % cls_dir)
        names = sorted(n for n in os.listdir(cls_dir)
                       if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        for name in names:
            img = Image.open(os.path.join(cls_dir, name)).convert("RGB")
            img = expand2square(img).resize((size, size), Image.BILINEAR)
            train_imgs.append(np.asarray(img, dtype=np.uint8))
            train_labels.append(label)
        print("   %-18s %3d images" % (cls, len(names)))

    # ---- test ----
    test_dir = os.path.join(root, "test")
    test_names = sorted(n for n in os.listdir(test_dir)
                        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    test_imgs = []
    for name in test_names:
        img = Image.open(os.path.join(test_dir, name)).convert("RGB")
        img = expand2square(img).resize((size, size), Image.BILINEAR)
        test_imgs.append(np.asarray(img, dtype=np.uint8))

    train_x = np.stack(train_imgs)
    test_x = np.stack(test_imgs)
    train_y = np.asarray(train_labels, dtype=np.int64)

    np.savez_compressed(cache_path,
                        train_x=train_x, train_y=train_y,
                        test_x=test_x, test_names=np.asarray(test_names),
                        classes=np.asarray(CLASSES))
    print("[cache] %d train / %d test -> %s (%.1fs, %.1f MB)"
          % (len(train_y), len(test_names), cache_path, time.time() - t0,
             os.path.getsize(cache_path) / 1e6))
    return cache_path


def load_cache(size=CACHE_SIZE, data_root=None):
    path = os.path.join(OUT, "cache_%d.npz" % size)
    if not os.path.exists(path):
        build_cache(data_root, size=size)
    data = np.load(path, allow_pickle=False)
    return (data["train_x"], data["train_y"], data["test_x"],
            [str(s) for s in data["test_names"]])


# =========================================================================== #
#  Stage 2 -- model zoo
# =========================================================================== #
def build_model(name, num_classes=NUM_CLASSES, pretrained=True):
    """Create a backbone with an ImageNet-initialised body and a fresh head."""
    from torchvision import models

    name = name.lower()

    def weights(enum_name):
        if not pretrained:
            return None
        return getattr(models, enum_name).DEFAULT

    if name == "resnet18":
        net = models.resnet18(weights=weights("ResNet18_Weights"))
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet34":
        net = models.resnet34(weights=weights("ResNet34_Weights"))
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet50":
        net = models.resnet50(weights=weights("ResNet50_Weights"))
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=weights("EfficientNet_B0_Weights"))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b1":
        net = models.efficientnet_b1(weights=weights("EfficientNet_B1_Weights"))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights=weights("ConvNeXt_Tiny_Weights"))
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=weights("MobileNet_V3_Large_Weights"))
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    elif name == "densenet121":
        net = models.densenet121(weights=weights("DenseNet121_Weights"))
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    elif name == "regnet_y_400mf":
        net = models.regnet_y_400mf(weights=weights("RegNet_Y_400MF_Weights"))
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "swin_t":
        net = models.swin_t(weights=weights("Swin_T_Weights"))
        net.head = nn.Linear(net.head.in_features, num_classes)
    elif name == "vit_b_16":
        net = models.vit_b_16(weights=weights("ViT_B_16_Weights"))
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
    else:
        raise ValueError("unknown model %r" % name)
    return net


#: input resolution and batch size that fit comfortably on a CPU
MODEL_PRESETS = {
    "resnet18":           dict(size=256, batch=32, lr_head=1e-3, lr_body=1e-4),
    "resnet34":           dict(size=256, batch=32, lr_head=1e-3, lr_body=1e-4),
    "resnet50":           dict(size=256, batch=24, lr_head=1e-3, lr_body=1e-4),
    "efficientnet_b0":    dict(size=288, batch=24, lr_head=1e-3, lr_body=1e-4),
    "efficientnet_b1":    dict(size=288, batch=20, lr_head=1e-3, lr_body=1e-4),
    "convnext_tiny":      dict(size=256, batch=20, lr_head=1e-3, lr_body=5e-5),
    "mobilenet_v3_large": dict(size=256, batch=32, lr_head=1e-3, lr_body=1e-4),
    "densenet121":        dict(size=256, batch=24, lr_head=1e-3, lr_body=1e-4),
    "regnet_y_400mf":     dict(size=256, batch=32, lr_head=1e-3, lr_body=1e-4),
    "swin_t":             dict(size=224, batch=16, lr_head=1e-3, lr_body=5e-5),
    "vit_b_16":           dict(size=224, batch=16, lr_head=1e-3, lr_body=5e-5),
}


def get_preset(name):
    return dict(MODEL_PRESETS.get(name.lower(),
                                  dict(size=256, batch=24, lr_head=1e-3, lr_body=1e-4)))


# =========================================================================== #
#  Stage 3 -- datasets and augmentation
# =========================================================================== #
class TrainDataset(Dataset):
    """Random-resized-crop + flips + rotation + colour jitter + erase.

    All of it happens on uint8 HWC arrays with numpy / PIL, which is fast
    enough to keep the CPU occupied with the forward/backward pass instead.
    """

    def __init__(self, images, labels, size, train=True, erase_p=0.25):
        self.images = images
        self.labels = labels
        self.size = size
        self.train = train
        self.erase_p = erase_p if train else 0.0
        self.mean = (MEAN * 255.0).astype(np.float32)
        self.std = (STD * 255.0).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def _augment(self, img):
        h, w = img.shape[:2]
        # random resized crop, scale tuned for plants that occupy part of the frame
        for _ in range(10):
            scale = random.uniform(0.45, 1.0)
            ratio = math.exp(random.uniform(math.log(0.75), math.log(1.333)))
            area = h * w * scale
            crop_h = int(round(math.sqrt(area / ratio)))
            crop_w = int(round(math.sqrt(area * ratio)))
            if crop_h <= h and crop_w <= w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                img = img[top:top + crop_h, left:left + crop_w]
                break
        pil = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
        if random.random() < 0.5:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
        angle = random.uniform(-180.0, 180.0)
        if abs(angle) > 1.0:
            pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        arr = np.asarray(pil, dtype=np.float32)
        # brightness / contrast / saturation jitter
        if random.random() < 0.8:
            arr *= random.uniform(0.8, 1.25)                       # brightness
            m = arr.mean()
            arr = (arr - m) * random.uniform(0.8, 1.25) + m        # contrast
        if random.random() < 0.5:
            arr += np.random.uniform(-12.0, 12.0, size=3).astype(np.float32)
        return arr

    def __getitem__(self, index):
        img = self.images[index]
        if self.train:
            arr = self._augment(img)
        else:
            arr = np.asarray(
                Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR),
                dtype=np.float32)

        arr = (arr - self.mean) / self.std
        if self.erase_p > 0 and random.random() < self.erase_p:
            eh = random.randint(self.size // 8, self.size // 3)
            ew = random.randint(self.size // 8, self.size // 3)
            top = random.randint(0, self.size - eh)
            left = random.randint(0, self.size - ew)
            arr[top:top + eh, left:left + ew] = 0.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor, int(self.labels[index])


class TestDataset(Dataset):
    """Deterministic views of the test images (used for TTA)."""

    def __init__(self, images, size, view=0):
        self.images = images
        self.size = size
        self.view = view
        self.mean = (MEAN * 255.0).astype(np.float32)
        self.std = (STD * 255.0).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pil = Image.fromarray(self.images[index])
        if self.view == 1:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.view == 2:
            pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.view == 3:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        elif self.view == 4:
            pil = pil.rotate(90, resample=Image.BILINEAR)
        elif self.view == 5:
            pil = pil.rotate(180, resample=Image.BILINEAR)
        elif self.view == 6:
            pil = pil.rotate(270, resample=Image.BILINEAR)
        elif self.view == 7:
            pil = pil.rotate(90, resample=Image.BILINEAR).transpose(Image.FLIP_LEFT_RIGHT)
        pil = pil.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32)
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))), index


# =========================================================================== #
#  Stage 4 -- stratified k-fold split (no sklearn)
# =========================================================================== #
def stratified_folds(labels, n_splits=5, seed=SEED):
    """Return a list of (train_idx, val_idx) with class proportions preserved."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    fold_of = np.empty(len(labels), dtype=np.int64)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        # round-robin so folds differ in size by at most one image
        for pos, i in enumerate(idx):
            fold_of[i] = pos % n_splits
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0])
            for f in range(n_splits)]


# =========================================================================== #
#  Stage 5 -- training one fold
# =========================================================================== #
def train_one(name, fold, folds=5, epochs=30, size=None, batch=None,
              lr_head=None, lr_body=None, weight_decay=1e-4, label_smooth=0.1,
              warmup=3, workers=0, seed=SEED, data_root=None, verbose=True,
              mixup=0.0, extra_x=None, extra_y=None, save_oof=True,
              ckpt_suffix=""):
    """Train a single (backbone, fold) pair; return validation accuracy.

    `extra_x` / `extra_y` append pseudo-labelled images to the *training* split
    only.  The validation indices still come from the real training set, and
    pseudo-labelled samples are excluded from every validation fold, so the
    reported agreement stays honest.

    `save_oof=False` skips writing the OOF file.  A pseudo-label model must use
    it: its validation images were part of the pool the pseudo-labels were
    chosen from, so saving those probabilities as "out-of-fold" would silently
    contaminate the decision-weight tuning that reads those files.
    """
    if torch is None:                                        # pragma: no cover
        raise RuntimeError("torch is required for training: %s" % _TORCH_ERR)

    preset = get_preset(name)
    size = size or preset["size"]
    batch = batch or preset["batch"]
    lr_head = lr_head or preset["lr_head"]
    lr_body = lr_body or preset["lr_body"]

    set_seed(seed + fold)
    train_x, train_y, _test_x, _names = load_cache(data_root=data_root)
    splits = stratified_folds(train_y, folds, seed)
    tr_idx, va_idx = splits[fold]

    tr_x, tr_y = train_x[tr_idx], train_y[tr_idx]
    if extra_x is not None and len(extra_x):
        tr_x = np.concatenate([tr_x, np.asarray(extra_x)], axis=0)
        tr_y = np.concatenate([tr_y, np.asarray(extra_y)], axis=0)
        if verbose:
            print("  [%s f%d] training on %d real + %d pseudo-labelled images"
                  % (name, fold, len(tr_idx), len(extra_x)), flush=True)

    train_ds = TrainDataset(tr_x, tr_y, size, train=True)
    val_ds = TrainDataset(train_x[va_idx], train_y[va_idx], size, train=False)

    # num_workers=0 keeps everything in-process: the dataset is already in RAM,
    # so worker processes would only add pickling overhead.
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, drop_last=False,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model(name).to(device)
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    # discriminative learning rates: body gently, head faster
    body, head = [], []
    head_names = ("fc", "classifier", "head", "heads")
    for pname, param in net.named_parameters():
        (head if pname.split(".")[0] in head_names else body).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": lr_body},
         {"params": head, "lr": lr_head}],
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr_body, lr_head],
        total_steps=epochs * max(1, len(train_loader)),
        pct_start=min(0.3, max(0.05, warmup / float(epochs))),
        anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        net.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)

            if mixup > 0 and random.random() < 0.5:
                lam = np.random.beta(mixup, mixup)
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                out = net(xb)
                loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[perm])
            else:
                loss = criterion(net(xb), yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        val_acc, val_loss = evaluate(net, val_loader, device, criterion)
        # track the macro-averaged metrics too: with balanced classes they are
        # the ones that actually reflect ranking quality, and they can diverge
        # from plain accuracy when the model drifts towards one class
        val_probs_ep = predict_loader(net, val_loader, device)
        val_f1 = macro_f1(val_probs_ep, train_y[va_idx])
        val_bal = balanced_accuracy(val_probs_ep, train_y[va_idx])
        history.append(dict(epoch=epoch + 1, train_loss=running / max(1, seen),
                            val_loss=val_loss, val_acc=val_acc,
                            val_macro_f1=val_f1, val_balanced_acc=val_bal,
                            lr=optimizer.param_groups[1]["lr"]))
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if verbose:
            # elapsed / remaining, so a long CPU run is not a black box
            done = (time.time() - t0) / 60.0
            left = done / (epoch + 1) * (epochs - epoch - 1)
            print("  [%s f%d] epoch %2d/%d  train %.4f  val %.4f  acc %.4f  f1 %.4f  "
                  "(%.1f min in, ~%.1f min left)%s"
                  % (name, fold, epoch + 1, epochs, running / max(1, seen),
                     val_loss, val_acc, val_f1, done, left,
                     "  *" if best_epoch == epoch + 1 else ""), flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)

    ckpt_dir = ensure_dir(os.path.join(OUT, "checkpoints"))
    tag = "%s_fold%d%s" % (name, fold, ckpt_suffix)
    ckpt = os.path.join(ckpt_dir, "%s.pt" % tag)
    torch.save(dict(model=name, fold=fold, size=size, state=net.state_dict(),
                    val_acc=best_acc, val_idx=va_idx, epochs=epochs, seed=seed,
                    suffix=ckpt_suffix, classes=CLASSES), ckpt)

    if save_oof:
        # validation probabilities of the best checkpoint (for OOF analysis)
        val_probs = predict_loader(net, val_loader, device)
        ensure_dir(os.path.join(OUT, "oof"))
        np.savez(os.path.join(OUT, "oof", "%s.npz" % tag),
                 probs=val_probs, idx=va_idx, labels=train_y[va_idx])

    print("  [%s f%d] best val acc %.4f at epoch %d  (%.1f min, ckpt %s)"
          % (name, fold, best_acc, best_epoch, (time.time() - t0) / 60.0,
             os.path.basename(ckpt)), flush=True)

    with open(os.path.join(OUT, "history_%s.json" % tag), "w") as fh:
        json.dump(dict(model=name, fold=fold, best_acc=best_acc,
                       best_epoch=best_epoch, size=size, epochs=epochs,
                       history=history), fh, indent=2)
    return best_acc


@torch.no_grad() if torch is not None else (lambda f: f)
def evaluate(net, loader, device, criterion=None):
    net.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        out = net(xb)
        if criterion is not None:
            loss_sum += criterion(out, yb).item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


@torch.no_grad() if torch is not None else (lambda f: f)
def predict_loader(net, loader, device):
    net.eval()
    outs = []
    for xb, yb in loader:
        xb = xb.to(device)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        outs.append(F.softmax(net(xb), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# =========================================================================== #
#  Metrics
#
# Plain accuracy is a misleading target here: the 5 training classes are
# perfectly balanced, so a model that over-predicts one class can keep a high
# accuracy while losing a lot on the macro-averaged scores that this kind of
# competition usually ranks on.  Both are therefore reported everywhere.
# =========================================================================== #
def confusion_matrix(probs, labels, n_cls=NUM_CLASSES):
    """rows = true class, columns = predicted class."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(labels, np.asarray(probs).argmax(1)):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1(probs, labels, n_cls=NUM_CLASSES):
    """Unweighted mean of the per-class F1 scores."""
    cm = confusion_matrix(probs, labels, n_cls)
    f1s = []
    for k in range(n_cls):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s))


def balanced_accuracy(probs, labels, n_cls=NUM_CLASSES):
    """Mean per-class recall (equals accuracy when the classes are balanced)."""
    cm = confusion_matrix(probs, labels, n_cls)
    recalls = [cm[k, k] / cm[k].sum() if cm[k].sum() else 0.0 for k in range(n_cls)]
    return float(np.mean(recalls))


def accuracy(probs, labels):
    return float((np.asarray(probs).argmax(1) == np.asarray(labels)).mean())


def classification_report(probs, labels, n_cls=NUM_CLASSES, title=None):
    """Print accuracy / balanced accuracy / macro-F1 plus a per-class table."""
    cm = confusion_matrix(probs, labels, n_cls)
    if title:
        print("\n" + title)
    print("  accuracy          %.4f" % accuracy(probs, labels))
    print("  balanced accuracy %.4f" % balanced_accuracy(probs, labels, n_cls))
    print("  macro F1          %.4f" % macro_f1(probs, labels, n_cls))
    print("  %-18s %6s %6s %8s %8s" % ("class", "n", "recall", "precision", "f1"))
    for k in range(n_cls):
        tp = cm[k, k]
        tot = cm[k].sum()
        pred_tot = cm[:, k].sum()
        rec = tp / tot if tot else 0.0
        prec = tp / pred_tot if pred_tot else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        print("  %-18s %6d %6.3f %9.3f %8.3f"
              % (CLASSES[k] if k < len(CLASSES) else str(k), tot, rec, prec, f1))
    return dict(accuracy=accuracy(probs, labels),
                balanced_accuracy=balanced_accuracy(probs, labels, n_cls),
                macro_f1=macro_f1(probs, labels, n_cls))


# =========================================================================== #
#  Decision tuning (prior / bias correction)
#
# Transfer learning on a small balanced subset tends to inherit the label prior
# of the pretraining data, which makes the model over-predict some classes and
# under-predict others.  Measuring that directly on the OOF probabilities and
# then rescaling each class before the argmax recovers a lot of macro-F1.
# =========================================================================== #
def class_distribution(probs, n_cls=NUM_CLASSES):
    """Predicted class proportions."""
    preds = np.asarray(probs).argmax(1)
    n = max(1, len(preds))
    return np.array([(preds == k).sum() / n for k in range(n_cls)])


def prior_weights(probs, target=None, n_cls=None, power=1.0):
    """Per-class multipliers that pull the predicted distribution onto `target`.

    `target` defaults to uniform, which is the right prior whenever the
    training set is balanced (it is: 100 images per class).

    The ratio is taken against the model's *mean softmax* rather than the hard
    predicted counts, because the softmax is a smoother and much lower-variance
    estimate of the same quantity.

    `n_cls` defaults to `probs.shape[1]`, so synthetic callers (unit tests,
    reduced-class experiments) do not have to remember to pass it.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D (n_samples, n_classes), got %r"
                         % (probs.shape,))
    if n_cls is None:
        n_cls = probs.shape[1]
    if probs.shape[1] != n_cls:
        raise ValueError("n_cls=%d does not match probs.shape[1]=%d"
                         % (n_cls, probs.shape[1]))
    if target is None:
        target = np.ones(n_cls) / n_cls
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (n_cls,):
        raise ValueError("target must have %d entries, got %r"
                         % (n_cls, target.shape))
    target = target / target.sum()
    mean_p = np.maximum(probs.mean(axis=0), 1e-12)
    w = (target / mean_p) ** power
    return w / w.mean()


def apply_weights(probs, weights):
    """Argmax over (probability x per-class weight)."""
    p = np.asarray(probs, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
    return p.argmax(1)


def prior_fix(probs, target=None, strength=1.0, n_cls=None, power=1.0):
    """Transductive prior correction using the unlabelled set's own predictions.

    Different tool from `tune_decision_weights`, and it matters which one you
    use:

    * `tune_decision_weights` fits multipliers on the out-of-fold predictions.
      That is the right thing when the evaluation set carries the same label
      skew as the training folds.  If the folds are (nearly) balanced, there is
      no skew to remove, and the optimiser is free to distort classes to chase
      a fraction of a point -- which then transfers badly to a test set whose
      skew is much larger.
    * this function measures the skew *on the set being predicted* and pushes
      its mean predicted distribution back towards `target`.

    No labels are used, so there is no leakage.  The assumption is that the
    unlabelled set follows the training prior -- sound here, because the
    training set is exactly 100 images per class.

    `strength` in [0, 1] interpolates between no correction and the full
    correction, which is the honest way to use a correction that cannot be
    validated on the target set.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if n_cls is None:
        n_cls = probs.shape[1]
    if strength <= 0:
        return probs.copy(), np.ones(n_cls), dict(strength=0.0)

    mean_p = np.maximum(probs.mean(axis=0), 1e-12)
    if target is None:
        target = np.ones(n_cls) / n_cls
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()

    w_full = (target / mean_p) ** power
    w_full = w_full / w_full.mean()
    w = w_full ** float(strength)                 # partial correction

    corrected = probs * w[None, :]
    corrected = corrected / np.maximum(corrected.sum(axis=1, keepdims=True), 1e-12)
    diag = dict(strength=float(strength),
                weights_full=w_full, weights_applied=w,
                before_dist=class_distribution(probs, n_cls),
                after_dist=class_distribution(corrected, n_cls))
    return corrected, w, diag


def tune_decision_weights(probs, labels, n_cls=NUM_CLASSES, power_grid=None,
                          max_iter=4, verbose=False):
    """Coordinate ascent on per-class multipliers, maximising balanced accuracy.

    The search happens in "power" space between the prior-correction weights
    `base` (computed once, from the mean softmax) and the current multipliers,
    so one step size means the same thing for every class.  Multipliers are
    normalised to mean 1, which removes the global scale ambiguity and leaves
    only n_cls-1 free parameters -- very few degrees of freedom for the ~500
    available out-of-fold samples.

    Cheap by construction: the response matrix `probs * base` is formed once,
    then each trial is a column rescale plus an argmax.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if power_grid is None:
        power_grid = np.concatenate([np.linspace(-1.0, 1.0, 25),
                                     np.linspace(-1.4, 1.4, 9)])
    if len(np.unique(labels)) != n_cls:
        raise ValueError("decision tuning needs every class present in the labels")

    base = prior_weights(probs, n_cls=n_cls, power=1.0)
    resp = probs * base[None, :]                 # (N, C), fixed for the whole search
    log_base = np.log(np.maximum(base, 1e-12))

    def score_of(w):
        return balanced_accuracy(resp * w[None, :], labels, n_cls)

    w = np.ones(n_cls)
    best = score_of(w)
    best_base_only = best

    for it in range(max_iter):
        improved = False
        for k in range(n_cls):
            for step in power_grid:
                trial = w.copy()
                trial[k] = np.exp(step * (1.0 + log_base[k]))
                trial = trial / trial.mean()
                s = score_of(trial)
                if s > best + 1e-9:
                    best, w, improved = s, trial, True
        if verbose:
            print("  [tune] iter %d -> balanced acc %.4f  w=%s"
                  % (it + 1, best, np.round(w, 3)))
        if not improved:
            break

    return dict(weights=w, base_weights=base, score=best,
                score_base_only=best_base_only, n=len(labels))


def validate_decision_weights(probs, labels, fold_id, n_cls=NUM_CLASSES,
                              max_iter=6, verbose=True):
    """Leave-one-fold-out check that decision tuning generalises.

    Weights are tuned on four folds and scored on the held-out fold, so the
    reported gain is not the gain of fitting the same data twice.

    Both balanced accuracy (the tuning objective) and macro-F1 (the metric a
    course leaderboard is most likely to rank on) are reported, so a change
    that helps one at the cost of the other is visible rather than hidden.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    fold_id = np.asarray(fold_id)
    folds = sorted(set(fold_id.tolist()))

    acc_before = acc_after = total = 0
    bal_before = bal_after = 0.0
    f1_before = f1_after = 0.0
    detail = []
    for f in folds:
        te = fold_id == f
        tr = ~te
        if len(np.unique(labels[tr])) < n_cls or len(np.unique(labels[te])) < n_cls:
            if verbose:
                print("  [val-weights] fold %s skipped (a class is missing)" % f)
            continue
        tuned = tune_decision_weights(probs[tr], labels[tr], n_cls, max_iter=max_iter)
        w = tuned["weights"] * tuned["base_weights"]

        n_te = int(te.sum())
        p_before, p_after = probs[te], probs[te] * w[None, :]
        acc_before += int((p_before.argmax(1) == labels[te]).sum())
        acc_after += int((p_after.argmax(1) == labels[te]).sum())
        bal_before += balanced_accuracy(p_before, labels[te], n_cls)
        bal_after += balanced_accuracy(p_after, labels[te], n_cls)
        f1_before += macro_f1(p_before, labels[te], n_cls)
        f1_after += macro_f1(p_after, labels[te], n_cls)
        total += n_te
        detail.append(dict(fold=f, n=n_te,
                           acc_before=accuracy(p_before, labels[te]),
                           acc_after=accuracy(p_after, labels[te]),
                           bal_before=balanced_accuracy(p_before, labels[te], n_cls),
                           bal_after=balanced_accuracy(p_after, labels[te], n_cls),
                           f1_before=macro_f1(p_before, labels[te], n_cls),
                           f1_after=macro_f1(p_after, labels[te], n_cls)))
        if verbose:
            print("  [val-weights] fold %s: acc %.4f -> %.4f | bal %.4f -> %.4f "
                  "| macroF1 %.4f -> %.4f"
                  % (f, detail[-1]["acc_before"], detail[-1]["acc_after"],
                     detail[-1]["bal_before"], detail[-1]["bal_after"],
                     detail[-1]["f1_before"], detail[-1]["f1_after"]))

    if total == 0:
        return dict(n=0, detail=[])
    nf = float(len(detail))
    return dict(n=int(total), detail=detail,
                acc_before=acc_before / total, acc_after=acc_after / total,
                bal_before=bal_before / nf, bal_after=bal_after / nf,
                f1_before=f1_before / nf, f1_after=f1_after / nf,
                delta_bal=(bal_after - bal_before) / nf,
                delta_f1=(f1_after - f1_before) / nf)


# =========================================================================== #
#  Stage 6 -- test-time prediction with augmentation
# =========================================================================== #
def available_folds(name, folds=None):
    """Folds of `name` that actually have a checkpoint, in ascending order."""
    ckpt_dir = os.path.join(OUT, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    limit = folds if folds is not None else 99
    return [f for f in range(limit)
            if os.path.exists(os.path.join(ckpt_dir, "%s_fold%d.pt" % (name, f)))]


def aggregate_oof(model_names, folds=5):
    """Collect every backbone's out-of-fold probabilities onto one sample axis.

    Each fold file stores the probabilities of the images that fold held out,
    so stacking them reconstructs one honest prediction per training image.
    Backbones with fewer folds simply leave `coverage` False where they have no
    prediction; those rows are excluded when tuning or scoring.

    Returns a dict with `probs` (n_models, n_train, C), `labels` (n_train,),
    `fold_id` (n_train,) and `coverage` (n_models, n_train).
    """
    n_train = None
    per_model = []
    for name in model_names:
        fold_files = []
        for fold in available_folds(name, folds):
            path = os.path.join(OUT, "oof", "%s_fold%d.npz" % (name, fold))
            if os.path.exists(path):
                fold_files.append((fold, path))
        if not fold_files:
            continue
        rows = []
        for fold, path in fold_files:
            d = np.load(path)
            rows.append((fold, d["idx"], d["probs"], d["labels"]))
            n_train = max(n_train or 0, int(d["idx"].max()) + 1)
        per_model.append((name, rows))

    if not per_model or not n_train:
        return dict(probs=np.zeros((0, 0, NUM_CLASSES)), labels=np.zeros(0, dtype=int),
                    fold_id=np.zeros(0, dtype=int),
                    coverage=np.zeros((0, 0), dtype=bool), names=[])

    probs = np.zeros((len(per_model), n_train, NUM_CLASSES), dtype=np.float64)
    coverage = np.zeros((len(per_model), n_train), dtype=bool)
    labels = np.full(n_train, -1, dtype=int)
    fold_id = np.full(n_train, -1, dtype=int)

    for mi, (_name, rows) in enumerate(per_model):
        for fold, idx, p, lab in rows:
            probs[mi, idx] = p
            coverage[mi, idx] = True
            labels[idx] = lab
            fold_id[idx] = fold

    return dict(probs=probs, labels=labels, fold_id=fold_id,
                coverage=coverage, names=[n for n, _r in per_model])


def ensemble_oof_probs(oof):
    """Mean OOF probabilities over the backbones that cover each image."""
    if oof["probs"].size == 0:
        return np.zeros((0, NUM_CLASSES)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    cov = oof["coverage"]
    w = cov.sum(axis=0)
    keep = w > 0
    summed = np.where(cov[:, :, None], oof["probs"], 0.0).sum(axis=0)
    mean = np.zeros_like(summed)
    mean[keep] = summed[keep] / w[keep, None]
    return mean[keep], oof["labels"][keep], oof["fold_id"][keep]


def fit_and_validate_weights(model_names, folds=5, n_cls=NUM_CLASSES, verbose=True):
    """Fit decision weights on all OOF data, then check them leave-one-fold-out.

    Returns dict(weights, base_weights, oof_probs, labels, fold_id,
                 before, after, validated) where `before`/`after` hold the
    accuracy, balanced accuracy and macro-F1 of the same OOF predictions with
    and without the correction.
    """
    oof = aggregate_oof(model_names, folds)
    probs, labels, fold_id = ensemble_oof_probs(oof)
    if len(labels) == 0 or len(np.unique(labels)) != n_cls:
        if verbose:
            print("[tune] not enough OOF coverage to tune decision weights")
        return None

    if verbose:
        classification_report(probs, labels, n_cls,
                              title="[tune] OOF before decision tuning")
        dist = class_distribution(probs, n_cls)
        print("  predicted distribution: %s"
              % "  ".join("%s %.3f" % (CLASSES[k], dist[k]) for k in range(n_cls)))

    tuned = tune_decision_weights(probs, labels, n_cls, verbose=verbose)
    weights = tuned["weights"] * tuned["base_weights"]
    before_pred = probs.argmax(1)
    after_pred = apply_weights(probs, weights)

    validated = validate_decision_weights(probs, labels, fold_id, n_cls, verbose=verbose)

    weighted = probs * weights[None, :]
    before = dict(accuracy=accuracy(probs, labels),
                  balanced_accuracy=balanced_accuracy(probs, labels, n_cls),
                  macro_f1=macro_f1(probs, labels, n_cls))
    after = dict(accuracy=accuracy(weighted, labels),
                 balanced_accuracy=balanced_accuracy(weighted, labels, n_cls),
                 macro_f1=macro_f1(weighted, labels, n_cls))

    if verbose:
        print("\n[tune] fitted weights: %s"
              % "  ".join("%s %.3f" % (CLASSES[k], weights[k]) for k in range(n_cls)))
        print("[tune] OOF accuracy          %.4f -> %.4f" % (before["accuracy"], after["accuracy"]))
        print("[tune] OOF balanced accuracy %.4f -> %.4f"
              % (before["balanced_accuracy"], after["balanced_accuracy"]))
        print("[tune] OOF macro F1          %.4f -> %.4f"
              % (before["macro_f1"], after["macro_f1"]))
        print("[tune] changed %d / %d OOF predictions"
              % (int((before_pred != after_pred).sum()), len(labels)))
        if validated["n"]:
            print("[tune] leave-one-fold-out check on %d images:" % validated["n"])
            print("        balanced accuracy %.4f -> %.4f (%+.4f)"
                  % (validated["bal_before"], validated["bal_after"], validated["delta_bal"]))
            print("        macro F1          %.4f -> %.4f (%+.4f)"
                  % (validated["f1_before"], validated["f1_after"], validated["delta_f1"]))
            if validated["delta_f1"] < 0 and validated["delta_bal"] < 0:
                print("        WARNING: tuning hurts on held-out folds -- "
                      "prefer a run with tune=False")
        else:
            print("[tune] leave-one-fold-out check skipped (needs >= 2 covered folds)")

    return dict(weights=weights, base_weights=tuned["base_weights"],
                oof_probs=probs, labels=labels, fold_id=fold_id,
                before=before, after=after, validated=validated)


def predict_test(model_names, folds=5, tta=8, batch=None, data_root=None,
                 tta_sizes=None, tune=True, verbose=True, include_pseudo=True,
                 prior_strength=0.0):
    """Average softmax over backbones x folds x TTA views.

    Fold coverage may differ between backbones (a cheap model can afford all
    five folds, an expensive one only three).  Each backbone is therefore
    averaged over *its own* folds first, and the backbones are then averaged
    with equal weight -- so a model with fewer folds is not silently
    down-weighted.

    `tta_sizes` adds multi-scale inference: every listed resolution contributes
    its own 8-view average, which helps because fine-grained plant texture
    responds differently to different resize scales.

    `tune` fits per-class decision weights on the out-of-fold predictions and
    applies them to the test probabilities.  This corrects the label-prior bias
    that transfer learning inherits, which is usually worth more on a
    macro-averaged leaderboard than any further architecture tweak.  Set it to
    False to get plain argmax behaviour.

    `include_pseudo` additionally averages in any round-2 (`_pl`) checkpoints
    written by `pseudo_label_train`, which doubles the ensemble size.

    `prior_strength` in (0, 1] turns on the transductive prior fix: the test
    predictions' own mean distribution is pushed back towards the training prior
    (uniform), using no labels.  It is a separate tool from `tune`, not a
    stacked one -- see `prior_fix` for when each applies.  The corrected
    probabilities are written to `test_priorfix.npz`.

    Returns (mean_probs, per_model_probs, test_names, oof_acc, weights).
    """
    if torch is None:                                        # pragma: no cover
        raise RuntimeError("torch is required: %s" % _TORCH_ERR)

    train_x, train_y, test_x, test_names = load_cache(data_root=data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs = []

    for name in model_names:
        # round 1, plus (optionally) the pseudo-label round from stage 6b
        variants = [(name, "")]
        if include_pseudo:
            variants.append((name + PL_SUFFIX, PL_SUFFIX))

        model_probs = []
        for tag_name, suffix in variants:
            for fold in available_folds(tag_name, folds):
                ckpt_path = os.path.join(OUT, "checkpoints",
                                         "%s_fold%d%s.pt" % (name, fold, suffix))
                if not os.path.exists(ckpt_path):
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                size = ckpt["size"]
                net = build_model(name, pretrained=False)
                net.load_state_dict(ckpt["state"])
                net = net.to(device).eval()

                # one 8-view average per resolution, then an average over scales
                sizes = list(tta_sizes) if tta_sizes else [size]
                fold_prob = np.zeros((len(test_x), NUM_CLASSES), dtype=np.float64)
                for scale in sizes:
                    for view in range(tta):
                        ds = TestDataset(test_x, scale, view=view)
                        loader = DataLoader(ds,
                                            batch_size=batch or get_preset(name)["batch"],
                                            shuffle=False, num_workers=0)
                        fold_prob += predict_loader(net, loader, device)
                fold_prob /= float(tta * len(sizes))
                model_probs.append(fold_prob)

                if verbose:
                    print("[predict] %s fold %d%s done (val acc %.4f)"
                          % (name, fold, suffix,
                             ckpt.get("val_acc", float("nan"))), flush=True)

        if model_probs:
            stacked = np.stack(model_probs)              # (variants, N, C)
            all_probs.append(stacked.mean(axis=0))
            np.savez(os.path.join(OUT, "test_%s.npz" % name), probs=stacked.mean(axis=0),
                     fold_probs=stacked, test_names=np.asarray(test_names))
            if verbose:
                print("[predict] %s: averaged %d model(s) -> backbone probability map"
                      % (name, len(model_probs)), flush=True)
        elif verbose:
            print("[predict] no checkpoints for %s -- skipped" % name)

    if not all_probs:
        raise RuntimeError("no checkpoints found under %s" % os.path.join(OUT, "checkpoints"))

    mean_probs = np.mean(np.stack(all_probs), axis=0)

    # ---- honest OOF score, and decision tuning from the same predictions ----
    # This reads the round-1 OOF files only.  Pseudo-label models deliberately
    # never write OOF files, so the weights below stay fit on clean predictions.
    oof = aggregate_oof(model_names, folds)
    oof_probs, oof_labels, _oof_folds = ensemble_oof_probs(oof)
    oof_acc = accuracy(oof_probs, oof_labels) if len(oof_labels) else float("nan")
    oof_f1 = macro_f1(oof_probs, oof_labels) if len(oof_labels) else float("nan")

    weights = np.ones(NUM_CLASSES)
    tuned_info = None
    if tune:
        # tuning is an optional refinement: never let it break prediction
        try:
            tuned_info = fit_and_validate_weights(model_names, folds, verbose=verbose)
            if tuned_info is not None:
                weights = tuned_info["weights"]
        except Exception as exc:
            print("[predict] decision tuning skipped: %s: %s"
                  % (type(exc).__name__, exc))

    # ---- transductive prior fix on the test set itself ----
    # Deliberately independent of the OOF tuning above: if the folds were
    # already balanced there is no skew for that step to remove, while the test
    # predictions can be badly skewed.  This one measures the skew where it
    # actually matters.
    prior_diag = None
    prior_probs = mean_probs
    if prior_strength and prior_strength > 0:
        try:
            prior_probs, prior_w, prior_diag = prior_fix(mean_probs, n_cls=NUM_CLASSES,
                                                         strength=prior_strength)
            if verbose:
                print("\n[prior] transductive prior fix on the test set "
                      "(strength %.2f)" % prior_strength)
                print("[prior] mean predicted distribution  %s"
                      % "  ".join("%s %.3f" % (CLASSES[k], prior_diag["before_dist"][k])
                                  for k in range(NUM_CLASSES)))
                print("[prior] multipliers                 %s"
                      % "  ".join("%s %.3f" % (CLASSES[k], prior_w[k])
                                  for k in range(NUM_CLASSES)))
                print("[prior] distribution after fix      %s"
                      % "  ".join("%s %.3f" % (CLASSES[k], prior_diag["after_dist"][k])
                                  for k in range(NUM_CLASSES)))
                print("[prior] no labels are used -- this only reshapes the decision "
                      "boundary towards the training prior (uniform).")
        except Exception as exc:
            print("[prior] prior fix skipped: %s: %s" % (type(exc).__name__, exc))
            prior_probs = mean_probs

    np.savez(os.path.join(OUT, "test_ensemble.npz"), probs=mean_probs,
             test_names=np.asarray(test_names), oof_acc=oof_acc, oof_macro_f1=oof_f1,
             weights=weights, classes=np.asarray(CLASSES))
    if prior_diag is not None:
        np.savez(os.path.join(OUT, "test_priorfix.npz"), probs=prior_probs,
                 test_names=np.asarray(test_names), oof_acc=oof_acc,
                 weights_applied=prior_diag["weights_applied"],
                 strength=prior_strength, classes=np.asarray(CLASSES))
    if verbose:
        print("\n[predict] ensemble OOF accuracy %.4f | macro F1 %.4f"
              % (oof_acc, oof_f1))
        dist_before = class_distribution(mean_probs)
        dist_after = class_distribution(mean_probs * weights[None, :])
        print("[predict] test distribution before tuning: %s"
              % "  ".join("%.1f%%" % (100 * d) for d in dist_before))
        print("[predict] test distribution after  tuning: %s"
              % "  ".join("%.1f%%" % (100 * d) for d in dist_after))
        ideal = 100.0 / NUM_CLASSES
        dev = lambda d: float(np.mean(np.abs(d * 100.0 - ideal)))
        print("[predict] mean deviation from uniform (lower is better): "
              "raw %.2f | oof-tuned %.2f | prior-fixed %.2f"
              % (dev(dist_before), dev(dist_after),
                 dev(class_distribution(prior_probs))))
    return mean_probs, all_probs, test_names, oof_acc, weights

# =========================================================================== #
#  Stage 6b -- pseudo-labelling (optional second round)
#
# 500 labelled images is the binding constraint here.  The 378 test images are
# unlabelled but come from exactly the distribution being graded, so adding the
# confident ones back as training data is the single biggest remaining lever.
#
# Honesty rules that this implementation follows:
#   * a pseudo-label is only used if the ensemble is confident AND all
#     backbones agree on the class;
#   * pseudo-labelled images never enter a validation fold, so the printed
#     validation accuracy is still measured on real labelled data;
#   * these runs write no OOF files (save_oof=False), because their validation
#     images belong to the pool the pseudo-labels were selected from.  Letting
#     them into outputs/oof would contaminate the decision-weight tuning.
# =========================================================================== #
PL_SUFFIX = "_pl"


def test_backbone_probs(model_names, expected_n=None):
    """Load the per-backbone test probability maps written by `predict_test`.

    Returns (list_of_arrays, names_found).  Used to require that every backbone
    agrees before a test image is accepted as a pseudo-label -- confidence of
    the averaged ensemble alone is not enough, because an average can be
    confident while the individual models disagree.
    """
    out, names = [], []
    for name in model_names:
        path = os.path.join(OUT, "test_%s.npz" % name)
        if not os.path.exists(path):
            continue
        p = np.load(path)["probs"]
        if expected_n is None or len(p) == expected_n:
            out.append(p)
            names.append(name)
    return out, names


def select_pseudo_labels(probs, min_conf=0.90, require_agreement=True,
                         per_model_probs=None):
    """Pick test images to treat as labelled.

    Returns (indices, labels, diagnostics).
    """
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(1)
    preds = probs.argmax(1)

    keep = conf >= min_conf
    agreement_applied = False
    if require_agreement and per_model_probs is not None and len(per_model_probs) > 1:
        stacks = [np.asarray(p).argmax(1) for p in per_model_probs]
        agree = np.array([len({s[i] for s in stacks}) == 1 for i in range(len(preds))])
        keep &= agree
        agreement_applied = True

    idx = np.where(keep)[0]
    diag = dict(total=len(probs), selected=len(idx),
                threshold=min_conf,
                agreement_applied=agreement_applied,
                n_backbones=len(per_model_probs) if per_model_probs else 0,
                mean_conf_selected=float(conf[idx].mean()) if len(idx) else float("nan"),
                class_counts={CLASSES[k]: int((preds[idx] == k).sum())
                              for k in range(NUM_CLASSES)})
    return idx, preds[idx], diag


def pseudo_label_train(model_names, folds=5, epochs=25, min_conf=0.90,
                       require_agreement=True, tta=8, batch=None,
                       data_root=None, seed=SEED, verbose=True):
    """Round 2: retrain every (backbone, fold) on real + confident test images.

    Must be called after `predict_test`, which produces the test probabilities
    and the decision weights used here.
    """
    if torch is None:                                        # pragma: no cover
        raise RuntimeError("torch is required: %s" % _TORCH_ERR)

    ens_path = os.path.join(OUT, "test_ensemble.npz")
    if not os.path.exists(ens_path):
        raise FileNotFoundError("run predict_test first (%s missing)" % ens_path)
    ens = np.load(ens_path)
    probs = ens["probs"]
    weights = ens["weights"] if "weights" in ens.files else np.ones(NUM_CLASSES)

    # apply the decision weights before thresholding, so the pseudo-labels are
    # chosen with the same decision rule that produced the submission
    per_backbone, _found = test_backbone_probs(model_names, expected_n=len(probs))
    if require_agreement and len(per_backbone) < 2:
        print("[pseudo] WARNING: only %d per-backbone probability map(s) found; "
              "the agreement gate needs at least 2, so it will be skipped."
              % len(per_backbone))
    idx, pl_labels, diag = select_pseudo_labels(
        probs * weights[None, :], min_conf=min_conf,
        require_agreement=require_agreement, per_model_probs=per_backbone)

    print("\n[pseudo] confidence threshold %.2f" % min_conf)
    print("[pseudo] backbone-agreement gate: %s (%d backbone(s))"
          % ("ON" if diag["agreement_applied"] else "OFF", diag["n_backbones"]))
    print("[pseudo] selected %d / %d test images (%.1f%%), mean confidence %.3f"
          % (diag["selected"], diag["total"],
             100.0 * diag["selected"] / max(1, diag["total"]),
             diag["mean_conf_selected"]))
    for cls, n in diag["class_counts"].items():
        print("           %-18s %4d" % (cls, n))
    if diag["selected"] == 0:
        print("[pseudo] nothing selected -- lower --min-conf or check the model")
        return None

    train_x, train_y, test_x, _names = load_cache(data_root=data_root)
    pl_x = test_x[idx]
    pl_y = pl_labels
    print("[pseudo] retraining %d (backbone, fold) runs with %d extra images each\n"
          % (len(model_names) * folds, len(pl_x)), flush=True)

    results = {}
    for name in model_names:
        for fold in range(folds):
            acc = train_one(name, fold, folds=folds, epochs=epochs,
                            batch=batch, seed=seed, data_root=data_root,
                            verbose=verbose, extra_x=pl_x, extra_y=pl_y,
                            save_oof=False, ckpt_suffix=PL_SUFFIX)
            results["%s_fold%d" % (name, fold)] = acc
            print("      -> round-2 %s fold %d: validation acc %.4f (real labels only)"
                  % (name, fold, acc), flush=True)

    mean_acc = float(np.mean(list(results.values()))) if results else float("nan")
    print("\n[pseudo] round-2 mean validation accuracy on real held-out data: %s"
          % ("%.4f" % mean_acc if results else "n/a (nothing trained)"))
    print("[pseudo] compare against the round-1 OOF score printed by predict_test.")
    return results


# =========================================================================== #
#  Stage 7 -- submission
# =========================================================================== #
def write_submission(probs, test_names, out_path, participant="", template=None,
                     weights=None):
    """Write ID,Category sorted the same way as the provided template.

    `weights` are the optional per-class decision multipliers from the OOF
    tuning step; without them the plain argmax is used.
    """
    order = list(range(len(test_names)))
    if template and os.path.exists(template):
        with open(template, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            tmpl_ids = [row[0] for row in reader if row]
        index = {n: i for i, n in enumerate(test_names)}
        missing = [t for t in tmpl_ids if t not in index]
        if missing:
            print("[submit] warning: %d template ids absent from test set" % len(missing))
        order = [index[t] for t in tmpl_ids if t in index]

    preds = apply_weights(probs, weights) if weights is not None else probs.argmax(axis=1)
    rows = [[test_names[i], CLASSES[preds[i]]] for i in order]
    write_csv(out_path, ["ID", "Category"], rows)

    counts = {c: 0 for c in CLASSES}
    for _i, c in rows:
        counts[c] += 1
    print("[submit] wrote %d rows -> %s" % (len(rows), out_path))
    for c in CLASSES:
        print("            %-18s %4d (%.1f%%)" % (c, counts[c], 100.0 * counts[c] / max(1, len(rows))))
    if participant:
        print("[submit] group name set to %r (must be your student id)" % participant)
    return out_path


# =========================================================================== #
#  CLI
# =========================================================================== #
def cmd_cache(args):
    build_cache(args.data, size=args.cache_size, force=args.force)


def cmd_train(args):
    models = args.model
    folds = list(range(args.folds)) if args.fold is None else [args.fold]
    results = {}
    for name in models:
        for fold in folds:
            acc = train_one(name, fold, folds=args.folds, epochs=args.epochs,
                            size=args.size, batch=args.batch, seed=args.seed,
                            data_root=args.data, mixup=args.mixup)
            results["%s_fold%d" % (name, fold)] = acc
    print("\n=== validation summary (accuracy) ===")
    for k, v in sorted(results.items()):
        print("  %-28s %.4f" % (k, v))
    print("  mean                          %.4f" % (np.mean(list(results.values()))))


def cmd_report(args):
    """Score the saved out-of-fold predictions, before and after tuning."""
    oof = aggregate_oof(args.model, args.folds)
    probs, labels, fold_id = ensemble_oof_probs(oof)
    if len(labels) == 0:
        raise FileNotFoundError(
            "no OOF files under %s/oof -- train first" % OUT)
    classification_report(probs, labels, title="=== OOF (no tuning) ===")
    dist = class_distribution(probs)
    print("  predicted distribution: %s"
          % "  ".join("%s %.3f" % (CLASSES[k], dist[k]) for k in range(NUM_CLASSES)))
    tuned = fit_and_validate_weights(args.model, args.folds, verbose=True)
    if tuned is not None:
        print("\n=== final decision weights ===")
        for k in range(NUM_CLASSES):
            print("  %-18s %.4f" % (CLASSES[k], tuned["weights"][k]))


def cmd_predict(args):
    probs, _per_model, names, oof_acc, weights = predict_test(
        args.model, folds=args.folds, tta=args.tta, batch=args.batch,
        data_root=args.data, tta_sizes=args.tta_sizes, tune=not args.no_tune,
        prior_strength=getattr(args, "prior_strength", 0.0))
    if args.out:
        write_submission(probs, names, args.out, args.name, args.template,
                         weights=weights)
        if args.out_plain:
            write_submission(probs, names, args.out_plain, args.name, args.template)
        # the transductive prior fix is also written out on its own, so the
        # three candidates can be compared before choosing one to submit
        if getattr(args, "out_priorfix", None):
            pf_path = os.path.join(OUT, "test_priorfix.npz")
            if os.path.exists(pf_path):
                pf = np.load(pf_path)
                write_submission(pf["probs"], [str(s) for s in pf["test_names"]],
                                 args.out_priorfix, args.name, args.template)
            else:
                print("[predict] %s missing -- prior fix was not applied" % pf_path)


def cmd_submit(args):
    path = os.path.join(OUT, "test_ensemble.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("run `predict` first (%s missing)" % path)
    data = np.load(path)
    weights = None
    if not args.no_tune and "weights" in data:
        weights = data["weights"]
        print("[submit] using tuned decision weights: %s"
              % "  ".join("%.3f" % w for w in weights))
    write_submission(data["probs"], [str(s) for s in data["test_names"]],
                     args.out, args.name, args.template, weights=weights)


def cmd_pseudo(args):
    """Round 2: retrain with confident test images added as pseudo-labels."""
    pseudo_label_train(args.model, folds=args.folds, epochs=args.epochs,
                       min_conf=args.min_conf,
                       require_agreement=not args.no_agreement,
                       tta=args.tta, batch=args.batch, data_root=args.data,
                       seed=args.seed)
    # refresh the submission so it already includes the round-2 models
    cmd_predict(args)


def cmd_all(args):
    cmd_cache(args)
    cmd_train(args)
    cmd_predict(args)

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plant seedling classification (transfer learning)")
    parser.add_argument("--data", default=None,
                        help="dataset root containing train/ and test/")
    parser.add_argument("--cache-size", type=int, default=CACHE_SIZE)

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cache", help="build the resized image cache")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("train", help="train one or more backbones")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None, help="train a single fold only")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--mixup", type=float, default=0.0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report",
                       help="score saved OOF predictions and fit decision weights")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("predict", help="predict the test set with TTA")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--tta-sizes", type=int, nargs="+", default=None,
                   help="extra resolutions to average over, e.g. --tta-sizes 256 320")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--no-tune", action="store_true",
                   help="skip the OOF decision-weight correction (plain argmax)")
    p.add_argument("--out", default=None, help="also write a submission csv here")
    p.add_argument("--out-plain", default=None,
                   help="additionally write an untuned submission for comparison")
    p.add_argument("--prior-strength", type=float, default=0.0,
                   help="0..1, transductive prior fix towards the training prior "
                        "(uniform); 0 disables it")
    p.add_argument("--out-priorfix", default=None,
                   help="additionally write the prior-fixed submission")
    p.add_argument("--name", default="", help="group name = your student id")
    p.add_argument("--template", default=None)
    p.set_defaults(func=cmd_predict)
    p = sub.add_parser("submit", help="write a submission from the saved ensemble")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--name", default="")
    p.add_argument("--template", default=None)
    p.add_argument("--no-tune", action="store_true",
                   help="ignore the saved decision weights")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("pseudo",
                       help="round 2: retrain with confident test images as pseudo-labels")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--min-conf", type=float, default=0.90,
                   help="ensemble confidence needed to accept a pseudo-label")
    p.add_argument("--no-agreement", action="store_true",
                   help="do not require all backbones to agree on the class")
    p.add_argument("--seed", type=int, default=SEED)
    # remaining options are forwarded to cmd_predict after the retraining
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--tta-sizes", type=int, nargs="+", default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--out-plain", default=None)
    p.add_argument("--name", default="")
    p.add_argument("--template", default=None)
    p.set_defaults(func=cmd_pseudo)

    p = sub.add_parser("all", help="cache + train + predict")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--tta-sizes", type=int, nargs="+", default=None)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--prior-strength", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.add_argument("--out-plain", default=None)
    p.add_argument("--out-priorfix", default=None)
    p.add_argument("--name", default="")
    p.add_argument("--template", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_all)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
