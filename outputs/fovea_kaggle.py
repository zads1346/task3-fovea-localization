# -*- coding: utf-8 -*-
# Kaggle 2026task3: complete grouped CV heatmap localization pipeline.
# Source and instructions are also available in fovea_kaggle.ipynb.
from __future__ import annotations

import os
import re
import gc
import sys
import json
import math
import random
import hashlib
import warnings
import platform
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from collections import defaultdict
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass
class Config:
    # Usually leave DATA_DIR empty: locate the attached competition automatically.
    INPUT_ROOT: str = "/kaggle/input"
    DATA_DIR: str | None = None
    LABEL_CSV: str | None = None
    SAMPLE_CSV: str | None = None
    OUTPUT_DIR: str = "/kaggle/working/fovea_run"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    GROUP_COLUMN: str | None = None  # Optional real patient/subject column in labels.
    COORDINATE_ORIGIN: int = 0  # Change to 1 only if annotation documentation confirms it.
    DUPLICATE_LABEL_TOLERANCE_PX: float = 5.0  # Preserve small label differences; keep duplicates in one fold.
    SEED: int = 2026
    N_FOLDS: int = 5
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    HEATMAP_SIGMA: float = 2.5
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    EPOCHS: int = 80
    PATIENCE: int = 12
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 3
    AMP: bool = True
    GRAD_CLIP: float = 1.0
    FREEZE_BN: bool = True
    COORD_SCALE: float = 512.0
    HEATMAP_LOSS_WEIGHT: float = 1.0
    COORD_LOSS_WEIGHT: float = 10.0
    VIS_LOSS_WEIGHT: float = 1.0
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = True
    PRETRAINED_WEIGHTS_PATH: str | None = None
    VISIBILITY_THRESHOLD: float = 0.5
    # False avoids reporting a threshold fitted on OOF labels as unbiased CV.
    TUNE_VISIBILITY_THRESHOLD: bool = False
    TTA: bool = True  # Original + horizontal flip; both decoded in original pixels.
    SHOW_PLOTS: bool = True
    REUSE_CHECKPOINTS: bool = False  # True only with identical data/config/code.
    SMOKE_TEST: bool = False  # 2 folds, 2 epochs, low resolution; NOT a scored model.


CFG = Config()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: some CUDA kernels do not have deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(0)


def validate_config(cfg):
    assert cfg.INPUT_SIZE >= 64 and cfg.INPUT_SIZE % 32 == 0
    assert cfg.HEATMAP_SIZE >= 8 and cfg.HEATMAP_SIGMA > 0
    assert cfg.N_FOLDS >= 2 and cfg.EPOCHS >= 1 and cfg.BATCH_SIZE >= 1
    assert cfg.COORDINATE_ORIGIN in (0, 1)
    assert cfg.COORD_SCALE > 0
    assert cfg.DUPLICATE_LABEL_TOLERANCE_PX >= 0
    assert 0 <= cfg.VISIBILITY_THRESHOLD <= 1
    return cfg


def csv_read(path):
    # Strings preserve leading zeroes in image names and exact submission IDs.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate CSV columns: {path}")
    return frame


def column_name(frame, expected):
    matches = [c for c in frame.columns if c.casefold() == expected.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected column {expected!r}; found {list(frame.columns)}")
    return matches[0]


def discover_data(cfg):
    base = Path(cfg.DATA_DIR or cfg.INPUT_ROOT)
    if not base.exists():
        raise FileNotFoundError(
            f"{base} does not exist. In Kaggle, Add Input -> Competition -> 2026task3, "
            "then set CFG.DATA_DIR if automatic discovery is ambiguous."
        )
    csvs = sorted(base.rglob("*.csv"))
    if cfg.LABEL_CSV:
        label_path = Path(cfg.LABEL_CSV)
    else:
        candidates = [p for p in csvs if p.name.casefold() == "fovea_localization_train_gt.csv"]
        if not candidates:
            candidates = []
            for p in csvs:
                try:
                    cols = set(pd.read_csv(p, nrows=0).columns.str.strip().str.casefold())
                    if {"data", "fovea_x", "fovea_y"} <= cols:
                        candidates.append(p)
                except Exception:
                    pass
        if len(candidates) != 1:
            raise ValueError(
                "Cannot uniquely locate labels. Set CFG.DATA_DIR or CFG.LABEL_CSV. "
                f"Candidates: {[str(p) for p in candidates]}"
            )
        label_path = candidates[0]
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    if cfg.SAMPLE_CSV:
        sample_path = Path(cfg.SAMPLE_CSV)
        root = base if cfg.DATA_DIR else Path(os.path.commonpath([label_path.parent, sample_path.parent]))
    else:
        # Search nearest enclosing directory first, avoiding other attached datasets.
        root = label_path.parent
        while True:
            candidates = [p for p in root.rglob("*.csv") if p.name.casefold() == "sample_submission.csv"]
            if candidates:
                if len(candidates) != 1:
                    raise ValueError(f"Several sample submissions under {root}; set CFG.SAMPLE_CSV.")
                sample_path = candidates[0]
                break
            if root == base or root.parent == root or base not in root.parents:
                raise FileNotFoundError("sample_submission.csv missing. Set CFG.SAMPLE_CSV explicitly.")
            root = root.parent
        if cfg.DATA_DIR:
            root = base
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    images = sorted(p for p in root.rglob("*") if p.suffix.casefold() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(
            f"No extracted images under {root}. Add the competition data in Kaggle; "
            "if your own upload contains only a ZIP, extract it and set CFG.DATA_DIR."
        )
    print(f"Data root: {root}\nLabels: {label_path}\nSample: {sample_path}\nImage files: {len(images)}")
    print("Image path examples:", [str(p.relative_to(root)) for p in images[:12]])
    return root, label_path, sample_path, images


def canonical_image_id(value):
    name = str(value).strip().replace("\\", "/").split("/")[-1]
    return Path(name).stem.casefold() if Path(name).suffix.casefold() in IMAGE_EXTENSIONS else name.casefold()


def parse_submission_id(value):
    text = str(value)
    # Remove only a recognized suffix; retain underscores and leading zeroes in image names.
    match = re.search(r"(?i)(?:[_\-\s]+)?fovea[_\-\s]*([xy])\s*$", text)
    if match is None:
        match = re.search(r"(?i)[_\-\s]+([xy])\s*$", text)
    if match is None:
        raise ValueError(f"Unrecognized ImageID {text!r}; expected image name + Fovea_X/Y.")
    name = text[:match.start()].rstrip("_- ")
    if not name:
        raise ValueError(f"Empty image name in ImageID {text!r}")
    return canonical_image_id(name), match.group(1).lower()


class ImageIndex(defaultdict):
    def __init__(self, root=None):
        super().__init__(list)
        self.root = Path(root) if root is not None else None


def build_image_index(paths, root=None):
    index = ImageIndex(root)
    for path in paths:
        path = Path(path)
        index[canonical_image_id(path.name)].append(path)
    return index


def image_split_role(path, root=None):
    """Use the nearest explicit split directory, limited to the dataset root."""
    path = Path(path)
    if root is not None:
        try:
            parts = path.relative_to(root).parts[:-1]
        except ValueError:
            raise ValueError(f"Image path {path} is outside the selected dataset root {root}.")
    else:
        parts = path.parts[:-1]
    for part in reversed(parts):
        tokens = set(re.split(r"[^a-z0-9]+", part.casefold()))
        is_train = bool(tokens & {"train", "training"})
        is_test = bool(tokens & {"test", "testing"})
        if is_train or is_test:
            return "ambiguous" if is_train and is_test else ("train" if is_train else "test")
    filename_role = re.match(r"^(train(?:ing)?|test(?:ing)?)[_\-\s.]", path.stem.casefold())
    if filename_role:
        return "train" if filename_role.group(1).startswith("train") else "test"
    return None


def numeric_image_signature(value, csv_identifier=False):
    """Single numeric block only: never extract a patient/date's trailing number."""
    key = canonical_image_id(value)
    if csv_identifier and re.fullmatch(r"[0-9]+\.0+", key):
        key = key.split(".")[0]  # Spreadsheet-exported integer, e.g. '1.0'.
    match = re.fullmatch(r"([^0-9]*)([0-9]+)([^0-9]*)", key)
    if match is None:
        return None
    prefix, number, suffix = match.groups()
    return prefix, number.lstrip("0") or "0", suffix


def resolve_image(index, image_id, role):
    if role not in {"train", "test"}:
        raise ValueError(f"Unknown image role: {role}")
    root = getattr(index, "root", None)
    all_paths = list(dict.fromkeys(path for paths in index.values() for path in paths))
    roles = {path: image_split_role(path, root) for path in all_paths}

    def scope(paths):
        # Never use an explicitly opposite-split image, even if its name is exact.
        eligible = [path for path in paths if roles[path] in (None, role)]
        explicit = [path for path in eligible if roles[path] == role]
        return explicit if explicit else eligible

    key = canonical_image_id(image_id)
    matches = scope(index.get(key, []))
    stage = "exact filename"
    if not matches:
        signature = numeric_image_signature(image_id, csv_identifier=True)
        if signature is not None:
            prefix, number, suffix = signature
            eligible = scope(all_paths)
            signatures = {path: numeric_image_signature(path.name) for path in eligible}
            # Prefer the same textual prefix/suffix: '01' -> '001', 'eye_1' -> 'eye_001'.
            matches = [path for path, item in signatures.items() if item == (prefix, number, suffix)]
            stage = "same prefix/suffix with normalized numeric ID"
            if not matches and prefix == "" and suffix == "":
                # CSV '1' may identify 'image_001.jpg'. Require a unique single-number
                # filename in this split; 'patient12_eye1' is deliberately not guessed.
                matches = [path for path, item in signatures.items() if item is not None and item[1] == number]
                stage = "unique numeric ID in a single-number filename"
    if len(matches) != 1:
        examples = [str(path.relative_to(root) if root is not None else path) for path in scope(all_paths)[:12]]
        raise ValueError(
            f"{role} image {image_id!r}: expected one unambiguous file at {stage}, "
            f"found {[str(p) for p in matches]}. Available {role} examples: {examples}. "
            "Check the label/file naming convention; no positional or substring matching is used."
        )
    return matches[0]


def read_rgb(path):
    # PIL supports non-ASCII local paths; do not apply EXIF rotations without changing labels.
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def prepare_tables(cfg):
    root, label_path, sample_path, images = discover_data(cfg)
    labels, sample = csv_read(label_path), csv_read(sample_path)
    dc, xc, yc = [column_name(labels, n) for n in ("data", "Fovea_X", "Fovea_Y")]
    ic, vc = [column_name(sample, n) for n in ("ImageID", "value")]
    if list(sample.columns) != [ic, vc]:
        raise ValueError("Expected sample_submission to contain exactly ImageID and value.")
    index = build_image_index(images, root=root)
    train = pd.DataFrame({
        "image_id": labels[dc].map(canonical_image_id),
        "source_id": labels[dc],
        "x": pd.to_numeric(labels[xc], errors="raise"),
        "y": pd.to_numeric(labels[yc], errors="raise"),
    })
    if not len(train) or train.image_id.duplicated().any():
        raise ValueError("Training labels are empty or contain repeated image IDs.")
    if not np.isfinite(train[["x", "y"]].to_numpy(float)).all():
        raise ValueError("Training coordinates contain NaN or infinity.")
    train["visible"] = ~((train.x == 0) & (train.y == 0))
    # Internally all image coordinates are zero-based pixel centres. Preserve sentinel.
    train.loc[train.visible, ["x", "y"]] -= cfg.COORDINATE_ORIGIN
    train["path"] = [str(resolve_image(index, name, "train")) for name in train.source_id]
    if train.path.duplicated().any():
        raise ValueError(f"Different training IDs map to the same file:\n{train.loc[train.path.duplicated(False), ['source_id', 'path']]}")

    if sample[ic].duplicated().any():
        raise ValueError("Duplicate submission ImageIDs.")
    parsed = [parse_submission_id(v) for v in sample[ic]]
    layout = pd.DataFrame(parsed, columns=["image_id", "axis"])
    counts = layout.groupby(["image_id", "axis"]).size().unstack(fill_value=0)
    if set(counts.columns) != {"x", "y"} or not (counts[["x", "y"]] == 1).all().all():
        raise ValueError("Each test image must have exactly one X row and one Y row.")
    test_ids = list(dict.fromkeys(layout.image_id))
    test = pd.DataFrame({"image_id": test_ids})
    test["path"] = [str(resolve_image(index, name, "test")) for name in test_ids]
    if test.path.duplicated().any():
        raise ValueError(f"Different test IDs map to the same file:\n{test.loc[test.path.duplicated(False), ['image_id', 'path']]}")
    if set(train.path) & set(test.path):
        raise ValueError("A physical image is referenced by both train and test. Check ID mapping.")
    print("Training ID -> file examples:", list(zip(train.source_id.head(8), train.path.map(lambda p: Path(p).name).head(8))))
    print("Test ID -> file examples:", list(zip(test.image_id.head(8), test.path.map(lambda p: Path(p).name).head(8))))

    for frame, role in ((train, "train"), (test, "test")):
        sizes = []
        hashes = []
        for path in frame.path:
            rgb = read_rgb(path)
            h, w = rgb.shape[:2]
            if min(w, h) < 2:
                raise ValueError(f"Invalid image dimensions: {path}")
            sizes.append((w, h))
            hashes.append(hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest())
        frame[["width", "height"]] = pd.DataFrame(sizes, index=frame.index)
        frame["pixel_hash"] = hashes
        if role == "train":
            bad = frame.visible & ((frame.x < 0) | (frame.y < 0) |
                                   (frame.x > frame.width - 1) | (frame.y > frame.height - 1))
            if bad.any():
                raise ValueError(f"Visible labels outside zero-based image bounds:\n{frame.loc[bad]}")
    if set(train.pixel_hash) & set(test.pixel_hash):
        warnings.warn("Exact pixel duplicates exist between train and test; inspect dataset provenance.")
    if not train.visible.any():
        raise ValueError("All training labels are (0,0): cannot train a supervised localization model.")

    group_column = cfg.GROUP_COLUMN
    if group_column is None:
        group_column = next((c for c in labels if c.casefold() in {"patient_id", "subject_id", "patient", "subject"}), None)
    if group_column is not None:
        if group_column not in labels:
            raise ValueError(f"GROUP_COLUMN {group_column!r} missing from label CSV.")
        raw_groups = labels[group_column].str.strip()
        if (raw_groups == "").any():
            raise ValueError("Patient group column contains empty values.")
        train["patient_group"] = raw_groups
    else:
        train["patient_group"] = train.image_id
        warnings.warn("No patient IDs available: grouping exact duplicates only; patient independence is unverified.")
    train["group"] = connected_groups(train)
    duplicate_audit = []
    for pixel_hash, group in train.groupby("pixel_hash"):
        if len(group) < 2:
            continue
        points = group[["x", "y"]].to_numpy(float)
        maximum_distance = float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max())
        consistent_visibility = group.visible.nunique() == 1
        ids = group.image_id.tolist()
        duplicate_audit.append({
            "pixel_hash": pixel_hash,
            "image_ids": ",".join(ids),
            "count": len(group),
            "max_label_distance_px": maximum_distance,
            "visibility_consistent": consistent_visibility,
            "policy": "keep original rows and coordinates; force same validation fold",
        })
        if not consistent_visibility:
            raise ValueError(f"Identical pixels have conflicting visible/invisible labels: {ids}")
        if maximum_distance > cfg.DUPLICATE_LABEL_TOLERANCE_PX:
            raise ValueError(
                f"Identical pixels {ids} have label differences of {maximum_distance:.3f}px, "
                f"above DUPLICATE_LABEL_TOLERANCE_PX={cfg.DUPLICATE_LABEL_TOLERANCE_PX}. "
                "Inspect the annotations before changing the tolerance."
            )
        if maximum_distance > 0:
            warnings.warn(
                f"Duplicate images {ids} differ in point annotation by {maximum_distance:.3f}px. "
                "Keeping both ORIGINAL labels and placing them in the SAME fold; see duplicate_label_audit.csv."
            )
    print(f"Train={len(train)}, test={len(test)}, submission rows={len(sample)}")
    print(f"Visible={int(train.visible.sum())}, invisible={int((~train.visible).sum())}, groups={train.group.nunique()}")
    if len(train) != 80 or len(test) != 20:
        warnings.warn("Counts differ from the published 80 train / 20 test. Verify the selected data.")
    return train, test, sample, layout, {"root": str(root), "label_csv": str(label_path),
                                      "sample_csv": str(sample_path), "id_column": ic, "value_column": vc,
                                      "duplicate_label_audit": duplicate_audit}


def connected_groups(frame):
    # Union both patient and duplicate relationships; neither may cross a fold.
    parent = list(range(len(frame)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for column in ("patient_group", "pixel_hash"):
        first = {}
        for i, key in enumerate(frame[column]):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    return [f"group_{find(i):04d}" for i in range(len(frame))]


def assign_folds(train, cfg):
    frame = train.copy()
    n_folds = min(cfg.N_FOLDS, frame.group.nunique())
    if n_folds < 2:
        raise ValueError("At least two independent groups are required for cross-validation.")
    class_groups = frame.groupby("visible").group.nunique()
    if len(class_groups) == 2 and class_groups.min() >= 2:
        n_folds = min(n_folds, int(class_groups.min()))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
        splits = splitter.split(frame, frame.visible.astype(int), frame.group)
    else:
        # Randomize group labels before deterministic GroupKFold assignment.
        names = frame.group.unique().copy()
        np.random.default_rng(cfg.SEED).shuffle(names)
        group_numbers = frame.group.map({name: i for i, name in enumerate(names)})
        splits = GroupKFold(n_splits=n_folds).split(frame, groups=group_numbers)
        if len(class_groups) == 2:
            warnings.warn("Too few independent invisible/visible groups for stratification; visibility CV is unreliable.")
    frame["fold"] = -1
    for fold, (tr, va) in enumerate(splits):
        if set(frame.iloc[tr].group) & set(frame.iloc[va].group):
            raise AssertionError("Group leakage in fold assignment.")
        if not frame.iloc[tr].visible.any():
            raise ValueError("A training fold has no visible examples; reduce folds or inspect group labels.")
        frame.loc[frame.index[va], "fold"] = fold
    assert (frame.fold >= 0).all()
    print(frame.groupby("fold").agg(images=("image_id", "size"), visible=("visible", "sum")))
    return frame


def letterbox_geometry(width, height, size):
    factor = min(size / width, size / height)
    new_width = min(size, max(1, int(round(width * factor))))
    new_height = min(size, max(1, int(round(height * factor))))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    sx, sy = new_width / width, new_height / height
    # OpenCV resize maps pixel centres: dst = scale*(src+0.5)-0.5.
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2],
                          [0, 0, 1]], dtype=np.float64)
    return new_width, new_height, left, top, transform


def letterbox(image, size):
    height, width = image.shape[:2]
    nw, nh, left, top, transform = letterbox_geometry(width, height, size)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if nw < width else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, transform


def gaussian_heatmap(xy, size, sigma):
    x, y = np.asarray(xy, dtype=np.float32) * (size - 1)
    axis = np.arange(size, dtype=np.float32)
    values = np.exp(-((axis[None, :] - x)**2 + (axis[:, None] - y)**2) / (2 * sigma**2))
    values /= max(float(values.sum()), 1e-12)
    return values[None].astype(np.float32)


class FoveaDataset(Dataset):
    def __init__(self, frame, cfg, training=False):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = read_rgb(row.path)
        height, width = image.shape[:2]
        size = self.cfg.INPUT_SIZE
        image, transform = letterbox(image, size)
        has_label = "visible" in self.frame.columns
        visible = bool(row.visible) if has_label else False
        original_xy = np.array([row.x, row.y], dtype=np.float64) if has_label else np.zeros(2)
        point = (transform @ np.r_[original_xy, 1.0])[:2] if visible else np.zeros(2)

        if self.training:
            # Retry if a visible point leaves the input; never relabel it invisible.
            for _ in range(8):
                rotate = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2),
                                                np.random.uniform(-15, 15),
                                                np.random.uniform(0.93, 1.07))
                rotate[:, 2] += np.random.uniform(-0.03, 0.03, 2) * size
                aug = np.vstack([rotate, [0, 0, 1]])
                if np.random.random() < 0.5:
                    flip = np.array([[-1, 0, size - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
                    aug = flip @ aug
                candidate = (aug @ np.r_[point, 1.0])[:2] if visible else point
                if not visible or np.all((candidate >= 0) & (candidate <= size - 1)):
                    image = cv2.warpAffine(image, aug[:2], (size, size), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    transform = aug @ transform
                    point = candidate
                    break
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-10, 10)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        xy = (point / (size - 1)).astype(np.float32) if visible else np.zeros(2, dtype=np.float32)
        target = gaussian_heatmap(xy, self.cfg.HEATMAP_SIZE, self.cfg.HEATMAP_SIGMA) if visible else \
            np.zeros((1, self.cfg.HEATMAP_SIZE, self.cfg.HEATMAP_SIZE), dtype=np.float32)
        image = image.astype(np.float32) / 255
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target_heatmap": torch.from_numpy(target),
            "target_xy": torch.from_numpy(xy),
            "visible": torch.tensor(float(visible), dtype=torch.float32),
            "orig_wh": torch.tensor([width, height], dtype=torch.float32),
            "input_to_orig": torch.from_numpy(np.linalg.inv(transform)[:2].astype(np.float32)),
            "image_id": str(row.image_id),
        }
        if has_label:
            item["x_true"] = torch.tensor(float(row.x), dtype=torch.float32)
            item["y_true"] = torch.tensor(float(row.y), dtype=torch.float32)
        return item


def make_loader(frame, cfg, training=False, seed_offset=0):
    generator = torch.Generator().manual_seed(cfg.SEED + seed_offset)
    return DataLoader(FoveaDataset(frame, cfg, training), batch_size=cfg.BATCH_SIZE,
                      shuffle=training, num_workers=cfg.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


"""Model and training helpers. Concatenate after the notebook config/data cells."""
import math
import socket
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_RESNET18_STATE = None
_RESNET18_DOWNLOAD_FAILED = False


def _load_torch_file(path, map_location="cpu"):
    # Our checkpoints contain tensors and primitive values, not pickled modules.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch installations.
        return torch.load(path, map_location=map_location)


def build_resnet18(pretrained=True, allow_download=True, weights_path=None):
    """Use explicit local weights/cache first; Internet is optional.

    Kaggle Internet=Off users can attach a dataset containing the official
    resnet18-f37072fd.pth and supply its path as weights_path. A failed download
    produces a clearly reported random-initialization fallback.
    """
    global _RESNET18_STATE, _RESNET18_DOWNLOAD_FAILED
    from torchvision.models import ResNet18_Weights, resnet18

    encoder = resnet18(weights=None)
    if not pretrained:
        return encoder
    if weights_path:
        explicit_path = Path(weights_path)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Pretrained weights do not exist: {explicit_path}")
        state = _load_torch_file(explicit_path)
        encoder.load_state_dict(state, strict=True)
        return encoder
    if _RESNET18_STATE is not None:
        encoder.load_state_dict(_RESNET18_STATE, strict=True)
        return encoder

    weights = ResNet18_Weights.DEFAULT
    cache_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    )
    try:
        if cache_path.is_file():
            state = _load_torch_file(cache_path)
        elif allow_download and not _RESNET18_DOWNLOAD_FAILED:
            # torchvision delegates to urllib without exposing its timeout.
            # Restrict only this synchronous download; restore the caller's
            # socket default even when a timeout or checksum error is raised.
            previous_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(20.0)
                state = weights.get_state_dict(progress=True, check_hash=True)
            finally:
                socket.setdefaulttimeout(previous_timeout)
        else:
            reason = "downloads disabled" if not allow_download else "previous download failed"
            warnings.warn(
                f"ResNet18 pretrained weights unavailable ({reason}); "
                "using random initialization. This may reduce accuracy on 80 images."
            )
            return encoder
        encoder.load_state_dict(state, strict=True)
        _RESNET18_STATE = state
    except Exception as exc:
        _RESNET18_DOWNLOAD_FAILED = True
        warnings.warn(
            f"Could not load ResNet18 pretrained weights: {type(exc).__name__}: {exc}. "
            "Using random initialization. Attach official weights for offline use."
        )
    return encoder


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvGNAct(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def heatmap_to_coords(logits):
    """Spatial softmax expectation in [0,1], retaining subpixel coordinates.

    A multi-peak heatmap can place its expectation between peaks; inspect OOF
    errors before changing the decoder or introducing a refinement model.
    """
    batch_size, _, height, width = logits.shape
    probabilities = logits.float().flatten(1).softmax(dim=1).reshape(
        batch_size, height, width
    )
    xs = torch.linspace(0, 1, width, device=logits.device, dtype=torch.float32)
    ys = torch.linspace(0, 1, height, device=logits.device, dtype=torch.float32)
    x = (probabilities.sum(dim=1) * xs).sum(dim=1)
    y = (probabilities.sum(dim=2) * ys).sum(dim=1)
    return torch.stack([x, y], dim=1)


class FoveaNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        allow_download=True,
        use_visibility=True,
        heatmap_size=128,
        freeze_bn=True,
        pretrained_weights_path=None,
    ):
        super().__init__()
        self.encoder = build_resnet18(
            pretrained=pretrained,
            allow_download=allow_download,
            weights_path=pretrained_weights_path,
        )
        self.encoder.fc = nn.Identity()
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )
        self.use_visibility = bool(use_visibility)
        self.freeze_bn = bool(freeze_bn)
        self.heatmap_size = int(heatmap_size)
        if not self.use_visibility:
            for parameter in self.visibility_head.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            # Small folds/batches give poor running-statistic estimates.
            # Affine BN weights still receive gradients.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(image)))
        c1 = e.layer1(e.maxpool(x))
        c2 = e.layer2(c1)
        c3 = e.layer3(c2)
        c4 = e.layer4(c3)
        decoded = self.decoder3(c4, c3)
        decoded = self.decoder2(decoded, c2)
        decoded = self.decoder1(decoded, c1)
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(
                logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_visibility:
            visibility_logits = self.visibility_head(c4).squeeze(1)
        else:
            # Disabled only when the training set is all visible. The data
            # validation/orchestrator rejects an all-invisible training fold.
            visibility_logits = logits.new_zeros(image.shape[0])
        return {
            "logits": logits,
            "coords": heatmap_to_coords(logits),
            "visibility_logits": visibility_logits,
        }


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def fovea_loss(output, batch, cfg, use_visibility):
    """Mean KL + original-pixel coordinate MSE + optional visibility BCE."""
    target_heatmap = batch["target_heatmap"].float()
    if target_heatmap.ndim == 3:
        target_heatmap = target_heatmap.unsqueeze(1)
    if target_heatmap.shape != output["logits"].shape:
        raise ValueError(
            f"Heatmap shape mismatch: target={target_heatmap.shape}, "
            f"prediction={output['logits'].shape}"
        )
    visible = batch["visible"].float().reshape(-1)
    n_visible = visible.sum().clamp_min(1.0)
    target_prob = target_heatmap.flatten(1)
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = output["logits"].float().flatten(1).log_softmax(dim=1)
    per_image_kl = (
        target_prob * (target_prob.clamp_min(1e-12).log() - log_prob)
    ).sum(dim=1)
    heatmap_loss = (per_image_kl * visible).sum() / n_visible

    # Convert input-coordinate differences to original pixels using the same
    # affine geometry as inference. No division by each image's own dimensions:
    # a 20-pixel error has the same loss on small and large originals.
    image_height, image_width = batch["image"].shape[-2:]
    input_extent = output["coords"].new_tensor([image_width - 1, image_height - 1])
    delta_input = (output["coords"] - batch["target_xy"].float()) * input_extent
    inverse_linear = batch["input_to_orig"].float()[:, :, :2]
    delta_original = torch.bmm(inverse_linear, delta_input.unsqueeze(-1)).squeeze(-1)
    per_image_coordinate = (
        delta_original / float(getattr(cfg, "COORD_SCALE", 512.0))
    ).square().mean(dim=1)
    coordinate_loss = (per_image_coordinate * visible).sum() / n_visible

    if use_visibility:
        visibility_loss = F.binary_cross_entropy_with_logits(
            output["visibility_logits"].float(), visible
        )
    else:
        visibility_loss = output["logits"].float().sum() * 0.0
    total = (
        float(getattr(cfg, "HEATMAP_LOSS_WEIGHT", 1.0)) * heatmap_loss
        + float(getattr(cfg, "COORD_LOSS_WEIGHT", 10.0)) * coordinate_loss
        + float(getattr(cfg, "VIS_LOSS_WEIGHT", 1.0)) * visibility_loss
    )
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "visibility": float(visibility_loss.detach()),
    }


def normalized_to_original(coords, batch, clamp=True):
    """Invert the recorded input-pixel transform, including resize offsets."""
    image_height, image_width = batch["image"].shape[-2:]
    extent = coords.new_tensor([image_width - 1, image_height - 1])
    input_pixels = coords.float() * extent
    homogeneous = torch.cat([input_pixels, torch.ones_like(input_pixels[:, :1])], dim=1)
    original = torch.bmm(
        batch["input_to_orig"].float(), homogeneous.unsqueeze(-1)
    ).squeeze(-1)
    if clamp:
        upper = (batch["orig_wh"].float() - 1.0).clamp_min(0.0)
        original = torch.minimum(original.clamp_min(0.0), upper)
    return original


def _autocast_context(device, cfg):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(getattr(cfg, "AMP", True)) and device.type == "cuda",
    )


def _prediction_rows(batch, original_coords, visibility_probability):
    coordinates = original_coords.detach().cpu().numpy()
    probabilities = visibility_probability.detach().cpu().numpy()
    ids = batch["image_id"]
    rows = []
    for i in range(len(coordinates)):
        row = {
            "image_id": str(ids[i]),
            "x": float(coordinates[i, 0]),
            "y": float(coordinates[i, 1]),
            "visibility_prob": float(probabilities[i]),
        }
        for field, destination in (
            ("visible", "visible_true"),
            ("x_true", "x_true"),
            ("y_true", "y_true"),
        ):
            if field in batch:
                value = batch[field][i]
                row[destination] = float(value.item() if torch.is_tensor(value) else value)
        rows.append(row)
    return rows


@torch.inference_mode()
def predict_loader(model, loader, device, cfg, tta=False):
    """Return raw original-coordinate predictions; threshold only after fusion.

    Horizontal TTA flips the full input canvas. Undo that flip in input space,
    then invert the affine transform for each view before averaging.
    """
    device = torch.device(device)
    model.eval()
    rows = []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        original_views = [normalized_to_original(output["coords"], batch, clamp=False)]
        probability_views = [
            output["visibility_logits"].float().sigmoid()
            if model.use_visibility
            else torch.ones(batch["image"].shape[0], device=device)
        ]
        if tta:
            with _autocast_context(device, cfg):
                flipped = model(torch.flip(batch["image"], dims=[-1]))
            corrected_coords = flipped["coords"].clone()
            corrected_coords[:, 0] = 1.0 - corrected_coords[:, 0]
            original_views.append(normalized_to_original(corrected_coords, batch, clamp=False))
            probability_views.append(
                flipped["visibility_logits"].float().sigmoid()
                if model.use_visibility
                else torch.ones(batch["image"].shape[0], device=device)
            )
        original = torch.stack(original_views).mean(dim=0)
        upper = (batch["orig_wh"].float() - 1.0).clamp_min(0.0)
        original = torch.minimum(original.clamp_min(0.0), upper)
        probability = torch.stack(probability_views).mean(dim=0)
        rows.extend(_prediction_rows(batch, original, probability))
    return rows


def prediction_rmse(rows, visibility_threshold=0.5, coordinate_origin=0):
    prediction = np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float64)
    truth = np.asarray([[row["x_true"], row["y_true"]] for row in rows], dtype=np.float64)
    probability = np.asarray([row["visibility_prob"] for row in rows], dtype=np.float64)
    predicted_visible = probability >= float(visibility_threshold)
    if coordinate_origin:
        if any("visible_true" not in row for row in rows):
            raise ValueError("coordinate_origin=1 requires each row's visible_true label.")
        true_visible = np.asarray([row["visible_true"] for row in rows], dtype=bool)
        # All internal coordinates are zero-based. Convert visible coordinates
        # only; the invisible sentinel remains (0, 0) in either convention.
        truth[true_visible] += float(coordinate_origin)
        prediction[predicted_visible] += float(coordinate_origin)
    prediction[~predicted_visible] = 0.0
    # Mean over all 2*N scalar coordinates, matching the long submission table.
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


@torch.inference_mode()
def evaluate_model(model, loader, device, cfg):
    model.eval()
    loss_sum, image_count, rows = 0.0, 0, []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        loss, _ = fovea_loss(output, batch, cfg, model.use_visibility)
        batch_size = batch["image"].shape[0]
        loss_sum += float(loss) * batch_size
        image_count += batch_size
        original = normalized_to_original(output["coords"], batch)
        probability = (
            output["visibility_logits"].float().sigmoid()
            if model.use_visibility
            else torch.ones(batch_size, device=device)
        )
        rows.extend(_prediction_rows(batch, original, probability))
    if image_count == 0:
        raise ValueError("Validation loader is empty.")
    score = prediction_rmse(
        rows,
        visibility_threshold=getattr(cfg, "VISIBILITY_THRESHOLD", 0.5),
        coordinate_origin=getattr(cfg, "COORDINATE_ORIGIN", 0),
    )
    return loss_sum / image_count, score


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = _load_torch_file(checkpoint_path, map_location="cpu")
    if bool(checkpoint.get("use_visibility", model.use_visibility)) != model.use_visibility:
        raise ValueError("Checkpoint and model use_visibility settings differ.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint


def fit_fold(model, train_loader, val_loader, cfg, device, fold, checkpoint_path):
    """Train one fold, checkpoint minimum fixed-threshold RMSE, restore best."""
    device = torch.device(device)
    model.to(device)
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; do not use drop_last=True on tiny folds.")
    epochs = int(getattr(cfg, "EPOCHS", 80))
    if epochs < 1:
        raise ValueError("EPOCHS must be >= 1.")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(getattr(cfg, "LR", 2e-4)),
        weight_decay=float(getattr(cfg, "WEIGHT_DECAY", 1e-4)),
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = min(
        int(getattr(cfg, "WARMUP_EPOCHS", 3)) * len(train_loader),
        max(0, total_steps - 1),
    )

    def learning_rate_factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    amp_enabled = bool(getattr(cfg, "AMP", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    patience = int(getattr(cfg, "PATIENCE", 12))
    best_score, best_epoch, bad_epochs = math.inf, -1, 0
    history = []
    for epoch in range(epochs):
        start_time = time.perf_counter()
        model.train()
        total_loss, image_count = 0.0, 0
        component_sums = {"heatmap": 0.0, "coordinate": 0.0, "visibility": 0.0}
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        for cpu_batch in train_loader:
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                output = model(batch["image"])
            # Spatial probabilities and losses are explicitly float32.
            loss, components = fovea_loss(output, batch, cfg, model.use_visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in fold={fold}, epoch={epoch + 1}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(getattr(cfg, "GRAD_CLIP", 1.0))
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # A scale reduction means AMP skipped the optimizer update.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            batch_size = batch["image"].shape[0]
            total_loss += float(loss.detach()) * batch_size
            image_count += batch_size
            for key, value in components.items():
                component_sums[key] += value * batch_size

        validation_loss, validation_rmse = evaluate_model(model, val_loader, device, cfg)
        if not math.isfinite(validation_rmse):
            raise FloatingPointError(f"Non-finite validation RMSE in fold={fold}.")
        improved = validation_rmse < best_score
        if improved:
            best_score, best_epoch, bad_epochs = validation_rmse, epoch + 1, 0
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": best_epoch,
                    "rmse": float(best_score),
                    "use_visibility": bool(model.use_visibility),
                    "heatmap_size": int(model.heatmap_size),
                    "input_size": int(getattr(cfg, "INPUT_SIZE", 512)),
                    "fold": int(fold),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        record = {
            "fold": int(fold),
            "epoch": epoch + 1,
            "train_loss": total_loss / image_count,
            "val_loss": validation_loss,
            "val_rmse": validation_rmse,
            "learning_rate": epoch_learning_rate,
            "seconds": time.perf_counter() - start_time,
            **{f"train_{key}": value / image_count for key, value in component_sums.items()},
        }
        history.append(record)
        print(
            f"fold {fold} | epoch {epoch + 1:03d}/{epochs} | "
            f"train {record['train_loss']:.4f} | val {validation_loss:.4f} | "
            f"RMSE {validation_rmse:.3f}px | best {best_score:.3f}px | "
            f"{record['seconds']:.1f}s" + (" *" if improved else ""),
            flush=True,
        )
        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.", flush=True)
            break
    load_model_checkpoint(model, checkpoint_path, device)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_rmse": best_score,
        "checkpoint_path": str(checkpoint_path),
    }


PIPELINE_VERSION = "2026task3-fovea-1.2-real-data"


def json_save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def audit_xml_boxes(train, root, output_dir, coordinate_origin=0):
    """Read optional Pascal VOC XML for audit only; CSV points remain the targets."""
    import xml.etree.ElementTree as ET
    xml_paths = sorted(Path(root).rglob("*.xml"))
    rows, errors = [], []
    train_index = build_image_index([Path(path) for path in train.path], root=root)
    path_to_id = dict(zip(train.path, train.image_id))
    for path in xml_paths:
        try:
            tree = ET.parse(path).getroot()
            if image_split_role(path, root) == "test":
                continue
            image_path = resolve_image(train_index, tree.findtext("filename") or path.stem, "train")
            key = path_to_id[str(image_path)]
            for obj in tree.findall(".//object"):
                box = obj.find("bndbox")
                if box is None:
                    continue
                values = [float(box.findtext(name)) - coordinate_origin
                          for name in ("xmin", "ymin", "xmax", "ymax")]
                if not np.isfinite(values).all() or values[2] < values[0] or values[3] < values[1]:
                    raise ValueError("Invalid bounding box")
                rows.append({"image_id": key, "class": obj.findtext("name") or "",
                             "xmin": values[0], "ymin": values[1],
                             "xmax": values[2], "ymax": values[3], "xml_path": str(path)})
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    columns = ["image_id", "class", "xmin", "ymin", "xmax", "ymax", "xml_path"]
    boxes = pd.DataFrame(rows, columns=columns)
    if len(boxes):
        boxes = boxes.merge(train[["image_id", "x", "y", "visible"]], on="image_id", validate="many_to_one")
        boxes["point_inside"] = boxes.visible & boxes.x.between(boxes.xmin, boxes.xmax) & boxes.y.between(boxes.ymin, boxes.ymax)
        boxes["box_center_distance_px"] = np.where(
            boxes.visible,
            np.hypot(boxes.x - (boxes.xmin + boxes.xmax) / 2,
                     boxes.y - (boxes.ymin + boxes.ymax) / 2), np.nan)
    boxes.to_csv(Path(output_dir) / "xml_box_audit.csv", index=False)
    json_save(errors, Path(output_dir) / "xml_parse_warnings.json")
    print(f"XML audit: {len(xml_paths)} files, {len(boxes)} boxes, {len(errors)} parse warnings.")
    return boxes


def plot_label_audit(train, boxes, output_dir, show=True):
    from matplotlib.patches import Rectangle
    for start in range(0, len(train), 16):
        block = train.iloc[start:start + 16]
        fig, axes = plt.subplots(math.ceil(len(block) / 4), 4, figsize=(16, 4 * math.ceil(len(block) / 4)),
                                 squeeze=False)
        for axis, (_, row) in zip(axes.flat, block.iterrows()):
            axis.imshow(read_rgb(row.path))
            if row.visible:
                axis.scatter(row.x, row.y, marker="+", s=120, color="lime", linewidths=2)
            if len(boxes):
                for _, box in boxes[boxes.image_id == row.image_id].iterrows():
                    axis.add_patch(Rectangle((box.xmin, box.ymin), box.xmax - box.xmin,
                                             box.ymax - box.ymin, fill=False, edgecolor="yellow"))
            axis.set_title(f"{row.image_id} | {'visible' if row.visible else '(0,0) invisible'}", fontsize=9)
            axis.axis("off")
        for axis in list(axes.flat)[len(block):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"labels_{start:03d}.png", dpi=120, bbox_inches="tight")
        if show and start == 0:
            plt.show()
        plt.close(fig)


def apply_visibility(frame, threshold, coordinate_origin=0):
    prediction = frame[["x", "y"]].to_numpy(dtype=np.float64).copy()
    keep = frame.visibility_prob.to_numpy(float) >= threshold
    prediction[keep] += coordinate_origin
    prediction[~keep] = 0.0
    return prediction


def truth_in_submission_coordinates(frame, coordinate_origin=0):
    truth = frame[["x_true", "y_true"]].to_numpy(dtype=np.float64).copy()
    truth[frame.visible_true.to_numpy(bool)] += coordinate_origin
    return truth


def score_frame(frame, threshold, coordinate_origin=0):
    return float(np.sqrt(np.mean((apply_visibility(frame, threshold, coordinate_origin) -
                                  truth_in_submission_coordinates(frame, coordinate_origin))**2)))


def choose_threshold(oof, cfg, visibility_supported):
    baseline = score_frame(oof, cfg.VISIBILITY_THRESHOLD, cfg.COORDINATE_ORIGIN)
    threshold = cfg.VISIBILITY_THRESHOLD
    results = {"fixed_threshold": threshold, "oof_rmse_fixed_threshold": baseline,
               "selected_threshold": threshold, "threshold_tuned": False,
               "score_note": "Development CV estimate: checkpoints selected on held-out fold RMSE."}
    if not cfg.TUNE_VISIBILITY_THRESHOLD:
        return threshold, results
    counts = oof.visible_true.astype(bool).value_counts()
    if not visibility_supported or len(counts) != 2 or counts.min() < 5:
        warnings.warn("Threshold tuning skipped: require at least five examples of each visibility class.")
        return threshold, results
    candidates = sorted(set(np.linspace(0.0, 0.95, 20).tolist() + [cfg.VISIBILITY_THRESHOLD]))
    table = [{"threshold": float(t), "rmse": score_frame(oof, t, cfg.COORDINATE_ORIGIN)}
             for t in candidates]
    winner = min(table, key=lambda row: (row["rmse"], abs(row["threshold"] - cfg.VISIBILITY_THRESHOLD)))
    threshold = winner["threshold"]
    results.update({"selected_threshold": threshold, "threshold_tuned": True,
                    "oof_rmse_threshold_tuned_optimistic": winner["rmse"],
                    "threshold_search": table,
                    "threshold_note": "Threshold selected on these OOF labels; tuned score is optimistic, not independent validation."})
    return threshold, results


def write_submission(sample, layout, prediction_frame, metadata, cfg, threshold):
    if prediction_frame.image_id.duplicated().any():
        raise ValueError("Duplicate predictions for a test image.")
    if set(prediction_frame.image_id) != set(layout.image_id):
        raise ValueError("Predicted image IDs do not match the sample submission.")
    prediction = apply_visibility(prediction_frame, threshold, cfg.COORDINATE_ORIGIN)
    mapping = {key: point for key, point in zip(prediction_frame.image_id, prediction)}
    values = [mapping[row.image_id][0 if row.axis == "x" else 1] for row in layout.itertuples()]
    result = sample.copy()
    result[metadata["value_column"]] = np.asarray(values, dtype=np.float64)
    assert len(result) == len(sample)
    assert result[metadata["id_column"]].equals(sample[metadata["id_column"]])
    if not np.isfinite(result[metadata["value_column"]]).all():
        raise ValueError("Submission contains non-finite values.")
    for row, point in zip(prediction_frame.itertuples(), prediction):
        if np.all(point == 0):
            continue
        low = cfg.COORDINATE_ORIGIN
        upper = np.array([row.width - 1, row.height - 1]) + low
        if np.any(point < low - 1e-6) or np.any(point > upper + 1e-6):
            raise ValueError(f"Out-of-bounds prediction: {row.image_id} -> {point}")
    path = Path(cfg.SUBMISSION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.6f")
    # Round-trip through disk: verify actual exported IDs and numeric values.
    reread = pd.read_csv(path, dtype={metadata["id_column"]: str}, keep_default_na=False)
    assert reread[metadata["id_column"]].tolist() == sample[metadata["id_column"]].tolist()
    assert np.isfinite(pd.to_numeric(reread[metadata["value_column"]], errors="raise")).all()
    print(f"Submission saved: {path} ({len(result)} rows)")
    print(result.head(8).to_string(index=False))
    return result


def plot_oof_errors(oof, train, threshold, cfg, output_dir):
    points = apply_visibility(oof, threshold, coordinate_origin=0)
    scored_points = apply_visibility(oof, threshold, cfg.COORDINATE_ORIGIN)
    scored_truth = truth_in_submission_coordinates(oof, cfg.COORDINATE_ORIGIN)
    errors = np.linalg.norm(scored_points - scored_truth, axis=1)
    report = oof.copy()
    report["pred_x"] = points[:, 0]
    report["pred_y"] = points[:, 1]
    report["euclidean_error_px"] = errors
    report = report.merge(train[["image_id", "path"]], on="image_id", validate="one_to_one")
    report.to_csv(Path(output_dir) / "oof_error_details.csv", index=False)
    worst = report.nlargest(min(12, len(report)), "euclidean_error_px")
    fig, axes = plt.subplots(math.ceil(len(worst) / 4), 4, figsize=(16, 4 * math.ceil(len(worst) / 4)), squeeze=False)
    for axis, row in zip(axes.flat, worst.itertuples()):
        axis.imshow(read_rgb(row.path))
        if row.visible_true:
            axis.scatter(row.x_true, row.y_true, marker="+", color="lime", s=120, label="truth")
        if row.visibility_prob >= threshold:
            axis.scatter(row.pred_x, row.pred_y, marker="x", color="red", s=90, label="prediction")
        axis.set_title(f"{row.image_id} | error {row.euclidean_error_px:.1f}px\np_visible={row.visibility_prob:.2f}", fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[len(worst):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "oof_worst_cases.png", dpi=120, bbox_inches="tight")
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return report


def run_pipeline(config):
    cfg = replace(config)
    if cfg.SMOKE_TEST:
        cfg.N_FOLDS, cfg.EPOCHS, cfg.PATIENCE = 2, 2, 2
        cfg.INPUT_SIZE, cfg.HEATMAP_SIZE, cfg.BATCH_SIZE = 128, 32, 4
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / "smoke_test")
        cfg.SUBMISSION_PATH = str(Path(cfg.SUBMISSION_PATH).with_name("submission_SMOKE_TEST.csv"))
        warnings.warn("SMOKE_TEST is for pipeline checks only. Its CSV is not a trained competition solution.")
    validate_config(cfg)
    seed_everything(cfg.SEED)
    cv2.setNumThreads(0)
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch {torch.__version__}")
    if device.type == "cpu" and not cfg.SMOKE_TEST:
        warnings.warn("No GPU detected. Enable a Kaggle GPU accelerator before full 5-fold training.")
    train, test, sample, layout, metadata = prepare_tables(cfg)
    mapping = pd.concat([
        train[["image_id", "path"]].assign(split="train"),
        test[["image_id", "path"]].assign(split="test"),
    ], ignore_index=True)
    mapping.to_csv(output_dir / "filename_mapping.csv", index=False)
    pd.DataFrame(metadata["duplicate_label_audit"], columns=[
        "pixel_hash", "image_ids", "count", "max_label_distance_px", "visibility_consistent", "policy"
    ]).to_csv(output_dir / "duplicate_label_audit.csv", index=False)
    train = assign_folds(train, cfg)

    # Use one consistent gate policy across folds. One isolated negative cannot
    # support a learnable/evaluable visibility head in every training fold.
    visibility_supported = bool((~train.visible).any()) and all(
        train.loc[train.fold != fold, "visible"].nunique() == 2 for fold in sorted(train.fold.unique()))
    if not visibility_supported:
        warnings.warn("Visibility gating disabled for ALL folds: insufficient invisible training examples. "
                      "The model cannot establish reliable detection of an unseen invisible class.")
    elif min(int(train.visible.sum()), int((~train.visible).sum())) < 5:
        warnings.warn("Visibility head enabled, but class counts are very small; assess its OOF failures carefully.")

    signature_config = asdict(cfg)
    for key in ("REUSE_CHECKPOINTS", "SHOW_PLOTS", "NUM_WORKERS"):
        signature_config.pop(key, None)
    fingerprint = {"version": PIPELINE_VERSION, "config": signature_config,
                   "training": train[["image_id", "pixel_hash", "x", "y", "fold", "group"]].to_dict("records"),
                   "visibility_supported": visibility_supported}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    json_save({"config": asdict(cfg), "metadata": metadata, "signature": signature,
               "visibility_supported": visibility_supported, "pipeline_version": PIPELINE_VERSION},
              output_dir / "run_config.json")
    json_save({"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
               "pandas": pd.__version__, "opencv": cv2.__version__, "platform": platform.platform(),
               "device": str(device)}, output_dir / "environment.json")
    train.to_csv(output_dir / "folds.csv", index=False)
    test.to_csv(output_dir / "test_manifest.csv", index=False)
    boxes = audit_xml_boxes(train, metadata["root"], output_dir, cfg.COORDINATE_ORIGIN)
    plot_label_audit(train, boxes, output_dir, show=cfg.SHOW_PLOTS)

    all_oof, all_test, fold_summary = [], [], []
    for fold in sorted(train.fold.unique()):
        fold = int(fold)
        seed_everything(cfg.SEED + fold)
        training = train[train.fold != fold].reset_index(drop=True)
        validation = train[train.fold == fold].reset_index(drop=True)
        checkpoint_path = output_dir / f"fold_{fold}_best.pt"
        can_reuse = cfg.REUSE_CHECKPOINTS and checkpoint_path.is_file()
        if can_reuse:
            old = _load_torch_file(checkpoint_path)
            if old.get("run_signature") != signature:
                raise ValueError(f"{checkpoint_path}: checkpoint/data/config mismatch. Disable REUSE_CHECKPOINTS.")
            del old
        model = FoveaNet(
            pretrained=cfg.PRETRAINED and not can_reuse,
            allow_download=cfg.ALLOW_WEIGHT_DOWNLOAD,
            use_visibility=visibility_supported,
            heatmap_size=cfg.HEATMAP_SIZE,
            freeze_bn=cfg.FREEZE_BN,
            pretrained_weights_path=cfg.PRETRAINED_WEIGHTS_PATH,
        ).to(device)
        validation_loader = make_loader(validation, cfg)
        if can_reuse:
            checkpoint = load_model_checkpoint(model, checkpoint_path, device)
            training_result = {"best_epoch": checkpoint["epoch"], "best_rmse": checkpoint["rmse"]}
            print(f"Reusing fold {fold}: epoch {checkpoint['epoch']}, RMSE {checkpoint['rmse']:.3f}")
            del checkpoint
        else:
            training_loader = make_loader(training, cfg, training=True, seed_offset=fold)
            training_result = fit_fold(model, training_loader, validation_loader, cfg, device, fold, checkpoint_path)
            pd.DataFrame(training_result["history"]).to_csv(output_dir / f"fold_{fold}_history.csv", index=False)
            checkpoint = _load_torch_file(checkpoint_path)
            checkpoint["run_signature"] = signature
            checkpoint["coordinate_origin"] = cfg.COORDINATE_ORIGIN
            torch.save(checkpoint, checkpoint_path)
            del checkpoint, training_loader

        # fit_fold restores the best checkpoint; reuse branch loads it explicitly.
        oof = pd.DataFrame(predict_loader(model, validation_loader, device, cfg, tta=cfg.TTA))
        oof["fold"] = fold
        oof.to_csv(output_dir / f"fold_{fold}_oof.csv", index=False)
        all_oof.append(oof)
        test_predictions = pd.DataFrame(predict_loader(model, make_loader(test, cfg), device, cfg, tta=cfg.TTA))
        test_predictions["fold"] = fold
        test_predictions.to_csv(output_dir / f"fold_{fold}_test.csv", index=False)
        all_test.append(test_predictions)
        fold_summary.append({"fold": fold, "best_epoch": training_result["best_epoch"],
                             "checkpoint_rmse_no_tta": training_result["best_rmse"],
                             "oof_rmse_with_configured_tta": score_frame(oof, cfg.VISIBILITY_THRESHOLD, cfg.COORDINATE_ORIGIN)})
        del model, validation_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(all_oof, ignore_index=True)
    assert len(oof) == len(train) and oof.image_id.nunique() == len(train)
    oof = oof.set_index("image_id").loc[train.image_id].reset_index()
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_summary).to_csv(output_dir / "fold_summary.csv", index=False)
    threshold, scores = choose_threshold(oof, cfg, visibility_supported)
    scores["n_folds"] = len(fold_summary)
    scores["folds"] = fold_summary
    scores["visibility_supported"] = visibility_supported
    scores["tta"] = cfg.TTA
    visible = oof.visible_true.astype(bool).to_numpy()
    distance_visible = np.linalg.norm(
        oof.loc[visible, ["x", "y"]].to_numpy() - oof.loc[visible, ["x_true", "y_true"]].to_numpy(), axis=1)
    scores["visible_raw_localization_median_px"] = float(np.median(distance_visible))
    scores["visible_raw_localization_p90_px"] = float(np.percentile(distance_visible, 90))
    predicted_visible = oof.visibility_prob.to_numpy() >= threshold
    scores["visible_rejected_count"] = int(np.sum(visible & ~predicted_visible))
    scores["invisible_accepted_count"] = int(np.sum(~visible & predicted_visible))
    json_save(scores, output_dir / "scores.json")
    print(json.dumps(scores, indent=2))
    plot_oof_errors(oof, train, threshold, cfg, output_dir)

    raw_test = pd.concat(all_test, ignore_index=True)
    raw_test.to_csv(output_dir / "test_predictions_all_folds.csv", index=False)
    # Fuse visible coordinates/probabilities first. Never average already-zeroed outputs.
    ensemble = raw_test.groupby("image_id", as_index=False)[["x", "y", "visibility_prob"]].mean()
    ensemble = test[["image_id", "width", "height"]].merge(ensemble, on="image_id", validate="one_to_one")
    ensemble.to_csv(output_dir / "test_ensemble_raw.csv", index=False)
    submission = write_submission(sample, layout, ensemble, metadata, cfg, threshold)
    print(f"Finished. Artifacts: {output_dir}")
    return {"config": cfg, "train": train, "test": test, "oof": oof,
            "test_predictions": ensemble, "scores": scores, "submission": submission}



def quick_sanity_checks():
    for width, height in [(1001, 777), (603, 1403), (512, 512)]:
        *_, forward = letterbox_geometry(width, height, 512)
        inverse = np.linalg.inv(forward)
        points = np.array([[0, 0, 1], [width - 1, height - 1, 1],
                           [(width - 1) / 2, (height - 1) / 2, 1]], dtype=float)
        restored = (inverse @ (forward @ points.T)).T
        assert np.allclose(restored, points, atol=1e-8)
    assert parse_submission_id("eye__001.jpg_Fovea_X") == ("eye__001", "x")
    assert parse_submission_id("patient left.PNG_Fovea_Y") == ("patient left", "y")
    assert parse_submission_id("0007_Fovea_X") == ("0007", "x")
    target = gaussian_heatmap([0.4, 0.6], 128, 2.5)
    assert np.isfinite(target).all() and np.isclose(target.sum(), 1.0)
    dummy = pd.DataFrame({"x": [20.0, 30.0], "y": [40.0, 50.0],
                          "visibility_prob": [0.8, 0.2]})
    assert np.array_equal(apply_visibility(dummy, 0.5), [[20, 40], [0, 0]])
    print("Geometry, heatmap, ID parsing and invisible-output checks passed.")


if __name__ == "__main__":
    quick_sanity_checks()
    RESULTS = run_pipeline(CFG)
