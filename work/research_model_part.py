"""Research dataset, strict pretrained model training, and distribution inference.

Concatenate after data_part.py and model_training_part.py. Inference explicitly
removes label columns. The uncertainty below is spatial heatmap dispersion, not
an automatically calibrated probability or an epistemic uncertainty estimate.
"""


def research_crop_transform(center_x, center_y, side, input_size):
    """Map original pixel centres into a square crop with pixel-edge side `side`.

    The crop covers [cx-side/2, cx+side/2] in pixel-edge coordinates relative
    to its centre. Output pixel centres are symmetrically located around
    (input_size-1)/2. cv2.warpAffine receives this forward transform and performs
    inverse sampling itself. Crops may extend beyond the original image.
    """
    side = float(side)
    if not np.isfinite([center_x, center_y, side]).all() or side <= 0:
        raise ValueError("Crop centre and side must be finite; side must be positive.")
    scale = float(input_size) / side
    centre = (float(input_size) - 1.0) / 2.0
    return np.array([[scale, 0.0, centre - scale * float(center_x)],
                     [0.0, scale, centre - scale * float(center_y)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class ResearchDataset(Dataset):
    """All-visible localization with optional prediction-centred fine crops.

    Fine training without crop_spec uses random GT-centred perturbations. This
    is a declared synthetic localization-error augmentation, not inner OOF
    predictions. Validation/inference never repairs a crop using the label.
    """

    def __init__(self, frame, cfg, training=False, stage="coarse", crop_spec=None):
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.frame = frame.reset_index(drop=True).copy()
        self.cfg, self.training, self.stage = cfg, bool(training), stage
        self.has_labels = {"x", "y", "visible"}.issubset(self.frame.columns)
        if self.training and not self.has_labels:
            raise ValueError("Training requires x, y and visible labels.")
        if self.has_labels and not self.frame.visible.astype(bool).all():
            raise ValueError("Research model supports this all-visible dataset only.")
        self.size = int(cfg.INPUT_SIZE if stage == "coarse" else cfg.FINE_INPUT_SIZE)
        self.hm_size = int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE)
        self.crops = None
        if crop_spec is not None:
            required = ["image_id", "crop_x", "crop_y", "crop_side"]
            if not set(required).issubset(crop_spec.columns):
                raise ValueError(f"crop_spec must contain {required}")
            self.crops = crop_spec[required].copy()
            self.crops["image_id"] = self.crops.image_id.astype(str)
            if self.crops.image_id.duplicated().any():
                raise ValueError("crop_spec contains duplicate image IDs.")
            self.crops = self.crops.set_index("image_id")
            missing = set(self.frame.image_id.astype(str)) - set(self.crops.index)
            if missing:
                raise ValueError(f"Missing crops for image IDs: {sorted(missing)}")
            values = self.crops[["crop_x", "crop_y", "crop_side"]].to_numpy(float)
            if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
                raise ValueError("Non-finite crop coordinates or non-positive crop side.")
        if stage == "fine" and not self.training and self.crops is None:
            raise ValueError("Fine validation/inference requires prediction-derived crop_spec.")
        self.roi_fractions = np.asarray(getattr(cfg, "ROI_FRACTIONS", (0.2, 0.3, 0.4)), float)
        if self.roi_fractions.size == 0 or not np.isfinite(self.roi_fractions).all() or (self.roi_fractions <= 0).any():
            raise ValueError("ROI_FRACTIONS must be finite positive numbers.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = read_rgb(row.path)
        height, width = image.shape[:2]
        if int(row.width) != width or int(row.height) != height:
            raise ValueError(f"Image dimensions changed: {row.path}")
        original = np.array([float(row.x), float(row.y)], np.float64) if self.has_labels else None
        crop_was_enlarged, initial_crop_miss = False, False
        crop_from_prediction = self.stage == "fine" and self.crops is not None
        if self.stage == "coarse":
            image, transform = letterbox(image, self.size)
            crop = np.array([(width - 1) / 2, (height - 1) / 2, max(width, height)], float)
        else:
            if self.crops is not None:
                crop = self.crops.loc[str(row.image_id), ["crop_x", "crop_y", "crop_side"]].to_numpy(float).copy()
                if self.training:
                    initial_crop_miss = bool(np.abs(original - crop[:2]).max() >= crop[2] / 2)
                    jitter = float(getattr(self.cfg, "OOF_CENTER_JITTER", 0.08))
                    if jitter < 0:
                        raise ValueError("OOF_CENTER_JITTER must be non-negative.")
                    crop[:2] += np.clip(np.random.normal(0, jitter, 2), -0.20, 0.20) * crop[2]
            else:
                side = float(np.random.choice(self.roi_fractions)) * min(width, height)
                jitter = float(getattr(self.cfg, "TRAIN_CENTER_JITTER", 0.30))
                if jitter < 0:
                    raise ValueError("TRAIN_CENTER_JITTER must be non-negative.")
                # Truncated normal errors cover off-centre targets without
                # training on crops that cannot contain the annotated point.
                offset = np.clip(np.random.normal(0, jitter, 2), -0.42, 0.42) * side
                crop = np.r_[original + offset, side]
            if self.training:
                # When optional prediction crops miss the point, increase the
                # *training* field of view; preserve the original coordinate.
                required_side = 2.5 * float(np.abs(original - crop[:2]).max()) + 2
                crop_was_enlarged = required_side > crop[2]
                crop[2] = max(crop[2], required_side)
            transform = research_crop_transform(*crop, self.size)
            image = cv2.warpAffine(image, transform[:2], (self.size, self.size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        point = (transform @ np.r_[original, 1.0])[:2] if original is not None else np.zeros(2)
        if self.training:
            for _ in range(8):
                affine = cv2.getRotationMatrix2D(((self.size - 1) / 2,) * 2,
                                                np.random.uniform(-12, 12),
                                                np.random.uniform(0.95, 1.05))
                affine[:, 2] += np.random.uniform(-0.025, 0.025, 2) * self.size
                augmentation = np.vstack([affine, [0, 0, 1]])
                if np.random.random() < 0.5:
                    augmentation = np.array([[-1, 0, self.size - 1], [0, 1, 0], [0, 0, 1]]) @ augmentation
                candidate = (augmentation @ np.r_[point, 1.0])[:2]
                if np.all((candidate >= 0) & (candidate <= self.size - 1)):
                    image = cv2.warpAffine(image, augmentation[:2], (self.size, self.size),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
                    transform = augmentation @ transform
                    point = candidate
                    break
            image = np.clip(image.astype(np.float32) * np.random.uniform(0.85, 1.15)
                            + np.random.uniform(-8, 8), 0, 255).astype(np.uint8)
        xy = (point / (self.size - 1)).astype(np.float32)
        # Labels outside validation ROIs stay outside; no clipping or crop repair.
        # Validation uses original-pixel errors, not this possibly tiny heatmap.
        target = gaussian_heatmap(xy, self.hm_size, self.cfg.HEATMAP_SIGMA) if self.has_labels else np.zeros((1, self.hm_size, self.hm_size), np.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target_heatmap": torch.from_numpy(target),
            "target_xy": torch.from_numpy(xy),
            "visible": torch.tensor(float(self.has_labels), dtype=torch.float32),
            "orig_wh": torch.tensor([width, height], dtype=torch.float32),
            "input_to_orig": torch.from_numpy(np.linalg.inv(transform)[:2].astype(np.float32)),
            "crop": torch.tensor(crop, dtype=torch.float32),
            "crop_from_prediction": torch.tensor(crop_from_prediction),
            "crop_was_enlarged": torch.tensor(crop_was_enlarged),
            "initial_crop_miss": torch.tensor(initial_crop_miss),
            "image_id": str(row.image_id),
        }
        if self.has_labels:
            item["x_true"] = torch.tensor(float(row.x), dtype=torch.float32)
            item["y_true"] = torch.tensor(float(row.y), dtype=torch.float32)
        return item


def make_research_loader(frame, cfg, training=False, stage="coarse", crop_spec=None, seed=0):
    return DataLoader(ResearchDataset(frame, cfg, training, stage, crop_spec),
                      batch_size=int(cfg.BATCH_SIZE), shuffle=training,
                      num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                      drop_last=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(int(seed)), persistent_workers=False)


def research_build_model(cfg, stage, device):
    """Require the explicitly resolved official weight file; no silent fallback."""
    path = getattr(cfg, "PRETRAINED_WEIGHTS_PATH", None)
    if cfg.PRETRAINED and (not path or not Path(path).is_file()):
        raise FileNotFoundError("Research training requires the attached official ResNet18 weight file. Resolve PRETRAINED_WEIGHTS_PATH first.")
    return FoveaNet(pretrained=bool(cfg.PRETRAINED), allow_download=False,
                    use_visibility=False, freeze_bn=bool(cfg.FREEZE_BN),
                    heatmap_size=int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE),
                    pretrained_weights_path=path if cfg.PRETRAINED else None).to(device)


@torch.inference_mode()
def research_validation_rmse(model, loader, cfg, device):
    model.eval()
    squared_error, coordinate_count = 0.0, 0
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        prediction = normalized_to_original(output["coords"], batch)
        truth = torch.stack([batch["x_true"], batch["y_true"]], dim=1)
        squared_error += float((prediction.double() - truth.double()).square().sum())
        coordinate_count += int(truth.numel())
    if not coordinate_count:
        raise ValueError("Validation set is empty.")
    return math.sqrt(squared_error / coordinate_count)


def fit_research_model(training, validation, cfg, device, stage, checkpoint_path,
                       train_crop=None, val_crop=None, epochs=None, seed=0):
    """Early-stop on inner data or refit for fixed epochs without validation.

    An integer epochs requests a final refit and the validation argument is
    intentionally never inspected. best_epoch is one-based, including refits.
    """
    device = torch.device(device)
    seed_everything(int(seed))
    fixed_epochs = epochs is not None
    total_epochs = int(epochs if fixed_epochs else (cfg.COARSE_EPOCHS if stage == "coarse" else cfg.FINE_EPOCHS))
    if total_epochs < 1 or len(training) == 0:
        raise ValueError("Training needs nonempty samples and at least one epoch.")
    train_loader = make_research_loader(training, cfg, True, stage, train_crop, seed)
    val_loader = None
    if not fixed_epochs:
        if validation is None or len(validation) == 0:
            raise ValueError("Early stopping needs a nonempty inner validation set.")
        if "group" in training and "group" in validation:
            if set(training.group.astype(str)) & set(validation.group.astype(str)):
                raise ValueError("Training/validation groups overlap.")
        val_loader = make_research_loader(validation, cfg, False, stage, val_crop, seed)
    model = research_build_model(cfg, stage, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(cfg.LR), weight_decay=float(cfg.WEIGHT_DECAY))
    steps = total_epochs * len(train_loader)
    warmup = min(int(cfg.WARMUP_EPOCHS) * len(train_loader), max(0, steps - 1))
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    amp = bool(cfg.AMP) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, bad_epochs = [], math.inf, 0, 0
    for epoch in range(1, total_epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        crop_counts = {"crop_from_prediction": 0, "crop_was_enlarged": 0, "initial_crop_miss": 0}
        for cpu_batch in train_loader:
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                output = model(batch["image"])
            loss, _ = fovea_loss(output, batch, cfg, use_visibility=False)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.GRAD_CLIP))
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            n = int(batch["image"].shape[0])
            total_loss += float(loss.detach()) * n
            count += n
            for key in crop_counts:
                crop_counts[key] += int(batch[key].sum())
        score = research_validation_rmse(model, val_loader, cfg, device) if not fixed_epochs else None
        if score is not None and not math.isfinite(score):
            raise FloatingPointError("Non-finite inner validation RMSE.")
        improved = fixed_epochs or score < best_score
        if improved:
            best_epoch, bad_epochs = epoch, 0
            if score is not None:
                best_score = float(score)
            if not fixed_epochs or epoch == total_epochs:
                torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "epoch": epoch, "stage": stage, "use_visibility": False,
                            "heatmap_size": model.heatmap_size,
                            "selection": "fixed_epochs_no_validation" if fixed_epochs else "inner_validation_rmse",
                            "rmse": score, "seed": int(seed)}, checkpoint_path)
        else:
            bad_epochs += 1
        record = {"stage": stage, "epoch": epoch, "train_loss": total_loss / count,
                  "val_rmse": score, "seconds": time.perf_counter() - started,
                  "learning_rate": optimizer.param_groups[0]["lr"], **crop_counts}
        history.append(record)
        validation_message = "fixed-epoch refit" if fixed_epochs else f"inner RMSE {score:.3f}px"
        print(f"{stage} | {checkpoint_path.stem} | epoch {epoch}/{total_epochs} | "
              f"loss {record['train_loss']:.4f} | {validation_message} | {record['seconds']:.1f}s", flush=True)
        if not fixed_epochs and int(cfg.PATIENCE) > 0 and bad_epochs >= int(cfg.PATIENCE):
            break
    load_model_checkpoint(model, checkpoint_path, device)
    info = {"best_epoch": best_epoch, "best_rmse": None if fixed_epochs else best_score,
            "history": history, "checkpoint_path": str(checkpoint_path),
            "fixed_epochs": fixed_epochs, "training_images": len(training),
            "crop_training_policy": ("prediction centre + jitter; enlarge around that centre when training target misses; never change labels"
                                     if stage == "fine" and train_crop is not None else
                                     "GT centre + truncated jitter" if stage == "fine" else "full image")}
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return model, info


def research_distribution_moments(logits, input_to_orig, input_size, undo_hflip=False):
    """Probability mean/covariance in original pixels, with affine propagation."""
    logits = logits.float()
    if undo_hflip:
        logits = torch.flip(logits, dims=[-1])
    b, _, h, w = logits.shape
    prob = logits.flatten(1).softmax(1)
    y, x = torch.meshgrid(torch.linspace(0, input_size - 1, h, device=logits.device),
                          torch.linspace(0, input_size - 1, w, device=logits.device), indexing="ij")
    grid = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    mean = prob @ grid
    delta = grid.unsqueeze(0) - mean.unsqueeze(1)
    covariance = torch.einsum("bn,bni,bnj->bij", prob, delta, delta)
    linear, shift = input_to_orig.float()[:, :, :2], input_to_orig.float()[:, :, 2]
    original_mean = torch.bmm(linear, mean.unsqueeze(-1)).squeeze(-1) + shift
    original_covariance = linear @ covariance @ linear.transpose(1, 2)
    return original_mean, original_covariance


@torch.inference_mode()
def predict_research(model, frame, cfg, device, stage="coarse", crop_spec=None):
    """No label access. Mixture covariance includes disagreement between TTA views.

    uncertainty = sqrt(largest eigenvalue of original-pixel covariance), in px.
    inference_seconds includes data loading and inference, averaged per image.
    Crop-edge fraction is min distance to the crop's four pixel edges / side;
    it may be negative if final image-boundary clipping moves a point out of ROI.
    """
    device = torch.device(device)
    # An explicit whitelist prevents accidental use of x/y/visible at inference.
    inference_frame = frame[["image_id", "path", "width", "height"]].copy()
    loader = make_research_loader(inference_frame, cfg, False, stage, crop_spec, int(cfg.SEED))
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, rows = time.perf_counter(), []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast_context(device, cfg):
            output = model(batch["image"])
        size = int(batch["image"].shape[-1])
        means, covariances = [], []
        mean, covariance = research_distribution_moments(output["logits"], batch["input_to_orig"], size)
        means.append(mean)
        covariances.append(covariance)
        if cfg.TTA:
            with _autocast_context(device, cfg):
                output_flip = model(torch.flip(batch["image"], dims=[-1]))
            mean, covariance = research_distribution_moments(output_flip["logits"], batch["input_to_orig"], size, True)
            means.append(mean)
            covariances.append(covariance)
        stacked = torch.stack(means)
        mean = stacked.mean(0)
        offsets = stacked - mean.unsqueeze(0)
        covariance = (torch.stack(covariances) + offsets.unsqueeze(-1) @ offsets.unsqueeze(-2)).mean(0)
        covariance = (covariance + covariance.transpose(1, 2)) / 2
        uncertainty = torch.linalg.eigvalsh(covariance).clamp_min(0)[:, -1].sqrt()
        mean = torch.minimum(mean.clamp_min(0), batch["orig_wh"].float() - 1)
        crop = batch["crop"]
        edge = 0.5 - ((mean - crop[:, :2]).abs() / crop[:, 2:3]).amax(1)
        for i, image_id in enumerate(batch["image_id"]):
            rows.append({"image_id": str(image_id), "x": float(mean[i, 0]), "y": float(mean[i, 1]),
                         "uncertainty": float(uncertainty[i]), "cov_xx": float(covariance[i, 0, 0]),
                         "cov_xy": float(covariance[i, 0, 1]), "cov_yy": float(covariance[i, 1, 1]),
                         "crop_edge_fraction": float(edge[i]) if stage == "fine" else np.nan})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    columns = ["image_id", "x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy", "crop_edge_fraction"]
    result = pd.DataFrame(rows, columns=columns)
    result["inference_seconds"] = (time.perf_counter() - started) / max(1, len(result))
    return result


def research_geometry_selfcheck():
    """Check half-pixel crop sampling, inverse affine, and flipped moments."""
    worst = 0.0
    for cx, cy, side, size in [(120.2, 90.8, 90, 128), (-10, 30, 121.5, 64),
                                (1450.3, 980.1, 600, 512)]:
        matrix = research_crop_transform(cx, cy, side, size)
        points = np.array([[cx, cy, 1], [cx + side * 0.21, cy - side * 0.3, 1]])
        mapped = (matrix @ points.T).T
        inverse = (np.linalg.inv(matrix) @ mapped.T).T
        worst = max(worst, float(np.abs(inverse - points).max()))
        np.testing.assert_allclose(mapped[0, :2], [(size - 1) / 2] * 2, atol=1e-8)
        np.testing.assert_allclose(inverse, points, atol=1e-8)
    logits = torch.full((1, 1, 9, 9), -50.0)
    logits[0, 0, 3, 6] = 50.0
    matrix = research_crop_transform(100, 150, 80, 64)
    inverse = torch.tensor(np.linalg.inv(matrix)[:2], dtype=torch.float32).unsqueeze(0)
    mean, covariance = research_distribution_moments(logits, inverse, 64)
    flipped_mean, flipped_cov = research_distribution_moments(torch.flip(logits, [-1]), inverse, 64, True)
    torch.testing.assert_close(mean, flipped_mean)
    torch.testing.assert_close(covariance, flipped_cov)
    assert torch.isfinite(covariance).all()
    # A coordinate ramp tests actual cv2 sampling, not just matrix inversion.
    ramp = np.tile(np.arange(200, dtype=np.float32), (180, 1))
    matrix = research_crop_transform(80, 90, 64, 64)
    sampled = cv2.warpAffine(ramp, matrix[:2], (64, 64), flags=cv2.INTER_LINEAR)
    expected_x = (np.linalg.inv(matrix) @ np.array([31, 31, 1]))[0]
    assert abs(float(sampled[31, 31]) - expected_x) <= 1 / 32 + 1e-5
    return {"max_inverse_error_px": worst, "hflip_distribution_roundtrip": True,
            "cv2_pixel_sampling": True}
