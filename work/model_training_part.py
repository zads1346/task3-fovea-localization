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
