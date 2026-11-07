"""Calibration, inference-only ROI decisions, and held-out research evaluation.

The calibration frame must be disjoint from every image used to train or select
the coarse/fine networks. Its in-sample fit scores are NOT validation scores.
All image alignment is by image_id; duplicate images are grouped by group.
"""
import math

import numpy as np
import pandas as pd


def _research_setting(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _research_frame(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(["image_id", *columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not len(frame):
        raise ValueError(f"{name} is empty")
    out = frame.copy()
    if out.image_id.isna().any():
        raise ValueError(f"{name} has null image_id")
    out["image_id"] = out.image_id.astype(str)
    if out.image_id.duplicated().any():
        raise ValueError(f"{name} has duplicate image_id")
    for col in columns:
        if col == "group":
            if out[col].isna().any():
                raise ValueError(f"{name} has null groups")
            out[col] = out[col].astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if not np.isfinite(out[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{col} contains NaN or infinity")
    return out


def _research_align(pred, truth, pred_columns=("x", "y"), truth_columns=("x", "y")):
    pred = _research_frame(pred, pred_columns, "predictions")
    truth = _research_frame(truth, truth_columns, "reference frame")
    if set(pred.image_id) != set(truth.image_id):
        raise ValueError("Prediction/reference image_id sets differ")
    truth = truth.set_index("image_id").loc[pred.image_id].reset_index()
    return pred.reset_index(drop=True), truth.reset_index(drop=True)


def _research_uncertainty(pred, floor_px):
    floor_px = float(floor_px)
    if not math.isfinite(floor_px) or floor_px <= 0:
        raise ValueError("UNCERTAINTY_FLOOR_PX must be finite and positive")
    value = pred.uncertainty.to_numpy(dtype=float)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("uncertainty must be finite and nonnegative")
    return np.maximum(value, floor_px)


def _research_higher_quantile(values, probability):
    values = np.asarray(values, dtype=float)
    probability = float(probability)
    if not len(values) or not np.isfinite(values).all() or not 0 < probability <= 1:
        raise ValueError("Invalid empirical quantile input")
    # Explicit order statistic avoids NumPy-version-dependent interpolation.
    return float(np.sort(values)[int(np.ceil((len(values) - 1) * probability))])


def fit_radius_calibration(pred, truth, cfg):
    """Fit an empirical square-radius multiplier on held-out calibration groups.

    uncertainty is sqrt(maximum eigenvalue of the original-pixel heatmap
    covariance). It is a prediction-spread proxy, not calibrated confidence.
    One maximum standardized residual per duplicate/patient group is used.
    No conformal or finite-sample coverage guarantee is claimed.
    """
    pred, truth = _research_align(pred, truth, ("x", "y", "uncertainty"),
                                  ("x", "y", "group"))
    coverage = float(_research_setting(cfg, "ROI_COVERAGE", 0.9))
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    uncertainty = _research_uncertainty(pred, floor_px)
    residual = np.abs(pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)).max(axis=1)
    standardized = residual / uncertainty
    groups = pd.DataFrame({"group": truth.group, "score": standardized})
    group_scores = groups.groupby("group", sort=True).score.max()
    probabilities = sorted(set([0.5, 0.8, 0.9, 0.95, coverage]))
    alpha_by_coverage = {
        str(p): _research_higher_quantile(group_scores.to_numpy(), p) for p in probabilities
    }
    alpha = alpha_by_coverage[str(coverage)]
    return {
        "method": "group_max_empirical_higher_quantile",
        "target_coverage": coverage,
        "alpha": alpha,
        "alpha_by_coverage": alpha_by_coverage,
        "uncertainty_floor_px": floor_px,
        "n_images": int(len(pred)),
        "n_groups": int(len(group_scores)),
        "calibration_image_ids": pred.image_id.tolist(),
        "calibration_group_ids": group_scores.index.astype(str).tolist(),
        "fit_image_coverage_before_margin_and_clamp": float(np.mean(standardized <= alpha)),
        "fit_group_coverage_before_margin_and_clamp": float(np.mean(group_scores.to_numpy() <= alpha)),
        "score_definition": "max(abs(dx),abs(dy))/max(uncertainty,floor_px); group maximum",
        "coverage_note": "Empirical calibration only; ROI bounds, radius clamps and domain shift can reduce actual coverage. Fit coverage is not held-out performance.",
    }


def make_roi_spec(pred, frame, cfg, mode="fixed", calibration=None, fixed_fraction=None):
    """Create centered square ROIs using predictions and dimensions only.

    crop_x/crop_y are CENTER coordinates in original pixels. Crops extending
    beyond the image use padding; effective image intersections are recorded.
    All modes apply side=clip(base_side+2*margin,min_fraction*short,max_fraction*short).
    A fixed fraction describes base_side before this common margin and clamp.
    """
    if mode not in {"fixed", "raw", "calibrated"}:
        raise ValueError(f"Unsupported ROI mode: {mode}")
    required = ("x", "y") if mode == "fixed" else ("x", "y", "uncertainty")
    pred, frame = _research_align(pred, frame, required, ("width", "height"))
    dimensions = frame[["width", "height"]].to_numpy(float)
    if np.any(dimensions <= 1):
        raise ValueError("Image dimensions must exceed one pixel")
    short = dimensions.min(axis=1)
    min_fraction = float(_research_setting(cfg, "ROI_MIN_FRACTION", 0.15))
    max_fraction = float(_research_setting(cfg, "ROI_MAX_FRACTION", 0.60))
    margin = float(_research_setting(cfg, "ROI_MARGIN_PX", 16.0))
    if not 0 < min_fraction <= max_fraction <= 1 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid ROI fraction bounds or pixel margin")
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    if mode == "fixed":
        fraction = float(fixed_fraction if fixed_fraction is not None else
                         _research_setting(cfg, "FIXED_ROI_FRACTION", 0.30))
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Fixed ROI fraction must be finite and positive")
        base_side = fraction * short
    else:
        if mode == "calibrated":
            if calibration is None or "alpha" not in calibration:
                raise ValueError("Calibrated ROIs require fitted alpha")
            multiplier = float(calibration["alpha"])
            floor_px = float(calibration.get("uncertainty_floor_px", floor_px))
        else:
            multiplier = float(_research_setting(cfg, "RAW_RADIUS_MULTIPLIER", 2.0))
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("Radius multiplier must be finite and nonnegative")
        base_side = 2 * multiplier * _research_uncertainty(pred, floor_px)
    proposed = base_side + 2 * margin
    side = np.clip(proposed, min_fraction * short, max_fraction * short)
    center = pred[["x", "y"]].to_numpy(float)
    if np.any(center < 0) or np.any(center > dimensions - 1 + 1e-6):
        raise ValueError("ROI centers must be within their original image")
    lower = center - side[:, None] / 2
    upper = center + side[:, None] / 2
    effective_lower = np.maximum(lower, 0)
    effective_upper = np.minimum(upper, dimensions - 1)
    # Areas are geometric fields in pixel-center coordinates; do not add +1.
    # Geometric crop area remains side**2 for matching the resampling field.
    effective_area = np.maximum(effective_upper[:, 0] - effective_lower[:, 0], 0) * np.maximum(effective_upper[:, 1] - effective_lower[:, 1], 0)
    return pd.DataFrame({
        "image_id": pred.image_id,
        "crop_x": center[:, 0], "crop_y": center[:, 1], "crop_side": side,
        "roi_x0": effective_lower[:, 0], "roi_y0": effective_lower[:, 1],
        "roi_x1": effective_upper[:, 0], "roi_y1": effective_upper[:, 1],
        "roi_area_px2": side ** 2, "roi_effective_area_px2": effective_area,
        "roi_area_fraction": side ** 2 / np.prod(dimensions, axis=1),
        "roi_at_min": proposed < min_fraction * short,
        "roi_at_max": proposed > max_fraction * short,
        "roi_has_padding": np.any(lower < 0, axis=1) | np.any(upper > dimensions - 1, axis=1),
        "roi_mode": mode,
    })


def roi_contains_truth(roi, truth):
    """Evaluation only. Never call this to select per-image inference behavior."""
    roi, truth = _research_align(roi, truth, ("roi_x0", "roi_y0", "roi_x1", "roi_y1"))
    point = truth[["x", "y"]].to_numpy(float)
    return pd.Series(
        (point[:, 0] >= roi.roi_x0.to_numpy(float)) & (point[:, 0] <= roi.roi_x1.to_numpy(float)) &
        (point[:, 1] >= roi.roi_y0.to_numpy(float)) & (point[:, 1] <= roi.roi_y1.to_numpy(float)),
        index=roi.image_id, name="contains_truth",
    )


def localization_metrics(pred, truth, roi=None):
    pred, truth = _research_align(pred, truth)
    delta = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(delta, axis=1)
    result = {
        "n_images": int(len(pred)), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mean_distance_px": float(distance.mean()),
        "median_distance_px": float(np.median(distance)),
        "p90_distance_px": float(np.quantile(distance, 0.90)),
        "max_distance_px": float(distance.max()),
    }
    if "group" in truth:
        result["n_groups"] = int(truth.group.nunique())
    if "fallback_used" in pred:
        result["fallback_rate"] = float(pred.fallback_used.astype(bool).mean())
    if roi is not None:
        roi, truth_roi = _research_align(roi, truth, ("crop_side", "roi_x0", "roi_y0", "roi_x1", "roi_y1"))
        result["roi_coverage"] = float(roi_contains_truth(roi, truth_roi).mean())
        result["mean_roi_area_px2"] = float(np.mean(roi.crop_side.to_numpy(float) ** 2))
        for column, key in [("roi_area_fraction", "mean_roi_area_fraction"),
                            ("roi_effective_area_px2", "mean_roi_effective_area_px2"),
                            ("roi_at_max", "roi_max_clamp_rate"),
                            ("roi_at_min", "roi_min_clamp_rate"),
                            ("roi_has_padding", "roi_padding_rate")]:
            if column in roi:
                value = pd.to_numeric(roi[column], errors="raise").to_numpy(float)
                if not np.isfinite(value).all():
                    raise ValueError(f"ROI metric {column} contains non-finite values")
                result[key] = float(value.mean())
    return result


def apply_fallback(coarse, fine, rule):
    """Use only prediction features at inference; no reference labels accepted."""
    coarse, fine = _research_align(coarse, fine, ("x", "y", "uncertainty"),
                                   ("x", "y", "crop_edge_fraction"))
    edge = fine.crop_edge_fraction.to_numpy(float)
    if np.any(edge < -1e-6) or np.any(edge > 0.5 + 1e-6):
        raise ValueError("crop_edge_fraction must be in [0,0.5]")
    spread = _research_uncertainty(coarse, rule.get("uncertainty_floor_px", 2.0))
    disagreement = np.linalg.norm(fine[["x", "y"]].to_numpy(float) - coarse[["x", "y"]].to_numpy(float), axis=1) / spread
    used = np.zeros(len(fine), dtype=bool)
    if rule.get("enabled", False):
        edge_threshold = float(rule["edge_threshold"])
        disagreement_threshold = float(rule["disagreement_threshold"])
        if not math.isfinite(edge_threshold) or not math.isfinite(disagreement_threshold):
            raise ValueError("Fallback thresholds must be finite")
        used = (edge < edge_threshold) | (disagreement > disagreement_threshold)
    result = fine.copy()
    result.loc[used, ["x", "y"]] = coarse.loc[used, ["x", "y"]].to_numpy(float)
    if "uncertainty" in result:
        result.loc[used, "uncertainty"] = coarse.loc[used, "uncertainty"].to_numpy(float)
    covariance_columns = [column for column in ("cov_xx", "cov_xy", "cov_yy")
                          if column in coarse and column in result]
    if covariance_columns:
        result.loc[used, covariance_columns] = coarse.loc[used, covariance_columns].to_numpy(float)
    result["fallback_used"] = used
    result["coarse_fine_disagreement_standardized"] = disagreement
    return result


def fit_fallback_rule(coarse, fine, truth, cfg):
    """Small prespecified search on calibration labels, with no-fallback default.

    Select a rule only if calibration RMSE improves and calibration P90 does
    not exceed the prespecified tolerance. This is NOT a per-image oracle and
    does not promise the same improvement on held-out images.
    """
    coarse, truth = _research_align(coarse, truth, ("x", "y", "uncertainty"), ("x", "y", "group"))
    coarse, fine = _research_align(coarse, fine, ("x", "y", "uncertainty"), ("x", "y", "crop_edge_fraction"))
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    baseline_rule = {"enabled": False, "edge_threshold": 0.0,
                     "disagreement_threshold": 1e9, "uncertainty_floor_px": floor_px}
    baseline_pred = apply_fallback(coarse, fine, baseline_rule)
    baseline = localization_metrics(baseline_pred, truth)
    min_gain = float(_research_setting(cfg, "FALLBACK_MIN_IMPROVEMENT_PX", 1e-6))
    p90_tolerance = float(_research_setting(cfg, "FALLBACK_P90_TOLERANCE", 0.0))
    if not math.isfinite(min_gain) or min_gain < 0 or not math.isfinite(p90_tolerance) or p90_tolerance < 0:
        raise ValueError("Fallback improvement/tolerance must be finite and nonnegative")
    table = []
    winner, best = baseline_rule, baseline
    edge_grid = _research_setting(cfg, "FALLBACK_EDGE_THRESHOLDS", (0.0, 0.03, 0.08))
    disagreement_grid = _research_setting(cfg, "FALLBACK_DISAGREEMENT_THRESHOLDS", (2.0, 4.0, 1e9))
    for edge in edge_grid:
        if not 0 <= float(edge) <= 0.5:
            raise ValueError("Fallback edge grid must be in [0,0.5]")
        for disagreement in disagreement_grid:
            if not math.isfinite(float(disagreement)) or float(disagreement) <= 0:
                raise ValueError("Fallback disagreement grid must be finite and positive")
            rule = {"enabled": True, "edge_threshold": float(edge),
                    "disagreement_threshold": float(disagreement), "uncertainty_floor_px": floor_px}
            metrics = localization_metrics(apply_fallback(coarse, fine, rule), truth)
            eligible = (metrics["rmse"] < baseline["rmse"] - min_gain and
                        metrics["p90_distance_px"] <= baseline["p90_distance_px"] * (1 + p90_tolerance) + 1e-9)
            table.append({**rule, "rmse": metrics["rmse"], "p90_distance_px": metrics["p90_distance_px"],
                          "fallback_rate": metrics["fallback_rate"], "eligible": bool(eligible)})
            if eligible and metrics["rmse"] < best["rmse"] - 1e-12:
                winner, best = rule, metrics
    return {
        **winner, "n_images": int(len(truth)), "n_groups": int(truth.group.nunique()),
        "calibration_image_ids": truth.image_id.tolist(),
        "calibration_group_ids": sorted(truth.group.unique().tolist()),
        "baseline_calibration_metrics": baseline, "selected_calibration_metrics": best,
        "search": table, "p90_relative_tolerance": p90_tolerance,
        "note": "Rule fitted on calibration labels only. Selection metrics are optimistic calibration-fit scores; evaluate all images on untouched outer validation folds.",
    }


def uncertainty_diagnostics(pred, truth, calibration=None, cfg=None):
    """Risk-coverage sorts by PREDICTED uncertainty, never by true error.

    The separately named oracle curve is a descriptive lower bound only.
    Report full-coverage RMSE regardless of any selective-risk curve.
    """
    pred, truth = _research_align(pred, truth, ("x", "y", "uncertainty"))
    uncertainty = _research_uncertainty(pred, _research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    delta = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
    squared = np.mean(delta ** 2, axis=1)
    distance = np.linalg.norm(delta, axis=1)
    ranks_u, ranks_e = pd.Series(uncertainty).rank(), pd.Series(distance).rank()
    rho = ranks_u.corr(ranks_e) if ranks_u.nunique() > 1 and ranks_e.nunique() > 1 else None
    predicted_order = np.argsort(uncertainty, kind="stable")
    oracle_order = np.argsort(distance, kind="stable")
    risk = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        retained = max(1, int(np.ceil(fraction * len(pred))))
        risk.append({"requested_retention": fraction, "actual_retention": retained / len(pred),
                     "n_retained": retained,
                     "rmse_by_predicted_uncertainty": float(np.sqrt(squared[predicted_order[:retained]].mean())),
                     "oracle_rmse_by_true_error": float(np.sqrt(squared[oracle_order[:retained]].mean()))})
    result = {"n_images": int(len(pred)), "full_coverage_rmse": float(np.sqrt(squared.mean())),
              "uncertainty_error_spearman": None if rho is None or not np.isfinite(rho) else float(rho),
              "uncertainty_definition": "sqrt(max eigenvalue of original-pixel heatmap covariance); spread proxy, not ERE or epistemic uncertainty",
              "risk_coverage": risk}
    if calibration is not None:
        if cfg is None:
            raise ValueError("cfg is required for actual bounded-ROI calibration diagnostics")
        coverage_rows = []
        for level, alpha in sorted(calibration.get("alpha_by_coverage", {}).items(), key=lambda item: float(item[0])):
            roi = make_roi_spec(pred, truth, cfg, "calibrated", {**calibration, "alpha": alpha})
            actual = roi_contains_truth(roi, truth)
            row = {"nominal_coverage": float(level), "alpha": float(alpha),
                   "actual_bounded_roi_image_coverage": float(actual.mean())}
            if "group" in truth:
                grouped = pd.DataFrame({"group": truth.group.to_numpy(), "covered": actual.to_numpy()})
                row["actual_bounded_roi_group_coverage"] = float(grouped.groupby("group").covered.min().mean())
            coverage_rows.append(row)
        result["bounded_roi_calibration_curve"] = coverage_rows
        result["calibration_note"] = "Actual coverage includes margin and min/max size clamps; empirical target is not a guarantee."
    return result


def paired_group_bootstrap(predictions_by_method, truth, baseline=None, n_bootstrap=2000, seed=2026):
    """Paired percentile intervals conditional on these fitted OOF predictions.

    Resample whole groups; preserve all labeled rows within each sampled group.
    This does not account for retraining variation or unknown patient overlap.
    """
    if not predictions_by_method:
        raise ValueError("At least one prediction method is required")
    names = list(predictions_by_method)
    baseline = names[0] if baseline is None else baseline
    if baseline not in names or int(n_bootstrap) < 100:
        raise ValueError("Invalid bootstrap baseline or fewer than 100 resamples")
    truth = _research_frame(truth, ("x", "y", "group"), "bootstrap truth")
    # Establish one reference order for all methods, regardless of their row order.
    reference_ids = truth.image_id.tolist()
    group_codes, group_names = pd.factorize(truth.group, sort=True)
    n_groups = len(group_names)
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    sums_by_method = []
    point_rmse = []
    for name in names:
        pred, aligned_truth = _research_align(predictions_by_method[name], truth)
        pred = pred.set_index("image_id").loc[reference_ids]
        error = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
        squared = np.mean(error ** 2, axis=1)
        sums_by_method.append(np.bincount(group_codes, weights=squared, minlength=n_groups))
        point_rmse.append(float(np.sqrt(squared.mean())))
    rng = np.random.default_rng(int(seed))
    multiplicity = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=int(n_bootstrap))
    denominator = multiplicity @ counts
    scores = np.sqrt((multiplicity @ np.asarray(sums_by_method).T) / denominator[:, None])
    baseline_index = names.index(baseline)
    records = []
    for index, name in enumerate(names):
        delta = scores[:, index] - scores[:, baseline_index]
        records.append({"method": name, "baseline": baseline, "rmse": point_rmse[index],
                        "rmse_ci_low": float(np.quantile(scores[:, index], 0.025)),
                        "rmse_ci_high": float(np.quantile(scores[:, index], 0.975)),
                        "delta_rmse": point_rmse[index] - point_rmse[baseline_index],
                        "delta_ci_low": float(np.quantile(delta, 0.025)),
                        "delta_ci_high": float(np.quantile(delta, 0.975)),
                        "n_images": int(len(truth)), "n_groups": int(n_groups),
                        "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                        "ci_note": "Paired group percentile bootstrap of fixed OOF predictions; excludes retraining variation and unobserved patient dependence."})
    return pd.DataFrame(records)
