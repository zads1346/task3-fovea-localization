def _research_cleanup(model=None):
    if model is not None:
        model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def research_group_holdout(frame, fraction, seed):
    groups = frame.group.unique().copy()
    if len(groups) < 5:
        raise ValueError("Too few groups for a development/calibration split.")
    np.random.default_rng(seed).shuffle(groups)
    count = min(len(groups) - 3, max(2, int(round(len(groups) * fraction))))
    selected = set(groups[:count])
    fit = frame.loc[~frame.group.isin(selected)].reset_index(drop=True)
    held = frame.loc[frame.group.isin(selected)].reset_index(drop=True)
    assert not set(fit.group) & set(held.group)
    return fit, held


def _research_predictions(frame, path):
    frame.to_csv(path, index=False)


def research_file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def research_runtime_code_digest():
    """Detect edited notebook function bodies without relying on source files.

    Cell execution filenames/line offsets are normalized so Run All can reuse
    the same code. Nested code objects are normalized recursively as well.
    """
    import types
    import inspect
    def normalize(value):
        # marshal encodes object sharing/interning flags which can change after
        # imports or training. Hash semantic values instead of those flags.
        if isinstance(value, types.CodeType):
            return {"bytecode": value.co_code.hex(), "constants": normalize(value.co_consts),
                    "names": value.co_names, "variables": value.co_varnames,
                    "free": value.co_freevars, "cells": value.co_cellvars,
                    "arguments": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
                    "flags": value.co_flags}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, frozenset):
            return {"frozenset": sorted([normalize(item) for item in value], key=lambda item: json.dumps(item, sort_keys=True))}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (float, complex)):
            return {type(value).__name__: repr(value)}
        if value is Ellipsis:
            return {"ellipsis": True}
        return value
    digest = hashlib.sha256()
    for name, value in sorted(list(globals().items())):
        entries = []
        if isinstance(value, types.FunctionType):
            value = inspect.unwrap(value)
        if isinstance(value, types.FunctionType) and value.__globals__ is globals():
            entries.append((name, value))
        elif isinstance(value, type) and value.__module__ == __name__:
            for member, function in sorted(vars(value).items()):
                if isinstance(function, (staticmethod, classmethod)):
                    function = function.__func__
                if isinstance(function, types.FunctionType):
                    function = inspect.unwrap(function)
                    entries.append((name + "." + member, function))
        for label, function in entries:
            digest.update(label.encode())
            digest.update(json.dumps(normalize(function.__code__), sort_keys=True, ensure_ascii=True).encode())
            digest.update(json.dumps(normalize(function.__defaults__), sort_keys=True, default=str).encode())
            digest.update(json.dumps(function.__kwdefaults__, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def research_training_environment():
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
            "device_type": "cuda" if torch.cuda.is_available() else "cpu"}


def research_cache_manifest(folder, paths):
    folder = Path(folder).resolve()
    result = {}
    for path in sorted(map(Path, paths)):
        resolved = path.resolve()
        relative = resolved.relative_to(folder).as_posix()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[relative] = {"bytes": resolved.stat().st_size, "sha256": research_file_digest(resolved)}
    if not result:
        raise ValueError("Cannot mark an empty cache as complete.")
    return result


def research_verify_cache(folder, manifest, required_paths=()):
    folder = Path(folder).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{folder}: cache has no integrity manifest; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
    missing = set(required_paths) - set(manifest)
    if missing:
        raise ValueError(f"Cache integrity manifest is missing required entries: {sorted(missing)}")
    for relative, expected in manifest.items():
        path = (folder / relative).resolve()
        try:
            path.relative_to(folder)
        except ValueError as exc:
            raise ValueError("Cache manifest escapes its output directory") from exc
        if (not path.is_file() or path.stat().st_size != expected.get("bytes") or
                research_file_digest(path) != expected.get("sha256")):
            raise ValueError(f"Cache integrity check failed: {path}. Restore the file or set REUSE_CHECKPOINTS=False.")


def research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop):
    def frame_payload(frame, crop=False):
        if frame is None:
            return None
        names = (["image_id", "crop_x", "crop_y", "crop_side"] if crop else
                 ["image_id", "pixel_hash", "x", "y", "visible", "group", "width", "height"])
        cols = [name for name in names if name in frame]
        return frame[cols].to_dict("records")
    config = asdict(cfg)
    # Worker count affects augmentation RNG. Keep it in the identity.
    for name in ("OUTPUT_DIR", "SUBMISSION_PATH", "REUSE_CHECKPOINTS", "SHOW_PLOTS",
                 "PRETRAINED_WEIGHTS_PATH", "INPUT_ROOT", "DATA_DIR", "LABEL_CSV", "SAMPLE_CSV"):
        config.pop(name, None)
    payload = {"source": globals().get("RESEARCH_SOURCE_HASH", RESEARCH_VERSION),
               "runtime_code": research_runtime_code_digest(),
               "environment": research_training_environment(), "config": config, "stage": stage,
               "seed": int(seed), "epochs": epochs, "training": frame_payload(training),
               "validation": frame_payload(validation) if epochs is None else None,
               "train_crop": frame_payload(train_crop, True),
               "val_crop": frame_payload(val_crop, True) if epochs is None else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _research_stage(training, validation, cfg, device, stage, folder, seed,
                    train_crop=None, val_crop=None, epochs=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "stage_complete.json"
    signature = research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop)
    if cfg.REUSE_CHECKPOINTS and marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"{folder}: cached stage configuration/data/code changed; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
        research_verify_cache(folder, record.get("files"), ("model.pt", "history.csv", "training_summary.json"))
        model = research_build_model(cfg, stage, device)
        load_model_checkpoint(model, folder / "model.pt", device)
        info = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
        info["history"] = pd.read_csv(folder / "history.csv").to_dict("records")
        info["reused_completed_stage"] = True
        print(f"Reuse completed training stage: {folder.name}", flush=True)
        return model, info
    if marker.exists():
        marker.unlink()
    model, info = fit_research_model(
        training, validation, cfg, device, stage, folder / "model.pt",
        train_crop=train_crop, val_crop=val_crop, epochs=epochs, seed=seed)
    pd.DataFrame(info["history"]).to_csv(folder / "history.csv", index=False)
    safe_info = {k: v for k, v in info.items() if k != "history"}
    json_save(safe_info, folder / "training_summary.json")
    files = [folder / name for name in ("model.pt", "history.csv", "training_summary.json")]
    record = {"signature": signature, "files": research_cache_manifest(folder, files),
              "resume_policy": "Reuse completed stages only; interrupted epochs restart this stage."}
    temporary = marker.with_suffix(".partial.json")
    json_save(record, temporary)
    temporary.replace(marker)
    return model, info


def research_crossfit_crops(development, cfg, device, folder, epochs, seed):
    """OOF coarse centres for fine TRAINING only, not for reporting accuracy.

    Epoch count is selected within development. These training augmentation
    predictions are not claimed to be an unbiased inner validation estimate.
    The separate calibration and outer validation groups never enter this step.
    """
    split_cfg = replace(cfg, N_FOLDS=cfg.INNER_CROP_FOLDS, SEED=seed)
    inner = assign_folds(development, split_cfg)
    parts, audit = [], []
    for f in sorted(inner.fold.unique()):
        fitting = inner.loc[inner.fold != f].reset_index(drop=True)
        held = inner.loc[inner.fold == f].reset_index(drop=True)
        assert not set(fitting.group) & set(held.group)
        model, _ = _research_stage(fitting, None, cfg, device, "coarse",
                                   folder / f"inner_{f}", seed + int(f), epochs=epochs)
        pred = predict_research(model, held, cfg, device, stage="coarse")
        parts.append(pred)
        audit.append({"inner_fold": int(f), "fit_ids": fitting.image_id.tolist(),
                      "held_ids": held.image_id.tolist(),
                      "fit_groups": fitting.group.astype(str).tolist(),
                      "held_groups": held.group.astype(str).tolist()})
        _research_cleanup(model)
        del model
    pred = pd.concat(parts, ignore_index=True)
    assert len(pred) == len(development) and pred.image_id.is_unique
    pred = pred.set_index("image_id").loc[development.image_id].reset_index()
    pred.to_csv(folder / "training_oof_centres.csv", index=False)
    json_save(audit, folder / "crossfit_audit.json")
    return make_roi_spec(pred, development, cfg, mode="fixed")


def _research_attach_truth(pred, truth, fold, experiment):
    cols = ["image_id", "x", "y", "group", "width", "height"]
    labels = truth[cols].rename(columns={"x": "x_true", "y": "y_true"})
    base = pred.drop(columns=[c for c in ("group", "width", "height", "x_true", "y_true")
                              if c in pred], errors="ignore")
    out = base.merge(labels, on="image_id", validate="one_to_one")
    out["fold"], out["experiment"] = fold, experiment
    out["error_px"] = np.hypot(out.x - out.x_true, out.y - out.y_true)
    return out


def _research_match_fixed_roi(coarse, frame, adaptive_roi, cfg):
    target = float(np.mean(adaptive_roi.crop_side.to_numpy() ** 2))
    # Search the BASE fraction before the common margin and clamp. Starting
    # at ROI_MIN_FRACTION would make a minimum-area adaptive crop unmatchable.
    low, high = 1e-8, float(cfg.ROI_MAX_FRACTION)
    for _ in range(40):
        middle = (low + high) / 2
        area = float(np.mean(make_roi_spec(coarse, frame, cfg, mode="fixed",
                                          fixed_fraction=middle).crop_side.to_numpy() ** 2))
        if area < target:
            low = middle
        else:
            high = middle
    fraction = (low + high) / 2
    matched = float(np.mean(make_roi_spec(coarse, frame, cfg, mode="fixed",
                                         fixed_fraction=fraction).crop_side.to_numpy() ** 2))
    return fraction, {"adaptive_mean_area_px2": target,
                                  "matched_mean_area_px2": matched,
                                  "relative_area_difference": abs(matched-target)/max(target, 1e-12),
                                  "selection_source": "calibration image geometry and predictions only"}


def run_research_fold(train, test, fold, cfg, device, output, signature):
    folder = output / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "complete.json"
    if cfg.REUSE_CHECKPOINTS and complete.exists():
        record = json.loads(complete.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"{folder}: completed run has a different configuration/data/code. Use a new OUTPUT_DIR.")
        required = [f"{name}_{kind}.csv" for name in cfg.EXPERIMENTS for kind in ("oof", "test")]
        required += ["radius_calibration.json", "fallback_rule.json", "fine_policy_selection.json",
                     "split_roles.csv", "split_audit.json", "coarse_outer_validation.csv", "metrics.csv",
                     "calibration_coarse.csv", "calibration_fine.csv", "calibration_roi.csv",
                     "calibration_health.json", "area_matched_control.json",
                     "coarse_final/model.pt", "fine_final/model.pt",
                     "coarse_final/stage_complete.json", "fine_final/stage_complete.json"]
        research_verify_cache(folder, record.get("files"), required)
        print(f"Reuse completed outer fold {fold}", flush=True)
        read = lambda name, kind: pd.read_csv(folder / f"{name}_{kind}.csv", dtype={"image_id": str, "group": str})
        return ({n: read(n, "oof") for n in cfg.EXPERIMENTS},
                {n: read(n, "test") for n in cfg.EXPERIMENTS}, record)

    # A failed forced rerun must never leave a stale completion marker that
    # legitimizes a mixture of old and newly written fold outputs.
    if complete.exists():
        complete.unlink()

    outer_train = train.loc[train.fold != fold].reset_index(drop=True)
    validation = train.loc[train.fold == fold].reset_index(drop=True)
    seed = cfg.SEED + int(fold) * 1000
    development, calibration = research_group_holdout(outer_train, cfg.CALIBRATION_FRACTION, seed)
    fitting, stopping = research_group_holdout(development, cfg.EARLY_STOP_FRACTION, seed + 1)
    for a, b in ((development, calibration), (outer_train, validation), (fitting, stopping)):
        assert not set(a.group) & set(b.group)
    assignments = []
    for role, frame in (("model_fit", fitting), ("early_stop", stopping),
                        ("calibration_only", calibration), ("outer_validation", validation)):
        assignments.append(frame[["image_id", "group"]].assign(role=role))
    pd.concat(assignments).to_csv(folder / "split_roles.csv", index=False)
    json_save({"outer_fold": int(fold), "development_ids": development.image_id.tolist(),
               "calibration_ids": calibration.image_id.tolist(),
               "validation_ids": validation.image_id.tolist(),
               "protocol": "Select epochs inside development; refit on development only; hold calibration aside; outer validation used once."},
              folder / "split_audit.json")
    print(f"Outer fold {fold}: development={len(development)} (fit={len(fitting)}, stop={len(stopping)}), "
          f"calibration={len(calibration)}, validation={len(validation)}", flush=True)

    # Select epochs without ever supplying outer validation/calibration labels.
    coarse_select, coarse_info = _research_stage(fitting, stopping, cfg, device, "coarse",
                                                folder / "coarse_select", seed + 10)
    stop_coarse = predict_research(coarse_select, stopping, cfg, device, stage="coarse")
    stop_roi = make_roi_spec(stop_coarse, stopping, cfg, mode="fixed")
    _research_cleanup(coarse_select)
    del coarse_select
    coarse_epochs = int(coarse_info["best_epoch"])

    if cfg.FINE_TRAIN_CROPS == "oof":
        fit_oof_crop = research_crossfit_crops(fitting, cfg, device, folder / "crop_crossfit",
                                              coarse_epochs, seed + 100)
        # Models producing fitting crops never train on early-stop groups.
        # Stopping crops come from the model fitted on fitting groups only.
        # All final fine-training centres are consequently image-held-out,
        # albeit produced using different training-set sizes.
        train_crop = pd.concat([fit_oof_crop, stop_roi], ignore_index=True)
    else:
        train_crop = None
    fit_crop = None if train_crop is None else train_crop.loc[train_crop.image_id.isin(fitting.image_id)]
    policy_candidates = ([False, True] if cfg.SELECT_FINE_CROP_POLICY and fit_crop is not None
                         else [bool(cfg.FINE_MULTISCALE_TRAIN)])
    policy_results = []
    for multiscale in policy_candidates:
        policy_name = "multiscale" if multiscale else "fixed"
        candidate_cfg = replace(cfg, FINE_MULTISCALE_TRAIN=multiscale)
        fine_select, candidate_info = _research_stage(
            fitting, stopping, candidate_cfg, device, "fine",
            folder / ("fine_select_" + policy_name), seed + 20,
            train_crop=fit_crop, val_crop=stop_roi)
        policy_results.append({"policy": policy_name, "multiscale": multiscale,
                               "best_rmse": candidate_info["best_rmse"],
                               "best_epoch": candidate_info["best_epoch"]})
        _research_cleanup(fine_select)
        del fine_select
    # Stable ties prefer the first, fixed-crop candidate. Never inspect outer
    # validation or calibration labels when choosing this training policy.
    fine_info = min(policy_results, key=lambda result: result["best_rmse"])
    fine_cfg = replace(cfg, FINE_MULTISCALE_TRAIN=fine_info["multiscale"])
    json_save({"selection_role": "inner_early_stop_only", "candidates": policy_results,
               "selected": fine_info, "stopping_image_ids": stopping.image_id.tolist(),
               "stopping_group_ids": sorted(stopping.group.astype(str).unique().tolist())},
              folder / "fine_policy_selection.json")
    fine_epochs = int(fine_info["best_epoch"])

    coarse, _ = _research_stage(development, None, cfg, device, "coarse",
                                folder / "coarse_final", seed + 30, epochs=coarse_epochs)
    coarse_cal = predict_research(coarse, calibration, cfg, device, stage="coarse")
    coarse_val = predict_research(coarse, validation, cfg, device, stage="coarse")
    coarse_test = predict_research(coarse, test, cfg, device, stage="coarse")
    coarse_val.to_csv(folder / "coarse_outer_validation.csv", index=False)
    _research_cleanup(coarse)
    del coarse
    radius_calibration = fit_radius_calibration(coarse_cal, calibration, cfg)
    roi_cal = make_roi_spec(coarse_cal, calibration, cfg, mode="calibrated", calibration=radius_calibration)
    matched_fraction, match_info = _research_match_fixed_roi(coarse_cal, calibration, roi_cal, cfg)

    fine, _ = _research_stage(development, None, fine_cfg, device, "fine",
                              folder / "fine_final", seed + 40, train_crop=train_crop, epochs=fine_epochs)
    fine_cal = predict_research(fine, calibration, cfg, device, stage="fine", crop_spec=roi_cal)
    fallback_rule = fit_fallback_rule(coarse_cal, fine_cal, calibration, cfg)
    # Both stages were fitted WITHOUT these calibration images. Parameters fit
    # here are developmental; only outer validation measures their generalization.
    coarse_cal.to_csv(folder / "calibration_coarse.csv", index=False)
    fine_cal.to_csv(folder / "calibration_fine.csv", index=False)
    roi_cal.to_csv(folder / "calibration_roi.csv", index=False)
    diagnostics = research_calibration_diagnostics(coarse_cal, calibration, roi_cal, radius_calibration, cfg)
    json_save(diagnostics, folder / "calibration_health.json")
    for message in diagnostics.get("warnings", []):
        warnings.warn(f"Fold {fold} calibration: {message}")
    json_save(radius_calibration, folder / "radius_calibration.json")
    json_save(fallback_rule, folder / "fallback_rule.json")
    json_save({"fixed_fraction": matched_fraction, **match_info}, folder / "area_matched_control.json")

    oof_parts, test_parts, metrics = {}, {}, []
    for kind, coarse_pred, truth in (("oof", coarse_val, validation), ("test", coarse_test, test)):
        preds = {"B0": coarse_pred}
        rois = {}
        modes = {"B1": "fixed", "B2": "raw", "B3": "calibrated", "C1": "fixed"}
        needed = set(cfg.EXPERIMENTS)
        if "B4" in needed:
            needed.add("B3")
        for name, mode in modes.items():
            if name not in needed:
                continue
            roi = make_roi_spec(coarse_pred, truth, cfg, mode=mode, calibration=radius_calibration,
                                fixed_fraction=matched_fraction if name == "C1" else None)
            pred = predict_research(fine, truth, cfg, device, stage="fine", crop_spec=roi)
            rois[name], preds[name] = roi, pred
            roi.to_csv(folder / f"{name}_{kind}_roi.csv", index=False)
        if "B4" in needed:
            preds["B4"] = apply_fallback(coarse_pred, preds["B3"], fallback_rule)
            rois["B4"] = rois["B3"]
        for name in cfg.EXPERIMENTS:
            pred = preds[name].copy()
            if kind == "oof":
                part = _research_attach_truth(pred, truth, fold, name)
                if name in rois:
                    roi = rois[name].set_index("image_id").loc[part.image_id]
                    part["crop_side"] = roi.crop_side.to_numpy()
                    part["roi_contains_truth"] = ((np.abs(part.x_true.to_numpy() - roi.crop_x.to_numpy()) <= roi.crop_side.to_numpy()/2) &
                                                   (np.abs(part.y_true.to_numpy() - roi.crop_y.to_numpy()) <= roi.crop_side.to_numpy()/2))
                oof_parts[name] = part
                m = localization_metrics(pred, truth, roi=rois.get(name))
                metrics.append({"fold": int(fold), "experiment": name, **m})
            else:
                part = pred.assign(fold=fold, experiment=name)
                test_parts[name] = part
            part.to_csv(folder / f"{name}_{kind}.csv", index=False)
    _research_cleanup(fine)
    del fine
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    record = {"signature": signature, "outer_fold": int(fold), "coarse_epochs": coarse_epochs,
              "selected_fine_training_policy": fine_info["policy"],
              "fine_epochs": fine_epochs, "n_development": len(development),
              "n_calibration": len(calibration), "n_validation": len(validation), "metrics": metrics}
    files = [p for p in folder.rglob("*") if p.is_file() and p.name not in
             {"complete.json", "complete.partial.json", "uncertainty_oof_diagnostics.json"}]
    record["files"] = research_cache_manifest(folder, files)
    marker = complete.with_suffix(".partial.json")
    json_save(record, marker)
    marker.replace(complete)  # Completion marker is written atomically, last.
    return oof_parts, test_parts, record


def research_summary(oof_by_experiment, train, cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    base = oof_by_experiment.get("B0")
    for name, pred in oof_by_experiment.items():
        metrics = localization_metrics(pred, train)
        metrics.update({"experiment": name, "description": EXPERIMENT_DESCRIPTIONS[name]})
        if "roi_contains_truth" in pred:
            if pred.roi_contains_truth.isna().any() or pred.crop_side.isna().any():
                raise ValueError(f"{name}: missing ROI evaluation values")
            metrics["roi_coverage"] = float(pred.roi_contains_truth.mean())
            metrics["mean_roi_area_px2"] = float(np.mean(pred.crop_side.to_numpy() ** 2))
        summaries.append(metrics)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    reference = "B0" if base is not None else next(iter(oof_by_experiment))
    bootstrap = paired_group_bootstrap(oof_by_experiment, train, baseline=reference,
                                        n_bootstrap=int(cfg.BOOTSTRAP_REPEATS), seed=cfg.SEED)
    bootstrap.to_csv(output / "paired_group_bootstrap.csv", index=False)
    bootstrap.rename(columns={"method": "experiment", "baseline": "reference",
                              "delta_rmse": "rmse_difference",
                              "delta_ci_low": "paired_group_bootstrap_ci_low",
                              "delta_ci_high": "paired_group_bootstrap_ci_high",
                              "n_bootstrap": "bootstrap_repeats"}).to_csv(output / "paired_comparisons.csv", index=False)
    if base is not None:
        diagnostics = uncertainty_diagnostics(base, train, cfg=cfg)
        diagnostics["note"] = "Pooled OOF raw heatmap spread across separately trained folds; no cases excluded from main RMSE. Fold-specific bounded ROI coverage is evaluated separately."
        pd.DataFrame(diagnostics["risk_coverage"]).to_csv(output / "uncertainty_risk_coverage.csv", index=False)
        json_save(diagnostics, output / "uncertainty_diagnostics.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(oof_by_experiment)
    axes[0].bar(names, [localization_metrics(p, train)["rmse"] for p in oof_by_experiment.values()])
    axes[0].set(ylabel="OOF coordinate RMSE (original pixels)", title="All validation images")
    distances = []
    for pred in oof_by_experiment.values():
        p, t = _research_align(pred, train)
        distances.append(np.linalg.norm(p[["x", "y"]].to_numpy(float) - t[["x", "y"]].to_numpy(float), axis=1))
    axes[1].boxplot(distances, showfliers=True)
    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names)
    axes[1].set(ylabel="Euclidean localization error (pixels)", title="Error distribution")
    fig.tight_layout()
    fig.savefig(output / "ablation_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))
    return summary


def research_uncertainty_evaluation(oof_by_experiment, train, cfg, output):
    """Evaluate each outer fold with ONLY that fold's saved calibration.

    No labels in this function are used to refit a multiplier or choose a rule.
    Counts, rather than unweighted fold averages, form the pooled coverage.
    The curve describes bounded ROI coverage, including min/max clamps and
    margins; it is not a guarantee or a calibrated full probability density.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = oof_by_experiment.get("B0")
    if base is None:
        files = sorted(output.glob("fold_*/coarse_outer_validation.csv"))
        if files:
            frames = []
            for path in files:
                frame = pd.read_csv(path, dtype={"image_id": str, "group": str})
                fold_from_path = int(path.parent.name.removeprefix("fold_"))
                if "fold" in frame and not (frame.fold == fold_from_path).all():
                    raise ValueError(f"Stored coarse prediction fold disagrees with folder: {path}")
                frame["fold"] = fold_from_path
                frames.append(frame)
            base = pd.concat(frames, ignore_index=True)
        else:
            result = {"status": "unavailable", "reason": "Coarse B0 OOF predictions were not retained; fine-model uncertainty must not substitute for coarse ROI uncertainty."}
            json_save(result, output / "uncertainty_calibration_diagnostics.json")
            return result
    base, truth = _research_align(base, train, ("x", "y", "uncertainty", "fold"),
                                  ("x", "y", "width", "height", "group"))
    group_folds = pd.DataFrame({"group": truth.group, "fold": base.fold}).groupby("group").fold.nunique()
    if (group_folds > 1).any():
        raise ValueError("A duplicate/patient group occurs in multiple outer validation folds")
    per_fold_rows, risk_rows, fold_reports = [], [], []
    nominal_levels = (0.50, 0.80, 0.90, 0.95)
    for fold in sorted(base.fold.unique()):
        if float(fold) != int(fold):
            raise ValueError("Outer fold must be an integer")
        folder = output / f"fold_{int(fold)}"
        calibration_path = folder / "radius_calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Missing saved calibration: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        indices = np.flatnonzero(base.fold.to_numpy() == fold)
        pred_fold, truth_fold = base.iloc[indices].reset_index(drop=True), truth.iloc[indices].reset_index(drop=True)
        if set(pred_fold.image_id) & set(map(str, calibration.get("calibration_image_ids", []))):
            raise ValueError("Calibration IDs overlap outer validation IDs")
        if set(truth_fold.group) & set(map(str, calibration.get("calibration_group_ids", []))):
            raise ValueError("Calibration groups overlap outer validation groups")
        report = uncertainty_diagnostics(pred_fold, truth_fold, calibration, cfg)
        report["outer_fold"] = int(fold)
        report["n_groups"] = int(truth_fold.group.nunique())
        report["calibration_n_images"] = int(calibration["n_images"])
        report["calibration_n_groups"] = int(calibration["n_groups"])
        json_save(report, folder / "uncertainty_oof_diagnostics.json")
        fold_reports.append(report)
        for entry in report["risk_coverage"]:
            risk_rows.append({"fold": int(fold), **entry})
        alpha_map = {float(level): float(alpha) for level, alpha in calibration["alpha_by_coverage"].items()}
        for level in nominal_levels:
            if level not in alpha_map:
                raise ValueError(f"Fold {fold}: saved calibration lacks nominal level {level}; do not refit on OOF labels")
            alpha = alpha_map[level]
            roi = make_roi_spec(pred_fold, truth_fold, cfg, mode="calibrated",
                                calibration={**calibration, "alpha": alpha})
            covered = roi_contains_truth(roi, truth_fold).to_numpy(bool)
            group_covered = pd.DataFrame({"group": truth_fold.group.to_numpy(), "covered": covered}).groupby("group").covered.min()
            per_fold_rows.append({
                "fold": int(fold), "nominal_coverage": float(level), "alpha": alpha,
                "n_images": int(len(pred_fold)), "covered_images": int(covered.sum()),
                "n_groups": int(len(group_covered)), "covered_groups": int(group_covered.sum()),
                "actual_image_coverage": float(covered.mean()),
                "actual_group_coverage": float(group_covered.mean()),
                "n_min_clamped": int(roi.roi_at_min.sum()), "n_max_clamped": int(roi.roi_at_max.sum()),
                "n_padded": int(roi.roi_has_padding.sum()),
                "sum_roi_area_px2": float(roi.roi_area_px2.sum()),
                "calibration_n_groups": int(calibration["n_groups"]),
            })
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(output / "uncertainty_calibration_by_fold.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "uncertainty_risk_coverage_by_fold.csv", index=False)
    pooled_rows = []
    for level in nominal_levels:
        rows = per_fold.loc[per_fold.nominal_coverage == level]
        n_images, n_groups = int(rows.n_images.sum()), int(rows.n_groups.sum())
        covered_images, covered_groups = int(rows.covered_images.sum()), int(rows.covered_groups.sum())
        pooled_rows.append({
            "nominal_coverage": float(level), "n_images": n_images, "covered_images": covered_images,
            "n_groups": n_groups, "covered_groups": covered_groups,
            "actual_image_coverage": covered_images / n_images,
            "actual_group_coverage": covered_groups / n_groups,
            "min_clamp_rate": int(rows.n_min_clamped.sum()) / n_images,
            "max_clamp_rate": int(rows.n_max_clamped.sum()) / n_images,
            "padding_rate": int(rows.n_padded.sum()) / n_images,
            "mean_roi_area_px2": float(rows.sum_roi_area_px2.sum() / n_images),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output / "uncertainty_calibration_curve.csv", index=False)
    result = {"status": "evaluated", "n_outer_folds": int(len(fold_reports)),
              "n_images": int(len(base)), "n_groups": int(truth.group.nunique()),
              "pooled_bounded_roi_coverage": pooled_rows, "outer_fold_diagnostics": fold_reports,
              "note": "Every OOF image uses its own outer fold's held-out calibration parameters. Counts are pooled across folds. No refitting on OOF labels; margins/clamps are included. Nominal coverage is an empirical target, not a finite-sample guarantee."}
    json_save(result, output / "uncertainty_calibration_diagnostics.json")
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="Nominal target")
    axis.plot(pooled.nominal_coverage, pooled.actual_image_coverage, "o-", label="Held-out images")
    axis.plot(pooled.nominal_coverage, pooled.actual_group_coverage, "s-", label="Held-out groups (all members)")
    axis.set(xlabel="Nominal empirical coverage", ylabel="Actual bounded ROI coverage",
             title="Outer-fold calibration assessment", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "uncertainty_calibration_curve.png", dpi=160)
    plt.close(fig)
    return result


def validate_research_config(cfg):
    validate_config(cfg)
    if cfg.FINE_INPUT_SIZE < 64 or cfg.FINE_INPUT_SIZE % 32 or cfg.FINE_HEATMAP_SIZE < 8:
        raise ValueError("FINE_INPUT_SIZE must be >=64 and divisible by32; FINE_HEATMAP_SIZE >=8.")
    if cfg.COARSE_EPOCHS < 1 or cfg.FINE_EPOCHS < 1:
        raise ValueError("Both training stages require at least one epoch.")
    if not (0 < cfg.ROI_MIN_FRACTION <= cfg.ROI_MAX_FRACTION <= 1 and cfg.ROI_MARGIN_PX >= 0):
        raise ValueError("ROI fractions must satisfy 0 < min <= max <= 1; margin must be nonnegative.")
    if not 0 < cfg.ROI_COVERAGE < 1:
        raise ValueError("ROI_COVERAGE must lie strictly between zero and one.")
    if cfg.NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be nonnegative.")


def research_preflight(config):
    validate_research_config(config)
    print("Geometry:", research_geometry_selfcheck())
    if not config.SMOKE_TEST and config.REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError("请先在Kaggle Notebook设置中启用GPU。CPU流程检查可设SMOKE_TEST=True；正式训练不建议使用CPU。")
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (smoke test)")
    print("Verified pretrained weights:", resolve_research_weights(config))
    train, test, sample, layout, metadata = prepare_tables(config)
    print(f"Data: {len(train)} training images, {train.pixel_hash.nunique()} unique images, {len(test)} test images, {len(sample)} submission rows")
    print("Default outer folds:", config.N_FOLDS, "| Submission method:", config.SUBMISSION_EXPERIMENT)
    print("All completed stages are checksummed; save the output directory to reuse them in a future session.")
    return {"train_images": len(train), "test_images": len(test), "submission_rows": len(sample)}


def run_research(config):
    cfg = replace(config)
    if cfg.SMOKE_TEST:
        cfg.N_FOLDS = cfg.SMOKE_FOLDS
        cfg.COARSE_EPOCHS = cfg.FINE_EPOCHS = cfg.SMOKE_EPOCHS
        cfg.INPUT_SIZE = cfg.FINE_INPUT_SIZE = cfg.SMOKE_INPUT_SIZE
        cfg.HEATMAP_SIZE = cfg.FINE_HEATMAP_SIZE = cfg.SMOKE_INPUT_SIZE // 4
        cfg.PATIENCE = cfg.SMOKE_EPOCHS
        cfg.NUM_WORKERS, cfg.BATCH_SIZE = 0, 4
        cfg.BOOTSTRAP_REPEATS = 100
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / "smoke_test")
        cfg.SUBMISSION_PATH = str(Path(cfg.SUBMISSION_PATH).with_name("submission_SMOKE_TEST.csv"))
        warnings.warn("SMOKE TEST: reduced training, not a competitive result. Formal run requires SMOKE_TEST=False.")
    validate_research_config(cfg)
    if not set(cfg.EXPERIMENTS) <= set(EXPERIMENT_DESCRIPTIONS) or not cfg.EXPERIMENTS:
        raise ValueError("Unknown or empty EXPERIMENTS.")
    if cfg.SUBMISSION_EXPERIMENT not in cfg.EXPERIMENTS:
        raise ValueError("SUBMISSION_EXPERIMENT must be listed in EXPERIMENTS.")
    if cfg.FINE_TRAIN_CROPS not in ("oof", "jitter"):
        raise ValueError("FINE_TRAIN_CROPS must be 'oof' or 'jitter'.")
    if not 0.1 <= cfg.CALIBRATION_FRACTION <= .4 or not .1 <= cfg.EARLY_STOP_FRACTION <= .4:
        raise ValueError("Holdout fractions must be between .1 and .4.")
    if cfg.INNER_CROP_FOLDS < 2 or cfg.BOOTSTRAP_REPEATS < 100:
        raise ValueError("INNER_CROP_FOLDS >= 2 and BOOTSTRAP_REPEATS >= 100 required.")
    seed_everything(cfg.SEED)
    cv2.setNumThreads(0)
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.REQUIRE_GPU and not cfg.SMOKE_TEST:
        raise RuntimeError("Enable a Kaggle GPU accelerator for the full experiment, or explicitly set REQUIRE_GPU=False.")
    cfg.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(cfg)
    train, test, sample, layout, metadata = prepare_tables(cfg)
    if not train.visible.all():
        raise ValueError("Research protocol requires visible fovea point labels. Invisible-class modelling needs a separate validated protocol.")
    train = assign_folds(train, cfg)
    if train.fold.nunique() != cfg.N_FOLDS:
        raise ValueError("Insufficient groups for the requested number of outer folds.")
    train.to_csv(output / "folds.csv", index=False)
    test.to_csv(output / "test_manifest.csv", index=False)
    pd.DataFrame(metadata["duplicate_label_audit"]).to_csv(output / "duplicate_label_audit.csv", index=False)
    pd.concat([train[["image_id", "path"]].assign(split="train"),
               test[["image_id", "path"]].assign(split="test")]).to_csv(output / "filename_mapping.csv", index=False)
    boxes = audit_xml_boxes(train, metadata["root"], output, cfg.COORDINATE_ORIGIN)
    if cfg.SHOW_PLOTS:
        plot_label_audit(train, boxes, output, show=False)
    signature_cfg = asdict(cfg)
    for key in ("REUSE_CHECKPOINTS", "SHOW_PLOTS", "PRETRAINED_WEIGHTS_PATH"):
        signature_cfg.pop(key, None)
    fingerprint = {"version": RESEARCH_VERSION, "source_hash": globals().get("RESEARCH_SOURCE_HASH", RESEARCH_VERSION),
                   "runtime_code": research_runtime_code_digest(), "environment": research_training_environment(),
                   "config": signature_cfg,
                   "train": train[["image_id", "pixel_hash", "x", "y", "group", "fold"]].to_dict("records"),
                   "test": test[["image_id", "pixel_hash"]].to_dict("records")}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    json_save({"config": asdict(cfg), "signature": signature, "version": RESEARCH_VERSION,
               "note": "Calibration groups remain held out from BOTH final networks. Outer labels never select epochs/ROI/fallback. Test labels are unavailable and unused."},
              output / "run_config.json")
    json_save({"python": sys.version, "torch": str(torch.__version__), "device": str(device),
               "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
               "platform": platform.platform()}, output / "environment.json")
    oof_all = {n: [] for n in cfg.EXPERIMENTS}
    test_all = {n: [] for n in cfg.EXPERIMENTS}
    folds = []
    for fold in sorted(train.fold.unique()):
        oof, tests, record = run_research_fold(train, test, int(fold), cfg, device, output, signature)
        folds.append(record)
        for name in cfg.EXPERIMENTS:
            oof_all[name].append(oof[name])
            test_all[name].append(tests[name])
    oof_combined = {}
    submissions = {}
    for name in cfg.EXPERIMENTS:
        pred = pd.concat(oof_all[name], ignore_index=True)
        if len(pred) != len(train) or not pred.image_id.is_unique:
            raise AssertionError("OOF must contain each labelled image exactly once.")
        pred = pred.set_index("image_id").loc[train.image_id].reset_index()
        oof_combined[name] = pred
        pred.to_csv(output / f"oof_{name}.csv", index=False)
        all_test = pd.concat(test_all[name], ignore_index=True)
        if not (all_test.groupby("image_id").size() == cfg.N_FOLDS).all():
            raise AssertionError("Each test image requires one prediction from every outer fold.")
        all_test.to_csv(output / f"test_all_folds_{name}.csv", index=False)
        average = all_test.groupby("image_id", as_index=False)[["x", "y"]].mean()
        ensemble = test[["image_id", "width", "height"]].merge(average, on="image_id", validate="one_to_one")
        ensemble["visibility_prob"] = 1.0
        ensemble.to_csv(output / f"test_ensemble_{name}.csv", index=False)
        export_cfg = replace(cfg, SUBMISSION_PATH=str(output / f"submission_{name}.csv"))
        submissions[name] = write_submission(sample, layout, ensemble, metadata, export_cfg, .5)
    summary = research_summary(oof_combined, train, cfg, output)
    research_uncertainty_evaluation(oof_combined, train, cfg, output)
    chosen = submissions[cfg.SUBMISSION_EXPERIMENT]
    Path(cfg.SUBMISSION_PATH).parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(cfg.SUBMISSION_PATH, index=False, float_format="%.6f")
    diagnostic = oof_combined[cfg.SUBMISSION_EXPERIMENT].copy()
    diagnostic["visibility_prob"], diagnostic["visible_true"] = 1.0, True
    plot_oof_errors(diagnostic, train, .5, cfg, output)
    json_save({"version": RESEARCH_VERSION, "smoke_test": cfg.SMOKE_TEST,
               "n_folds": cfg.N_FOLDS, "n_train": len(train), "n_test": len(test),
               "submission_rows": len(chosen), "submission_experiment": cfg.SUBMISSION_EXPERIMENT,
               "submission_path": cfg.SUBMISSION_PATH, "outer_folds": folds,
               "uncertainty": "Empirical error-radius calibration, not a theoretical conformal coverage guarantee.",
               "selection": "Submission method preselected in configuration, not auto-selected from outer OOF."},
              output / "run_summary.json")
    print(f"Complete: {cfg.SUBMISSION_PATH}")
    return {"config": cfg, "train": train, "test": test, "oof": oof_combined,
            "summary": summary, "submission": chosen, "submissions": submissions}
