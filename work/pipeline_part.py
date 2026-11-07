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
