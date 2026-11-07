"""Wider single-window refinement with selection-then-full-refit roles.

Append only to the original three-channel research engine. The previous dual
engine is passed explicitly for read-only SINGLE checkpoint inference. Import
does not start training or access external evaluation references.
"""
WIDE_VERSION = '7.0.0-wide-single-refit-20261027'
RESEARCH_VERSION = WIDE_VERSION
WIDE_SPLIT_SEEDS = (2026, 2027)
WIDE_FRACTION = .25
WIDE_WEIGHT = .25
WIDE_MIN_GAIN_PX = .5
WIDE_MIN_REL_GAIN = .02
WIDE_METHODS = ('COARSE', 'SINGLE', 'WIDE_DEV', 'WIDE_FULL', 'MATCHED_DEPLOY')
WIDE_PRIMARY_EXTRA = ('PREVIOUS_SUBMISSION', 'SUBMISSION_CANDIDATE')


def wr_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def wr_read(path):
    return pd.read_csv(path, dtype={'image_id': str, 'group': str})


def wr_align(prediction, ids):
    ids = list(map(str, ids))
    out = prediction.copy()
    out['image_id'] = out.image_id.astype(str)
    if out.image_id.duplicated().any() or len(ids) != len(set(ids)) or set(out.image_id) != set(ids):
        raise ValueError('Prediction/manifest IDs must be unique and identical.')
    return out.set_index('image_id').loc[ids].reset_index()


def wr_roi(centres, frame):
    """Prediction-centred exact quarter-short-side ROI; no label repair."""
    centres = wr_align(centres, frame.image_id)
    centre_columns = ['crop_x', 'crop_y'] if {'crop_x', 'crop_y'} <= set(centres) else ['x', 'y']
    xy = centres[centre_columns].to_numpy(float)
    dimensions = frame[['width', 'height']].to_numpy(float)
    if not np.isfinite(xy).all() or not np.isfinite(dimensions).all() or (dimensions <= 1).any():
        raise ValueError('Invalid source centres or image dimensions.')
    return pd.DataFrame({'image_id': frame.image_id.astype(str).to_numpy(),
                         'crop_x': xy[:, 0], 'crop_y': xy[:, 1],
                         'crop_side': WIDE_FRACTION * dimensions.min(axis=1)})


def wr_blend(base, wide, weight=WIDE_WEIGHT):
    wide = wr_align(wide, base.image_id)
    out = base[['image_id', 'x', 'y']].copy()
    out[['x', 'y']] = (1-weight)*base[['x', 'y']].to_numpy(float) + weight*wide[['x', 'y']].to_numpy(float)
    return out


def wr_metrics(prediction, truth, coarse=None, roi=None, source_roi=None):
    prediction = wr_align(prediction, truth.image_id)
    y = truth[['x', 'y']].to_numpy(float)
    residual = prediction[['x', 'y']].to_numpy(float) - y
    if not len(y) or not np.isfinite(residual).all():
        raise ValueError('Metrics require finite paired coordinates.')
    distance = np.linalg.norm(residual, axis=1)
    result = {'rmse': float(np.sqrt(np.mean(residual**2))),
              'mean_distance_px': float(distance.mean()), 'median_distance_px': float(np.median(distance)),
              'p90_distance_px': float(np.quantile(distance, .9)), 'p95_distance_px': float(np.quantile(distance, .95)),
              'max_distance_px': float(distance.max()), 'images': len(distance)}
    if coarse is not None:
        coarse = wr_align(coarse, truth.image_id)
        coarse_distance = np.linalg.norm(coarse[['x', 'y']].to_numpy(float)-y, axis=1)
        result['worse_than_coarse_fraction'] = float(np.mean(distance > coarse_distance+1e-9))
    for spec, key in ((roi, 'roi_coverage'), (source_roi, 'old_single_roi_coverage')):
        if spec is not None:
            spec = wr_align(spec, truth.image_id)
            inside = (np.abs(y-spec[['crop_x', 'crop_y']].to_numpy(float)) <= spec.crop_side.to_numpy(float)[:, None]/2).all(axis=1)
            result[key] = float(inside.mean())
    if roi is not None:
        result['roi_coverage_scope'] = 'current .25 proposal coverage; blends have no single physical ROI'
    return result


def wr_roles(train, fold, source_folder):
    """Preserve the old groups and explicitly relabel calibration reuse."""
    source_roles = wr_read(Path(source_folder)/'split_roles.csv')
    allowed = {'model_fit', 'early_stop', 'calibration_only', 'outer_validation'}
    if (not source_roles.image_id.is_unique or set(source_roles.image_id) != set(train.image_id) or
            set(source_roles.role) != allowed or not (source_roles.groupby('group').role.nunique() == 1).all()):
        raise ValueError('Invalid source role partition.')
    keyed = source_roles.set_index('image_id').loc[train.image_id]
    if not np.array_equal(keyed.group.astype(str).to_numpy(), train.group.astype(str).to_numpy()):
        raise ValueError('Source groups changed.')
    def take(role):
        ids = set(source_roles.loc[source_roles.role == role, 'image_id'])
        return train.loc[train.image_id.isin(ids)].reset_index(drop=True)
    fitting, stopping, selection, outer = [take(role) for role in
                                         ('model_fit', 'early_stop', 'calibration_only', 'outer_validation')]
    development = train.loc[train.image_id.isin(set(fitting.image_id)|set(stopping.image_id))].reset_index(drop=True)
    full = train.loc[train.fold != fold].reset_index(drop=True)
    if set(outer.image_id) != set(train.loc[train.fold == fold, 'image_id']):
        raise ValueError('Source outer fold differs.')
    if set(full.image_id) != set(development.image_id)|set(selection.image_id):
        raise ValueError('Full refit must contain development plus the former calibration group.')
    if set(full.group) & set(outer.group) or set(development.group) & set(selection.group):
        raise ValueError('Group roles overlap.')
    roles = source_roles.copy()
    roles.loc[roles.role == 'calibration_only', 'role'] = 'selection_then_final_fit'
    return fitting, stopping, development, selection, full, outer, source_roles, roles


def wr_source(train, test, fold, cfg, folder, source_folder, roles_tuple):
    """Verify reusable predictions/centres and their fitting-only crossfit lineage."""
    source_folder = Path(source_folder)
    fitting, stopping, development, selection, full, outer, source_roles, roles = roles_tuple
    complete = wr_json(source_folder/'complete.json')
    if complete['status'] != 'COMPLETE' or complete['fold'] != fold or complete['split_seed'] != cfg.SEED:
        raise ValueError('Source fold is not the expected completed experiment.')
    files = complete['files']
    required = ['split_roles.csv', 'training_centres.csv', 'coarse_calibration.csv', 'coarse_oof.csv',
                'coarse_lineage.json', 'crop_crossfit_audit.json', 'single/config.json',
                'single/fine_final/model.pt', 'single/fine_final/actual_training_inputs.csv',
                'SINGLE_oof.csv', 'oof_roi.csv']
    if cfg.SEED == 2026:
        required += ['coarse_test.csv', 'SINGLE_test.csv', 'SUBMISSION_CANDIDATE_oof.csv',
                     'SUBMISSION_CANDIDATE_test.csv', 'test_roi.csv']
    research_verify_cache(source_folder, {name: files[name] for name in required}, required)
    source_roles.to_csv(folder/'source_roles.csv', index=False)
    roles.to_csv(folder/'split_roles.csv', index=False)
    crossfit = wr_json(source_folder/'crop_crossfit_audit.json')
    seen = []
    for record in crossfit:
        fit_ids, held_ids = set(record['fit_ids']), set(record['held_ids'])
        if (fit_ids & held_ids or not (fit_ids|held_ids) <= set(fitting.image_id) or
                set(record['fit_groups']) & set(record['held_groups'])):
            raise ValueError('Fine centre crossfit is not fitting-only group-held-out.')
        seen.extend(record['held_ids'])
    if len(seen) != len(set(seen)) or set(seen) != set(fitting.image_id):
        raise ValueError('Crossfit must predict every fitting image once.')
    old_lineage = wr_json(source_folder/'coarse_lineage.json')
    coarse_checkpoint = Path(old_lineage['coarse_checkpoint'])
    coarse_stage = coarse_checkpoint.parent
    if cfg.SEED == 2026:
        coarse_parent = Path(old_lineage['source'])
        coarse_manifest = wr_json(coarse_parent/'complete.json')['files']
    else:
        coarse_parent = source_folder
        coarse_manifest = files
    coarse_required = ['coarse_final/model.pt', 'coarse_final/actual_training_inputs.csv']
    research_verify_cache(coarse_parent, {name: coarse_manifest[name] for name in coarse_required}, coarse_required)
    if coarse_stage.resolve() != (coarse_parent/'coarse_final').resolve():
        raise ValueError('Unexpected inherited coarse checkpoint path.')
    coarse_fit = wr_read(coarse_parent/'coarse_final/actual_training_inputs.csv')
    single_fit = wr_read(source_folder/'single/fine_final/actual_training_inputs.csv')
    if set(coarse_fit.image_id) != set(development.image_id) or set(single_fit.image_id) != set(development.image_id):
        raise ValueError('Inherited coarse/SINGLE saw an unexpected training role.')
    for actual in (coarse_fit, single_fit):
        if set(actual.image_id) & (set(selection.image_id)|set(outer.image_id)|set(test.image_id)):
            raise ValueError('Inherited reference leaked selection, outer or test images.')
    development_centres = wr_roi(wr_read(source_folder/'training_centres.csv'), development)
    coarse = {'selection': wr_align(wr_read(source_folder/'coarse_calibration.csv'), selection.image_id),
              'oof': wr_align(wr_read(source_folder/'coarse_oof.csv'), outer.image_id)}
    if cfg.SEED == 2026:
        coarse['test'] = wr_align(wr_read(source_folder/'coarse_test.csv'), test.image_id)
    extra_centres = wr_roi(coarse['selection'], selection)
    full_centres = wr_align(pd.concat([development_centres, extra_centres], ignore_index=True), full.image_id)
    development_centres.to_csv(folder/'training_centres_development.csv', index=False)
    full_centres.to_csv(folder/'training_centres_full.csv', index=False)
    for name, prediction in coarse.items():
        prediction.to_csv(folder/f'coarse_{name}.csv', index=False)
    json_save(crossfit, folder/'crop_crossfit_audit.json')
    lineage = {'source_folder': str(source_folder), 'source_complete_sha256': research_file_digest(source_folder/'complete.json'),
               'verified_source_files': {name: files[name] for name in required},
               'coarse_checkpoint': str(coarse_checkpoint), 'coarse_parent': str(coarse_parent),
               'verified_coarse_files': {name: coarse_manifest[name] for name in coarse_required},
               'fitting_centres': 'group-held-out crossfit within model_fit only',
               'stopping_centres': 'preserved from source coarse_select held stopping predictions',
               'extra_full_refit_centres': 'source coarse_final predictions excluded original calibration groups',
               'old_single_training_ids': single_fit.image_id.tolist(), 'old_coarse_training_ids': coarse_fit.image_id.tolist(),
               'source_calibration_role_reused': 'model selection then included in full final refit; never called independent calibration afterwards'}
    json_save(lineage, folder/'coarse_lineage.json')
    return development_centres, full_centres, coarse, lineage


def wr_fold(train, test, fold, cfg, output, previous_dir, old_engine):
    folder = Path(output)/f'fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(previous_dir)/f'split_{cfg.SEED}'/f'fold_{fold}'
    role_data = wr_roles(train, fold, source)
    fitting, stopping, development, selection_group, full, outer, _, _ = role_data
    json_save({'fold': fold, 'split_seed': cfg.SEED, 'config': asdict(cfg)}, folder/'fold_config.json')
    centres_dev, centres_full, coarse, lineage = wr_source(train, test, fold, cfg, folder, source, role_data)
    fit_crop = centres_dev.loc[centres_dev.image_id.isin(fitting.image_id)].reset_index(drop=True)
    stop_crop = centres_dev.loc[centres_dev.image_id.isin(stopping.image_id)].reset_index(drop=True)
    seed = cfg.SEED + fold*1000
    print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5 | wide_select | fit={len(fitting)} stop={len(stopping)} selection_then_refit={len(selection_group)} outer={len(outer)}', flush=True)
    selected, selected_info = _research_stage(fitting, stopping, cfg, 'cuda', 'fine', folder/'wide_select', seed+20,
                                               train_crop=fit_crop, val_crop=stop_crop)
    stop_prediction = predict_research(selected, stopping, cfg, 'cuda', 'fine', stop_crop)
    stop_prediction.to_csv(folder/'selected_stop_predictions.csv', index=False)
    if abs(wr_metrics(stop_prediction, stopping)['rmse']-selected_info['best_rmse']) >= 1e-3:
        raise ValueError('Selected stopping prediction did not reproduce early-stop metric.')
    _research_cleanup(selected)
    del selected
    epochs = int(selected_info['best_epoch'])
    print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5 | wide_dev | fixed {epochs} epochs on {len(development)} images', flush=True)
    wide_dev, _ = _research_stage(development, None, cfg, 'cuda', 'fine', folder/'wide_dev', seed+40,
                                  train_crop=centres_dev, epochs=epochs)
    roi_selection = wr_roi(coarse['selection'], selection_group)
    roi_selection.to_csv(folder/'calibration_roi.csv', index=False)
    wide_selection = predict_research(wide_dev, selection_group, cfg, 'cuda', 'fine', roi_selection)
    wide_selection.to_csv(folder/'calibration_wide_dev.csv', index=False)
    _research_cleanup(wide_dev)
    del wide_dev
    # Read-only comparison to the OLD matched SINGLE on the identical images,
    # using that model's original six-channel loader and .15 deployment ROI.
    old_config = old_engine.DualConfig(**wr_json(source/'single/config.json')['config'])
    old_config = old_engine.replace(old_config, PRETRAINED=False)
    if old_config.FINE_VARIANT != 'single' or old_config.SEED != cfg.SEED:
        raise ValueError('Expected the saved matched SINGLE reference.')
    old_single = old_engine.research_build_model(old_config, 'fine', 'cuda')
    old_engine.load_model_checkpoint(old_single, source/'single/fine_final/model.pt', 'cuda')
    old_roi = old_engine.dc_roi(coarse['selection'], selection_group)
    old_roi.to_csv(folder/'calibration_single_roi.csv', index=False)
    single_selection = old_engine.predict_research(old_single, selection_group, old_config, 'cuda', 'fine', old_roi)
    single_selection.to_csv(folder/'calibration_single.csv', index=False)
    _research_cleanup(old_single)
    del old_single
    single_metrics = wr_metrics(single_selection, selection_group)
    wide_metrics = wr_metrics(wide_selection, selection_group)
    gain = single_metrics['rmse'] - wide_metrics['rmse']
    accepted = bool(gain >= WIDE_MIN_GAIN_PX and gain >= WIDE_MIN_REL_GAIN*single_metrics['rmse'] and
                    wide_metrics['p90_distance_px'] <= single_metrics['p90_distance_px'] + 1e-9)
    decision = {'fold': fold, 'split_seed': cfg.SEED, 'accepted': accepted,
                'selected_epoch': epochs, 'early_stop_best_rmse': float(selected_info['best_rmse']),
                'stopping_ids': stopping.image_id.tolist(), 'selection_ids': selection_group.image_id.tolist(),
                'selection_groups': selection_group.group.astype(str).unique().tolist(),
                'single_metrics': single_metrics, 'wide_dev_metrics': wide_metrics, 'gain_px': gain,
                'minimum_gain_px': WIDE_MIN_GAIN_PX, 'minimum_relative_gain': WIDE_MIN_REL_GAIN,
                'p90_must_not_increase': True, 'weight_if_accepted': WIDE_WEIGHT,
                'decision_inputs': 'original held calibration groups, now selection_then_final_fit; no outer/test labels',
                'comparison': 'old SINGLE .15/256/64 against WIDE_DEV .25/384/96, before adding selection images',
                'full_refit_policy': 'add original calibration groups after freezing this decision; reuse selected epochs, never any old learned risk/coverage calibration',
                'full_training_ids': full.image_id.tolist()}
    # This gate record is persisted before FULL fitting and new outer inference.
    json_save(decision, folder/'selection.json')
    print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5 | gate accepted={accepted}, gain={gain:.4f}px | wide_full {epochs} epochs on {len(full)} images', flush=True)
    wide_full, _ = _research_stage(full, None, cfg, 'cuda', 'fine', folder/'wide_full', seed+40,
                                   train_crop=centres_full, epochs=epochs)
    _research_cleanup(wide_full)
    del wide_full
    outputs, metric_rows = {}, []
    for kind, frame in [('oof', outer)] + ([('test', test)] if cfg.SEED == 2026 else []):
        cp = coarse[kind]
        roi = wr_roi(cp, frame)
        source_roi = wr_align(wr_read(source/f'{kind}_roi.csv'), frame.image_id)
        roi.to_csv(folder/f'{kind}_roi.csv', index=False)
        source_roi.to_csv(folder/f'source_single_{kind}_roi.csv', index=False)
        pieces = {'COARSE': cp, 'SINGLE': wr_align(wr_read(source/f'SINGLE_{kind}.csv'), frame.image_id)}
        for name, stage in ([('WIDE_DEV', 'wide_dev'), ('WIDE_FULL', 'wide_full')] if kind == 'oof' else [('WIDE_FULL', 'wide_full')]):
            model = research_build_model(replace(cfg, PRETRAINED=False), 'fine', 'cuda')
            load_model_checkpoint(model, folder/stage/'model.pt', 'cuda')
            pieces[name] = predict_research(model, frame, cfg, 'cuda', 'fine', roi)
            _research_cleanup(model)
            del model
        pieces['MATCHED_DEPLOY'] = (wr_blend(pieces['SINGLE'], pieces['WIDE_FULL']) if accepted
                                    else pieces['SINGLE'][['image_id', 'x', 'y']].copy())
        if cfg.SEED == 2026:
            previous = wr_align(wr_read(source/f'SUBMISSION_CANDIDATE_{kind}.csv'), frame.image_id)
            pieces['PREVIOUS_SUBMISSION'] = previous
            pieces['SUBMISSION_CANDIDATE'] = (wr_blend(previous, pieces['WIDE_FULL']) if accepted
                                               else previous[['image_id', 'x', 'y']].copy())
        for method, prediction in pieces.items():
            saved = (_research_attach_truth(prediction, frame, fold, method) if kind == 'oof'
                     else prediction.assign(fold=fold, experiment=method))
            saved.to_csv(folder/f'{method}_{kind}.csv', index=False)
            outputs[(method, kind)] = saved
            if kind == 'oof':
                metric_rows.append({'split_seed': cfg.SEED, 'fold': fold, 'method': method,
                                     **wr_metrics(prediction, frame, cp, roi, source_roi)})
    pd.DataFrame(metric_rows).to_csv(folder/'metrics.csv', index=False)
    artifacts = [path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.json', '.csv', '.pt') and path.name != 'complete.json']
    json_save({'status': 'COMPLETE', 'fold': fold, 'split_seed': cfg.SEED, 'selection': decision,
               'full_refit_has_no_independent_calibration': True,
               'files': research_cache_manifest(folder, artifacts)}, folder/'complete.json')
    print(f'COMPLETE SPLIT {cfg.SEED} FOLD {fold+1}/5 | wide accepted={accepted}', flush=True)
    return outputs, decision


def wr_export(sample_template, ensemble, test, target):
    """Create all values from model predictions; template values are ignored."""
    submission = sample_template.copy()
    if list(submission.columns) != ['ImageID', 'value'] or len(submission) != 40 or not submission.ImageID.is_unique:
        raise ValueError('Expected the competition 40-row ImageID/value template.')
    ensemble = wr_align(ensemble, test.image_id).set_index('image_id')
    dimensions = test.set_index('image_id')
    values, seen = [], set()
    for label in submission.ImageID:
        iid, axis = str(label).split('_Fovea_')
        if axis not in ('X', 'Y') or iid not in ensemble.index:
            raise ValueError('Unknown template coordinate ID.')
        seen.add((iid, axis))
        value = float(ensemble.loc[iid, 'x' if axis == 'X' else 'y'])
        if not np.isfinite(value) or not 0 <= value <= dimensions.loc[iid, 'width' if axis == 'X' else 'height']-1:
            raise ValueError('Submission coordinate outside the image.')
        values.append(value)
    if seen != {(str(iid), axis) for iid in test.image_id for axis in ('X', 'Y')}:
        raise ValueError('Template must cover both coordinates of every test image.')
    submission['value'] = values
    submission.to_csv(target, index=False, float_format='%.6f')
    return submission


def run_wide_refine(config, previous_dir, previous_submission, old_engine):
    cfg = replace(config)
    output, previous_dir = Path(cfg.OUTPUT_DIR), Path(previous_dir)
    if output.resolve() == previous_dir.resolve() or previous_dir.resolve() in output.resolve().parents:
        raise ValueError('New experiment output must not be inside the previous experiment.')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'execution_report.json').exists() and wr_json(output/'execution_report.json').get('status') == 'COMPLETE':
        raise RuntimeError('This experiment is already complete; use a new directory rather than overwrite it.')
    assert torch.cuda.is_available() and not cfg.SMOKE_TEST and cfg.N_FOLDS == 5
    assert cfg.FINE_INPUT_SIZE == 384 and cfg.FINE_HEATMAP_SIZE == 96 and cfg.HEATMAP_SIGMA == 1.5
    assert not cfg.FINE_MULTISCALE_TRAIN and not cfg.FINE_MULTISCALE_EARLY_STOP
    assert cfg.FINE_EPOCHS == 30 and cfg.PATIENCE == 7 and cfg.BATCH_SIZE == 8 and cfg.MODEL_SEED_OFFSET == 10000
    assert cfg.ROI_MARGIN_PX == 0 and cfg.FIXED_ROI_FRACTION == .25
    cfg.PRETRAINED_WEIGHTS_PATH = resolve_research_weights(cfg)
    assert research_file_digest(cfg.PRETRAINED_WEIGHTS_PATH) == RESEARCH_WEIGHT_SHA256
    prior_execution = wr_json(previous_dir/'execution_report.json')
    prior_validation = wr_json(previous_dir/'validation_report.json')
    assert prior_execution['status'] == 'COMPLETE' and prior_validation['status'] == 'PASS'
    assert research_file_digest(previous_submission) == prior_execution['submission_sha256'] == prior_validation['submission_sha256']
    seed_everything(cfg.SEED)
    cv2.setNumThreads(0)
    train, test, sample, layout, metadata = prepare_tables(cfg)
    assert len(train) == 80 and len(test) == 20 and not set(train.pixel_hash) & set(test.pixel_hash)
    assert not set(train.image_id) & set(test.image_id)
    test.to_csv(output/'test_manifest.csv', index=False)
    sample.to_csv(output/'sample_template.csv', index=False)
    protocol = {'version': WIDE_VERSION, 'config': asdict(cfg), 'split_seeds': list(WIDE_SPLIT_SEEDS),
                'new_training_stages': 30, 'roi_short_side_fraction': WIDE_FRACTION,
                'gate_min_px': WIDE_MIN_GAIN_PX, 'gate_min_relative': WIDE_MIN_REL_GAIN,
                'gate_p90_must_not_increase': True, 'new_branch_weight': WIDE_WEIGHT,
                'previous_dir': str(previous_dir), 'previous_submission': str(previous_submission),
                'previous_submission_sha256': research_file_digest(previous_submission),
                'runtime_digest': research_runtime_code_digest(),
                'secondary_scope': 'complete fivefold stability audit only; no test inference or influence on primary submission',
                'role_change': 'old calibration_only selects WIDE_DEV versus old SINGLE, then joins WIDE_FULL fixed-epoch refit',
                'full_refit_uncertainty_calibration': 'none; old radius/risk rules are not reused on WIDE_FULL',
                'test_reference_labels_used': False}
    json_save(protocol, output/'run_protocol.json')
    started, summaries, decisions, submission = time.perf_counter(), [], {}, None
    for split_seed in WIDE_SPLIT_SEEDS:
        split_output = output/f'split_{split_seed}'
        split_output.mkdir(exist_ok=True)
        split_cfg = replace(cfg, SEED=split_seed, OUTPUT_DIR=str(split_output))
        split_train = assign_folds(train, split_cfg)
        old_folds = wr_align(wr_read(previous_dir/f'split_{split_seed}/folds.csv'), split_train.image_id)
        for column in ('fold', 'group', 'pixel_hash', 'x', 'y'):
            assert np.array_equal(split_train[column], old_folds[column]), column
        split_train.to_csv(split_output/'folds.csv', index=False)
        parts, decisions[str(split_seed)] = defaultdict(list), []
        for fold in range(5):
            fold_parts, decision = wr_fold(split_train, test, fold, split_cfg, split_output, previous_dir, old_engine)
            decisions[str(split_seed)].append(decision)
            for key, frame in fold_parts.items():
                parts[key].append(frame)
        coarse_oof = pd.concat(parts[('COARSE', 'oof')], ignore_index=True)
        roi = pd.concat([wr_read(split_output/f'fold_{fold}/oof_roi.csv') for fold in range(5)], ignore_index=True)
        source_roi = pd.concat([wr_read(split_output/f'fold_{fold}/source_single_oof_roi.csv') for fold in range(5)], ignore_index=True)
        for (method, kind), frames in parts.items():
            combined = pd.concat(frames, ignore_index=True)
            if kind == 'oof':
                assert len(combined) == 80 and combined.image_id.is_unique
                combined.to_csv(split_output/f'oof_{method}.csv', index=False)
                summaries.append({'split_seed': split_seed, 'method': method,
                                  **wr_metrics(combined, split_train, coarse_oof, roi, source_roi)})
            else:
                assert split_seed == 2026 and method != 'WIDE_DEV'
                assert len(combined) == 100 and (combined.groupby('image_id').size() == 5).all()
                combined.to_csv(split_output/f'test_folds_{method}.csv', index=False)
                ensemble = combined.groupby('image_id', as_index=False)[['x', 'y']].mean()
                ensemble.to_csv(split_output/f'test_ensemble_{method}.csv', index=False)
                if method == 'PREVIOUS_SUBMISSION':
                    reconstructed = wr_export(sample, ensemble, test, output/'previous_reconstructed.csv')
                    previous = pd.read_csv(previous_submission).set_index('ImageID').loc[reconstructed.ImageID]
                    assert np.max(np.abs(reconstructed.value.to_numpy()-previous.value.to_numpy())) < 1.1e-6
                if method == 'SUBMISSION_CANDIDATE':
                    submission = wr_export(sample, ensemble, test, output/'submission.csv')
        pd.DataFrame([row for row in summaries if row['split_seed'] == split_seed]).to_csv(split_output/'summary.csv', index=False)
    assert submission is not None
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'ablation_summary.csv', index=False)
    json_save({'status': 'COMPLETE', 'split_seeds': list(WIDE_SPLIT_SEEDS), 'completed_folds': 10,
               'new_training_stages': 30, 'decisions': decisions,
               'submission_sha256': research_file_digest(output/'submission.csv'),
               'elapsed_seconds': time.perf_counter()-started,
               'peak_cuda_mb': torch.cuda.max_memory_allocated()/2**20, 'gpu': torch.cuda.get_device_name(0),
               'full_refit_has_no_independent_calibration': True,
               'secondary_submission_influence': False}, output/'execution_report.json')
    print(summary.to_string(index=False), flush=True)
    print('COMPLETE SUBMISSION:', output/'submission.csv', flush=True)
    return {'summary': summary, 'submission': submission, 'decisions': decisions}
