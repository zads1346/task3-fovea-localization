"""CPU-only checks for wide ROI geometry and reused training-side provenance.

The importing run.py has no training side effects. All coordinates used here
are labelled training data or previously stored training-side coarse outputs.
No external score/reference files or test images are read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from run import *
import time


def read(path):
    return m.pd.read_csv(path, dtype={'image_id': str, 'group': str})


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def align(frame, ids):
    assert frame.image_id.is_unique and set(frame.image_id) == set(ids)
    return frame.set_index('image_id').loc[list(ids)].reset_index()


def main():
    started = time.perf_counter()
    assert not torch.cuda.is_available(), 'This validation must not use the GPU.'
    assert cfg.FINE_INPUT_SIZE == 384 and cfg.FINE_HEATMAP_SIZE == 96
    assert cfg.HEATMAP_SIGMA == 1.5 and cfg.INPUT_SIZE == 256 and cfg.HEATMAP_SIZE == 64
    assert not cfg.FINE_MULTISCALE_TRAIN and not cfg.FINE_MULTISCALE_EARLY_STOP
    assert not cfg.SELECT_FINE_CROP_POLICY and cfg.ROI_MARGIN_PX == 0
    assert cfg.OOF_CENTER_JITTER == .08
    assert m.research_file_digest(cfg.PRETRAINED_WEIGHTS_PATH) == m.RESEARCH_WEIGHT_SHA256
    folds = []
    example = None
    coverage = []
    all_source_paths = set()
    for split_seed in (2026, 2027):
        source = PREVIOUS / f'split_{split_seed}'
        train_path = source / 'folds.csv'
        train = read(train_path)
        all_source_paths.add(train_path)
        assert len(train) == 80 and train.image_id.is_unique
        assert set(train.image_id) == {str(i) for i in range(1, 81)}
        oof_prediction = read(source / 'oof_COARSE.csv')
        all_source_paths.add(source / 'oof_COARSE.csv')
        new_roi = m.wr_roi(oof_prediction, train)
        reordered = train.iloc[::-1].reset_index(drop=True)
        reversed_roi = m.wr_roi(oof_prediction.iloc[::-1], reordered)
        assert list(reversed_roi.image_id) == list(reordered.image_id)
        assert m.np.array_equal(align(reversed_roi, train.image_id)[['crop_x','crop_y','crop_side']],
                               new_roi[['crop_x','crop_y','crop_side']])
        source_aligned = align(oof_prediction, train.image_id)
        assert m.np.array_equal(new_roi[['crop_x','crop_y']].to_numpy(),
                               source_aligned[['x','y']].to_numpy())
        expected_side = .25 * train[['width','height']].min(axis=1).to_numpy(float)
        assert m.np.array_equal(new_roi.crop_side.to_numpy(), expected_side)
        bare = train[['image_id','path','width','height']].copy()
        truth_mutated = train.copy()
        truth_mutated['x'] = -12345.
        truth_mutated['y'] = 99999.
        for frame in (bare, truth_mutated):
            candidate = m.wr_roi(oof_prediction, frame)
            assert candidate.equals(new_roi), 'ROI construction must not depend on labels.'
        difference = m.np.abs(train[['x','y']].to_numpy(float) - new_roi[['crop_x','crop_y']].to_numpy(float)).max(axis=1)
        margins = new_roi.crop_side.to_numpy()/2 - difference
        coverage.append({'split_seed': split_seed, 'covered': int((margins >= 0).sum()),
                         'total': len(train), 'minimum_edge_margin_px': float(margins.min()),
                         'side_min_px': float(new_roi.crop_side.min()),
                         'side_max_px': float(new_roi.crop_side.max())})
        for fold in range(5):
            folder = source / f'fold_{fold}'
            role_path = folder / 'split_roles.csv'
            centre_path = folder / 'training_centres.csv'
            cal_path = folder / 'coarse_calibration.csv'
            lineage_path = folder / 'coarse_lineage.json'
            crossfit_path = folder / 'crop_crossfit_audit.json'
            all_source_paths.update([role_path, centre_path, cal_path, lineage_path, crossfit_path])
            roles = read(role_path)
            role_ids = {name: set(roles.loc[roles.role == name, 'image_id']) for name in
                        ('model_fit','early_stop','calibration_only','outer_validation')}
            fit_ids, stop_ids, cal_ids, outer_ids = [role_ids[name] for name in
                        ('model_fit','early_stop','calibration_only','outer_validation')]
            assert roles.image_id.is_unique and set(roles.image_id) == set(train.image_id)
            assert (roles.groupby('group').role.nunique() == 1).all()
            development_ids = fit_ids | stop_ids
            full_ids = development_ids | cal_ids
            assert len(full_ids) == 64 and not full_ids & outer_ids
            centres = read(centre_path)
            assert centres.image_id.is_unique and set(centres.image_id) == development_ids
            cal_predictions = read(cal_path)
            assert cal_predictions.image_id.is_unique and set(cal_predictions.image_id) == cal_ids
            development = train.loc[train.image_id.isin(development_ids)].reset_index(drop=True)
            selection = train.loc[train.image_id.isin(cal_ids)].reset_index(drop=True)
            new_development_roi = m.wr_roi(centres, development)
            new_cal_roi = m.wr_roi(cal_predictions, selection)
            expected_centres = align(centres, development.image_id)
            assert m.np.array_equal(new_development_roi[['crop_x','crop_y']].to_numpy(),
                                   expected_centres[['crop_x','crop_y']].to_numpy())
            full = train.loc[train.image_id.isin(full_ids)].reset_index(drop=True)
            full_roi = m.wr_roi(m.pd.concat([new_development_roi, new_cal_roi], ignore_index=True), full)
            assert len(full_roi) == 64 and full_roi.image_id.is_unique
            assert list(full_roi.image_id) == list(full.image_id)
            crossfit = js(crossfit_path)
            held = []
            for item in crossfit:
                used, unseen = set(map(str,item['fit_ids'])), set(map(str,item['held_ids']))
                assert used <= fit_ids and unseen <= fit_ids and not used & unseen
                assert not (used | unseen) & (stop_ids | cal_ids | outer_ids)
                assert not set(map(str,item['fit_groups'])) & set(map(str,item['held_groups']))
                held.extend(unseen)
            assert len(held) == len(set(held)) == len(fit_ids) and set(held) == fit_ids
            lineage = js(lineage_path)
            checkpoint = Path(lineage['coarse_checkpoint'])
            actual_path = checkpoint.parent / 'actual_training_inputs.csv'
            all_source_paths.add(actual_path)
            actual = read(actual_path)
            assert set(actual.image_id) == development_ids
            assert not set(actual.image_id) & (cal_ids | outer_ids)
            assert not set(actual.group) & set(roles.loc[roles.image_id.isin(cal_ids|outer_ids), 'group'])
            folds.append({'split_seed': split_seed, 'fold': fold,
                          'fitting':len(fit_ids), 'stopping':len(stop_ids),
                          'development':len(development_ids), 'selection_then_final_fit':len(cal_ids),
                          'full_fine_training':len(full_ids), 'outer':len(outer_ids),
                          'coarse_calibration_training_overlap':0,
                          'crossfit_stopping_training_overlap':0,
                          'full_centres_held_out_from_their_predicting_coarse_models':True})
            if example is None:
                example = (development.head(2).reset_index(drop=True), new_development_roi.copy())

    frame, all_crops = example
    crop = all_crops.loc[all_crops.image_id.isin(frame.image_id)].copy()
    no_truth = frame[['image_id','path','width','height']].copy()
    validation = m.ResearchDataset(frame, cfg, False, 'fine', crop)
    unlabelled = m.ResearchDataset(no_truth, cfg, False, 'fine', crop)
    deliberately_outside = frame.copy()
    deliberately_outside['x'] = frame['x'] + 10000.
    deliberately_outside['y'] = frame['y'] - 10000.
    changed_truth = m.ResearchDataset(deliberately_outside, cfg, False, 'fine', crop)
    transform_max_error = 0.
    for index in range(len(frame)):
        labelled, unknown, outside = validation[index], unlabelled[index], changed_truth[index]
        assert labelled['image'].shape == (3, 384, 384)
        assert labelled['target_heatmap'].shape == (1, 96, 96)
        assert torch.isfinite(labelled['image']).all()
        assert torch.isfinite(labelled['target_heatmap']).all()
        assert not bool(labelled['crop_was_enlarged']) and not bool(outside['crop_was_enlarged'])
        for key in ('image','input_to_orig','crop','crop_final_side'):
            assert torch.equal(labelled[key], unknown[key]), key
            assert torch.equal(labelled[key], outside[key]), key
        assert ((outside['target_xy'] < 0) | (outside['target_xy'] > 1)).any()
        assert 'x_true' not in unknown
        inverse = m.np.vstack([labelled['input_to_orig'].numpy().astype(float), [0.,0.,1.]])
        input_point = m.np.r_[labelled['target_xy'].numpy().astype(float) * 383., 1.]
        restored = inverse @ input_point
        original = frame.iloc[index][['x','y']].to_numpy(float)
        error = float(m.np.max(m.np.abs(restored[:2] - original)))
        transform_max_error = max(transform_max_error, error)
        assert error < 3e-3
        cx, cy, side = labelled['crop'].numpy().astype(float)
        direct = m.research_crop_transform(cx, cy, side, 384)
        for pixel in ([0.,0.,1.], [383.,383.,1.], [191.5,191.5,1.]):
            assert m.np.max(m.np.abs(direct @ m.np.linalg.inv(direct) @ pixel - pixel)) < 1e-9

    # Base 3-channel architecture and original-coordinate inference agree
    # exactly when labels are absent, present, or intentionally replaced.
    local_cfg = m.replace(cfg, PRETRAINED=False, AMP=False, BATCH_SIZE=2, NUM_WORKERS=0)
    m.seed_everything(1829)
    model = m.research_build_model(local_cfg, 'fine', 'cpu').eval()
    with torch.no_grad():
        example_tensor = validation[0]['image'].unsqueeze(0)
        output = model(example_tensor)
    assert output['logits'].shape == (1,1,96,96)
    assert output['coords'].shape == (1,2)
    predictions = [m.predict_research(model, source, local_cfg, 'cpu', 'fine', crop)
                   for source in (frame, no_truth, deliberately_outside)]
    for other in predictions[1:]:
        assert m.np.array_equal(predictions[0][['x','y','uncertainty']].to_numpy(),
                               other[['x','y','uncertainty']].to_numpy())
    del model

    # Exact spacing implied by crop S/side and normalized target / (S-1).
    baseline_sigma_px = 1.5 * (256-1)/(64-1) * 300/256
    new_sigma_px = 1.5 * (384-1)/(96-1) * 500/384
    heatmap = m.gaussian_heatmap(m.np.array([.5,.5], dtype=m.np.float32),96,1.5)[0]
    grid = m.np.arange(96, dtype=float)
    marginal = heatmap.sum(axis=0)
    mean = float((marginal * grid).sum())
    observed_sigma_cells = float(m.np.sqrt((marginal * (grid-mean)**2).sum()))
    assert abs(observed_sigma_cells-1.5) < 1e-5
    report = {'status':'PASS', 'gpu_used':False, 'external_reference_or_test_images_read':False,
              'engine_sha256':m.research_file_digest(WORK/'engine.py'),
              'configuration':{'fine_input_size':384,'fine_heatmap_size':96,'fine_sigma_cells':1.5,
                               'coarse_input_size_unchanged':256,'coarse_heatmap_size_unchanged':64,
                               'roi_short_side_fraction':.25,'fine_channels':3},
              'checks':{'roi_label_independent':True,'roi_centres_unchanged':True,
                        'roi_and_labels_keep_frame_order':True,'full_refit_inputs_are_64_non_outer_training_images':True,
                        'all_source_coarse_models_exclude_selection_group':True,
                        'stopping_groups_never_enter_crop_crossfit_training':True,
                        'validation_never_enlarges_or_repairs_roi':True,
                        'validation_pixels_and_transform_independent_of_labels':True,
                        'predictions_identical_with_absent_or_replaced_labels':True,
                        'original_coordinate_roundtrip_max_error_px':transform_max_error},
              'physical_scale_example_for_shortside_2000':{
                  'baseline_roi_side_px':300,'wide_roi_side_px':500,
                  'baseline_sigma_original_px':baseline_sigma_px,'wide_sigma_original_px':new_sigma_px,
                  'sigma_original_ratio':new_sigma_px/baseline_sigma_px,
                  'baseline_original_px_per_input_px':300/256,'wide_original_px_per_input_px':500/384,
                  'baseline_jitter_sigma_px':.08*300,'wide_jitter_sigma_px':.08*500,
                  'observed_gaussian_sigma_cells':observed_sigma_cells},
              'source_oof_geometric_coverage':coverage,'fold_role_audits':folds,
              'source_file_digests':{str(path):m.research_file_digest(path) for path in sorted(all_source_paths)},
              'scope':'CPU functional validation of fixed configuration and source training-side lineage; no training or candidate selection.',
              'seconds':time.perf_counter()-started}
    (WORK/'geometry_cpu_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('source_file_digests','fold_role_audits')},indent=2),flush=True)


if __name__ == '__main__':
    main()
