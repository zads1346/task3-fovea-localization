"""Label-free multiscale diagnostics and a small held-out calibration rule.

Append after the base research engine and model.py. This module never reads
external score references, chooses an outer-validation method, or starts jobs.
"""

import time

DUAL_RISK_VERSION = '1.0.0-fixed-four-policy-loog'
DUAL_RISK_PROBE_RATIO = 1.25
DUAL_RISK_PEAK_EXCLUSION_RADIUS = 3
DUAL_RISK_SCALE_THRESHOLD = 0.05
DUAL_RISK_PEAK_RATIO_THRESHOLD = 0.80
DUAL_RISK_PEAK_SCALE_THRESHOLD = 0.025
DUAL_RISK_TTA_THRESHOLD = 0.025
DUAL_RISK_MIN_GAIN_PX = 0.5
DUAL_RISK_MIN_RELATIVE_GAIN = 0.02
DUAL_RISK_POLICIES = ('fine_only', 'risk_half', 'risk_coarse', 'global_quarter')
DUAL_RISK_FEATURE_COLUMNS = ('scale_disagreement_norm', 'tta_disagreement_norm',
                             'secondary_peak_ratio')


def dual_risk_constants():
    return {'probe_side_ratio': DUAL_RISK_PROBE_RATIO,
            'secondary_peak_exclusion_radius_cells': DUAL_RISK_PEAK_EXCLUSION_RADIUS,
            'scale_threshold': DUAL_RISK_SCALE_THRESHOLD,
            'peak_ratio_threshold': DUAL_RISK_PEAK_RATIO_THRESHOLD,
            'peak_scale_threshold': DUAL_RISK_PEAK_SCALE_THRESHOLD,
            'tta_threshold': DUAL_RISK_TTA_THRESHOLD,
            'minimum_absolute_gain_px': DUAL_RISK_MIN_GAIN_PX,
            'minimum_relative_gain': DUAL_RISK_MIN_RELATIVE_GAIN,
            'policies_in_tie_break_order': list(DUAL_RISK_POLICIES)}


def dual_secondary_peak_ratio(probability, radius=DUAL_RISK_PEAK_EXCLUSION_RADIUS):
    """Secondary cell maximum outside the main peak's radius / main maximum.

    This is a simple multimodality proxy, not a count of separate modes or a
    calibrated probability of a localization failure. Coordinates are cells.
    """
    if probability.ndim != 4 or probability.shape[1] != 1 or radius < 3:
        raise ValueError('Expected Bx1xHxW probability and exclusion radius >= 3.')
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError('Peak probabilities must be finite and nonnegative.')
    b, _, h, w = probability.shape
    flat = probability[:, 0].reshape(b, -1)
    peak_value, peak_index = flat.max(1)
    if (peak_value <= 0).any():
        raise ValueError('Probability maps must have positive mass.')
    yy, xx = torch.meshgrid(torch.arange(h, device=probability.device),
                            torch.arange(w, device=probability.device), indexing='ij')
    peak_y, peak_x = peak_index // w, peak_index % w
    outside = ((xx[None] - peak_x[:, None, None]) ** 2 +
               (yy[None] - peak_y[:, None, None]) ** 2) > radius ** 2
    secondary = probability[:, 0].masked_fill(~outside, 0).flatten(1).max(1).values
    return secondary / peak_value


@torch.inference_mode()
def _dual_risk_batch(model, batch, cfg, device, diagnostics):
    """Exactly the base probability-mixture decoder, plus optional diagnostics."""
    image = batch['image']
    size = int(image.shape[-1])
    with _autocast_context(device, cfg):
        logits = model(image)['logits']
    mean, covariance = research_distribution_moments(logits, batch['input_to_orig'], size)
    means, covariances = [mean], [covariance]
    probabilities = [logits.float().flatten(1).softmax(1).reshape(logits.shape)] if diagnostics else []
    if bool(cfg.TTA):
        with _autocast_context(device, cfg):
            flipped_logits = model(torch.flip(image, dims=[-1]))['logits']
        flipped_mean, flipped_covariance = research_distribution_moments(
            flipped_logits, batch['input_to_orig'], size, True)
        means.append(flipped_mean)
        covariances.append(flipped_covariance)
        if diagnostics:
            aligned_logits = torch.flip(flipped_logits.float(), dims=[-1])
            probabilities.append(aligned_logits.flatten(1).softmax(1).reshape(aligned_logits.shape))
    stacked = torch.stack(means)
    mixture_mean = stacked.mean(0)
    offsets = stacked - mixture_mean.unsqueeze(0)
    mixture_covariance = (torch.stack(covariances) + offsets.unsqueeze(-1) @ offsets.unsqueeze(-2)).mean(0)
    mixture_covariance = (mixture_covariance + mixture_covariance.transpose(1, 2)) / 2
    clipped_mean = torch.minimum(mixture_mean.clamp_min(0), batch['orig_wh'].float() - 1)
    if not diagnostics:
        return clipped_mean, mixture_covariance, None, None
    tta_disagreement = (torch.linalg.vector_norm(means[0] - means[1], dim=1)
                        if len(means) == 2 else torch.zeros(len(image), device=image.device))
    mean_probability = torch.stack(probabilities).mean(0)
    secondary = dual_secondary_peak_ratio(mean_probability)
    return clipped_mean, mixture_covariance, tta_disagreement, secondary


@torch.inference_mode()
def dual_predict_features(model, frame, cfg, device, crop_spec):
    """Fine prediction with no label access and four forwards per TTA batch.

    The wider-scale pass supplies a diagnostic only. Base x/y/covariance are
    unchanged from normal-ROI inference. No truth-dependent ROI repair occurs.
    Context construction is delegated to model.py's fine dataset.
    """
    device = torch.device(device)
    required = ['image_id', 'path', 'width', 'height']
    if not set(required).issubset(frame.columns) or not len(frame):
        raise ValueError('Inference requires a nonempty image manifest.')
    inference_frame = frame[required].copy()
    if inference_frame.image_id.isna().any():
        raise ValueError('Inference image IDs cannot be null.')
    inference_frame['image_id'] = inference_frame.image_id.astype(str)
    if inference_frame.image_id.duplicated().any():
        raise ValueError('Inference image IDs must be unique.')
    crop_columns = ['image_id', 'crop_x', 'crop_y', 'crop_side']
    crop = _research_frame(crop_spec[crop_columns], crop_columns[1:], 'diagnostic crops')
    if not set(inference_frame.image_id).issubset(crop.image_id) or (crop.crop_side <= 0).any():
        raise ValueError('Crop IDs or side lengths are invalid.')
    crop = crop.set_index('image_id').loc[inference_frame.image_id].reset_index()
    probe_crop = crop.copy()
    probe_crop['crop_side'] = probe_crop.crop_side * DUAL_RISK_PROBE_RATIO
    loader = make_research_loader(inference_frame, cfg, False, 'fine', crop, int(cfg.SEED))
    probe_loader = make_research_loader(inference_frame, cfg, False, 'fine', probe_crop, int(cfg.SEED))
    if len(loader) != len(probe_loader):
        raise ValueError('Scale-probe loader lengths differ.')
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started, rows = time.perf_counter(), []
    for cpu_batch, cpu_probe in zip(loader, probe_loader):
        if list(cpu_batch['image_id']) != list(cpu_probe['image_id']):
            raise ValueError('Scale-probe prediction IDs are misaligned.')
        batch, probe = _to_device(cpu_batch, device), _to_device(cpu_probe, device)
        mean, covariance, tta_disagreement, secondary = _dual_risk_batch(model, batch, cfg, device, True)
        probe_mean, _, _, _ = _dual_risk_batch(model, probe, cfg, device, False)
        side = batch['crop'][:, 2]
        scale_disagreement = torch.linalg.vector_norm(mean - probe_mean, dim=1) / side
        tta_disagreement = tta_disagreement / side
        uncertainty = torch.linalg.eigvalsh(covariance).clamp_min(0)[:, -1].sqrt()
        edge = 0.5 - ((mean - batch['crop'][:, :2]).abs() / side[:, None]).amax(1)
        for i, iid in enumerate(batch['image_id']):
            rows.append({'image_id': str(iid), 'x': float(mean[i, 0]), 'y': float(mean[i, 1]),
                         'uncertainty': float(uncertainty[i]), 'cov_xx': float(covariance[i, 0, 0]),
                         'cov_xy': float(covariance[i, 0, 1]), 'cov_yy': float(covariance[i, 1, 1]),
                         'crop_edge_fraction': float(edge[i]),
                         'scale_disagreement_norm': float(scale_disagreement[i]),
                         'tta_disagreement_norm': float(tta_disagreement[i]),
                         'secondary_peak_ratio': float(secondary[i]),
                         'scale_probe_x': float(probe_mean[i, 0]), 'scale_probe_y': float(probe_mean[i, 1]),
                         'crop_side': float(side[i]), 'scale_probe_side': float(probe['crop'][i, 2])})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = pd.DataFrame(rows)
    if len(result) != len(inference_frame) or set(result.image_id) != set(inference_frame.image_id):
        raise ValueError('Incomplete feature predictions.')
    numeric = result.drop(columns='image_id').to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError('Nonfinite fine diagnostic prediction.')
    result['inference_seconds'] = (time.perf_counter() - started) / len(result)
    return result


def dual_risk_mask(fine):
    fine = _research_frame(fine, DUAL_RISK_FEATURE_COLUMNS, 'fine risk features')
    scale = fine.scale_disagreement_norm.to_numpy(float)
    tta = fine.tta_disagreement_norm.to_numpy(float)
    peak = fine.secondary_peak_ratio.to_numpy(float)
    if np.any(scale < 0) or np.any(tta < 0) or np.any((peak < 0) | (peak > 1 + 1e-6)):
        raise ValueError('Risk features outside their mathematical range.')
    return ((scale > DUAL_RISK_SCALE_THRESHOLD) |
            ((peak > DUAL_RISK_PEAK_RATIO_THRESHOLD) & (scale > DUAL_RISK_PEAK_SCALE_THRESHOLD)) |
            (tta > DUAL_RISK_TTA_THRESHOLD))


def _dual_policy_weights(fine, policy):
    if policy not in DUAL_RISK_POLICIES:
        raise ValueError(f'Unknown risk policy: {policy}')
    risk = dual_risk_mask(fine)
    if policy == 'fine_only':
        return np.zeros(len(fine)), risk
    if policy == 'risk_half':
        return risk.astype(float) * .5, risk
    if policy == 'risk_coarse':
        return risk.astype(float), risk
    return np.full(len(fine), .25), risk


def _dual_align_predictions(coarse, fine):
    fine = _research_frame(fine, ('x', 'y', *DUAL_RISK_FEATURE_COLUMNS), 'fine predictions')
    coarse = _research_frame(coarse, ('x', 'y'), 'coarse predictions')
    if set(coarse.image_id) != set(fine.image_id):
        raise ValueError('Coarse/fine prediction ID sets differ.')
    coarse = coarse.set_index('image_id').loc[fine.image_id].reset_index()
    return coarse.reset_index(drop=True), fine.reset_index(drop=True)


def _dual_risk_metrics(prediction, truth):
    residual = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != 2 or not len(residual) or not np.isfinite(residual).all():
        raise ValueError('Risk metrics need finite paired x/y arrays.')
    distances = np.linalg.norm(residual, axis=1)
    return {'rmse': float(np.sqrt(np.mean(residual ** 2))),
            'p90_distance_px': float(np.quantile(distances, .9)),
            'max_distance_px': float(distances.max()), 'rows': int(len(residual)),
            'sse': float(np.sum(residual ** 2))}


def apply_risk_rule(coarse, fine, rule):
    """Apply an already fitted rule without truth access or per-image edits."""
    coarse, fine = _dual_align_predictions(coarse, fine)
    if rule.get('feature_constants') != dual_risk_constants():
        raise ValueError('Risk-rule constants do not match this frozen implementation.')
    policy = rule.get('selected_policy') if bool(rule.get('enabled', False)) else 'fine_only'
    weight, risk = _dual_policy_weights(fine, policy)
    c = coarse[['x', 'y']].to_numpy(float)
    f = fine[['x', 'y']].to_numpy(float)
    prediction = weight[:, None] * c + (1 - weight[:, None]) * f
    out = fine.copy()
    out[['x', 'y']] = prediction
    out['risk_trigger'] = risk
    out['coarse_weight'] = weight
    out['fallback_used'] = weight > 0
    out['risk_policy'] = policy
    covariance_columns = ('cov_xx', 'cov_xy', 'cov_yy')
    if any(col in fine or col in coarse for col in covariance_columns):
        if not set(covariance_columns).issubset(fine) or not set(covariance_columns).issubset(coarse):
            raise ValueError('Both distributions need all covariance entries for a mixture.')
        for source in (coarse, fine):
            if not np.isfinite(source[list(covariance_columns)].to_numpy(float)).all():
                raise ValueError('Covariances must be finite.')
        dc, df = c - prediction, f - prediction
        xx = weight * (coarse.cov_xx.to_numpy() + dc[:, 0] ** 2) + (1-weight) * (fine.cov_xx.to_numpy() + df[:, 0] ** 2)
        xy = weight * (coarse.cov_xy.to_numpy() + dc[:, 0] * dc[:, 1]) + (1-weight) * (fine.cov_xy.to_numpy() + df[:, 0] * df[:, 1])
        yy = weight * (coarse.cov_yy.to_numpy() + dc[:, 1] ** 2) + (1-weight) * (fine.cov_yy.to_numpy() + df[:, 1] ** 2)
        largest = .5 * (xx + yy + np.sqrt(np.maximum((xx-yy) ** 2 + 4*xy ** 2, 0)))
        out['cov_xx'], out['cov_xy'], out['cov_yy'] = xx, xy, yy
        out['uncertainty'] = np.sqrt(np.maximum(largest, 0))
    # crop_edge_fraction and diagnostic probes intentionally remain properties
    # of the original fine prediction, not recalibrated mixture confidence.
    return out


def fit_risk_rule(coarse, fine, truth, cfg=None):
    """Choose among four fixed rules on independent calibration groups only.

    Leave-one-group-out evaluates the selection procedure, not four thresholds
    optimized against outer OOF or test answers. It still has small-sample
    selection uncertainty; no finite-sample guarantee is asserted.
    """
    coarse, fine = _dual_align_predictions(coarse, fine)
    truth = _research_frame(truth, ('x', 'y', 'group'), 'calibration truth')
    if set(truth.image_id) != set(fine.image_id):
        raise ValueError('Calibration truth/prediction ID sets differ.')
    truth = truth.set_index('image_id').loc[fine.image_id].reset_index()
    groups = truth.group.astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    minimum_groups = int(getattr(cfg, 'FALLBACK_MIN_GROUPS', 12)) if cfg is not None else 12
    if minimum_groups < 2:
        raise ValueError('At least two calibration groups are required.')
    c, f, target = (value[['x', 'y']].to_numpy(float) for value in (coarse, fine, truth))
    risk = dual_risk_mask(fine)
    candidates, arrays = [], {}
    for policy in DUAL_RISK_POLICIES:
        weight, _ = _dual_policy_weights(fine, policy)
        prediction = weight[:, None] * c + (1-weight[:, None]) * f
        arrays[policy] = prediction
        candidates.append({'policy': policy, **_dual_risk_metrics(prediction, target),
                           'changed_prediction_rows': int(np.any(prediction != f, axis=1).sum()),
                           'nonzero_coarse_weight_rows': int((weight > 0).sum()),
                           'risk_trigger_rows': int(risk.sum())})
    # Stable candidate order resolves exact ties in favor of less intervention.
    selected_index = min(range(len(candidates)), key=lambda i: (candidates[i]['rmse'], i))
    selected_policy = candidates[selected_index]['policy']
    crossfit_rows, crossfit_predictions = [], np.zeros_like(f)
    if len(unique_groups) >= 2:
        for group in unique_groups:
            held = groups == group
            fitting = ~held
            training_scores = []
            for policy in DUAL_RISK_POLICIES:
                training_scores.append({'policy': policy, **_dual_risk_metrics(arrays[policy][fitting], target[fitting])})
            winner = min(range(len(training_scores)), key=lambda i: (training_scores[i]['rmse'], i))
            policy = DUAL_RISK_POLICIES[winner]
            crossfit_predictions[held] = arrays[policy][held]
            crossfit_rows.append({'held_group': group,
                                  'fit_groups': sorted(set(groups[fitting])),
                                  'fit_image_ids': truth.image_id[fitting].tolist(),
                                  'held_image_ids': truth.image_id[held].tolist(),
                                  'selected_policy': policy, 'fitting_candidate_scores': training_scores,
                                  'held_metrics': _dual_risk_metrics(arrays[policy][held], target[held])})
        crossfit_metrics = _dual_risk_metrics(crossfit_predictions, target)
    else:
        crossfit_metrics = None
    baseline = candidates[0]
    gain = baseline['rmse'] - crossfit_metrics['rmse'] if crossfit_metrics else None
    enough_groups = len(unique_groups) >= minimum_groups
    criterion = (crossfit_metrics is not None and gain >= DUAL_RISK_MIN_GAIN_PX and
                 gain >= DUAL_RISK_MIN_RELATIVE_GAIN * baseline['rmse'] and
                 crossfit_metrics['p90_distance_px'] <= baseline['p90_distance_px'] + 1e-9)
    enabled = bool(enough_groups and criterion and selected_policy != 'fine_only')
    reason = ('accepted_by_calibration_group_crossfit' if enabled else
              'too_few_calibration_groups' if not enough_groups else
              'no_nontrivial_full_calibration_policy' if selected_policy == 'fine_only' else
              'crossfit_gain_or_p90_gate_failed')
    return {'version': DUAL_RISK_VERSION, 'enabled': enabled, 'selected_policy': selected_policy,
            'effective_policy': selected_policy if enabled else 'fine_only', 'reason': reason,
            'feature_constants': dual_risk_constants(), 'minimum_groups': minimum_groups,
            'calibration_images': int(len(truth)), 'calibration_groups': len(unique_groups),
            'calibration_ids': truth.image_id.tolist(), 'calibration_group_ids': unique_groups,
            'candidate_scores': candidates, 'baseline_fine_metrics': baseline,
            'crossfit_selected_procedure_metrics': crossfit_metrics,
            'crossfit_gain_px': gain, 'crossfit_audit': crossfit_rows,
            'selection_role': 'independent_calibration_groups_only',
            'crossfit_protocol': 'Each held group uses a policy selected by RMSE on all other calibration groups; fixed 4 candidates and stable tie break.',
            'acceptance': 'Group-crossfit RMSE gain >=0.5px AND >=2%; group-crossfit P90 must not increase; minimum calibration-group count required.',
            'uncertainty_scope': 'Mixture covariance is descriptive, not a calibrated coverage guarantee.',
            'limitations': 'Small calibration sets and repeated development can cause selection noise; no outer or test labels may be used to select this rule.'}
