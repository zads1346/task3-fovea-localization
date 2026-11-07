"""CPU-only contract tests for diagnostics and calibration-rule selection."""
from pathlib import Path
import os
import sys
import types
import json
import hashlib

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
source = (ROOT / 'work/restore_20261027/engine.py').read_text(encoding='utf-8') + '\n\n' + (WORK / 'risk.py').read_text(encoding='utf-8')
module = types.ModuleType('risk_cpu_test_engine')
module.__file__ = str(WORK / 'risk.py')
sys.modules[module.__name__] = module
exec(compile(source, '<risk_cpu_test_engine>', 'exec'), module.__dict__)
m = module
m.torch.set_num_threads(2)
assert not m.torch.cuda.is_initialized()
checks = []


def passed(name):
    checks.append(name)
    print('PASS:', name, flush=True)


def assert_raises(function, exception=ValueError):
    try:
        function()
    except exception:
        return
    raise AssertionError('Expected exception')


# Risk threshold strictness is part of the frozen rule, not a tuned parameter.
feature_frame = m.pd.DataFrame({
    'image_id': [str(i) for i in range(8)],
    'scale_disagreement_norm': [.05, .0500001, .025, .0250001, .0250001, 0, 0, .1],
    'tta_disagreement_norm': [0, 0, 0, 0, 0, .025, .0250001, 0],
    'secondary_peak_ratio': [0, 0, 1, .8, .800001, 0, 0, 0]})
assert m.dual_risk_mask(feature_frame).tolist() == [False, True, False, False, True, False, True, True]
assert_raises(lambda: m.dual_risk_mask(feature_frame.assign(secondary_peak_ratio=1.1)))
passed('Strict fixed feature thresholds and feature bounds')

# Peak suppression must ignore neighbors and preserve a distant secondary peak.
prob = m.torch.zeros(1, 1, 11, 11)
prob[0, 0, 5, 5], prob[0, 0, 5, 6], prob[0, 0, 0, 0] = 1, .99, .7
assert abs(float(m.dual_secondary_peak_ratio(prob)[0]) - .7) < 1e-6
assert_raises(lambda: m.dual_secondary_peak_ratio(prob, radius=2))
passed('Secondary peak excludes a radius-three neighborhood')

n = 16
truth = m.pd.DataFrame({'image_id': [str(i) for i in range(n)], 'x': m.np.arange(n) * 3.0,
                      'y': m.np.arange(n) * 2.0, 'group': [str(i) for i in range(n)]})
fine = truth[['image_id', 'x', 'y']].copy()
fine['x'] += 10
fine['scale_disagreement_norm'] = .1
fine['tta_disagreement_norm'] = 0.0
fine['secondary_peak_ratio'] = .1
fine['crop_edge_fraction'] = .3
fine['cov_xx'], fine['cov_xy'], fine['cov_yy'], fine['uncertainty'] = 4., 0., 4., 2.
coarse = truth[['image_id', 'x', 'y']].copy()
coarse['cov_xx'], coarse['cov_xy'], coarse['cov_yy'], coarse['uncertainty'] = 9., 0., 9., 3.
cfg = m.ResearchConfig()
cfg.FALLBACK_MIN_GROUPS = 12
rule = m.fit_risk_rule(coarse.iloc[::-1], fine, truth.iloc[::-1], cfg)
assert rule['enabled'] and rule['selected_policy'] == 'risk_coarse'
assert rule['crossfit_selected_procedure_metrics']['rmse'] == 0
assert len(rule['crossfit_audit']) == n
for split in rule['crossfit_audit']:
    assert split['held_group'] not in split['fit_groups']
    assert not set(split['held_image_ids']) & set(split['fit_image_ids'])
output = m.apply_risk_rule(coarse.iloc[::-1], fine, rule)
assert m.np.array_equal(output[['x', 'y']], truth[['x', 'y']])
assert m.np.all(output.coarse_weight == 1)
assert m.np.all(output.uncertainty == 3)
passed('ID alignment, group-held-out selection, accepted rule and covariance')

# Every copy of a duplicate group is held out together.
grouped = truth.copy()
grouped.loc[grouped.image_id.isin(['0', '1']), 'group'] = 'shared'
group_rule = m.fit_risk_rule(coarse, fine, grouped, cfg)
held = next(row for row in group_rule['crossfit_audit'] if row['held_group'] == 'shared')
assert set(held['held_image_ids']) == {'0', '1'}
assert not {'0', '1'} & set(held['fit_image_ids'])
passed('Duplicate group exclusion in calibration crossfit')

# Exact ties choose fine, and zero discrepancy must never invent improvement.
same = fine.copy()
same[['x', 'y']] = coarse[['x', 'y']].to_numpy()
tie_rule = m.fit_risk_rule(coarse, same, truth, cfg)
assert not tie_rule['enabled'] and tie_rule['selected_policy'] == 'fine_only'
out_tie = m.apply_risk_rule(coarse, same, tie_rule)
assert m.np.array_equal(out_tie[['x', 'y']], same[['x', 'y']])
assert m.np.all(out_tie.coarse_weight == 0)
assert m.np.all(out_tie.uncertainty == same.uncertainty)
small_rule = m.fit_risk_rule(coarse.head(6), fine.head(6), truth.head(6), cfg)
assert not small_rule['enabled'] and small_rule['reason'] == 'too_few_calibration_groups'
passed('Tie-to-fine, disabled exact identity, and insufficient calibration groups')

half_rule = {'enabled': True, 'selected_policy': 'risk_half', 'feature_constants': m.dual_risk_constants()}
half = m.apply_risk_rule(coarse, fine, half_rule)
assert m.np.allclose(half.x, truth.x + 5)
assert m.np.allclose(half.cov_xx, .5 * (9 + 25) + .5 * (4 + 25))
assert m.np.allclose(half.cov_yy, 6.5)
assert_raises(lambda: m.apply_risk_rule(coarse.iloc[:-1], fine, half_rule))
assert_raises(lambda: m.apply_risk_rule(coarse, m.pd.concat([fine, fine.iloc[[0]]]), half_rule))
assert_raises(lambda: m.apply_risk_rule(coarse, fine, {**half_rule, 'feature_constants': {}}))
passed('Mixture moments, duplicate/missing IDs and frozen-rule constants')

metric = m._dual_risk_metrics(m.np.array([[3, 4], [0, 0]]), m.np.zeros((2, 2)))
assert metric['sse'] == 25 and metric['rmse'] == 2.5
passed('Coordinate RMSE and SSE, not mean Euclidean distance')

# Mock data construction provides real affine transforms and equivariant
# logits, while proving inference drops every label column before loading.
loader_calls = []


def mock_loader(frame, cfg, training, stage, crop_spec, seed):
    assert list(frame.columns) == ['image_id', 'path', 'width', 'height']
    assert list(crop_spec.columns) == ['image_id', 'crop_x', 'crop_y', 'crop_side']
    assert training is False and stage == 'fine'
    loader_calls.append({'frame_columns': list(frame.columns), 'crop_sides': crop_spec.crop_side.tolist()})
    size = 32
    images, transforms, crops, dims, ids = [], [], [], [], []
    indexed = crop_spec.set_index('image_id')
    for row in frame.itertuples(index=False):
        crop = indexed.loc[row.image_id, ['crop_x', 'crop_y', 'crop_side']].to_numpy(float)
        transform = m.research_crop_transform(*crop, size)
        world_target = m.np.array([100 + int(row.image_id), 90 + int(row.image_id), 1.])
        point = (transform @ world_target)[:2]
        yy, xx = m.torch.meshgrid(m.torch.arange(size), m.torch.arange(size), indexing='ij')
        logits = -((xx-point[0])**2 + (yy-point[1])**2) / (2 * 1.8**2)
        image = m.torch.zeros(6, size, size)
        image[0] = logits
        images.append(image)
        transforms.append(m.torch.tensor(m.np.linalg.inv(transform)[:2], dtype=m.torch.float32))
        crops.append(m.torch.tensor(crop, dtype=m.torch.float32))
        dims.append(m.torch.tensor([row.width, row.height], dtype=m.torch.float32))
        ids.append(row.image_id)
    return [{'image': m.torch.stack(images), 'input_to_orig': m.torch.stack(transforms),
             'crop': m.torch.stack(crops), 'orig_wh': m.torch.stack(dims), 'image_id': ids}]


class MockModel(m.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        assert image.shape[1] == 6
        return {'logits': image[:, :1]}


m.make_research_loader = mock_loader
frame = m.pd.DataFrame({'image_id': ['1', '2'], 'path': ['unused1', 'unused2'], 'width': [240, 240],
                       'height': [200, 200], 'x': [999, 999], 'y': [-999, -999], 'visible': [True, True]})
crop = m.pd.DataFrame({'image_id': ['2', '1'], 'crop_x': [100., 100.], 'crop_y': [100., 100.],
                      'crop_side': [100., 100.], 'x_true': [12345, 12345]})
cfg.TTA, cfg.AMP = True, False
mock = MockModel()
pred = m.dual_predict_features(mock, frame, cfg, 'cpu', crop)
assert mock.calls == 4
assert m.np.max(pred.scale_disagreement_norm) < 1e-5
assert m.np.max(pred.tta_disagreement_norm) < 1e-5
assert m.np.allclose(pred[['x', 'y']].to_numpy(), [[101, 91], [102, 92]], atol=5e-4)
assert m.np.all(pred.scale_probe_side == pred.crop_side * 1.25)
label_changed = m.dual_predict_features(mock, frame.assign(x=-1e9, y=1e9, visible=False), cfg, 'cpu', crop)
numeric_columns = [c for c in pred if c not in ('image_id', 'inference_seconds')]
assert m.np.array_equal(pred[numeric_columns], label_changed[numeric_columns])
batch = mock_loader(frame[['image_id', 'path', 'width', 'height']], cfg, False, 'fine',
                    crop[['image_id', 'crop_x', 'crop_y', 'crop_side']], cfg.SEED)[0]
base_mean, base_covariance = m.research_predict_batch(mock, batch, cfg, m.torch.device('cpu'))
assert m.np.array_equal(pred[['x', 'y']].to_numpy(), base_mean.numpy())
assert m.np.array_equal(pred.cov_xx.to_numpy(), base_covariance[:, 0, 0].numpy())
passed('No-label feature loader, four-forward TTA, scale probe and exact base decoder identity')

cfg.TTA = False
mock = MockModel()
no_tta = m.dual_predict_features(mock, frame, cfg, 'cpu', crop)
assert mock.calls == 2 and m.np.all(no_tta.tta_disagreement_norm == 0)
passed('No-TTA mode uses exactly two forwards and zero TTA disagreement')

assert not m.torch.cuda.is_initialized()
report = {'status': 'PASS', 'device': 'cpu', 'cuda_initialized': False, 'checks': checks,
          'source_sha256': hashlib.sha256((WORK / 'risk.py').read_bytes()).hexdigest(),
          'scope': 'Pure numerical rule tests and mock-equivariant model inference; no training or external test references.'}
(WORK / 'risk_cpu_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
