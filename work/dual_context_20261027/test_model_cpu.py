"""CPU-only geometry, architecture, gradients and real-train-image checks."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
import sys, types, hashlib, json, time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'work/gpu_test_deps'))
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_available()


def main():
    started = time.perf_counter()
    base = (ROOT / 'work/restore_20261027/engine.py').read_text(encoding='utf-8')
    source = (WORK / 'model.py').read_text(encoding='utf-8')
    module = types.ModuleType('dual_context_cpu_test_engine')
    module.__file__ = str(WORK / 'engine_cpu_test.py')
    sys.modules[module.__name__] = module
    exec(compile(base + '\n\n' + source, module.__file__, 'exec'), module.__dict__)
    m = module
    cfg_dict = json.loads((ROOT / 'outputs/restored_20261027/seed_10000/run_config.json').read_text())['config']
    cfg = m.DualConfig(**cfg_dict)
    cfg.FINE_VARIANT = 'single'
    cfg.CONTEXT_RATIO = 2.
    cfg.HEATMAP_SIGMA = 1.5
    cfg.NUM_WORKERS = 0
    cfg.BATCH_SIZE = 2
    official = Path(cfg.PRETRAINED_WEIGHTS_PATH)
    assert hashlib.sha256(official.read_bytes()).hexdigest() == m.RESEARCH_WEIGHT_SHA256
    checks = {}

    # Local/context world coordinates agree through reflection, rotation,
    # translation, off-centre and edge-crossing crops.
    max_geo_error = 0.
    for cx, cy, side, angle, flip in [(100., 90., 300., 0., False),
                                      (10., 8., 450., 12., True),
                                      (1250., 900., 620., -9., False)]:
        local = m.research_crop_transform(cx, cy, side, 256)
        augmentation = m.np.vstack([m.cv2.getRotationMatrix2D((127.5, 127.5), angle, 1.03), [0., 0., 1.]])
        augmentation[:2, 2] += [3.2, -4.5]
        if flip:
            augmentation = m.np.array([[-1., 0., 255.], [0., 1., 0.], [0., 0., 1.]]) @ augmentation
        local = augmentation @ local
        mapping = m.context_output_transform(256, 2.)
        context = mapping @ local
        for world in ([cx, cy, 1.], [cx - 72., cy + 34., 1.], [cx + 100., cy - 120., 1.]):
            local_xy = local @ world
            context_xy = context @ world
            expected = mapping @ local_xy
            error = float(m.np.max(m.np.abs(context_xy - expected)))
            inverse_error = float(m.np.max(m.np.abs(m.np.linalg.inv(context) @ context_xy - world)))
            max_geo_error = max(max_geo_error, error, inverse_error)
    assert max_geo_error < 1e-9
    checks['physical_augmentation_alignment_max_error_px'] = max_geo_error

    max_feature_error = 0.
    for stride in (4, 8, 16, 32):
        n = 256 // stride
        yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32) * stride,
                                torch.arange(n, dtype=torch.float32) * stride, indexing='ij')
        coordinate_planes = torch.stack([xx, yy]).unsqueeze(0)
        aligned = m.align_context_feature(coordinate_planes, 256, stride, 2.)
        expected = 127.5 + (coordinate_planes - 127.5) / 2.
        error = float((aligned - expected).abs().max())
        max_feature_error = max(max_feature_error, error)
    assert max_feature_error < 4e-5
    checks['feature_grid_registration_max_error_input_px'] = max_feature_error

    # A repeated Notebook cell must retain the original coarse entrypoints.
    original_model, original_loader = m._dual_context_base_model, m._dual_context_base_loader
    exec(compile(source, str(WORK / 'model.py'), 'exec'), m.__dict__)
    assert m._dual_context_base_model is original_model
    assert m._dual_context_base_loader is original_loader
    coarse_cfg = m.replace(cfg, PRETRAINED=False, HEATMAP_SIGMA=2.5)
    coarse = m.research_build_model(coarse_cfg, 'coarse', 'cpu')
    assert type(coarse) is m.FoveaNet
    del coarse
    checks['coarse_dispatch_and_repeated_cell_safe'] = True

    # Two arms receive exactly the same six channels and consume the same RNG.
    train = m.pd.read_csv(ROOT / 'outputs/restored_20261027/seed_10000/folds.csv', dtype={'image_id': str})
    roles = m.pd.read_csv(ROOT / 'outputs/restored_20261027/seed_10000/fold_0/split_roles.csv', dtype={'image_id': str})
    fitting_ids = roles.loc[roles.role == 'model_fit', 'image_id']
    frame = train.loc[train.image_id.isin(fitting_ids)].head(2).reset_index(drop=True)
    assert len(frame) == 2 and set(frame.image_id) <= {str(i) for i in range(1, 81)}
    crops = m.pd.read_csv(ROOT / 'outputs/restored_20261027/seed_10000/fold_0/fine_final/actual_training_crops.csv', dtype={'image_id': str})
    crops = crops.loc[crops.image_id.isin(frame.image_id)].copy()
    cfg.FINE_MULTISCALE_TRAIN = False
    m.seed_everything(5901)
    single_item = m.DualContextDataset(frame, cfg, True, 'fine', crops)[0]
    single_next_random = m.np.random.random(4)
    m.seed_everything(5901)
    dual_item = m.DualContextDataset(frame, m.replace(cfg, FINE_VARIANT='dual'), True, 'fine', crops)[0]
    dual_next_random = m.np.random.random(4)
    assert torch.equal(single_item['image'], dual_item['image'])
    assert m.np.array_equal(single_next_random, dual_next_random)
    m.seed_everything(5901)
    old_item = m.ResearchDataset(frame, cfg, True, 'fine', crops)[0]
    assert torch.equal(single_item['image'][:3], old_item['image'])
    for key in ('target_xy', 'target_heatmap', 'input_to_orig', 'crop'):
        assert torch.equal(single_item[key], old_item[key]), key
    assert single_item['image'].shape == (6, 256, 256)
    local_inverse = m.np.vstack([single_item['input_to_orig'].numpy(), [0., 0., 1.]])
    context_inverse = m.np.vstack([single_item['context_input_to_orig'].numpy(), [0., 0., 1.]])
    derived_context_inverse = local_inverse @ m.np.linalg.inv(m.context_output_transform(256, 2.))
    assert m.np.max(m.np.abs(context_inverse - derived_context_inverse)) < 2e-4
    no_labels = frame[['image_id', 'path', 'width', 'height']]
    inference = m.DualContextDataset(no_labels, cfg, False, 'fine', crops)[0]
    assert 'x_true' not in inference and torch.isfinite(inference['image']).all()
    labelled_inference = m.DualContextDataset(frame, cfg, False, 'fine', crops)[0]
    assert torch.equal(inference['image'], labelled_inference['image'])
    assert torch.equal(inference['crop'], labelled_inference['crop'])
    checks['single_dual_identical_inputs_and_rng'] = True
    checks['local_pixels_labels_transform_unchanged'] = True
    checks['inference_inputs_independent_of_truth'] = True

    # The single arm is exactly the base network, under equal initialization.
    m.seed_everything(102046)
    single = m.research_build_model(cfg, 'fine', 'cpu').eval()
    m.seed_everything(102046)
    old = m._dual_context_base_model(cfg, 'fine', 'cpu').eval()
    batch = torch.stack([single_item['image'], inference['image']])
    with torch.no_grad():
        single_out = single(batch)
        old_out = old(batch[:, :3])
        mutated = batch.clone()
        mutated[:, 3:] = torch.randn_like(mutated[:, 3:])
        single_context_mutated = single(mutated)
    assert torch.equal(single_out['logits'], old_out['logits'])
    assert torch.equal(single_out['logits'], single_context_mutated['logits'])
    weights = torch.load(official, map_location='cpu', weights_only=True)
    assert torch.equal(single.encoder.conv1.weight.detach(), weights['conv1.weight'])
    checks['single_arm_exact_original_architecture'] = True
    del old, single, single_out, old_out, single_context_mutated

    m.seed_everything(102046)
    dual = m.research_build_model(m.replace(cfg, FINE_VARIANT='dual'), 'fine', 'cpu')
    assert torch.equal(dual.encoder.conv1.weight.detach(), weights['conv1.weight'])
    dual.train()
    assert all(not module.training for module in dual.encoder.modules() if isinstance(module, torch.nn.BatchNorm2d))
    # Gradients are checked through both views, shared encoder and all fusions.
    grad_input = batch[:1].clone().requires_grad_(True)
    output = dual(grad_input)
    assert output['logits'].shape == (1, 1, 64, 64)
    assert output['coords'].shape == (1, 2) and torch.isfinite(output['coords']).all()
    output['logits'].square().mean().backward()
    context_input_gradient = float(grad_input.grad[:, 3:].abs().sum())
    local_input_gradient = float(grad_input.grad[:, :3].abs().sum())
    encoder_gradient = float(dual.encoder.conv1.weight.grad.abs().sum())
    fusion_gradients = [float(layer.weight.grad.abs().sum()) for layer in dual.context_fusions]
    assert min(context_input_gradient, local_input_gradient, encoder_gradient, *fusion_gradients) > 0
    groups = m.research_optimizer_groups(dual, cfg)
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in dual.parameters() if p.requires_grad}
    dual.eval()
    with torch.no_grad():
        original = dual(batch[:1])['logits']
        changed = dual(mutated[:1])['logits']
    context_effect = float((original - changed).abs().max())
    assert context_effect > 1e-6
    checks.update({'official_encoder_weights_loaded': True,
                   'frozen_encoder_batchnorm': True,
                   'context_input_gradient_l1': context_input_gradient,
                   'local_input_gradient_l1': local_input_gradient,
                   'shared_encoder_gradient_l1': encoder_gradient,
                   'fusion_gradient_l1': fusion_gradients,
                   'context_change_logit_max_effect': context_effect,
                   'optimizer_covers_shared_encoder_once': True,
                   'dual_parameter_count': sum(p.numel() for p in dual.parameters())})

    # Shared six-channel inference, including simultaneous horizontal-flip TTA.
    cfg_dual = m.replace(cfg, FINE_VARIANT='dual', TTA=True)
    predictions = m.predict_research(dual, no_labels, cfg_dual, 'cpu', 'fine', crops)
    assert len(predictions) == 2 and predictions.image_id.is_unique
    assert m.np.isfinite(predictions[['x', 'y', 'uncertainty']].to_numpy()).all()
    report = {'status': 'PASS', 'gpu_used': False, 'test_labels_or_images_used': False,
              'model_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'checks': checks, 'seconds': time.perf_counter() - started,
              'scope': 'CPU functional checks using synthetic geometry and two fitting images only; no model selection or formal training.'}
    (WORK / 'model_cpu_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
