@dataclass
class OptimizedConfig(ResearchConfig):
    OUTPUT_DIR: str = "/kaggle/working/optimized_20261026"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    INPUT_SIZE: int = 384
    HEATMAP_SIZE: int = 96
    FINE_INPUT_SIZE: int = 384
    FINE_HEATMAP_SIZE: int = 96
    COARSE_EPOCHS: int = 35
    FINE_EPOCHS: int = 30
    COARSE_MIN_EPOCHS: int = 10
    FINE_MIN_EPOCHS: int = 8
    PATIENCE: int = 10
    BACKBONE_LR_MULTIPLIER: float = 0.25
    ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)
    FINE_VALIDATION_ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)
    FINE_MULTISCALE_TRAIN: bool = True
    FINE_MULTISCALE_EARLY_STOP: bool = True
    SELECT_FINE_CROP_POLICY: bool = False
    COARSE_BLEND_WEIGHT: float = 0.5
    OOF_CENTER_JITTER: float = 0.10
    NUM_WORKERS: int = 0
    BATCH_SIZE: int = 8
    AMP: bool = True
    USE_EMA: bool = False
    EARLY_STOP_FRACTION: float = 0.20
    REUSE_CHECKPOINTS: bool = True
    SHOW_PLOTS: bool = False
    SMOKE_TEST: bool = False
    N_FOLDS: int = 5


CFG = OptimizedConfig()
RESEARCH_VERSION = "4.0.0-20261026"
RESEARCH_SOURCE_HASH = "20261026-full-outer-fit-multiscale-blend-v1"


@torch.inference_mode()
def optimized_validation_score(model, loaders, cfg, device, stage):
    """Evaluate exactly the deployed TTA + multiscale + fixed blend on inner data."""
    model.eval()
    predictions, reference_truth, reference_ids = [], None, None
    scale_scores = {}
    for name, loader in loaders:
        pp, tt, ids = [], [], []
        for cpu_batch in loader:
            batch = _to_device(cpu_batch, device)
            point, _ = research_predict_batch(model, batch, cfg, device)
            pp.append(point.double().cpu().numpy())
            tt.append(torch.stack([batch["x_true"], batch["y_true"]], dim=1).double().cpu().numpy())
            ids.extend(list(cpu_batch['image_id']))
        prediction, truth = np.concatenate(pp), np.concatenate(tt)
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        else:
            if ids != reference_ids or not np.array_equal(reference_truth, truth):
                raise AssertionError("Multiscale validation rows must be aligned.")
        predictions.append(prediction)
        scale_scores[name] = float(np.sqrt(np.mean((prediction-truth)**2)))
    mean = np.mean(predictions, axis=0)
    if stage == 'fine':
        dataset = loaders[0][1].dataset
        coarse = dataset.crops.loc[reference_ids, ['crop_x', 'crop_y']].to_numpy(float)
        weight = float(cfg.COARSE_BLEND_WEIGHT)
        if not 0 <= weight <= 1:
            raise ValueError("Invalid fixed blend weight.")
        mean = weight*coarse + (1-weight)*mean
    score = float(np.sqrt(np.mean((mean-reference_truth)**2)))
    scale_scores['deployed_combination'] = score
    return score, scale_scores


def optimized_stage(training, validation, cfg, device, stage, folder, seed,
                    train_crop=None, val_crop=None, epochs=None):
    """Persist the exact IDs actually passed to each training/selection call."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if validation is not None and epochs is None:
        assert not set(training.group)&set(validation.group)
    columns=['image_id','group','path','pixel_hash']
    training[columns].to_csv(folder/'actual_training_inputs.csv',index=False)
    if validation is not None and epochs is None:
        validation[columns].to_csv(folder/'actual_early_stop_inputs.csv',index=False)
    if train_crop is not None:
        train_crop.to_csv(folder/'actual_training_crops.csv',index=False)
    if val_crop is not None and epochs is None:
        val_crop.to_csv(folder/'actual_early_stop_crops.csv',index=False)
    return _research_stage(training,validation,cfg,device,stage,folder,seed,train_crop,val_crop,epochs)


def optimized_crossfit(fitting,cfg,device,folder,epochs,seed):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    inner=assign_folds(fitting,replace(cfg,N_FOLDS=cfg.INNER_CROP_FOLDS,SEED=seed))
    predictions,audit=[],[]
    for fold in sorted(inner.fold.unique()):
        tr=inner.loc[inner.fold!=fold].reset_index(drop=True)
        va=inner.loc[inner.fold==fold].reset_index(drop=True)
        assert not set(tr.group)&set(va.group)
        model,_=optimized_stage(tr,None,cfg,device,'coarse',folder/f'inner_{fold}',seed+int(fold),epochs=epochs)
        predictions.append(predict_research(model,va,cfg,device,'coarse'))
        audit.append({'inner_fold':int(fold),'fit_ids':tr.image_id.tolist(),'held_ids':va.image_id.tolist(),
                      'fit_groups':tr.group.tolist(),'held_groups':va.group.tolist()})
        _research_cleanup(model);del model
    coarse=pd.concat(predictions,ignore_index=True).set_index('image_id').loc[fitting.image_id].reset_index()
    assert coarse.image_id.is_unique and len(coarse)==len(fitting)
    coarse.to_csv(folder/'training_oof_centres.csv',index=False)
    json_save(audit,folder/'crossfit_audit.json')
    return make_roi_spec(coarse,fitting,cfg,mode='fixed')


def optimized_inference(fine,frame,coarse,cfg,device):
    """No label-dependent crop repair; combine three predefined fields of view."""
    frame=frame[['image_id','path','width','height']].copy()
    center=coarse.set_index('image_id').loc[frame.image_id].reset_index()
    views=[]
    for fraction in cfg.FINE_VALIDATION_ROI_FRACTIONS:
        roi=make_roi_spec(center,frame,cfg,mode='fixed',fixed_fraction=float(fraction))
        pred=predict_research(fine,frame,cfg,device,'fine',roi)
        assert pred.image_id.tolist()==frame.image_id.tolist()
        views.append(pred.assign(roi_fraction=float(fraction)))
    all_views=pd.concat(views,ignore_index=True)
    mean=all_views.groupby('image_id',sort=False)[['x','y']].mean().loc[frame.image_id].reset_index()
    blend=mean.copy()
    weight=float(cfg.COARSE_BLEND_WEIGHT)
    blend[['x','y']]=weight*center[['x','y']].to_numpy()+(1-weight)*mean[['x','y']].to_numpy()
    return mean,blend,all_views


def run_optimized(config):
    cfg=replace(config)
    if cfg.SMOKE_TEST:
        raise ValueError("This release is a genuine training/inference run; use a separate smoke output for debugging.")
    if cfg.N_FOLDS!=5 or cfg.COARSE_BLEND_WEIGHT!=0.5:
        raise ValueError("This frozen experiment prespecifies five folds and an equal coarse/fine blend.")
    if tuple(cfg.ROI_FRACTIONS)!=tuple(cfg.FINE_VALIDATION_ROI_FRACTIONS):
        raise ValueError("Training and inference scale sets must agree.")
    if not torch.cuda.is_available():
        raise RuntimeError("A working GPU is required for this training run.")
    seed_everything(cfg.SEED);cv2.setNumThreads(0)
    device=torch.device('cuda');out=Path(cfg.OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    cfg.PRETRAINED_WEIGHTS_PATH=resolve_research_weights(cfg)
    train,test,sample,layout,metadata=prepare_tables(cfg)
    assert train.visible.all() and not set(train.image_id)&set(test.image_id)
    if set(train.pixel_hash)&set(test.pixel_hash):
        raise ValueError("Exact train/test pixel duplicates require investigation before training.")
    train=assign_folds(train,cfg)
    train.to_csv(out/'folds.csv',index=False);test.to_csv(out/'test_manifest.csv',index=False)
    identity={'config':asdict(cfg),'source':research_runtime_code_digest(),'environment':research_training_environment(),
              'train':train[['image_id','pixel_hash','x','y','group','fold']].to_dict('records'),
              'test':test[['image_id','pixel_hash']].to_dict('records')}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['signature']!=signature:
        raise ValueError('Output folder belongs to another configuration/code/data; choose a new folder.')
    json_save({'config':asdict(cfg),'signature':signature,'protocol':'Full outer training fold refit; no uncertainty calibration; fixed 50/50 coarse and multiscale fine fusion.',
               'test_labels_used':False,'submission_method_prespecified':'BLEND'},config_path)
    json_save(research_training_environment(),out/'environment.json')
    oof_all={k:[] for k in ['COARSE','FINE_MULTISCALE','BLEND']};test_all={k:[] for k in oof_all}
    for fold in sorted(train.fold.unique()):
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        marker=folder/'complete.json'
        if marker.exists() and cfg.REUSE_CHECKPOINTS:
            record=json.loads(marker.read_text())
            if record['signature']!=signature:raise ValueError('Fold identity mismatch.')
            research_verify_cache(folder,record['files'])
            for method in oof_all:
                oof_all[method].append(pd.read_csv(folder/f'{method}_oof.csv',dtype={'image_id':str,'group':str}))
                test_all[method].append(pd.read_csv(folder/f'{method}_test.csv',dtype={'image_id':str}))
            print('Reuse completed outer fold',fold,flush=True)
            continue
        outer=train.loc[train.fold!=fold].reset_index(drop=True)
        validation=train.loc[train.fold==fold].reset_index(drop=True)
        seed=cfg.SEED+int(fold)*1000
        fitting,stopping=research_group_holdout(outer,cfg.EARLY_STOP_FRACTION,seed+1)
        assert not set(outer.group)&set(validation.group)
        assert not set(fitting.group)&set(stopping.group)
        assert not (set(outer.image_id)|set(validation.image_id))&set(test.image_id)
        roles=pd.concat([fitting[['image_id','group']].assign(role='model_fit'),
                         stopping[['image_id','group']].assign(role='early_stop'),
                         validation[['image_id','group']].assign(role='outer_validation')])
        roles.to_csv(folder/'split_roles.csv',index=False)
        print(f'OUTER FOLD {fold}: fit={len(fitting)}, stop={len(stopping)}, final_train={len(outer)}, outer_validation={len(validation)}',flush=True)
        coarse_select,ci=optimized_stage(fitting,stopping,cfg,device,'coarse',folder/'coarse_select',seed+10)
        stop_prediction=predict_research(coarse_select,stopping,cfg,device,'coarse')
        stop_roi=make_roi_spec(stop_prediction,stopping,cfg,mode='fixed')
        stop_prediction.to_csv(folder/'early_stop_coarse.csv',index=False)
        _research_cleanup(coarse_select);del coarse_select
        fit_roi=optimized_crossfit(fitting,cfg,device,folder/'crop_crossfit',ci['best_epoch'],seed+100)
        full_roi=pd.concat([fit_roi,stop_roi],ignore_index=True)
        assert set(full_roi.image_id)==set(outer.image_id) and full_roi.image_id.is_unique
        fine_select,fi=optimized_stage(fitting,stopping,cfg,device,'fine',folder/'fine_select',seed+20,train_crop=fit_roi,val_crop=stop_roi)
        _research_cleanup(fine_select);del fine_select
        coarse,_=optimized_stage(outer,None,cfg,device,'coarse',folder/'coarse_final',seed+30,epochs=ci['best_epoch'])
        coarse_val=predict_research(coarse,validation,cfg,device,'coarse')
        coarse_test=predict_research(coarse,test,cfg,device,'coarse')
        _research_cleanup(coarse);del coarse
        fine,_=optimized_stage(outer,None,cfg,device,'fine',folder/'fine_final',seed+40,epochs=fi['best_epoch'],train_crop=full_roi)
        metrics=[]
        for kind,frame,cp in [('oof',validation,coarse_val),('test',test,coarse_test)]:
            fp,bp,views=optimized_inference(fine,frame,cp,cfg,device)
            views.to_csv(folder/f'fine_scales_{kind}.csv',index=False)
            for method,pred in [('COARSE',cp),('FINE_MULTISCALE',fp),('BLEND',bp)]:
                if kind=='oof':
                    part=_research_attach_truth(pred,frame,int(fold),method)
                    oof_all[method].append(part)
                    metrics.append({'fold':int(fold),'method':method,**localization_metrics(pred,frame)})
                else:
                    part=pred.assign(fold=int(fold),method=method);test_all[method].append(part)
                part.to_csv(folder/f'{method}_{kind}.csv',index=False)
        _research_cleanup(fine);del fine
        pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
        record={'signature':signature,'fold':int(fold),'final_training_images':len(outer),
                'coarse_epochs':ci['best_epoch'],'fine_epochs':fi['best_epoch'],'metrics':metrics}
        record['files']=research_cache_manifest(folder,[p for p in folder.iterdir() if p.is_file() and p.name!='complete.json'])
        json_save(record,marker)
        print('COMPLETED OUTER FOLD',fold,flush=True)
    metrics=[];combined={}
    for method in oof_all:
        pred=pd.concat(oof_all[method],ignore_index=True)
        assert len(pred)==len(train) and pred.image_id.is_unique
        pred=pred.set_index('image_id').loc[train.image_id].reset_index()
        pred.to_csv(out/f'oof_{method}.csv',index=False);combined[method]=pred
        metrics.append({'method':method,**localization_metrics(pred,train)})
        tests=pd.concat(test_all[method],ignore_index=True)
        assert (tests.groupby('image_id').size()==5).all()
        tests.to_csv(out/f'test_all_folds_{method}.csv',index=False)
        average=tests.groupby('image_id',as_index=False)[['x','y']].mean()
        ensemble=test[['image_id','width','height']].merge(average,on='image_id',validate='one_to_one')
        ensemble['visibility_prob']=1.0
        ensemble.to_csv(out/f'test_ensemble_{method}.csv',index=False)
        path=cfg.SUBMISSION_PATH if method=='BLEND' else str(out/f'submission_{method}.csv')
        write_submission(sample,layout,ensemble,metadata,replace(cfg,SUBMISSION_PATH=path),.5)
    results=pd.DataFrame(metrics);results.to_csv(out/'ablation_summary.csv',index=False)
    paired_group_bootstrap(combined,train,baseline='COARSE',n_bootstrap=1000,seed=cfg.SEED).to_csv(out/'paired_group_bootstrap.csv',index=False)
    diag=combined['BLEND'].copy();diag['visibility_prob']=1.0;diag['visible_true']=True
    plot_oof_errors(diag,train,.5,cfg,out)
    json_save({'status':'COMPLETE','n_train':len(train),'n_test':len(test),'n_folds':5,'stages_per_fold':7,
               'submission_method':'BLEND','coarse_weight':cfg.COARSE_BLEND_WEIGHT,
               'metrics':metrics,'submission_sha256':research_file_digest(cfg.SUBMISSION_PATH)},out/'run_summary.json')
    print(results.to_string(index=False),flush=True)
    return {'config':cfg,'summary':results,'oof':combined,'train':train,'test':test}
