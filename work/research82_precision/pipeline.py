# Appended to the audited baseline engine and fine_variants.py; standalone in Kaggle.
REFINEMENT_VERSION = '5.1.0-research82-precision-20261027'
RESEARCH_VERSION = REFINEMENT_VERSION
REFINEMENT_VARIANTS = ('resnet18_narrow',)
REFINEMENT_WEIGHT = .20
REFINEMENT_MIN_GAIN_PX = .5
REFINEMENT_MIN_REL_GAIN = .02

def refinement_read(path):
    return pd.read_csv(path, dtype={'image_id': str, 'group': str})

def refinement_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def refinement_variant_config(cfg, variant, folder):
    if variant != 'resnet18_narrow':
        raise ValueError(variant)
    # The sole treatment is narrower Gaussian supervision for fine training.
    # Keep the 256 input, 64 heatmap, ResNet18, LR, seed, splits, and gates fixed.
    return replace(cfg, FINE_VARIANT='resnet18', OUTPUT_DIR=str(folder),
                   FINE_HEATMAP_SIZE=64, HEATMAP_SIGMA=1.5)

def refinement_blend(a, b, weight_b):
    assert a.image_id.is_unique and b.image_id.is_unique
    assert set(a.image_id) == set(b.image_id)
    aligned = b.set_index('image_id').loc[a.image_id]
    out = a[['image_id', 'x', 'y']].copy()
    out[['x', 'y']] = ((1-weight_b)*a[['x','y']].to_numpy()
                      +weight_b*aligned[['x','y']].to_numpy())
    return out

def refinement_metrics(pred):
    error = pred[['x','y']].to_numpy()-pred[['x_true','y_true']].to_numpy()
    distance = np.linalg.norm(error, axis=1)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'median_distance_px':float(np.median(distance)),
            'p90_distance_px':float(np.quantile(distance,.9)), 'max_distance_px':float(distance.max())}

def refinement_submit(reference_csv, predictions, test, target):
    sub = pd.read_csv(reference_csv)
    assert list(sub.columns) == ['ImageID','value'] and len(sub)==2*len(test)
    assert sub.ImageID.is_unique and predictions.image_id.is_unique
    coords=predictions.set_index('image_id'); dimensions=test.set_index('image_id')
    values=[]
    for name in sub.ImageID:
        iid,axis=name.split('_Fovea_'); assert axis in ('X','Y')
        value=float(coords.loc[iid,'x' if axis=='X' else 'y'])
        assert np.isfinite(value) and 0 <= value <= dimensions.loc[iid,'width' if axis=='X' else 'height']-1
        values.append(value)
    sub['value']=values
    sub.to_csv(target,index=False,float_format='%.6f')
    return sub

def refinement_fold(train, test, fold, cfg, old_run, source_run, output):
    source=Path(source_run)/f'fold_{fold}'; old=Path(old_run)/f'fold_{fold}'
    folder=Path(output)/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    saved=refinement_json(source/'complete.json')['files']
    needed=['coarse_final/model.pt','coarse_select/model.pt','calibration_coarse.csv','radius_calibration.json',
        'B0_oof.csv','B0_test.csv','fine_policy_selection.json','fine_final/actual_training_crops.csv',
        'split_roles.csv','split_audit.json','crop_crossfit/crossfit_audit.json']
    research_verify_cache(source,{name:saved[name] for name in needed},needed)
    for parent in (source,old):
        original=refinement_json(parent/'complete.json')['files']
        research_verify_cache(parent,{n:original[n] for n in ('B4_oof.csv','B4_test.csv')})
    roles=refinement_read(source/'split_roles.csv')
    def role(name):
        ids=set(roles.loc[roles.role==name,'image_id'])
        return train.loc[train.image_id.isin(ids)].reset_index(drop=True)
    fitting,stopping,calibration,validation=map(role,('model_fit','early_stop','calibration_only','outer_validation'))
    development=pd.concat([fitting,stopping]).sort_values('image_id',key=lambda s:s.map(int)).reset_index(drop=True)
    # Preserve the original input row order, including deterministic data-loader permutation semantics.
    development=train.loc[train.image_id.isin(development.image_id)].reset_index(drop=True)
    assert not set(development.group)&set(calibration.group)
    assert not set(train.loc[train.fold!=fold,'group'])&set(validation.group)
    assert set(validation.image_id)==set(train.loc[train.fold==fold,'image_id'])
    assert not set(train.image_id)&set(test.image_id)
    roles.to_csv(folder/'split_roles.csv',index=False)
    crop=refinement_read(source/'fine_final/actual_training_crops.csv')
    assert set(crop.image_id)==set(development.image_id) and crop.image_id.is_unique
    fitting_crop=crop.loc[crop.image_id.isin(fitting.image_id)]
    stopping_crop=crop.loc[crop.image_id.isin(stopping.image_id)]
    coarse_cal=refinement_read(source/'calibration_coarse.csv')
    radius=refinement_json(source/'radius_calibration.json')
    roi_cal=make_roi_spec(coarse_cal,calibration,cfg,mode='calibrated',calibration=radius)
    proxy=refinement_json(source/'fine_policy_selection.json')['selected']['best_rmse']
    lineage={'inherited_source':str(source),'verified_source_files':{n:saved[n] for n in needed},
        'coarse_and_training_centres':'Inherited byte-verified ResNet18 models and image-held-out centres.',
        'baseline_proxy_inner_score':proxy,'proxy_is_8p2_ensemble_score':False}
    json_save(lineage,folder/'coarse_lineage.json')
    models={}; candidates=[]
    seed=int(cfg.SEED)+int(fold)*1000
    for variant in REFINEMENT_VARIANTS:
        destination=folder/variant;destination.mkdir(exist_ok=True)
        variant_cfg=refinement_variant_config(cfg,variant,destination)
        json_save({'config':asdict(variant_cfg)},destination/'variant_config.json')
        policies=[]
        for multiscale in (False,True):
            name='multiscale' if multiscale else 'fixed'
            candidate_cfg=replace(variant_cfg,FINE_MULTISCALE_TRAIN=multiscale)
            print(f'FOLD {fold+1}/5 | {variant} | selection {name}',flush=True)
            model,info=_research_stage(fitting,stopping,candidate_cfg,'cuda','fine',
                destination/('fine_select_'+name),seed+20,train_crop=fitting_crop,val_crop=stopping_crop)
            policies.append({'policy':name,'multiscale':multiscale,'best_rmse':info['best_rmse'],'best_epoch':info['best_epoch']})
            _research_cleanup(model);del model
        selected=min(policies,key=lambda x:x['best_rmse'])
        final_cfg=replace(variant_cfg,FINE_MULTISCALE_TRAIN=selected['multiscale'])
        json_save({'candidates':policies,'selected':selected,'selection_role':'inner_early_stop_only',
                   'stopping_ids':stopping.image_id.tolist()},destination/'policy_selection.json')
        print(f'FOLD {fold+1}/5 | {variant} | final refit {selected["best_epoch"]} epochs',flush=True)
        model,info=_research_stage(development,None,final_cfg,'cuda','fine',destination/'fine_final',
            seed+40,train_crop=crop,epochs=int(selected['best_epoch']))
        fine_cal=predict_research(model,calibration,final_cfg,'cuda','fine',roi_cal)
        rule=fit_fallback_rule(coarse_cal,fine_cal,calibration,final_cfg)
        fine_cal.to_csv(destination/'calibration_fine.csv',index=False)
        json_save(rule,destination/'fallback_rule.json')
        json_save({'config':asdict(final_cfg)},destination/'final_config.json')
        models[variant]={'config':final_cfg,'checkpoint':destination/'fine_final/model.pt','rule':rule}
        candidates.append({'variant':variant,**selected,'model_sha256':research_file_digest(models[variant]['checkpoint'])})
        _research_cleanup(model);del model
    winner=min(candidates,key=lambda x:x['best_rmse'])
    gain=float(proxy-winner['best_rmse'])
    accepted=bool(gain>=REFINEMENT_MIN_GAIN_PX and gain>=REFINEMENT_MIN_REL_GAIN*proxy)
    selection={'fold':int(fold),'candidates':candidates,'selected':winner['variant'],
        'proxy_resnet18_inner_rmse':proxy,'proxy_is_8p2_ensemble':False,'gain_px':gain,
        'accepted':accepted,'blend_weight':REFINEMENT_WEIGHT if accepted else 0.0,
        'criterion':'At least 0.5px AND 2% improvement on this fold internal stopping score; no outer/test score used.'}
    # Freeze architecture/gate before any outer/test evaluation of new fine models.
    json_save(selection,folder/'selection.json')
    parts={}; metrics=[]
    for kind,frame in (('oof',validation),('test',test)):
        baseline=refinement_blend(refinement_read(old/f'B4_{kind}.csv'),refinement_read(source/f'B4_{kind}.csv'),.25)
        coarse=refinement_read(source/f'B0_{kind}.csv')
        outputs={'BASELINE8P2':baseline}
        for variant in REFINEMENT_VARIANTS:
            state=models[variant];model=research_build_model(state['config'],'fine','cuda')
            load_model_checkpoint(model,state['checkpoint'],'cuda')
            roi=make_roi_spec(coarse,frame,state['config'],mode='calibrated',calibration=radius)
            fp=predict_research(model,frame,state['config'],'cuda','fine',roi)
            outputs[variant]=apply_fallback(coarse,fp,state['rule'])
            _research_cleanup(model);del model
        outputs['GUARDED20']=refinement_blend(baseline,outputs[winner['variant']],REFINEMENT_WEIGHT) if accepted else baseline.copy()
        for name,pred in outputs.items():
            result=_research_attach_truth(pred,frame,fold,name) if kind=='oof' else pred.assign(fold=fold,experiment=name)
            result.to_csv(folder/f'{name}_{kind}.csv',index=False)
            parts[(name,kind)]=result
            if kind=='oof':metrics.append({'fold':int(fold),'method':name,**refinement_metrics(result)})
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    json_save({'status':'COMPLETE','fold':int(fold),'selection':selection,'metrics':metrics},folder/'complete.json')
    print('COMPLETED FOLD',fold+1,selection,flush=True)
    return parts,selection

def run_refinement(config, old_run, source_run, reference_csv):
    cfg=replace(config);output=Path(cfg.OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    assert not cfg.SMOKE_TEST and cfg.N_FOLDS==5 and cfg.INPUT_SIZE==cfg.FINE_INPUT_SIZE==256
    if not torch.cuda.is_available():raise RuntimeError('This experiment requires a GPU.')
    cfg.PRETRAINED_WEIGHTS_PATH=resolve_research_weights(cfg)
    assert research_file_digest(cfg.PRETRAINED_WEIGHTS_PATH)==RESEARCH_WEIGHT_SHA256
    seed_everything(cfg.SEED);cv2.setNumThreads(0)
    train,test,sample,layout,metadata=prepare_tables(cfg)
    train=assign_folds(train,cfg)
    original=refinement_read(Path(source_run)/'folds.csv').set_index('image_id').loc[train.image_id]
    for name in ('fold','group','pixel_hash','x','y'):assert np.array_equal(train[name],original[name]),name
    assert not set(train.pixel_hash)&set(test.pixel_hash)
    train.to_csv(output/'folds.csv',index=False);test.to_csv(output/'test_manifest.csv',index=False)
    protocol={'version':REFINEMENT_VERSION,'config':asdict(cfg),'variants':list(REFINEMENT_VARIANTS),
        'blend_weight':REFINEMENT_WEIGHT,'minimum_absolute_gain_px':REFINEMENT_MIN_GAIN_PX,
        'minimum_relative_gain':REFINEMENT_MIN_REL_GAIN,'proxy':'Single seed10000 ResNet18 inner score, not the 8.2 ensemble inner score.',
        'old_run':str(old_run),'source_run':str(source_run),'baseline_csv':str(reference_csv),
        'baseline_csv_sha256':research_file_digest(reference_csv),'runtime_digest':research_runtime_code_digest(),
        'scope':'Repeated development OOF, no external retinal labels or fovea-trained weights; no claim of universally best algorithm.'}
    json_save(protocol,output/'run_protocol.json')
    started=time.perf_counter();all_parts=defaultdict(list);selections=[]
    for fold in range(5):
        parts,selection=refinement_fold(train,test,fold,cfg,old_run,source_run,output)
        selections.append(selection)
        for key,part in parts.items():all_parts[key].append(part)
    summaries=[];ensembles={}
    for name in ('BASELINE8P2',)+REFINEMENT_VARIANTS+('GUARDED20',):
        oof=pd.concat(all_parts[(name,'oof')],ignore_index=True)
        assert len(oof)==len(train) and oof.image_id.is_unique
        oof.to_csv(output/f'oof_{name}.csv',index=False)
        summaries.append({'method':name,**refinement_metrics(oof)})
        combined=pd.concat(all_parts[(name,'test')],ignore_index=True)
        assert (combined.groupby('image_id').size()==5).all()
        combined.to_csv(output/f'test_all_folds_{name}.csv',index=False)
        avg=combined.groupby('image_id',as_index=False)[['x','y']].mean()
        avg.to_csv(output/f'test_ensemble_{name}.csv',index=False);ensembles[name]=avg
    # Verify inherited baseline equals the actual score-8.2 file before exporting a candidate.
    baseline_check=refinement_submit(reference_csv,ensembles['BASELINE8P2'],test,output/'submission_baseline_reconstructed.csv')
    original=pd.read_csv(reference_csv)
    assert np.max(np.abs(baseline_check.value-original.value))<1.1e-6
    submission=refinement_submit(reference_csv,ensembles['GUARDED20'],test,output/'submission.csv')
    summary=pd.DataFrame(summaries);summary.to_csv(output/'ablation_summary.csv',index=False)
    json_save({'status':'COMPLETE','folds':5,'new_training_stages':15,'selected_folds':selections,
        'submission_sha256':research_file_digest(output/'submission.csv'),'elapsed_seconds':time.perf_counter()-started,
        'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20,'gpu':torch.cuda.get_device_name(0)},output/'execution_report.json')
    print(summary.to_string(index=False),flush=True)
    print('COMPLETE SUBMISSION:',output/'submission.csv',flush=True)
    return {'summary':summary,'submission':submission,'selections':selections,'config':cfg}
