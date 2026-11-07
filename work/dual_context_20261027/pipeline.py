"""Paired two-split experiment; append after the base, model and risk modules."""
DUAL_VERSION = '6.0.0-dual-context-20261027'
RESEARCH_VERSION = DUAL_VERSION
DUAL_SPLIT_SEEDS = (2026, 2027)
DUAL_WEIGHT = .20
DUAL_MIN_GAIN = .5
DUAL_MIN_REL_GAIN = .02
DUAL_METHODS = ('COARSE', 'SINGLE', 'DUAL_RAW', 'DUAL_RISK', 'MATCHED_DEPLOY')


def dc_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dc_read(path):
    return pd.read_csv(path, dtype={'image_id': str, 'group': str})


def dc_align(pred, ids):
    assert pred.image_id.is_unique and set(pred.image_id) == set(ids)
    return pred.set_index('image_id').loc[list(ids)].reset_index()


def dc_blend(first, second, weight):
    second = dc_align(second, first.image_id)
    out = first[['image_id', 'x', 'y']].copy()
    out[['x', 'y']] = (1-weight)*first[['x','y']].to_numpy() + weight*second[['x','y']].to_numpy()
    return out


def dc_roi(centres, frame):
    """Exact deployment side: 15% of the short side, no hidden margin."""
    centres = dc_align(centres, frame.image_id)
    x = centres['crop_x' if 'crop_x' in centres else 'x'].to_numpy(float)
    y = centres['crop_y' if 'crop_y' in centres else 'y'].to_numpy(float)
    side = .15 * frame[['width', 'height']].min(axis=1).to_numpy(float)
    return pd.DataFrame({'image_id': frame.image_id.to_numpy(), 'crop_x': x, 'crop_y': y, 'crop_side': side})


def dc_metrics(pred, truth, coarse=None, roi=None):
    pred = dc_align(pred, truth.image_id)
    difference = pred[['x','y']].to_numpy(float) - truth[['x','y']].to_numpy(float)
    distance = np.linalg.norm(difference, axis=1)
    result = {'rmse': float(np.sqrt(np.mean(difference**2))),
              'mean_distance_px': float(distance.mean()), 'median_distance_px': float(np.median(distance)),
              'p90_distance_px': float(np.quantile(distance,.9)), 'p95_distance_px': float(np.quantile(distance,.95)),
              'max_distance_px': float(distance.max()), 'images': len(distance)}
    if coarse is not None:
        coarse = dc_align(coarse, truth.image_id)
        coarse_distance = np.linalg.norm(coarse[['x','y']].to_numpy(float)-truth[['x','y']].to_numpy(float),axis=1)
        result['worse_than_coarse_fraction'] = float(np.mean(distance > coarse_distance+1e-9))
    if roi is not None:
        roi = dc_align(roi, truth.image_id)
        inside = (np.abs(truth[['x','y']].to_numpy(float)-roi[['crop_x','crop_y']].to_numpy(float)) <= roi.crop_side.to_numpy(float)[:,None]/2).all(axis=1)
        result['roi_coverage'] = float(inside.mean())
    return result


def dc_roles(train, fold, cfg, source=None):
    if source is not None:
        roles = dc_read(Path(source)/f'fold_{fold}/split_roles.csv')
        def take(role):
            return train.loc[train.image_id.isin(roles.loc[roles.role==role,'image_id'])].reset_index(drop=True)
        fitting, stopping, calibration, outer = [take(r) for r in ('model_fit','early_stop','calibration_only','outer_validation')]
        development = train.loc[train.image_id.isin(set(fitting.image_id)|set(stopping.image_id))].reset_index(drop=True)
    else:
        outer = train.loc[train.fold==fold].reset_index(drop=True)
        available = train.loc[train.fold!=fold].reset_index(drop=True)
        development, calibration = research_group_holdout(available,cfg.CALIBRATION_FRACTION,cfg.SEED+fold*1000)
        fitting, stopping = research_group_holdout(development,cfg.EARLY_STOP_FRACTION,cfg.SEED+fold*1000+1)
        roles = pd.concat([fr[['image_id','group']].assign(role=role) for role,fr in
                  (('model_fit',fitting),('early_stop',stopping),('calibration_only',calibration),('outer_validation',outer))],ignore_index=True)
    assert roles.image_id.is_unique and set(roles.image_id)==set(train.image_id)
    assert (roles.groupby('group').role.nunique()==1).all()
    assert set(outer.image_id)==set(train.loc[train.fold==fold,'image_id'])
    return fitting,stopping,development,calibration,outer,roles


def dc_coarse(fitting, stopping, development, calibration, outer, test, fold, cfg, folder, source=None):
    """Primary reuses verified held-out centres; secondary trains its own coarse models."""
    seed = cfg.SEED + fold*1000
    if source is not None:
        original = Path(source)/f'fold_{fold}'
        files = dc_json(original/'complete.json')['files']
        required = ['coarse_select/model.pt','coarse_final/model.pt','fine_final/actual_training_crops.csv',
                    'crop_crossfit/crossfit_audit.json','calibration_coarse.csv','B0_oof.csv','B0_test.csv','split_roles.csv']
        research_verify_cache(original,{name:files[name] for name in required},required)
        audit = dc_json(original/'crop_crossfit/crossfit_audit.json')
        held=[]
        for record in audit:
            fit_ids, held_ids = set(record['fit_ids']),set(record['held_ids'])
            assert fit_ids <= set(fitting.image_id) and held_ids <= set(fitting.image_id)
            assert not fit_ids & held_ids
            assert not (fit_ids|held_ids) & set(stopping.image_id)
            assert not set(record['fit_groups']) & set(record['held_groups'])
            held.extend(record['held_ids'])
        assert len(held)==len(set(held))==len(fitting) and set(held)==set(fitting.image_id)
        centres = dc_align(dc_read(original/'fine_final/actual_training_crops.csv'),development.image_id)
        crops = dc_roi(centres,development)
        predictions = {'calibration':dc_align(dc_read(original/'calibration_coarse.csv'),calibration.image_id),
                       'oof':dc_align(dc_read(original/'B0_oof.csv'),outer.image_id),
                       'test':dc_align(dc_read(original/'B0_test.csv'),test.image_id)}
        lineage = {'mode':'verified_primary_reuse','source':str(original),'files':{name:files[name] for name in required},
                   'coarse_checkpoint':str(original/'coarse_final/model.pt'),
                   'crossfit_fit_and_held_are_model_fit_only':True,'early_stop_groups_in_crossfit_training':0}
        json_save(audit,folder/'crop_crossfit_audit.json')
    else:
        select,info = _research_stage(fitting,stopping,cfg,'cuda','coarse',folder/'coarse_select',seed+10)
        stop_pred = predict_research(select,stopping,cfg,'cuda','coarse')
        stop_roi = dc_roi(stop_pred,stopping)
        _research_cleanup(select);del select
        fit_roi = research_crossfit_crops(fitting,cfg,'cuda',folder/'crop_crossfit',int(info['best_epoch']),seed+100)
        crops = dc_roi(pd.concat([fit_roi,stop_roi],ignore_index=True),development)
        coarse,_ = _research_stage(development,None,cfg,'cuda','coarse',folder/'coarse_final',seed+30,epochs=int(info['best_epoch']))
        predictions = {name:predict_research(coarse,frame,cfg,'cuda','coarse') for name,frame in [('calibration',calibration),('oof',outer)]}
        # No secondary test inference: this split is an audit, never a submission component.
        _research_cleanup(coarse);del coarse
        lineage={'mode':'new_secondary_models','coarse_checkpoint':str(folder/'coarse_final/model.pt'),
                 'coarse_epoch':int(info['best_epoch']),'early_stop_groups_in_crossfit_training':0}
        audit=dc_json(folder/'crop_crossfit/crossfit_audit.json')
        json_save(audit,folder/'crop_crossfit_audit.json')
    crops.to_csv(folder/'training_centres.csv',index=False)
    json_save(lineage,folder/'coarse_lineage.json')
    for name,pred in predictions.items(): pred.to_csv(folder/f'coarse_{name}.csv',index=False)
    return crops,predictions


def dc_fold(train,test,fold,cfg,output,old_run,source_run):
    folder=Path(output)/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    source=source_run if cfg.SEED==2026 else None
    fitting,stopping,development,calibration,outer,roles=dc_roles(train,fold,cfg,source)
    roles.to_csv(folder/'split_roles.csv',index=False)
    json_save({'fold':fold,'split_seed':cfg.SEED,'config':asdict(cfg)},folder/'fold_config.json')
    print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5: fit={len(fitting)}, stop={len(stopping)}, calibration={len(calibration)}, outer={len(outer)}',flush=True)
    crops,coarse=dc_coarse(fitting,stopping,development,calibration,outer,test,fold,cfg,folder,source)
    fit_crop=crops.loc[crops.image_id.isin(fitting.image_id)]
    stop_crop=crops.loc[crops.image_id.isin(stopping.image_id)]
    arm_states={};selections={}
    for arm in ('single','dual'):
        arm_folder=folder/arm;arm_folder.mkdir(exist_ok=True)
        arm_cfg=replace(cfg,FINE_VARIANT=arm,HEATMAP_SIGMA=1.5,OUTPUT_DIR=str(arm_folder))
        json_save({'config':asdict(arm_cfg)},arm_folder/'config.json')
        print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5 | {arm} | deployment-scale selection',flush=True)
        model,info=_research_stage(fitting,stopping,arm_cfg,'cuda','fine',arm_folder/'fine_select',cfg.SEED+fold*1000+20,
                                  train_crop=fit_crop,val_crop=stop_crop)
        stop_pred=predict_research(model,stopping,arm_cfg,'cuda','fine',stop_crop)
        stop_pred.to_csv(arm_folder/'selected_stop_predictions.csv',index=False)
        measured=dc_metrics(stop_pred,stopping)['rmse']
        assert abs(measured-info['best_rmse'])<1e-3
        selections[arm]={'best_epoch':int(info['best_epoch']),'best_rmse':float(info['best_rmse']),
                         'stopping_ids':stopping.image_id.tolist(),'metric':'actual deployment-scale original-coordinate RMSE with TTA'}
        _research_cleanup(model);del model
        print(f'SPLIT {cfg.SEED} FOLD {fold+1}/5 | {arm} | final refit {info["best_epoch"]} epochs',flush=True)
        model,_=_research_stage(development,None,arm_cfg,'cuda','fine',arm_folder/'fine_final',cfg.SEED+fold*1000+40,
                               train_crop=crops,epochs=int(info['best_epoch']))
        _research_cleanup(model);del model
        arm_states[arm]={'config':arm_cfg,'checkpoint':arm_folder/'fine_final/model.pt'}
    gain=selections['single']['best_rmse']-selections['dual']['best_rmse']
    accepted=bool(gain>=DUAL_MIN_GAIN and gain>=DUAL_MIN_REL_GAIN*selections['single']['best_rmse'])
    selection={'fold':fold,'split_seed':cfg.SEED,'arms':selections,'gain_px':gain,'accepted':accepted,
               'minimum_gain_px':DUAL_MIN_GAIN,'minimum_relative_gain':DUAL_MIN_REL_GAIN,'weight_if_accepted':DUAL_WEIGHT,
               'decision_inputs':'inner stopping data only; no calibration, outer or test labels'}
    json_save(selection,folder/'selection.json')
    # Both fit and selection roles are excluded from this separate risk calibration set.
    state=arm_states['dual'];model=research_build_model(replace(state['config'],PRETRAINED=False),'fine','cuda')
    load_model_checkpoint(model,state['checkpoint'],'cuda')
    roi_cal=dc_roi(coarse['calibration'],calibration)
    cal_features=dual_predict_features(model,calibration,state['config'],'cuda',roi_cal)
    cal_features.to_csv(folder/'calibration_dual_features.csv',index=False)
    roi_cal.to_csv(folder/'calibration_roi.csv',index=False)
    risk_rule=fit_risk_rule(coarse['calibration'],cal_features,calibration,state['config'])
    json_save(risk_rule,folder/'risk_rule.json')
    _research_cleanup(model);del model
    outputs={};metrics=[]
    kinds=[('oof',outer)] + ([('test',test)] if cfg.SEED==2026 else [])
    for kind,frame in kinds:
        cp=coarse[kind];roi=dc_roi(cp,frame);roi.to_csv(folder/f'{kind}_roi.csv',index=False)
        parts={'COARSE':cp}
        for arm in ('single','dual'):
            state=arm_states[arm];model=research_build_model(replace(state['config'],PRETRAINED=False),'fine','cuda')
            load_model_checkpoint(model,state['checkpoint'],'cuda')
            if arm=='single': parts['SINGLE']=predict_research(model,frame,state['config'],'cuda','fine',roi)
            else:
                parts['DUAL_RAW']=dual_predict_features(model,frame,state['config'],'cuda',roi)
                parts['DUAL_RISK']=apply_risk_rule(cp,parts['DUAL_RAW'],risk_rule)
            _research_cleanup(model);del model
        parts['MATCHED_DEPLOY']=dc_blend(parts['SINGLE'],parts['DUAL_RISK'],DUAL_WEIGHT) if accepted else parts['SINGLE'][['image_id','x','y']].copy()
        if cfg.SEED==2026:
            old=dc_read(Path(old_run)/f'fold_{fold}/B4_{kind}.csv')
            original=dc_read(Path(source_run)/f'fold_{fold}/B4_{kind}.csv')
            parts['BASELINE8P2']=dc_blend(old,original,.25)
            parts['SUBMISSION_CANDIDATE']=dc_blend(parts['BASELINE8P2'],parts['DUAL_RISK'],DUAL_WEIGHT) if accepted else parts['BASELINE8P2'].copy()
        for name,pred in parts.items():
            saved=_research_attach_truth(pred,frame,fold,name) if kind=='oof' else pred.assign(fold=fold,experiment=name)
            saved.to_csv(folder/f'{name}_{kind}.csv',index=False)
            outputs[(name,kind)]=saved
            if kind=='oof': metrics.append({'fold':fold,'split_seed':cfg.SEED,'method':name,**dc_metrics(pred,frame,cp,roi)})
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    artifacts=list(folder.rglob('*.json'))+list(folder.rglob('*.csv'))+list(folder.rglob('*.pt'))
    artifacts=[p for p in artifacts if p.name!='complete.json']
    json_save({'status':'COMPLETE','fold':fold,'split_seed':cfg.SEED,'selection':selection,
               'files':research_cache_manifest(folder,artifacts)},folder/'complete.json')
    print(f'COMPLETE SPLIT {cfg.SEED} FOLD {fold+1}/5 | dual accepted={accepted} | inner gain={gain:.4f}px',flush=True)
    return outputs,selection


def dc_export(reference,ensemble,test,target):
    submission=pd.read_csv(reference)
    assert list(submission.columns)==['ImageID','value'] and len(submission)==40 and submission.ImageID.is_unique
    prediction=ensemble.set_index('image_id');dimensions=test.set_index('image_id');values=[]
    for label in submission.ImageID:
        iid,axis=label.split('_Fovea_');assert axis in ('X','Y')
        value=float(prediction.loc[iid,'x' if axis=='X' else 'y'])
        assert np.isfinite(value) and 0<=value<=float(dimensions.loc[iid,'width' if axis=='X' else 'height'])-1
        values.append(value)
    submission['value']=values;submission.to_csv(target,index=False,float_format='%.6f')
    return submission


def run_dual_context(config,old_run,source_run,baseline_csv):
    cfg=replace(config);output=Path(cfg.OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and not cfg.SMOKE_TEST and cfg.N_FOLDS==5
    assert cfg.HEATMAP_SIGMA==2.5 and cfg.CONTEXT_RATIO==2.0 and cfg.INPUT_SIZE==cfg.FINE_INPUT_SIZE==256
    assert not cfg.FINE_MULTISCALE_TRAIN and not cfg.FINE_MULTISCALE_EARLY_STOP
    cfg.PRETRAINED_WEIGHTS_PATH=resolve_research_weights(cfg)
    assert research_file_digest(cfg.PRETRAINED_WEIGHTS_PATH)==RESEARCH_WEIGHT_SHA256
    seed_everything(cfg.SEED);cv2.setNumThreads(0)
    train,test,sample,layout,metadata=prepare_tables(cfg)
    assert len(train)==80 and len(test)==20 and not set(train.pixel_hash)&set(test.pixel_hash)
    test.to_csv(output/'test_manifest.csv',index=False)
    json_save({'version':DUAL_VERSION,'config':asdict(cfg),'split_seeds':list(DUAL_SPLIT_SEEDS),
               'new_training_stages':65,'local_short_side_fraction':.15,'context_ratio':2.0,
               'gate_min_px':DUAL_MIN_GAIN,'gate_min_relative':DUAL_MIN_REL_GAIN,'new_branch_weight':DUAL_WEIGHT,
               'old_run':str(old_run),'source_run':str(source_run),'baseline_csv':str(baseline_csv),
               'baseline_sha256':research_file_digest(baseline_csv),'runtime_digest':research_runtime_code_digest(),
               'secondary_scope':'complete fivefold stability audit only; no test inference or submission influence',
               'test_reference_labels_used':False},output/'run_protocol.json')
    started=time.perf_counter();all_summaries=[];decisions={};submission=None
    for split_seed in DUAL_SPLIT_SEEDS:
        split_out=output/f'split_{split_seed}';split_out.mkdir(exist_ok=True)
        split_cfg=replace(cfg,SEED=split_seed,OUTPUT_DIR=str(split_out))
        split_train=assign_folds(train,split_cfg);split_train.to_csv(split_out/'folds.csv',index=False)
        if split_seed==2026:
            original=dc_align(dc_read(Path(source_run)/'folds.csv'),split_train.image_id)
            for column in ('fold','group','pixel_hash','x','y'): assert np.array_equal(split_train[column],original[column]),column
        else:
            previous=dc_align(dc_read(output/'split_2026/folds.csv'),split_train.image_id)
            assert np.any(previous.fold.to_numpy()!=split_train.fold.to_numpy())
        parts=defaultdict(list);decisions[str(split_seed)]=[]
        for fold in range(5):
            fold_parts,selection=dc_fold(split_train,test,fold,split_cfg,split_out,old_run,source_run)
            decisions[str(split_seed)].append(selection)
            for key,value in fold_parts.items():parts[key].append(value)
        coarse_oof=pd.concat(parts[('COARSE','oof')],ignore_index=True)
        combined_roi=pd.concat([dc_read(split_out/f'fold_{f}/oof_roi.csv') for f in range(5)],ignore_index=True)
        for (name,kind),frames in parts.items():
            combined=pd.concat(frames,ignore_index=True)
            if kind=='oof':
                assert len(combined)==80 and combined.image_id.is_unique
                combined.to_csv(split_out/f'oof_{name}.csv',index=False)
                all_summaries.append({'split_seed':split_seed,'method':name,**dc_metrics(combined,split_train,coarse_oof,combined_roi)})
            else:
                assert len(combined)==100 and (combined.groupby('image_id').size()==5).all()
                combined.to_csv(split_out/f'test_folds_{name}.csv',index=False)
                ensemble=combined.groupby('image_id',as_index=False)[['x','y']].mean()
                ensemble.to_csv(split_out/f'test_ensemble_{name}.csv',index=False)
                if name=='BASELINE8P2':
                    baseline_check=dc_export(baseline_csv,ensemble,test,output/'baseline_reconstructed.csv')
                    assert np.max(np.abs(baseline_check.value-pd.read_csv(baseline_csv).value))<1.1e-6
                if name=='SUBMISSION_CANDIDATE':
                    submission=dc_export(baseline_csv,ensemble,test,output/'submission.csv')
        pd.DataFrame([r for r in all_summaries if r['split_seed']==split_seed]).to_csv(split_out/'summary.csv',index=False)
    summary=pd.DataFrame(all_summaries);summary.to_csv(output/'ablation_summary.csv',index=False)
    assert submission is not None
    json_save({'status':'COMPLETE','split_seeds':list(DUAL_SPLIT_SEEDS),'completed_folds':10,'new_training_stages':65,
               'decisions':decisions,'submission_sha256':research_file_digest(output/'submission.csv'),
               'elapsed_seconds':time.perf_counter()-started,'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20,
               'gpu':torch.cuda.get_device_name(0)},output/'execution_report.json')
    print(summary.to_string(index=False),flush=True)
    print('COMPLETE SUBMISSION:',output/'submission.csv',flush=True)
    return {'summary':summary,'submission':submission,'decisions':decisions}
