"""Fixed, prespecified coordinate ensemble. No fitting on test labels or OOF tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def combine_runs(baseline_dir, additional_dir, output_dir, baseline_csv, candidate_csv):
    baseline_dir,additional_dir,output_dir=map(Path,(baseline_dir,additional_dir,output_dir))
    output_dir.mkdir(parents=True,exist_ok=True)
    read=lambda p:pd.read_csv(p,dtype={'image_id':str,'group':str})
    scores=[]
    def metrics(df):
        residual=df[['x','y']].to_numpy()-df[['x_true','y_true']].to_numpy()
        distance=np.linalg.norm(residual,axis=1)
        return {'coordinate_rmse_px':float(np.sqrt(np.mean(residual**2))),
                'median_distance_px':float(np.median(distance)),
                'p90_distance_px':float(np.quantile(distance,.9)),
                'max_distance_px':float(distance.max())}
    a=read(baseline_dir/'oof_B4.csv').sort_values('image_id').reset_index(drop=True)
    b=read(additional_dir/'oof_B4.csv').set_index('image_id').loc[a.image_id].reset_index()
    assert a.image_id.is_unique and b.image_id.is_unique and len(a)==len(b)==80
    for c in ['fold','group','x_true','y_true']:assert np.array_equal(a[c],b[c]),c
    c=a.copy()
    c[['x','y']]=.75*a[['x','y']].to_numpy()+.25*b[['x','y']].to_numpy()
    c['experiment']='BASELINE75_SEED25'
    c['error_px']=np.hypot(c.x-c.x_true,c.y-c.y_true)
    # Old single-model uncertainty, fallback flags, and ROI sizes do not describe this ensemble.
    c=c[['image_id','x','y','x_true','y_true','group','fold','width','height','experiment','error_px']]
    c.to_csv(output_dir/'oof_candidate.csv',index=False)
    changes=c[['image_id','fold','group','x_true','y_true','width','height']].copy()
    for name,df in [('baseline',a),('additional_seed',b),('candidate',c)]:
        changes[name+'_error_px']=np.hypot(df.x-df.x_true,df.y-df.y_true)
    changes['error_change_px']=changes.candidate_error_px-changes.baseline_error_px
    changes.sort_values('error_change_px',ascending=False).to_csv(output_dir/'oof_error_changes.csv',index=False)
    for name,df in [('baseline_8p9',a),('additional_seed',b),('candidate_75_25',c)]:
        scores.append({'method':name,**metrics(df)})
    summary=pd.DataFrame(scores);summary.to_csv(output_dir/'comparison.csv',index=False)
    fold_rows=[]
    for f in sorted(a.fold.unique()):
        fold_rows.append({'fold':int(f),**{name:metrics(df.loc[df.fold==f])['coordinate_rmse_px']
            for name,df in [('baseline',a),('additional_seed',b),('candidate',c)]}})
    pd.DataFrame(fold_rows).to_csv(output_dir/'fold_comparison.csv',index=False)
    pa=read(baseline_dir/'test_ensemble_B4.csv').sort_values('image_id').reset_index(drop=True)
    pb=read(additional_dir/'test_ensemble_B4.csv').set_index('image_id').loc[pa.image_id].reset_index()
    assert len(pa)==len(pb)==20 and pa.image_id.is_unique and pb.image_id.is_unique
    pc=pa[['image_id','width','height']].copy()
    pc[['x','y']]=.75*pa[['x','y']].to_numpy()+.25*pb[['x','y']].to_numpy()
    pc.to_csv(output_dir/'test_ensemble_candidate.csv',index=False)
    sub=pd.read_csv(baseline_csv)
    assert list(sub.columns)==['ImageID','value'] and len(sub)==40 and sub.ImageID.is_unique
    pc=pc.set_index('image_id')
    values=[]
    for key in sub.ImageID:
        iid,axis=key.split('_Fovea_');assert axis in ('X','Y')
        value=float(pc.loc[iid,'x' if axis=='X' else 'y'])
        assert np.isfinite(value) and 0<=value<=pc.loc[iid,'width' if axis=='X' else 'height']-1
        values.append(value)
    candidate=sub.copy();candidate['value']=values
    candidate.to_csv(candidate_csv,index=False,float_format='%.6f')
    diff=pa[['image_id','x','y']].rename(columns={'x':'baseline_x','y':'baseline_y'})
    diff[['candidate_x','candidate_y']]=pc.loc[diff.image_id,['x','y']].to_numpy()
    diff['displacement_px']=np.hypot(diff.candidate_x-diff.baseline_x,diff.candidate_y-diff.baseline_y)
    diff.to_csv(output_dir/'test_prediction_changes.csv',index=False)
    # Fixed-prediction group bootstrap, descriptive only, not retraining uncertainty.
    groups=a.group.unique();indices={g:np.flatnonzero(a.group.to_numpy()==g) for g in groups}
    err_a=((a[['x','y']].to_numpy()-a[['x_true','y_true']].to_numpy())**2).mean(axis=1)
    err_c=((c[['x','y']].to_numpy()-c[['x_true','y_true']].to_numpy())**2).mean(axis=1)
    rng=np.random.default_rng(427);deltas=[]
    for _ in range(2000):
        ix=np.concatenate([indices[g] for g in rng.choice(groups,len(groups),replace=True)])
        deltas.append(float(np.sqrt(err_c[ix].mean())-np.sqrt(err_a[ix].mean())))
    report={'weights':{'baseline':.75,'additional_seed':.25},'weights_selected_before_training':True,
       'scores':scores,'rmse_delta_candidate_minus_baseline':scores[2]['coordinate_rmse_px']-scores[0]['coordinate_rmse_px'],
       'paired_group_bootstrap_95pct_delta_interval':np.quantile(deltas,[.025,.975]).tolist(),
       'mean_test_displacement_px':float(diff.displacement_px.mean()),'max_test_displacement_px':float(diff.displacement_px.max()),
       'oof_images_improved':int((changes.error_change_px<0).sum()),
       'oof_images_worsened':int((changes.error_change_px>0).sum()),
       'worst_candidate_oof_images':changes.nlargest(5,'candidate_error_px').to_dict('records'),
       'scope':'Repeated development OOF; not independent validation. Leaderboard scores are user-reported; candidate leaderboard score unknown.',
       'no_ensemble_uncertainty_coverage_claim':True,'active_baseline_not_overwritten':True}
    (output_dir/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    ROOT=Path(__file__).resolve().parents[2]
    OUT=ROOT/'outputs/restored_20261027'
    print(combine_runs(ROOT/'outputs/trained_submission_20261025_fixed',OUT/'seed_10000',OUT,
                      OUT/'submission_restored_8p9.csv',OUT/'submission_candidate.csv').to_string(index=False))
