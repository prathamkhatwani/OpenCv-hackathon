"""python -m app.ml.train --data events.json --output models/fusion.json
Splits must be assigned by drive AND physical site before training.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,brier_score_loss,log_loss
from app.ml.features import FEATURES,VISUAL_FEATURES,vector
from app.ml.infer import raw_score,sigmoid


def validate(records,allow_synthetic=False):
    if not records:raise ValueError('Empty dataset')
    ids=set();groups={};sites={}
    for r in records:
        for k in ['event_id','drive_id','site_id','split','pothole_exists','impact_associated','contact_label','source_kind','features']:
            if k not in r:raise ValueError(f'Missing {k}')
        if r['event_id'] in ids:raise ValueError('Duplicate event_id')
        ids.add(r['event_id'])
        if r['split'] not in ('train','validation','test'):raise ValueError('Invalid split')
        if r['source_kind']!='real' and not allow_synthetic:raise ValueError('Synthetic data requires --allow-synthetic; output will be experimental')
        for key,mapping in [('drive_id',groups),('site_id',sites)]:
            if not r[key]:raise ValueError(f'Empty {key}')
            if mapping.setdefault(r[key],r['split'])!=r['split']:raise ValueError(f'{key} leaks across splits')
        if r['contact_label'] not in ('yes','no','unknown'):raise ValueError('Invalid contact label')
        if r['pothole_exists'] not in (0,1) or r['impact_associated'] not in (None,0,1):raise ValueError('Invalid labels')
        if r['impact_associated']==1 and (r['pothole_exists']!=1 or r['contact_label']!='yes'):raise ValueError('Associated impact requires independently verified pothole and contact')
        vector(r['features'])


def metrics(y,p):
    return {'n':len(y),'precision':float(precision_score(y,p>=.5,zero_division=0)),'recall':float(recall_score(y,p>=.5,zero_division=0)),'f1':float(f1_score(y,p>=.5,zero_division=0)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1]))}


def export(model,names,mean,scale):
    spec={'features':names,'mean':mean.tolist(),'scale':scale.tolist()}
    if isinstance(model,LogisticRegression):spec.update(kind='logistic',coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))
    else:
        prior=model.init_.class_prior_[1]
        spec.update(kind='gradient_boosting',initial=float(np.log(prior/(1-prior))),learning_rate=model.learning_rate,trees=[])
        for estimator in model.estimators_[:,0]:
            t=estimator.tree_;spec['trees'].append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()})
    return spec


def train(records):
    artifact={'schema_version':1,'source_kind':'real' if all(r['source_kind']=='real' for r in records) else 'synthetic','models':{},'test_metrics':{},'selection_metrics':{},'seed':42,'feature_schema':FEATURES}
    for target,names in [('pothole_exists',VISUAL_FEATURES),('impact_associated',FEATURES)]:
        sets={s:[r for r in records if r['split']==s and r[target] is not None] for s in ('train','validation','test')}
        for split,rows in sets.items():
            if len(rows)<10 or len({r[target] for r in rows})<2:raise ValueError(f'{target}: {split} requires at least 10 labeled events and both classes')
        x={s:np.array([vector(r['features'],names) for r in rows]) for s,rows in sets.items()};y={s:np.array([r[target] for r in rows]) for s,rows in sets.items()}
        mean=x['train'].mean(0);scale=np.maximum(x['train'].std(0),1e-6);z={s:(xx-mean)/scale for s,xx in x.items()}
        candidates=[LogisticRegression(max_iter=2000,random_state=42),GradientBoostingClassifier(n_estimators=60,max_depth=2,min_samples_leaf=3,random_state=42)]
        scores=[]
        for model in candidates:
            model.fit(z['train'],y['train']);scores.append(float(log_loss(y['validation'],model.predict_proba(z['validation'])[:,1])))
        chosen=candidates[int(np.argmin(scores))]
        calibration=LogisticRegression(C=1.,max_iter=1000).fit(chosen.decision_function(z['validation']).reshape(-1,1),y['validation'])
        spec=export(chosen,names,mean,scale);spec['calibration']={'coef':float(calibration.coef_[0,0]),'intercept':float(calibration.intercept_[0])}
        probs=np.array([sigmoid(spec['calibration']['coef']*raw_score(spec,r['features'])+spec['calibration']['intercept']) for r in sets['test']])
        # Verify exported inference against sklearn before writing deployable artifact.
        expected=calibration.predict_proba(chosen.decision_function(z['test']).reshape(-1,1))[:,1]
        if not np.allclose(probs,expected,atol=1e-8):raise ValueError('Export parity failed')
        artifact['models'][target]=spec;artifact['test_metrics'][target]=metrics(y['test'],probs)
        artifact['selection_metrics'][target]={'validation_log_loss':dict(zip(['logistic','gradient_boosting'],scores)),'selected':spec['kind']}
        if target=='pothole_exists':
            ix=[i for i,r in enumerate(sets['test']) if r['pothole_exists']==1 and r['contact_label']=='no']
            artifact['test_metrics']['no_contact_pothole_recall']={'n':len(ix),'recall':float(np.mean(probs[ix]>=.5)) if ix else None}
    artifact['limitations']=['Validation set is reused for model selection and calibration; use larger nested splits for strong claims.','Event classifier only: does not measure missed visual proposals. Evaluate end-to-end detector recall separately.','No automatic production release; examine subgroup errors and dataset rights.']
    return artifact


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True);parser.add_argument('--allow-synthetic',action='store_true');args=parser.parse_args()
    raw=Path(args.data).read_bytes();records=json.loads(raw);validate(records,args.allow_synthetic);artifact=train(records)
    artifact['dataset_sha256']=hashlib.sha256(raw).hexdigest()
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    print(json.dumps({'output':str(p),'source_kind':artifact['source_kind'],'test_metrics':artifact['test_metrics']},indent=2))
if __name__=='__main__':main()
