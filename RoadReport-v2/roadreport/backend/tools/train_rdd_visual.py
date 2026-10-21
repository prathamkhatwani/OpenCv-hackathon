"""Train a small CPU visual baseline on prepared RDD2022 data. No fake IMU labels."""
import argparse,collections,hashlib,json
from pathlib import Path
import cv2,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,log_loss
from app.ml.visual import crop_features,context_crop,VisualModel

def train(root,output):
    root=Path(root);raw=(root/'manifest.json').read_bytes();records=json.loads(raw)
    data={s:[] for s in ('train','validation','test')};seen={};hash_splits={}
    for r in records:
        if seen.setdefault(r['group'],r['split'])!=r['split']:raise ValueError('Group leakage')
        if hash_splits.setdefault(r['image_sha256'],r['split'])!=r['split']:raise ValueError('Image leakage')
        image=cv2.imread(str(root/r['image_path']))
        if image is None:raise ValueError('Missing image '+r['image_path'])
        rows=[]
        for o in r['objects']:
            rows.append((o['bbox'],int(o['category']=='D40'),'annotated'))
        # One bottom-half road patch; avoid overlap with every known damage box.
        rng=np.random.default_rng(int(r['image_sha256'][:8],16));h,w=image.shape[:2]
        side=max(16,min(h,w)//5)
        for _ in range(12):
            x=int(rng.integers(0,max(1,w-side)));y=int(rng.integers(h//2,max(h//2+1,h-side)))
            b=[x,y,x+side,min(h,y+side)]
            if not any(min(b[2],o['bbox'][2])>max(b[0],o['bbox'][0]) and min(b[3],o['bbox'][3])>max(b[1],o['bbox'][1]) for o in r['objects']):
                rows.append((b,0,'unannotated_background'));break
        for box,label,kind in rows:
            crop=context_crop(image,box)
            if min(crop.shape[:2])<8:continue
            data[r['split']].append((crop_features(crop),label,kind))
    for split,rows in data.items():
        if len(rows)<20 or len(set(r[1] for r in rows))<2:raise ValueError('Each split needs at least 20 crops and both classes: '+split)
    x={s:np.array([r[0] for r in rows]) for s,rows in data.items()};y={s:np.array([r[1] for r in rows]) for s,rows in data.items()}
    mean=x['train'].mean(0);scale=np.maximum(x['train'].std(0),1e-5);z={s:(v-mean)/scale for s,v in x.items()}
    choices=[]
    for c in [.01,.1,1.]:
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=1000,random_state=42).fit(z['train'],y['train'])
        choices.append((log_loss(y['validation'],model.predict_proba(z['validation'])[:,1]),c,model))
    loss,c,model=min(choices,key=lambda v:v[0])
    cal=LogisticRegression(C=1.,max_iter=1000).fit(model.decision_function(z['validation']).reshape(-1,1),y['validation'])
    probs=cal.predict_proba(model.decision_function(z['test']).reshape(-1,1))[:,1];pred=probs>=.5
    metrics={'n':len(pred),'pothole_crops':int(y['test'].sum()),'precision':float(precision_score(y['test'],pred,zero_division=0)),'recall':float(recall_score(y['test'],pred)),'f1':float(f1_score(y['test'],pred)),'balanced_accuracy':float(balanced_accuracy_score(y['test'],pred)),'log_loss':float(log_loss(y['test'],probs))}
    artifact={'schema':'rdd2022-hog-logistic-v1','dataset_manifest_sha256':hashlib.sha256(raw).hexdigest(),'source_summary':json.loads((root/'summary.json').read_text()),'seed':42,'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'mean':mean.tolist(),'scale':scale.tolist(),'calibration':{'coef':float(cal.coef_[0,0]),'intercept':float(cal.intercept_[0])},'test_metrics':metrics,'selected_C':c,'selection_validation_log_loss':float(loss),'crop_counts':{s:dict(collections.Counter(str(r[1]) for r in rows)) for s,rows in data.items()},'limitations':['Pilot crop classifier, not detector mAP.','Filename blocks are not verified independent sites/drives.','Unannotated background patches can contain missed labels.','Validation reused for hyperparameter choice and calibration.','No contact, no shock, no temporal labels; cannot validate sensor fusion.','Scores are displayed only and never alter decisions.']}
    # Check the exported arithmetic independently before writing.
    exported_raw=((x['test']-mean)/scale)@np.array(artifact['coef'])+artifact['intercept']
    exported=1/(1+np.exp(-np.clip(artifact['calibration']['coef']*exported_raw+artifact['calibration']['intercept'],-40,40)))
    assert np.allclose(exported,probs,atol=1e-8)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    VisualModel(out)
    report={k:v for k,v in artifact.items() if k not in ('coef','intercept','mean','scale','calibration')}
    report['model_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();out.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();train(a.data,a.output)
