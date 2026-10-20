"""Portable JSON inference; never unpickle model uploads."""
import json,math,hashlib
from pathlib import Path
from app.ml.features import vector

def sigmoid(x):return 1/(1+math.exp(-max(-40,min(40,float(x)))))

def raw_score(spec,features):
    x=vector(features,spec['features'])
    x=(x-spec['mean'])/spec['scale']
    if spec['kind']=='logistic':return float(sum(x*spec['coef'])+spec['intercept'])
    score=spec['initial']
    for tree in spec['trees']:
        i=0
        for _ in range(128):
            if tree['left'][i]<0:
                score+=spec['learning_rate']*tree['value'][i];break
            i=tree['left'][i] if x[tree['feature'][i]]<=tree['threshold'][i] else tree['right'][i]
        else:raise ValueError('Invalid model tree')
    return score

class FusionModel:
    def __init__(self,path):
        p=Path(path)
        if p.stat().st_size>5_000_000:raise ValueError('Model file too large')
        raw=p.read_bytes();self.data=json.loads(raw)
        if self.data.get('schema_version')!=1:raise ValueError('Unsupported model schema')
        self.sha256=hashlib.sha256(raw).hexdigest()
        self.experimental=self.data.get('source_kind')!='real'

    def predict(self,features):
        result={}
        for name,spec in self.data['models'].items():
            raw=raw_score(spec,features);cal=spec['calibration']
            result[name]=sigmoid(cal['coef']*raw+cal['intercept'])
        return result

    def info(self):
        return {'state':'experimental' if self.experimental else 'trained_pilot','sha256':self.sha256,'source_kind':self.data.get('source_kind'),'metrics':self.data.get('test_metrics'),'note':'Dataset-specific pilot; no production accuracy guarantee'}
