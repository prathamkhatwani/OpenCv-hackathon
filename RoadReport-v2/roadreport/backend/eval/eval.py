"""Same-universe event benchmark with missed detections and unmatched predictions.
python -m eval.eval --labels labels.json --predictions predictions.json --output metrics.json
Manual event matching must be frozen before inspecting threshold results.
"""
import argparse,json
from pathlib import Path

def evaluate(labels,predictions):
    truth={r['event_id']:r for r in labels}
    if len(truth)!=len(labels):raise ValueError('Duplicate ground-truth events')
    seen=set()
    for p in predictions:
        if p['prediction_id'] in seen:raise ValueError('Duplicate prediction IDs')
        seen.add(p['prediction_id'])
        if p.get('event_id') is not None and p['event_id'] not in truth:raise ValueError('Unknown event link')
    output={}
    for mode in ['vision_positive','shock_positive','fusion_positive']:
        matched=set();fp=0
        for p in predictions:
            if not p[mode]:continue
            eid=p.get('event_id')
            if eid in truth and truth[eid]['pothole_exists'] and eid not in matched:matched.add(eid)
            else:fp+=1
        positive={k for k,v in truth.items() if v['pothole_exists']};tp=len(matched);fn=len(positive-matched)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
        subgroup={k for k,v in truth.items() if v['pothole_exists'] and v['contact_label']=='no'}
        output[mode]={'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'no_contact_recall':len(subgroup&matched)/len(subgroup) if subgroup else None,'no_contact_events':len(subgroup)}
    output['coverage']={'ground_truth_events':len(labels),'predictions':len(predictions),'needs_review':sum(bool(p.get('needs_review')) for p in predictions),'note':'Review items count negative unless explicitly marked positive; report resolved-human metrics separately. Ground truth must include every defect, including undetected events.'}
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=evaluate(json.loads(Path(a.labels).read_text()),json.loads(Path(a.predictions).read_text()))
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
