"""Join independently annotated labels to exported observed features, never auto-label."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--exports',nargs='+',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True);a=p.parse_args()
features={}
for path in a.exports:
    export=json.loads(Path(path).read_text())
    for finding in export['findings']:
        if finding['type']=='pothole':
            features[finding['id']]=json.loads(finding['measurements_json'])['features']
rows=json.loads(Path(a.labels).read_text());result=[]
for row in rows:
    if row['event_id'] not in features:raise ValueError('Label has no observed pothole candidate: '+row['event_id'])
    result.append(dict(row,features=features[row['event_id']]))
Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False))
print(f'Joined {len(result)} candidate events. Missed detections belong in the separate end-to-end benchmark.')
