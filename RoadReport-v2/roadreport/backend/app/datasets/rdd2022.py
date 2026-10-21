"""RDD2022 VOC ingestion with source checksums and deterministic blocked splits.
python -m app.datasets.rdd2022 --archive RDD2022_India.zip --output data/rdd2022
"""
import argparse,collections,hashlib,json,re,zipfile
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET
import cv2,numpy as np
CLASSES={'D00':0,'D10':1,'D20':2,'D40':3}
NAMES=['longitudinal_crack','transverse_crack','alligator_crack','pothole']
SOURCE='https://github.com/sekilab/RoadDamageDetector'

def split_for(country,stem):
    match=re.search(r'(\d+)$',stem)
    if not match:raise ValueError('Cannot assign sequence block: '+stem)
    group=f'{country}:block-{int(match[1])//100}'
    bucket=int(hashlib.sha256(('roadreport-rdd-v1:'+group).encode()).hexdigest()[:8],16)%10
    return group,('train' if bucket<7 else 'validation' if bucket<9 else 'test')

def parse_voc(raw):
    if len(raw)>2_000_000 or b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:raise ValueError('Unsafe annotation XML')
    root=ET.fromstring(raw);w=int(root.findtext('size/width'));h=int(root.findtext('size/height'))
    if not 1<=w<=10000 or not 1<=h<=10000:raise ValueError('Invalid image dimensions')
    objects=[];unknown=[]
    for obj in root.findall('object'):
        category=obj.findtext('name','').strip()
        if category not in CLASSES:unknown.append(category);continue
        box=obj.find('bndbox')
        if box is None:raise ValueError('Missing bbox')
        # VOC integer coordinates -> zero-based, half-open xyxy. Zero lower bounds accepted.
        xmin,ymin,xmax,ymax=[int(float(box.findtext(k))) for k in ('xmin','ymin','xmax','ymax')]
        x1,y1=max(0,xmin-1),max(0,ymin-1);x2,y2=xmax,ymax
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid bounding box')
        objects.append({'category':category,'class_id':CLASSES[category],'bbox':[x1,y1,x2,y2]})
    return w,h,objects,unknown

def prepare(archive,output):
    archive=Path(archive);out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be new/empty to prevent stale split mixing')
    out.mkdir(parents=True,exist_ok=True)
    with archive.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
    records=[];skipped=[];seen_hashes={};unknown_counts=collections.Counter()
    with zipfile.ZipFile(archive) as z:
        members=z.infolist();by_name={m.filename:m for m in members}
        if len(by_name)!=len(members):raise ValueError('Duplicate archive members')
        for m in members:
            p=PurePosixPath(m.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in m.filename:raise ValueError('Unsafe archive path')
        xmls=sorted(m.filename for m in members if m.filename.lower().endswith('.xml') and '/train/' in '/'+m.filename)
        if not xmls:raise ValueError('No labeled train XML files; official test images are unlabeled')
        for xml in xmls:
            p=PurePosixPath(xml);parts=p.parts;country=parts[parts.index('train')-1]
            if not re.fullmatch(r'[A-Za-z_]+',country):raise ValueError('Invalid domain name')
            prefix='/'.join(parts[:parts.index('train')+1]);image_name=f'{prefix}/images/{p.stem}.jpg'
            if image_name not in by_name:skipped.append({'xml':xml,'reason':'missing_image'});continue
            if by_name[xml].file_size>2_000_000 or by_name[image_name].file_size>40_000_000:raise ValueError('Oversized member')
            raw=z.read(xml)
            try:w,h,objects,unknown=parse_voc(raw)
            except (ValueError,TypeError,ET.ParseError) as e:skipped.append({'xml':xml,'reason':str(e)});continue
            unknown_counts.update(unknown)
            # Do not silently turn excluded category boxes into negative background.
            if unknown:skipped.append({'xml':xml,'reason':'unsupported_categories','categories':unknown});continue
            image=z.read(image_name);image_hash=hashlib.sha256(image).hexdigest()
            if image_hash in seen_hashes:
                skipped.append({'xml':xml,'reason':'duplicate_image','original':seen_hashes[image_hash]});continue
            decoded=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape[:2]!=(h,w):skipped.append({'xml':xml,'reason':'image_dimensions_or_decode_mismatch'});continue
            group,split=split_for(country,p.stem);stem=f'{country}_{p.stem}'
            dest=out/'images'/split/(stem+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(image)
            labels=out/'labels'/split/(stem+'.txt');labels.parent.mkdir(parents=True,exist_ok=True)
            lines=[]
            for obj in objects:
                x1,y1,x2,y2=obj['bbox'];lines.append(f"{obj['class_id']} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            labels.write_text('\n'.join(lines))
            records.append({'image_id':stem,'image_path':dest.relative_to(out).as_posix(),'image_sha256':image_hash,'xml_sha256':hashlib.sha256(raw).hexdigest(),'source_image':image_name,'source_xml':xml,'country':country,'width':w,'height':h,'group':group,'split':split,'objects':objects,'source_kind':'real','contact_label':None,'impact_associated':None,'telemetry':None})
            seen_hashes[image_hash]=xml
    if not records:raise ValueError('No valid labeled records')
    classes=collections.Counter(o['category'] for r in records for o in r['objects']);splits={s:{'images':sum(r['split']==s for r in records),'boxes':dict(collections.Counter(o['category'] for r in records if r['split']==s for o in r['objects']))} for s in ('train','validation','test')}
    summary={'dataset':'RDD2022','source':SOURCE,'archive_name':archive.name,'archive_sha256':digest,'images':len(records),'class_counts':dict(classes),'splits':splits,'skipped':len(skipped),'unsupported_categories':dict(unknown_counts),'split_policy':'Filename sequence blocks of 100, hashed 70/20/10; exact duplicates removed. Not verified drive/site disjoint. Official test images excluded (unlabeled).','sensor_labels':'Unavailable; never fabricate contact or impact labels.'}
    (out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'skipped.json').write_text(json.dumps(skipped,indent=2))
    (out/'data.yaml').write_text('path: '+json.dumps(str(out.resolve()))+'\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES)))
    for split in ('train','validation','test'):
        subset=[r for r in records if r['split']==split];annotations=[]
        for i,r in enumerate(subset):
            for o in r['objects']:
                x1,y1,x2,y2=o['bbox'];annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':o['class_id'],'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
        coco={'images':[{'id':i,'file_name':r['image_path'],'width':r['width'],'height':r['height']} for i,r in enumerate(subset)],'annotations':annotations,'categories':[{'id':i,'name':n} for i,n in enumerate(NAMES)]}
        (out/f'{split}.coco.json').write_text(json.dumps(coco))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a.archive,a.output),indent=2))
