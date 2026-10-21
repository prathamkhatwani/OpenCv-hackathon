import json,zipfile
import cv2,numpy as np,pytest
from app.datasets.rdd2022 import parse_voc,prepare,split_for
from app.ml.visual import crop_features,VisualModel

def xml(label='D40',box=(1,1,20,20)):
    a,b,c,d=box
    return f'<annotation><size><width>32</width><height>32</height></size><object><name>{label}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object></annotation>'.encode()

def test_voc_mapping_and_bad_geometry():
    w,h,objects,unknown=parse_voc(xml())
    assert (w,h)==(32,32) and objects[0]['class_id']==3 and objects[0]['bbox']==[0,0,20,20] and not unknown
    with pytest.raises(ValueError):parse_voc(xml(box=(20,20,10,10)))
    with pytest.raises(ValueError):parse_voc(b'<!DOCTYPE foo><annotation/>')
    assert parse_voc(xml('D99'))[3]==['D99']

def test_block_split():
    assert split_for('India','India_000001')==split_for('India','India_000099')
    assert split_for('India','India_000100')[0]!=split_for('India','India_000099')[0]

def test_import_real_contract_no_fake_sensor_labels(tmp_path):
    # Synthetic archive tests importer structure only, not real model performance.
    source=tmp_path/'fixture.zip';_,image=cv2.imencode('.jpg',np.zeros((32,32,3),dtype=np.uint8))
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('India/train/annotations/xmls/India_000001.xml',xml())
        z.writestr('India/train/images/India_000001.jpg',image.tobytes())
        z.writestr('India/test/images/India_009999.jpg',image.tobytes())
    summary=prepare(source,tmp_path/'prepared')
    assert summary['images']==1 and summary['class_counts']=={'D40':1}
    row=json.loads((tmp_path/'prepared/manifest.json').read_text())[0]
    assert row['telemetry'] is None and row['contact_label'] is None and row['impact_associated'] is None
    assert (tmp_path/'prepared'/row['image_path']).is_file()
    with pytest.raises(ValueError,match='empty'):prepare(source,tmp_path/'prepared')

def test_archive_path_guard(tmp_path):
    source=tmp_path/'bad.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('../outside.xml',xml())
    with pytest.raises(ValueError,match='Unsafe'):prepare(source,tmp_path/'prepared')

def test_visual_weights_validation(tmp_path):
    with pytest.raises(ValueError):crop_features(np.zeros((0,0,3),dtype=np.uint8))
    p=tmp_path/'bad.json';p.write_text(json.dumps({'schema':'rdd2022-hog-logistic-v1','coef':[1],'mean':[0],'scale':[0]}))
    with pytest.raises(ValueError):VisualModel(p)
