"""Reviewed evidence export, with explicit source/measurement limitations."""
import json
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,KeepTogether
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
FONT_DIR=Path(__file__).parent/'fonts'
pdfmetrics.registerFont(TTFont('RoadSans',str(FONT_DIR/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('RoadSans-Bold',str(FONT_DIR/'DejaVuSans-Bold.ttf')))



def generate_report(job,findings,image_dir,output_path):
    styles=getSampleStyleSheet();parts=[]
    for style in styles.byName.values():
        style.fontName='RoadSans-Bold' if style.name.startswith('Heading') or style.name=='Title' else 'RoadSans'
    styles['BodyText'].fontSize=9
    styles['BodyText'].leading=13
    def text(s,style='BodyText'):
        parts.append(Paragraph(escape(str(s)),styles[style]));parts.append(Spacer(1,8))
    opts=json.loads(job.get('options_json') or '{}')
    text('RoadReport • Evidence assessment','Title')
    text(f"Job: {job['id']} | Source: {opts.get('source_kind','real')} | Location: {job.get('location_text') or 'Not provided'}")
    text('Advisory inspection report. Visual severity and heuristic scores are not measured depth, repair prices or calibrated probabilities. Camera motion is a proxy. No-contact does not disprove a pothole.')
    selected=[f for f in findings if f['status'] in ('confirmed','approved')]
    text(f'{len(selected)} supported or human-approved findings; pending/rejected/dismissed candidates excluded.','Heading2')
    for i,f in enumerate(selected,1):
        m=json.loads(f.get('measurements_json') or '{}')
        text(f"{i}. {f['type'].capitalize()} at {f['timestamp_s']:.2f}s — {f['severity_label']} visual severity",'Heading2')
        text(f"Verification: {'Human-approved' if f['status']=='approved' else 'Automated visual baseline'} | Evidence: {m.get('evidence_class','visual_only')} | Score: {f['confidence']:.3f}")
        text(m.get('reason',''))
        text(f"Contact: {m.get('contact',{}).get('state','unknown')}; motion source: {m.get('impact',{}).get('source','unavailable')}")
        p=Path(image_dir)/Path(f.get('image_key') or '').name
        if p.is_file():
            from PIL import Image as PILImage
            with PILImage.open(p) as im:w,h=im.size
            parts.append(Image(str(p),width=6*inch,height=min(3.8*inch,6*inch*h/w),kind='proportional'))
            parts.append(Spacer(1,12))
    text('Limitations','Heading2')
    text('Fixed camera calibration and straight-path assumption are required for contact estimation. Water, shadows, night, stabilization and suspension differences may invalidate evidence. Ground truth and local inspection are required before maintenance decisions. Real-world fusion accuracy has not been established by the bundled software tests.')
    SimpleDocTemplate(output_path,rightMargin=54,leftMargin=54,topMargin=45,bottomMargin=45).build(parts)
    return output_path
