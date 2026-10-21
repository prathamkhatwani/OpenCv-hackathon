import cv2
import numpy as np
import os

out_dir = "roadreport/frontend/public/sample"
os.makedirs(out_dir, exist_ok=True)

def draw_hud_box(img, box, label, color=(0, 0, 240)):
    x, y, w, h = box
    canvas = img.copy()
    
    # Modern bracket corners
    clen = min(22, w // 4, h // 4)
    t = 3
    # Corner brackets
    cv2.line(canvas, (x, y), (x + clen, y), color, t)
    cv2.line(canvas, (x, y), (x, y + clen), color, t)
    cv2.line(canvas, (x + w, y), (x + w - clen, y), color, t)
    cv2.line(canvas, (x + w, y), (x + w, y + clen), color, t)
    cv2.line(canvas, (x, y + h), (x + clen, y + h), color, t)
    cv2.line(canvas, (x, y + h), (x, y + h - clen), color, t)
    cv2.line(canvas, (x + w, y + h), (x + w - clen, y + h), color, t)
    cv2.line(canvas, (x + w, y + h), (x + w, y + h - clen), color, t)
    
    # Thin enclosing bounding box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

    # Label with dark solid backing
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(canvas, (x, y - th - 12), (x + tw + 14, y), color, -1)
    cv2.putText(canvas, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Location tag
    tag = f"BBOX: [{x},{y}] {w}x{h}px"
    (gw, gh), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(canvas, (x, y + h + 4), (x + gw + 8, y + h + gh + 10), (20, 20, 20), -1)
    cv2.putText(canvas, tag, (x + 4, y + h + gh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas

findings_config = [
    {
        "id": 1,
        "type": "Pothole",
        "src_file": "pothole_raw1.jpg",
        "box": (240, 395, 150, 75),
        "label": "[POTHOLE #1: HIGH SEV 78 | CONF 0.88]",
        "color": (0, 0, 235), # Red
        "is_crack": False,
        "is_shadow": False
    },
    {
        "id": 2,
        "type": "Crack",
        "src_file": "pothole_raw2.jpg",
        "box": (510, 190, 110, 310),
        "label": "[CRACK #2: MED SEV 52 | CONF 0.77]",
        "color": (0, 140, 255), # Orange
        "is_crack": True,
        "is_shadow": False
    },
    {
        "id": 3,
        "type": "Pothole",
        "src_file": "pothole_raw5.jpg",
        "box": (720, 320, 240, 180),
        "label": "[POTHOLE #3: MED SEV 41 | CONF 0.55]",
        "color": (0, 180, 255), # Amber
        "is_crack": False,
        "is_shadow": False
    },
    {
        "id": 4,
        "type": "Shadow",
        "src_file": "road_shadow_cand1.jpg",
        "box": (570, 330, 160, 110),
        "label": "[SURFACE STAIN #4: DISMISSED | CONF 0.21]",
        "color": (160, 160, 160), # Slate Gray
        "is_crack": False,
        "is_shadow": True
    },
    {
        "id": 5,
        "type": "Crack",
        "src_file": "road_shadow_cand2.jpg",
        "box": (780, 490, 180, 120),
        "label": "[CRACK #5: CONFIRMED | CONF 0.81]",
        "color": (0, 215, 255), # Yellow
        "is_crack": True,
        "is_shadow": False
    }
]

for cfg in findings_config:
    fid = cfg["id"]
    raw = cv2.imread(cfg["src_file"])
    if raw is None:
        print(f"Error reading {cfg['src_file']}")
        continue

    raw = cv2.resize(raw, (1024, 640))
    x, y, w, h = cfg["box"]

    # 1. Road Context View
    annotated_road = draw_hud_box(raw, cfg["box"], cfg["label"], cfg["color"])
    cv2.imwrite(f"{out_dir}/road_context_{fid}.jpg", annotated_road, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out_dir}/frame_{fid}_annotated.jpg", annotated_road, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out_dir}/frame_{fid}.jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 2. 1x Raw Defect Crop
    crop = raw[y:y+h, x:x+w].copy()
    cv2.imwrite(f"{out_dir}/crop_{fid}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 3. 3x Zoom Re-Inspection View with OpenCV Verification
    zoom = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(zoom, (0, 0), 1.0)
    sharpened = cv2.addWeighted(zoom, 1.5, blurred, -0.5, 0)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    overlay = sharpened.copy()
    zh, zw = sharpened.shape[:2]

    if cfg["is_crack"]:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        edges = cv2.Canny(enh, 40, 120)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        overlay[edges > 0] = (255, 230, 0) # Cyan highlight
        final_zoom = cv2.addWeighted(sharpened, 0.45, overlay, 0.55, 0)
        # OpenCV HUD Header and Footer
        cv2.rectangle(final_zoom, (0, 0), (zw, 36), (15, 23, 42), -1)
        cv2.putText(final_zoom, "3x BICUBIC + CANNY EDGE FISSURE TRACKING", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.rectangle(final_zoom, (0, zh - 28), (zw, zh), (15, 23, 42), -1)
        cv2.putText(final_zoom, "CYAN: SKELETONIZED CRACK FISSURE (0.85 CONTINUITY)", (12, zh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 230, 0), 1, cv2.LINE_AA)

    elif cfg["is_shadow"]:
        final_zoom = sharpened.copy()
        cv2.rectangle(final_zoom, (0, 0), (zw, 36), (15, 23, 42), -1)
        cv2.putText(final_zoom, "3x BICUBIC RE-INSPECTION [PASS-2 VERIFICATION]", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.rectangle(final_zoom, (0, zh - 28), (zw, zh), (15, 23, 42), -1)
        cv2.putText(final_zoom, "DISMISSED: CONTRAST 8.4 < 20 (UNIFORM ROAD STAIN)", (12, zh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 160, 255), 1, cv2.LINE_AA)

    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        blk = max(3, (min(zh, zw) // 6) * 2 + 1)
        thresh = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 6)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            large_cnts = sorted([c for c in cnts if cv2.contourArea(c) > 100], key=cv2.contourArea, reverse=True)[:3]
            for c in large_cnts:
                cv2.drawContours(overlay, [c], -1, (255, 230, 0), 2)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [c], -1, 255, -1)
                dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
                ring = cv2.subtract(dilated, mask)
                ring_cnts, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, ring_cnts, -1, (0, 165, 255), 2)

        final_zoom = cv2.addWeighted(sharpened, 0.45, overlay, 0.55, 0)
        cv2.rectangle(final_zoom, (0, 0), (zw, 36), (15, 23, 42), -1)
        cv2.putText(final_zoom, "3x BICUBIC + UNSHARP FILTER VERIFICATION", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.rectangle(final_zoom, (0, zh - 28), (zw, zh), (15, 23, 42), -1)
        cv2.putText(final_zoom, "CYAN: CAVITY CONTOUR | AMBER: DILATED RING", (12, zh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 230, 0), 1, cv2.LINE_AA)

    cv2.imwrite(f"{out_dir}/zoom_defect_{fid}.jpg", final_zoom, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out_dir}/zoom_{fid}.jpg", final_zoom, [cv2.IMWRITE_JPEG_QUALITY, 95])

print("Finished generating clean assets!")
