import cv2
import os
import numpy as np

os.makedirs('roadreport/frontend/public/sample', exist_ok=True)
cap = cv2.VideoCapture('roadreport/frontend/public/sample_dashcam.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

defects = [
    {'id': 1, 'type': 'Pothole', 't': 4.2, 'sev': 'High', 'score': 78, 'conf': 0.86, 'box': [580, 480, 140, 75]},
    {'id': 2, 'type': 'Crack', 't': 9.8, 'sev': 'Medium', 'score': 52, 'conf': 0.77, 'box': [420, 510, 220, 65]},
    {'id': 3, 'type': 'Pothole', 't': 14.5, 'sev': 'Medium', 'score': 41, 'conf': 0.55, 'box': [660, 490, 110, 60]},
    {'id': 4, 'type': 'Shadow', 't': 17.3, 'sev': 'Low', 'score': 0, 'conf': 0.21, 'box': [380, 460, 160, 70]},
    {'id': 5, 'type': 'Crack', 't': 21.0, 'sev': 'Low', 'score': 24, 'conf': 0.81, 'box': [490, 520, 180, 55]}
]

for d in defects:
    frame_no = int(d['t'] * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        print('Could not read frame for', d['id'])
        continue

    fh, fw = frame.shape[:2]
    x, y, w, h = d['box']
    x = max(0, min(x, fw - w))
    y = max(0, min(y, fh - h))

    # 1. Save raw frame
    cv2.imwrite(f"roadreport/frontend/public/sample/frame_{d['id']}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2. Save annotated frame with real OpenCV box & label
    annotated = frame.copy()
    color = (0, 0, 255) if d['sev'] == 'High' else (0, 140, 255) if d['sev'] == 'Medium' else (0, 215, 255)
    if d['type'] == 'Shadow':
        color = (120, 120, 120)

    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
    lbl = f"{d['type']} ({d['sev']} {d['conf']:.2f})"
    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 8, y), color, -1)
    cv2.putText(annotated, lbl, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(f"roadreport/frontend/public/sample/frame_{d['id']}_annotated.jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3. Save raw 1x crop from actual asphalt
    crop = frame[y:y+h, x:x+w].copy()
    cv2.imwrite(f"roadreport/frontend/public/sample/crop_{d['id']}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4. Save agent 3x zoom re-inspection crop with real OpenCV processing
    zoom = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(zoom, (0, 0), 1.0)
    sharpened = cv2.addWeighted(zoom, 1.5, blurred, -0.5, 0)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    overlay = sharpened.copy()

    if d['type'] == 'Pothole':
        blk = max(3, (min(gray.shape) // 4) * 2 + 1)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 6)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            mc = max(cnts, key=cv2.contourArea)
            cv2.drawContours(overlay, [mc], -1, (255, 255, 0), 3)
            mask_in = np.zeros_like(gray)
            cv2.drawContours(mask_in, [mc], -1, 255, -1)
            dilated = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
            ring_cnts, _ = cv2.findContours(cv2.subtract(dilated, mask_in), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, ring_cnts, -1, (0, 165, 255), 2)
    elif d['type'] == 'Crack':
        edges = cv2.Canny(gray, 30, 90)
        overlay[edges > 0] = (0, 220, 255)

    final_zoom = cv2.addWeighted(sharpened, 0.45, overlay, 0.55, 0)
    cv2.imwrite(f"roadreport/frontend/public/sample/zoom_{d['id']}.jpg", final_zoom, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Processed finding {d['id']}: real frame, crop, and zoom saved")

cap.release()
print("All real assets successfully generated!")
