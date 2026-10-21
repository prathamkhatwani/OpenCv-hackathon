import cv2
import numpy as np
import imageio.v3 as iio
import os
import shutil

print("Starting 50 FPS smooth video compilation with realistic defects...")

# Source video: project_video.mp4 (1280x720, 25fps, 1260 frames)
cap = cv2.VideoCapture('project_video.mp4')
src_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We will create a 50.0 FPS video
target_fps = 50.0

# Load defect templates and masks
p_raw1 = cv2.imread('pothole_raw1.jpg')
p_mask1 = cv2.imread('pothole_mask1.png', cv2.IMREAD_GRAYSCALE)
# Large pothole crop (Finding 1)
large_pothole_crop = p_raw1[395:470, 240:390] # 75x150
large_pothole_mask = p_mask1[395:470, 240:390]
_, large_pothole_mask = cv2.threshold(large_pothole_mask, 10, 255, cv2.THRESH_BINARY)
# If mask is sparse, create solid cavity with edge
if cv2.countNonZero(large_pothole_mask) < 200:
    cv2.ellipse(large_pothole_mask, (75, 37), (65, 30), 0, 0, 360, 255, -1)

# Large pothole #2 (Finding 5)
p_raw5 = cv2.imread('pothole_raw5.jpg')
p_raw5 = cv2.resize(p_raw5, (1024, 640))
p_mask5 = cv2.imread('pothole_mask5.png', cv2.IMREAD_GRAYSCALE)
large_pothole2_crop = p_raw5[320:480, 720:940] # 160x220
large_pothole2_mask = np.zeros(large_pothole2_crop.shape[:2], dtype=np.uint8)
cv2.ellipse(large_pothole2_mask, (110, 80), (95, 65), -10, 0, 360, 255, -1)

# Small pothole crop (Finding 3)
p_raw3 = cv2.imread('pothole_raw3.jpg')
p_raw3 = cv2.resize(p_raw3, (1024, 640))
small_pothole_crop = p_raw3[310:355, 620:670] # 45x50
small_pothole_mask = np.zeros(small_pothole_crop.shape[:2], dtype=np.uint8)
cv2.ellipse(small_pothole_mask, (25, 22), (20, 16), 0, 0, 360, 255, -1)

# Crack crop (Finding 2)
p_raw2 = cv2.imread('pothole_raw2.jpg')
p_raw2 = cv2.resize(p_raw2, (1024, 640))
crack_crop = p_raw2[190:480, 510:600] # 290x90
crack_mask = np.zeros(crack_crop.shape[:2], dtype=np.uint8)
cv2.ellipse(crack_mask, (45, 145), (35, 135), 0, 0, 360, 255, -1)

# Shadow crop (Finding 4)
shadow_raw = cv2.imread('road_shadow_cand1.jpg')
shadow_raw = cv2.resize(shadow_raw, (1024, 640))
shadow_crop = shadow_raw[330:440, 570:730] # 110x160
shadow_mask = np.zeros(shadow_crop.shape[:2], dtype=np.uint8)
cv2.ellipse(shadow_mask, (80, 55), (70, 45), -15, 0, 360, 255, -1)

def blend_defect(frame, defect_img, defect_mask, center_x, center_y, width, height, darkness=0.75):
    fh, fw = frame.shape[:2]
    w = max(10, int(width))
    h = max(6, int(height))
    
    x1 = int(center_x - w // 2)
    y1 = int(center_y - h // 2)
    x2 = x1 + w
    y2 = y1 + h
    
    if x1 < 0 or y1 < 0 or x2 >= fw or y2 >= fh:
        return frame, None
    
    res_img = cv2.resize(defect_img, (w, h))
    res_mask = cv2.resize(defect_mask, (w, h))
    blurred_mask = cv2.GaussianBlur(res_mask, (9, 9), 3.0).astype(np.float32) / 255.0
    
    road_patch = frame[y1:y2, x1:x2].astype(np.float32)
    road_mean = max(1.0, float(road_patch.mean()))
    def_mean = max(1.0, float(res_img.mean()))
    
    tone_adjusted = res_img.astype(np.float32) * (road_mean / def_mean) * darkness
    m3 = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)
    
    blended = (tone_adjusted * m3 + road_patch * (1.0 - m3)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    return frame, (x1, y1, w, h)

# Defects schedule (in seconds)
# Each defect appears, grows as car approaches, and passes
defect_events = [
    {
        'id': 1,
        'type': 'pothole_large',
        'start_s': 3.5,
        'peak_s': 4.6,
        'end_s': 5.2,
        'img': large_pothole_crop,
        'mask': large_pothole_mask,
        'start_pos': (610, 480, 35, 18),
        'end_pos': (560, 560, 220, 110),
        'darkness': 0.70
    },
    {
        'id': 2,
        'type': 'crack',
        'start_s': 10.0,
        'peak_s': 11.2,
        'end_s': 12.0,
        'img': crack_crop,
        'mask': crack_mask,
        'start_pos': (640, 490, 25, 60),
        'end_pos': (600, 570, 90, 260),
        'darkness': 0.82
    },
    {
        'id': 3,
        'type': 'pothole_small',
        'start_s': 16.8,
        'peak_s': 17.8,
        'end_s': 18.5,
        'img': small_pothole_crop,
        'mask': small_pothole_mask,
        'start_pos': (625, 485, 15, 12),
        'end_pos': (590, 545, 65, 45),
        'darkness': 0.78
    },
    {
        'id': 4,
        'type': 'shadow',
        'start_s': 23.5,
        'peak_s': 24.6,
        'end_s': 25.3,
        'img': shadow_crop,
        'mask': shadow_mask,
        'start_pos': (605, 490, 40, 25),
        'end_pos': (540, 555, 180, 110),
        'darkness': 0.84
    },
    {
        'id': 5,
        'type': 'pothole_large2',
        'start_s': 30.0,
        'peak_s': 31.2,
        'end_s': 32.0,
        'img': large_pothole2_crop,
        'mask': large_pothole2_mask,
        'start_pos': (630, 480, 40, 28),
        'end_pos': (680, 570, 240, 160),
        'darkness': 0.72
    }
]

# We will read frames and generate 50fps video
# Source is 25fps, so 1 source frame generates 2 output frames
print(f"Reading {total_frames} frames from source video...")
all_frames = []

frame_idx = 0
last_frame = None

key_finding_frames = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    t_sec = frame_idx / src_fps
    
    # Check if any defect is active at this time
    active_boxes = {}
    for ev in defect_events:
        if ev['start_s'] <= t_sec <= ev['end_s']:
            prog = (t_sec - ev['start_s']) / max(0.01, (ev['end_s'] - ev['start_s']))
            # Ease in quadratic (approaching vehicle accelerates defect motion downward)
            eased = prog * prog
            
            x = ev['start_pos'][0] + (ev['end_pos'][0] - ev['start_pos'][0]) * eased
            y = ev['start_pos'][1] + (ev['end_pos'][1] - ev['start_pos'][1]) * eased
            w = ev['start_pos'][2] + (ev['end_pos'][2] - ev['start_pos'][2]) * eased
            h = ev['start_pos'][3] + (ev['end_pos'][3] - ev['start_pos'][3]) * eased
            
            frame, box = blend_defect(frame, ev['img'], ev['mask'], x, y, w, h, ev['darkness'])
            if box is not None:
                active_boxes[ev['id']] = box
                
            # If close to peak, save as key finding frame
            if abs(t_sec - ev['peak_s']) < 0.04 and ev['id'] not in key_finding_frames:
                key_finding_frames[ev['id']] = (frame.copy(), box, t_sec)

    # 50 fps interpolation: emit frame + blended transition frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if last_frame is not None:
        # Interpolated intermediate frame for silky smooth 50 fps
        interp = cv2.addWeighted(last_frame, 0.5, frame_rgb, 0.5, 0)
        all_frames.append(interp)
        
    all_frames.append(frame_rgb)
    last_frame = frame_rgb
    
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  Processed {frame_idx}/{total_frames} frames ({t_sec:.1f}s)...")

cap.release()

print(f"Total output frames: {len(all_frames)} at 50 FPS ({len(all_frames)/50.0:.1f}s)")

# Write H.264 video
out_path = 'roadreport/frontend/public/sample_dashcam.mp4'
print(f"Writing 50 FPS H.264 video to {out_path}...")
iio.imwrite(
    out_path,
    all_frames,
    fps=50.0,
    codec='libx264',
    pixelformat='yuv420p',
    output_params=['-preset', 'veryfast', '-crf', '22']
)

print(f"Video created successfully! Size: {os.path.getsize(out_path)} bytes")

# Also copy to dist/
shutil.copy(out_path, 'roadreport/frontend/dist/sample_dashcam.mp4')
print("Copied video to dist/")

# Save key finding frames and annotated road context images
out_sample_dir = 'roadreport/frontend/public/sample'

print("Generating updated road context views for key findings...")
for ev in defect_events:
    fid = ev['id']
    if fid in key_finding_frames:
        k_frame, k_box, k_t = key_finding_frames[fid]
    else:
        continue
        
    kx, ky, kw, kh = k_box
    
    # Save clean road frame
    cv2.imwrite(f"{out_sample_dir}/frame_{fid}.jpg", k_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Create annotated road context image with OpenCV HUD box
    annotated = k_frame.copy()
    
    if fid == 1:
        color = (0, 0, 235) # Red (High)
        label = f"[POTHOLE #1: CRITICAL LARGE | SEV: 92 | CONF: 0.94]"
    elif fid == 2:
        color = (0, 140, 255) # Orange (Medium)
        label = f"[CRACK #2: ASPHALT JOINT | SEV: 52 | CONF: 0.77]"
    elif fid == 3:
        color = (0, 215, 100) # Green (Low)
        label = f"[POTHOLE #3: SHALLOW PIT | SEV: 24 | CONF: 0.62]"
    elif fid == 4:
        color = (150, 150, 150) # Gray (Dismissed)
        label = f"[SURFACE STAIN #4: DISMISSED | CONF: 0.21]"
    elif fid == 5:
        color = (0, 0, 235) # Red (High)
        label = f"[POTHOLE #5: CRITICAL LARGE | SEV: 88 | CONF: 0.91]"
        
    # Draw corner brackets & label
    clen = min(24, kw // 4, kh // 4)
    t = 3
    cv2.line(annotated, (kx, ky), (kx + clen, ky), color, t)
    cv2.line(annotated, (kx, ky), (kx, ky + clen), color, t)
    cv2.line(annotated, (kx + kw, ky), (kx + kw - clen, ky), color, t)
    cv2.line(annotated, (kx + kw, ky), (kx + kw, ky + clen), color, t)
    cv2.line(annotated, (kx, ky + kh), (kx + clen, ky + kh), color, t)
    cv2.line(annotated, (kx, ky + kh), (kx, ky + kh - clen), color, t)
    cv2.line(annotated, (kx + kw, ky + kh), (kx + kw - clen, ky + kh), color, t)
    cv2.line(annotated, (kx + kw, ky + kh), (kx + kw, ky + kh - clen), color, t)
    cv2.rectangle(annotated, (kx, ky), (kx + kw, ky + kh), color, 1)
    
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(annotated, (kx, ky - th - 12), (kx + tw + 14, ky), color, -1)
    cv2.putText(annotated, label, (kx + 7, ky - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    
    tag = f"BBOX: [{kx},{ky}] {kw}x{kh}px @ {k_t:.1f}s"
    (gw, gh), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(annotated, (kx, ky + kh + 4), (kx + gw + 8, ky + kh + gh + 10), (20, 20, 20), -1)
    cv2.putText(annotated, tag, (kx + 4, ky + kh + gh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    cv2.imwrite(f"{out_sample_dir}/road_context_{fid}.jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out_sample_dir}/frame_{fid}_annotated.jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Also save 1x crop from this frame
    pad = 10
    cy1 = max(0, ky - pad)
    cy2 = min(k_frame.shape[0], ky + kh + pad)
    cx1 = max(0, kx - pad)
    cx2 = min(k_frame.shape[1], kx + kw + pad)
    crop = k_frame[cy1:cy2, cx1:cx2]
    cv2.imwrite(f"{out_sample_dir}/crop_{fid}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

print("Finished compiling 50 FPS video and updated assets!")
