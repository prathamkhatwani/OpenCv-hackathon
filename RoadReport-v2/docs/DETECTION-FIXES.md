# Detection integration revision — 26 October 2026

Base repository commit: e48f721. No changes pushed to GitHub.

## Root causes corrected

1. The API advertised YOLO, but `run_pipeline` called classical pothole/crack
   functions. It now calls the trained OpenCV DNN detector directly once per frame.
2. The tracker required strictly downward movement and ended a track immediately
   on a missed observation. It now uses predicted boxes, global one-to-one
   assignment and a bounded recovery interval. IDs are scoped to each job.
3. When the agent saved no findings, a fallback recreated raw tracks as confirmed.
   That fallback is removed. Zero findings is a valid result.
4. Heuristic reinspection could raise a neural confidence score. Confirmation now
   requires temporal support from model outputs; sharpening is evidence viewing only.
5. The frontend expected `measurements.track`, but it was not populated. The agent
   now persists normalized boxes and timestamps, observation counts and track IDs.
6. The player drew only the selected finding, with a narrow timestamp window.
   It now renders all active findings using video-frame callbacks (animation-frame
   fallback), short-interval interpolation and explicit seek updates.
7. Video dimensions and duration were missing from the job response. They are now
   read from the actual video and exposed to the player to avoid aspect-ratio drift.
8. Removed fabricated pothole contrast/solidity/ring measurements. Severity remains
   a visual area proxy, not a physical depth measurement.

## Precision policy

Model proposal threshold: 0.45 (was 0.18). Same-class NMS IoU: 0.45.
Automatic confirmation: at least three observations with confidence >= 0.60 and
median track confidence >= 0.60. Repeated candidates with max score >= 0.50 may
be sent for review; isolated flashes are dismissed. These are conservative
engineering defaults, not thresholds calibrated on a new held-out dataset.
They can reduce recall and miss subtle potholes. Confidence is not a calibrated
probability that a pothole exists.

The bundled model and its upstream benchmark metadata are unchanged. Upstream
benchmark numbers must not be presented as new measurements for this revision.
No fresh YOLO training was run. The training script was improved and syntax-checked,
but running it and measuring target-domain precision remain outstanding.

## Validation

- Backend suite: 23 tests passed, including real ONNX loading/inference, ROI
  filtering, NMS, tracking, API/database integration and regression checks.
- Five JavaScript overlay tests passed: interpolation, lifetimes, reverse seeking,
  long-gap suppression and empty tracks.
- Frontend production build and lint passed.
- Integration tests decode a generated textured video and exercise real tracking,
  agent decisions, persistence, image generation and API serialization. Detector
  outputs are stubbed in those tests; they do not measure real-world accuracy.
- Model inference also ran on repository JPEGs. Those images have no verified
  ground-truth labels, so no precision/recall claim is made from that smoke test.
- Browser automation was attempted, but Chromium downloads returned corrupt
  archives in this environment. End-to-end browser playback was not verified.
- No Windows execution, GPU training or AWS deployment validation was performed.

## Limits

Tracking can split after >1 second with no observations or under abrupt scene
changes. Nearby/overlapping objects can still be misassociated. Cross-trip or
out-of-view deduplication is not guaranteed. Overlays intentionally disappear
across observation gaps >0.5 seconds instead of inventing locations.

Detection still samples sharp frames at roughly four per second, with the
existing 240-frame cap. Use short clips (up to 60 seconds); longer recordings
may only be partially analyzed. Existing telemetry/calibration/event modules
were not connected as part of this fix. The scripted Demo remains synthetic.
