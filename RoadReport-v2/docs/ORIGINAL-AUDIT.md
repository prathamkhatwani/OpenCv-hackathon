# RoadReport: repository audit and competition proposal

Reviewed 20 October 2026. Proposal only; no application source changes or model training performed.

Repository: https://github.com/prathamkhatwani/opencv

Snapshot: main, commit `1826b47ce071d0df907de289084818a39e2d5d41`. Inventory: 134 tracked files, including 85 raster images and one video. All application source/configuration and documentation were reviewed; raster assets were inspected as contact sheets, SVGs structurally inspected, and video metadata plus six representative frames inspected. This is not an exhaustive frame-by-frame video review, an AWS deployment audit, or a browser interaction test. Only main exists in the fetched remote branch inventory. Both project READMEs are identical.

## Recommendation

Keep the useful React/FastAPI/OpenCV foundation. Fix its integration and evidence problems first, then add contact-aware visual/motion fusion, an evidence replay, and bounded adaptive re-inspection. This is a credible competition direction; winning cannot be predicted.

## Competition brief

- Devpost header: October 26, 23:45 PDT; body: 23:59. Earlier cutoff converts to October 27, 12:15 IST. Our target: October 26 evening IST.
- Required: substantive OpenCV 5, meaningful AWS execution, report, accessible code, pinned dependencies/instructions, architecture, working endpoint or arranged screen-share, video at most five minutes, evaluation including failures. [S1]
- Overall scoring: technical 30%; innovation 20%; impact 20%; UX, documentation and cloud/responsibility 10% each. Cash: $5,000/$3,000/$2,000 plus COOL and Agentic Vision $1,000 each. One overall award can combine with special awards. Participants retain IP; submitted materials receive broad promotional/use licensing. Eligibility forms, third-party rights and truthful results matter. Rules permit ages 13+ with guardian permission below majority; Devpost overview says majority. [S2]
- Official OpenCV page distinguishes $12,000 prizes from $8,250 credits, but contradicts itself on teams: five versus four. It also lists August 17 registration closure and required proposal language. Verify your existing registration/proposal with organizers. [S3]
- Grant selection is not necessary to win. Notifications have already gone out. [S4]
- Grant count/check-in dates conflict: Devpost says 50 teams and October 7–14; OpenCV says 55 participants and September 21–October 2. Judging October 27–November 9; announcement November 10. Do not silently choose between conflicting eligibility instructions. [S1,S3]
- Agentic eligibility requires evidence to change a later action; COOL eligibility requires the core workload on Graviton/Arm with reproducible comparisons. These are optional for overall awards. [S1]

## What exists today

The intended workflow is upload → storage → frame sampling → classical pothole/crack detection → temporal tracking → rule-driven re-inspection → findings → human review → PDF.

Potholes are found with CLAHE, bilateral filtering, adaptive thresholding, morphology, contour geometry and interior-versus-ring contrast. Cracks use Canny edges, lane-color rejection, connected components and skeletonization. These are classical CV heuristics, not trained ML. Tracking uses spatial displacement, size growth and template matching. Scores called confidence are handcrafted values, not calibrated probabilities.

The agent is a deterministic policy that can select a stricter crop inspection. That is a useful starting point for a perception/action workflow; it is not a trained reasoning model. AWS S3 support and EC2 scripts exist, but no deployed endpoint or account evidence was provided.

## Verification performed

| Check | Observed result | Meaning |
|---|---|---|
| Existing backend tests | 6 passed, one dependency warning | Tests cover CRUD/storage/API and mocked integration; passing does not prove real CV integration. |
| Real tracker adapter with a minimal detection | KeyError `frame` after fallback | The original flat-versus-nested input error is masked by retrying reversed arguments. |
| Real agent adapter | TypeError: missing callback/job/frame arguments | Production call does not match `run_agent`. |
| Real PDF adapter | TypeError: missing findings/image_dir/output_path | Production endpoint does not supply required image directory. |
| Unmocked short synthetic-video pipeline | Job marked failed at agent invocation | Its frames were rejected by blur filtering; this checks the zero-retained-frame route, not detector accuracy. |
| Local-storage containment helper | `../outside.txt` resolves outside storage | Path containment is missing; no private file was read. |
| Frontend build | Passed, Vite 8.3.4 | Compilation succeeds. |
| Frontend lint | Seven warnings, no errors | Includes hook dependencies, unused code and recursive callback warning. |
| Fresh dependency install | OpenCV 5.0.0, package 5.0.0.93 | A fresh install now selects 5, but deployed environments are unverified and dependencies are unpinned. |
| Sample video | H.264, 1280×720, 50 fps, 2,519 frames, 50.38 s | Asset generation code interpolates source footage and composites defects. |
| Git status after audit | No tracked modifications | No requested implementation was started. |

Dependencies were installed into a temporary Python environment and ignored frontend node_modules; generated build/test material is not a project change.

## Findings to fix before adding features

### P0: application integration

1. **Tracker input contract.** `pipeline.py` appends every detection into one flat list. `track.py` expects one list per sampled frame. `_invoke_tracker` catches TypeError and retries reversed inputs, hiding the original bug. Use a typed explicit interface and preserve empty frame lists.
2. **Agent input contract.** `run_agent(tracks, add_step_fn, create_finding_fn, job_id, frame_data_list)` cannot be invoked by the current `_invoke_agent`. Supply explicit adapters rather than guessing signatures.
3. **Agent database callbacks.** The positional logging calls use a different order from `db.add_agent_step`. The creation callback receives a rich dictionary including arrays, while `db.create_finding` expects named scalar fields and returns a record, not just an ID. Fix both contracts.
4. **Trace identifiers.** Agent logs use `track_N`, while the API looks up steps by persisted finding UUID. Introduce one persistent finding/track relationship. Preserve dismissed evidence too.
5. **Report adapter.** `report.generate_report` requires `image_dir`. Download the selected S3 evidence to a temporary directory and pass it explicitly.
6. **Review/report correctness.** The endpoint currently passes every finding to the report. The report labels all non-confirmed statuses as Human-approved. Filter approved/confirmed records, preserve human-review provenance, and never describe pending/rejected findings as verified. Review updates also leave job summaries stale.

### P0: evidence integrity

7. **Silent demo substitution.** `App.jsx` swaps in SAMPLE_FINDINGS after upload failure and after a successful result with zero findings. Keep a separate opt-in synthetic demo. Real failures must remain failures; no detections needs an explicit empty state.
8. **Invented measurements in UI.** The reinspection component computes initial confidence as final confidence × 0.7. App invents persistence from frame_index and default boxes/scores. Replace with actual values or unavailable. Zero values must not be overwritten by `||` defaults.
9. **Simulated location/telemetry.** RouteMap interpolates a fictional corridor and adds Math.random jitter. The player hardcodes speed, GPS, resolution and LIVE. Use synchronized telemetry or label schematic/unknown. Changing an address does not currently geocode its default coordinates.
10. **Unsupported benchmark/depth claims.** README lists precise precision/recall improvements without committed supporting results, dataset manifest or split. Its depth/volume claims conflict with the report's own limitations. Brightness differences are not dB or physical depth; bicubic upscaling cannot recover independently measured detail.
11. **Asset provenance.** Scripts composite road defects with manually chosen scores. Some scratch images show stock/news watermarks; rights are not established by their presence in the repo. Keep provenance/license records or replace with team-owned footage. Current sample #5 zoom is labeled as a crack while the frontend classifies sample #5 as a pothole; sample #3 also has mismatched source configurations.

### P1: reliability, evaluation and deployment

12. **Motion signal discarded.** 4 fps sampling and blur rejection are unsuitable as the only shock stream. Preserve original-timestamp high-rate motion measurements separately.
13. **Tracking fragility.** Positive downward motion is mandatory; stationary objects, turns, jitter and one missed sample can end a track. Use ego-motion compensation and bounded missing-frame tolerance. Avoid ndarray-containing dictionary equality when locating a best detection; retain its index.
14. **Severity is a proxy.** Screen area changes with distance; darkness changes with lighting. Present a visual severity proxy unless calibrated physical measurements exist. Repair costs and aggregate road-health formula are estimates without validated local rates or length normalization.
15. **Evaluation denominators differ.** First pass scores retained sampled frames, treating unlabeled frames as empty ground truth; agent scoring visits only labeled frames and uses one best frame per track. This can hide agent false positives and count a persistent object inconsistently. Evaluate the same event units/coverage with exhaustive labels. Include skipped/blurred cases in coverage. `--frames-dir` is documented but absent from the parser.
16. **Security controls.** Constrain resolved storage paths to the storage root; restrict local serving to local mode/authorized objects. Add access checks, bounded concurrent work, video duration/decoded-frame limits, HTTPS, specific CORS origins and sanitized errors before public deployment.
17. **Reproducibility.** Pin Python dependencies and retain npm lockfile (currently ignored). Record cv2 version/build, model hashes, calibration and dataset split. setup.sh assumes backend directly under ~/roadreport, but this repo has an extra roadreport directory. Its GET /api/jobs health command has no matching GET endpoint. OS-packaged Node compatibility must be validated.
18. **Resource handling.** All retained frame/color arrays are held in RAM; long uploads are only size-limited. FastAPI background tasks are not a durable job queue. For a small demo, bound jobs and recover interrupted states; use a worker queue if capacity permits.
19. **Player coordinates.** Backend detections are resized to width 960; player assumes 1280×720. Store coordinate space and frame-specific tracks; respect letterboxing and actual video duration.
20. **Frontend lifecycle.** Cancel polling/replay timers, revoke object URLs, clean up Leaflet instances and retain review failures rather than showing success. Consolidate older unused upload/results components. Add keyboard control and focus management to comparison/modal interactions.

## Proposed ML integration: separate defect and contact

The key distinction is physical: a pothole can exist without either tyre crossing it. Conversely, a shock can be caused by a speed bump, joint, braking or a loose mount. The system should infer defect existence, contact likelihood, and impact association separately.

### Data flow

1. Run visual candidate detection on sampled sharp frames. Keep the classical detector as a baseline; add a lightweight trained visual detector/classifier after obtaining suitable data and weight rights.
2. Track each candidate across native-timestamp frames and keep its mask or box trajectory.
3. Estimate two tyre corridors in the road plane using camera calibration, mount pose, vehicle track width and tyre width. A bounding box around the whole vehicle path is insufficient because it includes the gap between tyres. Curves require a changing path estimate. If calibration is unreliable, output unknown contact.
4. Estimate a contact-time interval from geometry and speed/trajectory. The frame where a pothole is first visible is earlier than wheel contact; rear axle timing adds another possible delay. Avoid one arbitrary delay for every car.
5. Build a high-rate camera-motion stream before stabilization/blur filtering: track static scene points, robustly fit background motion, exclude nearby moving objects and measure fast vertical/rotational excursions relative to the slower travel trend. OpenCV optical flow provides the tracking primitive [S6]. Low texture, rolling shutter, stabilization and mount motion must reduce confidence.
6. Prefer an optional synchronized phone IMU stream (accelerometer and gyro) mounted rigidly in the vehicle. Correct orientation/gravity and preserve units, timestamps, missing-sample masks and time-offset uncertainty. Aim for approximately 100 Hz if the device supports it; measure actual delivered frequency. Video-only motion remains a proxy and is not independent confirmation of physical acceleration.
7. Associate impacts only with candidates whose tyre overlap and contact-time intervals are compatible. A jolt must not validate every pothole visible in that frame.
8. Run the fusion model, calibrate its output on held-out validation data, then return a decision plus evidence provenance and quality.

### Model choice

**First implementation: small trained gradient-boosted trees over event features**, with a logistic baseline. Train separate heads/models for visual defect probability and impact/contact association. Features: visual model score; track duration; changing mask/area; left/right overlap; contact-time uncertainty; background-motion pulse energy, duration and rotation; optional vertical acceleration/jerk, gyro and frequency features; speed; mount quality; available-modality flags. Use observed time intervals, not frame count as a substitute for time.

Learn interactions between visual evidence, contact likelihood and shock. Enforce a decision-policy guard: absent shock cannot by itself reject a candidate with low/unknown tyre overlap or poor sensor quality. Missing IMU is not zero acceleration.

**Later, only with enough diverse paired recordings:** small 1D temporal convolutional network for the IMU/motion windows and a visual embedding branch, fused with geometry. A large transformer is not justified by the current dataset or deadline. Proposed architectures are hypotheses to benchmark, not claims of achieved accuracy.

### Case matrix and labels

| Event | Ground truth | Expected system behavior |
|---|---|---|
| Pothole, left/right tyre contact, aligned jolt | defect=yes, contact=yes | Impact-supported pothole, if evidence quality passes. |
| Pothole between tyres | defect=yes, contact=no | Retain visual detection; explain that a jolt was not expected. |
| Pothole safely passed to one side | defect=yes, contact=no | Retain visual detection; record avoidance only if path is observable. |
| Tyre contact but weak response at low speed/soft suspension | defect=yes, contact=yes | No-shock is weak negative evidence; do not automatically reject. |
| Pothole visible, unrelated speed bump causes jolt | defect=yes, association=no | Keep visual defect; do not falsely claim impact support. |
| Shadow/stain plus jolt from another source | defect=no, association=no | Reject or request review; shock alone is insufficient. |
| Speed bump, manhole, expansion joint or rough road | pothole=no, shock may=yes | Classify other road event or uncertain. |
| Camera handled, loose mount, braking or turning | pothole may=no | Detect poor motion reliability; downweight proxy. |
| Water-filled/occluded candidate | existence may=unknown | Request inspection; never assume darkness proves depth. |
| Stabilized video, missing timestamps/IMU, blur/night | quality impaired | Fall back to visual evidence and expose uncertainty. |

Annotate actual pothole identity independently of shock. Store contact as yes/no/unknown, side, expected/actual event times, geometry quality, road/location/drive IDs, vehicle/mount IDs, source rights and annotator confidence. Never label every shock a pothole or every no-shock event a negative.

### Training and validation plan

RDD2022 contains road images including India and pothole/crack categories [S5]; it can support visual training, but image boxes alone cannot train synchronized shock association. The current repo has no fusion dataset, sensor recordings, training script or trained fusion weights.

Start with team-owned synchronized drives or properly licensed paired data. A planning target of 300–800 independently labeled events across positives, no-contact positives and difficult negatives is a pilot objective, not a guarantee of generalization. If the deadline allows only a smaller set, report it as a limited pilot. Do not deliberately strike hazardous potholes for collection; use ordinary safe journeys and existing recordings.

Split by physical road segment/pothole and drive, not neighboring frames. Hold out a vehicle/mount if possible. Keep every augmentation and repeated pass of a pothole within one split. Fit normalization and calibration without touching the test set. Synthetic compositing may test plumbing and augment training, but cannot substitute for real held-out shock evidence.

Compare classical vision, learned vision, naive vision-plus-shock, contact-aware video fusion, and fusion with IMU on the same real events. Report precision/recall, no-contact pothole recall, false confirmations from unrelated jolts, contact association accuracy, uncertainty/review rate, calibration, coverage, runtime, and subgroup sample sizes. Count unclear samples separately; do not quietly discard hard conditions. Use grouped confidence intervals where feasible. Release the manifest, seeds, exact weights/configuration and limitations.

## Feature proposal and priority

| Feature | Reviewer experience | Scope |
|---|---|---|
| Evidence replay | Video, tyre corridors, predicted contact interval, motion/IMU plot and decision trace scrub together. | Core |
| No-contact proof case | A visible pothole passes between tyres without a jolt and is correctly retained. | Core |
| Adaptive inspection agent | Disagreement triggers native-resolution neighboring-frame extraction, alternate-view comparison, sensor-quality checks, then human review if unresolved. | Core |
| Independent-pass corroboration | Two drives support one road defect while correlated frames from one drive count as one observation. GPS uncertainty and appearance matching prevent blind merging. | Stretch |
| Repair verification | Before/after imagery aligned to the same location; clear visibility and repeated passes support closure. Obscured views remain inconclusive. | Stretch |
| Budget planner | An explicit budget ranks reviewed repairs by transparent risk/cost assumptions and displays uncertainty. It does not invent depth or government prices. | Later |
| Counterfactual explanation | Offline diagram explains why a particular wheel path missed the defect. Labels simulated alternatives clearly; no live steering recommendations. | Optional UX |
| COOL comparison | Same input/workload on baseline versus COOL/Graviton with logs, output checks and measurements. | Only after core works |

The adaptive agent needs bounded tool calls and a stopping rule. For example: insufficient visual evidence → fetch nearby original frames → re-run detection → check track/contact alignment → compare motion reliability → corroborate or ask review. Record actual tool inputs/outputs and reasons. Do not manufacture confidence gains; contradictory evidence should sometimes reduce confidence.

For the competition, focus on the overall and Agentic Vision routes. Merely adding a chatbot or AWS service logos does not strengthen the evidence. Keep the first deployment simple: existing API plus a bounded processing worker on AWS, private S3 artifacts and operational logs. Exact instance choice/cost should follow a measured workload. Avoid promising a large deep-model training run or unnecessary multi-service architecture.

## Proposed implementation map, pending approval

| Area | Planned change |
|---|---|
| pipeline.py / agent.py / db.py | Explicit interfaces, stable evidence IDs, valid trace callbacks, review provenance, correct status summaries. |
| main.py / storage.py / report.py | Validated uploads and telemetry, safe path handling, report image materialization and reviewed-only export. |
| cv/frames.py | Separate sharp-frame visual sampling from high-rate motion extraction; timestamps, quality and coordinate metadata. |
| New cv/motion.py | Robust background feature tracking and motion pulse extraction. |
| New cv/contact.py | Calibration, projected tyre corridors, overlap and contact-time intervals. |
| New ml/features.py / train.py / infer.py | Dataset validation, grouped split, baseline/fusion training, calibration, model versioning and missing-modality handling. |
| New telemetry ingestion | Timestamped accelerometer/gyro/GPS sidecar schema and offset quality checks. |
| App.jsx / player / map / inspector | Explicit demo mode, real empty/error states, accurate overlays, synchronized evidence chart and confidence provenance. |
| eval/ and tests/ | Real integration contract tests, held-out event evaluation and no-contact/shock-negative cases. |
| infra / dependencies / README | Repeatable AWS runtime, dependency locks, honest benchmark results, asset/model/data attribution and deployment instructions. |

Implementation can build the data collection and training pipeline before real recordings arrive. It cannot honestly deliver a validated trained fusion model without representative paired data. Until then, mark any rule-based fusion as a baseline and any synthetic experiment as synthetic.

## Working schedule after approval

- October 20: repair integration, truthfulness, storage containment and reproducibility; establish one unmocked upload-to-report path.
- October 21: collect/label safe real recordings; add timestamp, calibration and telemetry contracts.
- October 22: implement motion/contact features and a small baseline/fusion training run if paired labels are ready.
- October 23: evidence replay and adaptive inspection; measure no-contact and shock-negative cases.
- October 24: AWS deployment, held-out comparisons and failures; stretch work only if the core is stable.
- October 25: technical report, architecture, demo recording and reproducibility checks.
- October 26: freeze, verify judge access and submit early.

This schedule is conditional on data/access being available, not a delivery guarantee. If data is late, prioritize a robust measured visual system, motion baseline and candid limitations over unsupported trained-model claims.

Suggested demo: explain the problem; process a real clip; show an impact-supported event; show a no-contact pothole; show a speed-bump/shadow failure avoided; inspect the agent's next action; show real evaluation and AWS execution; finish with human-reviewed evidence export. Keep within the official video length.

## Sources

- S1: https://opencv26.devpost.com/ — deadline, requirements, paths and schedule, accessed October 20, 2026.
- S2: https://opencv26.devpost.com/rules — scoring, prizes, rights and conduct, accessed October 20, 2026.
- S3: https://opencv.org/opencv-ai-competition-2026/ — official competition overview; contains contradictions identified above.
- S4: https://opencv26.devpost.com/updates — compute-grant notification update.
- S5: https://arxiv.org/abs/2209.08538 — RDD2022 dataset paper. Verify the exact dataset distribution license before training.
- S6: https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html — optical-flow principles and tracking primitives. Validate APIs against the pinned OpenCV 5 build.
- S7: https://docs.opencv.org/5.0/ — OpenCV 5 documentation.

Repository-specific statements derive from the inspected commit and local checks, not from its README claims alone.

## Complete tracked-file inventory

Each row below accounts for a tracked file. Image inspection establishes visual content, not ownership, authenticity, ground truth or measured performance.

| File | Contents and assessment |
|---|---|
| `.gitignore` | Ignores environments, videos, data and package-lock.json; retaining a dependency lock is required for repeatability. |
| `README.md` | 577-line architecture/setup/API/method guide, embedded MIT license; unsupported benchmark table, OpenCV 4 reference and depth claims need correction. |
| `create_50fps_smooth_dashcam.py` | Composites defect patches into absent project_video.mp4; interpolates 25→50 fps, writes demo video and manually scored annotations. Does not simulate validated vehicle dynamics. |
| `extract_real_assets.py` | Extracts frames at hardcoded event times, adds fixed boxes/scores and enhanced crops; its schedule differs from the current demo. |
| `fig1.png` | Reference illustration/road segmentation figure; origin and usage rights not established in repository. 1155×467 pixels. |
| `fig2.png` | Reference illustration/road segmentation figure; origin and usage rights not established in repository. 649×415 pixels. |
| `fig3.png` | Reference illustration/road segmentation figure; origin and usage rights not established in repository. 870×549 pixels. |
| `fig4.png` | Reference illustration/road segmentation figure; origin and usage rights not established in repository. 864×366 pixels. |
| `fig5.png` | Reference illustration/road segmentation figure; origin and usage rights not established in repository. 628×473 pixels. |
| `generate_perfect_assets.py` | Creates annotated context and zoom images from handpicked source photos/configuration; labels are fixed strings. Config differs from current samples 3 and 5. |
| `inspect_10s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_12s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_15s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_18s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_20s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_24s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_2s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_4s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_6s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `inspect_8s.jpg` | Inspection frame of heavily damaged rural road with puddles; no synchronized sensor/contact labels. 1280×720 pixels. |
| `pothole_mask1.png` | Sparse binary reference mask; used or available for demo compositing, not contact annotation. 1024×640 pixels. |
| `pothole_mask2.png` | Sparse binary reference mask; used or available for demo compositing, not contact annotation. 1024×640 pixels. |
| `pothole_mask3.png` | Sparse binary reference mask; used or available for demo compositing, not contact annotation. 1024×640 pixels. |
| `pothole_mask4.png` | Sparse binary reference mask; used or available for demo compositing, not contact annotation. 1024×640 pixels. |
| `pothole_mask5.png` | Sparse binary reference mask; used or available for demo compositing, not contact annotation. 1024×640 pixels. |
| `pothole_raw1.jpg` | Reference road photo containing candidate pavement damage/patches; source rights and semantics need verification. 1024×640 pixels. |
| `pothole_raw2.jpg` | Reference road photo containing candidate pavement damage/patches; source rights and semantics need verification. 1024×640 pixels. |
| `pothole_raw3.jpg` | Reference road photo containing candidate pavement damage/patches; source rights and semantics need verification. 1024×640 pixels. |
| `pothole_raw4.jpg` | Reference road photo containing candidate pavement damage/patches; source rights and semantics need verification. 1024×640 pixels. |
| `pothole_raw5.jpg` | Reference road photo containing candidate pavement damage/patches; source rights and semantics need verification. 1024×640 pixels. |
| `preview_detect.mp4.jpg` | Preview from another source video, some with stock/news overlays; not an evaluation record. 640×360 pixels. |
| `preview_final.mp4.jpg` | Preview from another source video, some with stock/news overlays; not an evaluation record. 1280×720 pixels. |
| `preview_kc_road.avi.jpg` | Preview from another source video, some with stock/news overlays; not an evaluation record. 1280×720 pixels. |
| `preview_kc_sample.mp4.jpg` | Preview from another source video, some with stock/news overlays; not an evaluation record. 1280×720 pixels. |
| `preview_video_1.mp4.jpg` | Preview from another source video, some with stock/news overlays; not an evaluation record. 720×1280 pixels. |
| `project_f0.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `raw_road.jpg` | Road surface/shadow reference image; must independently label actual defect presence. 1024×630 pixels. |
| `road2_11s.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road2_1s.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road2_3s.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road2_5s.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road2_8s.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road2_pothole_full.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `road_shadow_cand1.jpg` | Road surface/shadow reference image; must independently label actual defect presence. 1024×640 pixels. |
| `road_shadow_cand2.jpg` | Road surface/shadow reference image; must independently label actual defect presence. 1024×640 pixels. |
| `roadreport/README.md` | Byte-identical copy of root README; same documentation issues. |
| `roadreport/backend/app/__init__.py` | Package docstring only. |
| `roadreport/backend/app/agent.py` | Handwritten confidence triage and stricter crop inspection; callback contracts, trace IDs, persistence and confidence provenance need repair. |
| `roadreport/backend/app/cv/__init__.py` | CV package docstring only. |
| `roadreport/backend/app/cv/crack.py` | CLAHE/bilateral/Canny, marking suppression, component skeletonization and geometry score; requires representative road testing. |
| `roadreport/backend/app/cv/frames.py` | 4 fps target, 960-pixel width, 240 retained-frame cap, fixed trapezoid/glare and blur masks. Need separate native-time motion stream. |
| `roadreport/backend/app/cv/pothole.py` | Classical dark contour detector with area/solidity/aspect/contrast gates and weighted heuristic confidence; no trained model. |
| `roadreport/backend/app/cv/severity.py` | Area/contrast or crack geometry weighted proxy; no calibrated depth or physical severity. |
| `roadreport/backend/app/cv/track.py` | Greedy same-type spatial/template tracker; expects nested detections. No gap tolerance or proper ego-motion model. |
| `roadreport/backend/app/db.py` | SQLite jobs/findings/agent_steps with parameterized values and WAL; lacks telemetry/calibration/model version and human-review history. |
| `roadreport/backend/app/main.py` | FastAPI upload/status/findings/review/report/local-file endpoints; report invocation, trace lookup, review status and access controls need repair. |
| `roadreport/backend/app/pipeline.py` | Orchestrates extraction/detection/tracking/agent/artifact export; tracker/agent interfaces broken; dynamic retries hide errors. |
| `roadreport/backend/app/report.py` | ReportLab report layout; requires image_dir, mislabels nonconfirmed statuses, does not itself filter rejected findings. Image/table width also merits render QA after wiring repair. |
| `roadreport/backend/app/storage.py` | S3/local transfer and presigning; local path joining lacks resolved-root containment. |
| `roadreport/backend/eval/__init__.py` | Evaluation package docstring only. |
| `roadreport/backend/eval/eval.py` | IoU matching and separate raw/agent evaluations; inconsistent frame/event universes invalidate direct claimed improvement comparisons. |
| `roadreport/backend/eval/labels_template.csv` | Three illustrative box annotations over frames 0 and 30; not a labeled evaluation corpus or synchronized sensor dataset. |
| `roadreport/backend/requirements.txt` | Ten unpinned Python dependencies; no training framework or test dependencies; install resolved OpenCV 5 during this audit. |
| `roadreport/backend/tests/test_backend_core.py` | Six passing tests; end-to-end detector/tracker/agent and report generator are mocked, missing real interface failures. |
| `roadreport/frontend/.gitignore` | Standard frontend artifacts/editor exclusions. |
| `roadreport/frontend/.oxlintrc.json` | React/Oxc lint rules; current run produces seven warnings. |
| `roadreport/frontend/README.md` | Uncustomized React/Vite template guide. |
| `roadreport/frontend/index.html` | RoadReport title and Google font loading; starter favicon remains. |
| `roadreport/frontend/package.json` | React/Vite/Axios/Leaflet and Oxlint scripts; version ranges, no tracked lockfile. |
| `roadreport/frontend/public/favicon.svg` | Vite-style starter vector icon; no analysis logic. |
| `roadreport/frontend/public/icons.svg` | Starter social/documentation SVG symbols; no analysis logic. |
| `roadreport/frontend/public/sample/crop_1.jpg` | Demo defect crop; hardcoded event asset, not automatically generated evidence from a new upload. 132×76 pixels. |
| `roadreport/frontend/public/sample/crop_2.jpg` | Demo defect crop; hardcoded event asset, not automatically generated evidence from a new upload. 66×147 pixels. |
| `roadreport/frontend/public/sample/crop_3.jpg` | Demo defect crop; hardcoded event asset, not automatically generated evidence from a new upload. 50×42 pixels. |
| `roadreport/frontend/public/sample/crop_4.jpg` | Demo defect crop; hardcoded event asset, not automatically generated evidence from a new upload. 112×76 pixels. |
| `roadreport/frontend/public/sample/crop_5.jpg` | Demo defect crop; hardcoded event asset, not automatically generated evidence from a new upload. 127×92 pixels. |
| `roadreport/frontend/public/sample/frame_1.jpg` | Composited sample-video keyframe; matches synthetic event demonstration. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_1_annotated.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_2.jpg` | Composited sample-video keyframe; matches synthetic event demonstration. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_2_annotated.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_3.jpg` | Composited sample-video keyframe; matches synthetic event demonstration. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_3_annotated.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_4.jpg` | Composited sample-video keyframe; matches synthetic event demonstration. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_4_annotated.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_5.jpg` | Composited sample-video keyframe; matches synthetic event demonstration. 1280×720 pixels. |
| `roadreport/frontend/public/sample/frame_5_annotated.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/road_context_1.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/road_context_2.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/road_context_3.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/road_context_4.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/road_context_5.jpg` | Composited demo road context with manually assigned detection/severity label. 1280×720 pixels. |
| `roadreport/frontend/public/sample/zoom_1.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 450×225 pixels. |
| `roadreport/frontend/public/sample/zoom_2.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 330×930 pixels. |
| `roadreport/frontend/public/sample/zoom_3.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 720×540 pixels. |
| `roadreport/frontend/public/sample/zoom_4.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 480×330 pixels. |
| `roadreport/frontend/public/sample/zoom_5.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 540×360 pixels. |
| `roadreport/frontend/public/sample/zoom_defect_1.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 450×225 pixels. |
| `roadreport/frontend/public/sample/zoom_defect_2.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 330×930 pixels. |
| `roadreport/frontend/public/sample/zoom_defect_3.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 720×540 pixels. |
| `roadreport/frontend/public/sample/zoom_defect_4.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 480×330 pixels. |
| `roadreport/frontend/public/sample/zoom_defect_5.jpg` | Enhanced demo crop with fixed overlay/text. Some source/configuration mismatches with active sample findings; not measured confidence. 540×360 pixels. |
| `roadreport/frontend/public/sample_dashcam.mp4` | H.264 1280×720, 50fps, 50.38s/2519 frames; inspected six timestamps. Composited synthetic-event demonstration, not paired impact ground truth. |
| `roadreport/frontend/src/App.css` | 1,408-line theme/layout/player/map/inspector stylesheet; duplicate detection selector and undefined --text token; responsive rules and reduced-motion support exist. |
| `roadreport/frontend/src/App.jsx` | Active 1,111-line app; hardcoded demo, silent fallback, heuristic costs/health, fabricated persistence and review success handling need correction. |
| `roadreport/frontend/src/api.js` | Axios upload/status/findings/review/report client; truthy checks omit legitimate zero coordinates. |
| `roadreport/frontend/src/assets/hero.png` | Starter decorative layered illustration; not road evidence. 343×361 pixels. |
| `roadreport/frontend/src/assets/react.svg` | React starter vector logo; no analysis logic. |
| `roadreport/frontend/src/assets/vite.svg` | Vite starter vector logo; no analysis logic. |
| `roadreport/frontend/src/components/FindingCard.jsx` | Older finding/timeline card used by ResultsView, not active App route. |
| `roadreport/frontend/src/components/ReinspectionVisualizer.jsx` | Road/crop zoom and split view; fabricates before confidence and assumes sample image paths for real UUIDs. |
| `roadreport/frontend/src/components/ResultsView.jsx` | Older API-driven results/poll/review/report component; not imported by active App; its empty-state behavior is preferable to sample substitution. |
| `roadreport/frontend/src/components/RouteMap.jsx` | Leaflet satellite/street map; route and defect coordinates are simulated with random jitter; cleanup absent. |
| `roadreport/frontend/src/components/SynchronizedVideoPlayer.jsx` | Playback/seek/timestamp markers; hardcoded telemetry, fixed box coordinate dimensions and static event window instead of actual per-frame tracks. |
| `roadreport/frontend/src/components/UploadView.jsx` | Older upload/metadata/drag-drop/validation component, not imported by active App. |
| `roadreport/frontend/src/index.css` | 13-line base reset/body stylesheet. |
| `roadreport/frontend/src/main.jsx` | React StrictMode bootstrap and CSS imports. |
| `roadreport/frontend/vite.config.js` | React plugin and /api development proxy to localhost:8000. |
| `roadreport/infra/nginx.conf` | HTTP static hosting/API reverse proxy, upload/timeouts and asset caching; HTTPS not configured here. |
| `roadreport/infra/roadreport.service` | Uvicorn systemd unit, hardcoded bucket/region, single process; no proof this has been deployed. |
| `roadreport/infra/setup.sh` | EC2-oriented Ubuntu bootstrap; source-directory assumption mismatches repository nesting; default OS Node version unvalidated; health URL has no GET handler. |
| `sample_p_12s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `sample_p_20s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `sample_p_28s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `sample_p_35s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `sample_p_45s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `sample_p_5s.jpg` | Highway source/reference frame for sample generation; no shock or tyre-contact labels. 1280×720 pixels. |
| `shadow_road.jpg` | Road surface/shadow reference image; must independently label actual defect presence. 216×138 pixels. |
| `test_car_f0.jpg` | Road/vehicle reference still; the two files have identical content. 768×432 pixels. |
| `test_frame.jpg` | Road/vehicle reference still; the two files have identical content. 768×432 pixels. |
| `test_mask_blend.jpg` | Compositing experiment with inserted/boxed defect on highway image; synthetic. 1280×720 pixels. |
| `test_pothole_blend.jpg` | Compositing experiment with inserted/boxed defect on highway image; synthetic. 1280×720 pixels. |
| `test_road2_frame.jpg` | Rural damaged-road reference, with stock watermark visible in several files; provenance/license needed. 1920×1080 pixels. |
| `test_road_context1.jpg` | Manually annotated source photo for inspection-panel experiment. 1024×640 pixels. |
| `test_zoom_defect1.jpg` | Enhanced crop/contour overlay experiment; not an independently verified physical measurement. 480×285 pixels. |
