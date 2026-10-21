# RoadReport v2 — evidence before action

A road-inspection prototype that combines visual candidates, estimated tyre paths and time-aligned camera/IMU motion. **A pothole remains a pothole when the wheels miss it.**

Built from `prathamkhatwani/opencv`, main commit `1826b47ce071d0df907de289084818a39e2d5d41`. The original repository is unchanged. This package replaces the broken integration and demonstration-only results with an explicit evidence workflow.

## Try the included demo

Double-click **START-DEMO.bat** on Windows, or run `npm ci` then `npm run demo` in `roadreport/frontend`. It opens the explicit 18-second synthetic simulation at http://127.0.0.1:5173/?demo=1 . No backend is required. See [DEMO.md](DEMO.md) for the three cases, controls and asset provenance.

## What is implemented

- OpenCV 5 classical pothole/crack detection, temporal tracks, stable finding IDs and per-finding action traces.
- Native-frame optical-flow motion analysis, independent of the 4 fps sharp-frame visual stream.
- Calibrated straight-road tyre corridors, no-contact classification, predicted front/rear contact windows, and ambiguity handling when several candidates share an impact.
- Optional time-aligned, gravity-removed vertical IMU evidence. Camera movement is explicitly a proxy, not physical acceleration or proof of wheel contact.
- Two-head trainable event classifier: visual existence and impact association. Logistic regression versus gradient boosting selection, calibration, grouped splits, JSON model export and inference. Existence uses visual features only so absent shock does not suppress a missed-by-tyres pothole.
- Adaptive agent action: uncertain observations trigger inspection of three native-resolution neighboring frames; unresolved cases go to human review.
- Synchronized video, observed boxes, calibrated corridors, motion plots, actual GPS, source labels, evidence crops, saved reviews, inspection history, PDF and JSON exports.
- Browser camera/sensor recorder; strict sidecar validation; protected evidence URLs; bounded queue and upload sizes; restart recovery; local or private S3 storage.

**Model status:** RDD2022 India is integrated, and a real-data-trained HOG/logistic visual crop classifier is bundled. Its probability appears in finding details, traces and reports as a display-only pilot. It does not replace the classical proposal detector. The separate shock/event fusion model remains `baseline_untrained`: RDD2022 has no synchronized IMU, tyre-contact or impact labels. See [dataset integration and reproduction](docs/RDD2022.md) and the [paired-event training guide](docs/TRAINING.md).

**Not implemented:** curved-path reconstruction, dense segmentation/depth, learned visual proposal detector, fleet consensus, verified repair tracking, cost optimization, or validated Graviton/COOL benchmarks. The original audit lists these as possible follow-up work; they are not hidden behind placeholder UI.

## Start on Windows (PowerShell)

Install Python **3.12** and Node **24**. Extract the ZIP, open its `RoadReport-v2` folder in VS Code, then open a terminal.

Backend terminal:

```powershell
cd roadreport/backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend terminal, starting from the extracted root:

```powershell
cd roadreport/frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. The backend must remain running. Upload a 1–60 second video under 100 MB. Telemetry/calibration are optional; missing values remain unknown. `http://127.0.0.1:8000/api/health` should report OpenCV 5. Do not run multiple backend workers.

Linux/macOS: use `python3.12 -m venv .venv`, then `.venv/bin/python` for backend commands. If a platform has no wheel for a pinned dependency, use Linux x86-64/container; that is the tested runtime. No Windows runtime execution was performed during packaging.

## Verify locally

From `roadreport/backend`, using the environment above:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full lock includes test/training dependencies. From `roadreport/frontend`, run `npm run build` and `npm run lint`. See [verification record](docs/VERIFICATION.md) for what was actually exercised.

## Record paired evidence

Use the Capture tab on a phone through **HTTPS** (or localhost on the recording device). Grant camera/motion permissions, fix the phone rigidly, and record ordinary journeys. Never deliberately strike road defects. The recorder stops after 30 seconds and downloads video and sensor JSON separately. Upload both under Inspect. GPS appears only if permission and measurements exist.

Browser sensor/video alignment is approximate. Check the offset and uncertainty; stabilization, mount flex, vehicle suspension, phone orientation and browser sampling affect the signal. No model can establish reliable contact without valid measurements. If a browser recording lacks readable duration metadata, remux it:

```bash
ffmpeg -i drive.webm -c:v libx264 -pix_fmt yuv420p -an drive.mp4
```

Verify that the first-frame time has not shifted relative to telemetry after conversion. Do not silently adjust an offset to manufacture a match.

## Documentation

- [Architecture and decision behavior](docs/ARCHITECTURE.md)
- [Training, labeling, calibration and evaluation](docs/TRAINING.md)
- [AWS deployment and submission checklist](docs/DEPLOYMENT.md)
- [Verification and known limits](docs/VERIFICATION.md)
- [Original file-by-file audit and proposal](docs/ORIGINAL-AUDIT.md)

Original composite demonstration footage, stock/watermarked scratch images and asset-generation scripts are omitted from this clean release because their provenance was not established. They remain in the original GitHub repository. No private credentials, user footage, database, raw RDD2022 images or node_modules are included. The portable RDD2022 visual pilot weights and measured crop metrics are included. Dependencies retain their upstream licenses; review dataset/media rights and the project's own licensing before public submission.
