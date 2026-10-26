# RoadReport — supervised detection and persistent tracking

This revision fixes the real upload pipeline, repeated findings and video overlays.
See `docs/DETECTION-FIXES.md` for verification and limitations. Older architecture
and training documents describe additional experimental modules; they are not a
statement that every module is connected to this upload pipeline.

## Run on Windows

Install Python 3.12 and Node.js 24. Extract the ZIP, open this directory in VS Code.

Backend terminal:

```powershell
cd roadreport/backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:USE_LOCAL_STORAGE="true"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend terminal:

```powershell
cd roadreport/frontend
npm ci
npm run dev
```

Open http://localhost:5173, choose **Inspect**, upload a short road recording and
wait for analysis to finish. Play the result video: every visible tracked finding
has a box and stable ID, including findings you have not selected. Green boxes
are supported findings; amber boxes require review. The finding list has one
entry per track. Re-analyze older uploads to generate the new track metadata.

Linux/macOS: use `python3 -m venv .venv` and `.venv/bin/python` instead.

The supplied ONNX weights are already trained on RDD2022. Inference runs through
OpenCV DNN on CPU; inference does not require PyTorch or a GPU. Missing/corrupt
weights fail the job rather than falling back to dark-patch detection.

The **Demo** tab remains a clearly labeled scripted synthetic illustration. Use
**Inspect** for actual supervised detection. This revision does not turn the
scripted demo into a real-world accuracy benchmark.

## What changed

- Actual uploads now use the bundled supervised YOLO model, once per sampled frame.
- Global one-to-one motion matching retains identities through stationary frames,
  upward jitter and short missed-detection gaps (up to one second).
- Repeated observations produce one persisted finding; nearby separate objects
  can remain separate tracks.
- Confirmation requires at least three strong model observations. Interpolating
  or sharpening a crop cannot artificially raise confidence.
- The backend saves normalized time-stamped track boxes. The player draws all
  active findings and interpolates short observed intervals (at most 0.5 seconds).
- Model inference is locked across concurrent jobs. Same-class NMS suppresses
  duplicate proposals without suppressing a different damage class.

## Further supervised training

Use a separate training environment with `requirements-training.txt`. Prepare
labeled RDD2022 data using the existing downloader, or provide the prepared
archive explicitly:

```bash
python -m pip install -r requirements-training.txt
python -m tools.download_pothole_dataset --output data/pothole_dataset
python -m tools.train_yolo --data data/pothole_dataset/data.yaml --epochs 100 --device auto
```

The script selects CUDA when available, uses a fixed seed, validates the four
class names, evaluates validation/test splits separately, and exports to
`runs/train/candidate_export`. It does not overwrite the shipped model. Review
held-out results and your own labeled road clips before replacing the matching
ONNX and JSON pair in `app/models/`. Include negative examples such as shadows,
patches, drains and manholes; split your own clips by drive/site, not adjacent
frames. Add their genuine human annotations before training.

No new detector training run was performed for this ZIP. The supplied weights
are unchanged; the main correction is that the real pipeline finally uses them.
There is no measured new precision/recall result for the user's videos.

## Verification

Backend: `python -m pytest -q` (install `requirements-dev.txt` first).
Frontend: `npm run build`, `npm run lint`, `node --test tests/*.test.mjs`.

Tracking is local to a continuous observation. A pothole that fully leaves view
and is encountered again after a long gap can receive a new ID. Reliable deduplication
across revisits needs calibrated location or stronger visual re-identification.
