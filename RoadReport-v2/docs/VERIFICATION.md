# Verification record

Executed 20 October 2026 in Linux x86-64, Python 3.12, Node 24.19.0, OpenCV runtime 5.0.0 (package 5.0.0.93), Vite 8.3.4.

| Check | Result | What it establishes |
|---|---|---|
| Backend regression suite | 15 passed; added empty-visual regression separately passed (16 tests total in delivered suite) | Functional contracts and cases below |
| Native-frame decoding | 40 native motion samples versus at most 8 visual samples on generated fixture | Shock stream is not restricted to visual sample rate |
| Real API pipeline | Generated video uploaded, decoded, detected, tracked, reinspected, persisted, reviewed, exported | No detector/tracker/agent/report mocks in integration test |
| No-contact policy | Between-tyres geometry retained without shock; missing calibration unknown | Intended no-contact behavior on known synthetic geometry |
| Timing and ambiguity | Two axle windows independently covered; unrelated pulse ignored; ambiguous pulse reviewed | Contact-time association guards |
| ML training/export | Two candidate algorithms trained on explicitly synthetic fixtures, selected/calibrated and exported; inference parity checked | Pipeline works, not real-world model quality |
| Dataset integrity | Synthetic flag required; drive leakage rejected | Basic training-data validation |
| Empty retained-frame path | Completed with zero findings and coverage warning | No fabricated sample fallback |
| Storage/auth/sidecars | Traversal, invalid timestamps/calibration, missing key and bad video rejected | Tested input/access controls, not a full penetration test |
| Event benchmark | Missed event and duplicate prediction correctly counted | Same-universe evaluation accounting |
| Calibration CLI | Synthetic measured-correspondence fixture fitted and emitted disabled calibration | Tool execution, not real-camera calibration |
| Frontend production build | Passed | Modules compile and assets bundle |
| Frontend lint | Passed with no warnings/errors | Configured static checks |
| PDF export | Empty and image-containing reports generated; image-containing report rendered and visually inspected | Adapter integration and readable layout |
| ZIP | Archive integrity and excluded-path checks performed during packaging | Deliverable completeness/cleanliness |

One upstream warning remains: Starlette's TestClient deprecates its httpx adapter in favor of httpx2. Tests currently pass with the pinned httpx version. This is not a reported application failure.

## Not verified

- Browser interaction/screenshot QA: Playwright browser installation failed because the downloaded browser archives were invalid. Build/lint are not substitutes for full UI testing.
- Real phone camera/DeviceMotion/GPS permissions, timestamps, sensor orientation and browser recording codecs. Validate on the team's actual phone before collecting training data.
- AWS account execution, S3 role/bucket policies, Docker image build, Caddy certificates and public judge access. Configuration and instructions are supplied, but these were not deployed here.
- Windows dependency execution, ARM/Graviton compatibility or COOL performance.
- Real held-out precision/recall, tyre contact truth, physical depth, repair cost or operational safety. No paired real training dataset was supplied.

## Known engineering limitations

The detector remains classical CV. Its fixed road ROI, heuristic thresholds, greedy tracking and bbox-based geometry can miss or misclassify defects. Tracks tolerate short gaps but do not perform full ego-motion compensation. Reinspection reuses the best box location in nearby native frames, so rapid motion can shift the crop. Optical flow does not fully exclude all moving objects. Camera stabilization may hide jolts; mount vibration may invent them. Calibration and constant straight-path speed assumptions cannot validate turns or uneven geometry.

There is one in-process analysis worker with a bounded queue; no durable retry system or multi-user accounts. Signed evidence links expire after one hour; reopen the job to renew them. External map tiles and optional web fonts require network access; analysis does not. A GPS trace is the vehicle route, not a surveyed pothole coordinate.

The existence head is deliberately visual-only to preserve no-contact positives. The association head can support attribution only with appropriate data, timing, geometry and held-out validation. All synthetic fixture scores are software checks and must not be presented as competition accuracy results.
