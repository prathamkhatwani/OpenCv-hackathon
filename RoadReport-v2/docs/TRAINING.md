# Training a contact-aware event classifier

## Status and experimental design

The delivered code is trainable; it is not a field-validated trained model. Two heads predict `pothole_exists` and `impact_associated`. Existence uses initial visual score, reinspection score, track duration and observation count only. Association additionally uses contact overlap/availability, camera pulse/quality, IMU peak/availability and speed. When no reinspection is requested, its feature equals the initial visual score; the UI explicitly says Not requested. This explicitly prevents missing shock from becoming an existence feature. It cannot recover potholes that the classical candidate detector never proposes.

Train logistic regression and gradient boosting; select each head by validation log loss; fit sigmoid calibration on validation; evaluate once on the untouched test set. JSON export is checked against sklearn inference before saving. Never load a pickle or an untrusted model file. A synthetic artifact is labeled experimental and excluded from impact-confirmation gates.

Split by **both complete drive and physical site**, with held-out vehicles, mounts and road conditions wherever possible. The validator rejects drives/sites appearing in multiple splits, duplicate IDs, nonfinite/missing features, inconsistent contact/impact labels, or fewer than 10 labeled samples per class-containing split/head. These are minimum software checks, not a sufficient scientific sample size. A useful pilot needs hundreds of diverse events, enough examples per subgroup, and a separately collected test set. Reused validation for selection and calibration is a limitation; larger datasets should use nested splits or a separate calibration partition.

## Cases to collect and independently annotate

| Case | Existence | Contact | Impact association |
|---|---:|---|---|
| Pothole crossed by tyre, attributable measured pulse | 1 | yes | 1 |
| Pothole between tyres or outside both strips | 1 | no | 0 |
| Pothole crossed, no clear pulse due to suspension | 1 | yes | 0 or unknown (`null`) |
| Pothole avoided by driver | 1 | no/unknown | 0/unknown; fixed geometry invalid on turn |
| Shadow, patch, puddle or manhole | 0 | no/unknown | 0 |
| Speed bump, joint, braking or loose mount | 0 | as observed | 0 for pothole association |
| Real pothole with unrelated nearby shock | 1 | as observed | 0/unknown |
| Several defects in one contact window | independently label each | as observed | unknown unless separable |
| Blurred/occluded/night/wet/stabilized footage | independently assess | unknown if unresolved | unknown |

Use independent road inspection, controlled safe stationary imagery, or other trustworthy evidence for existence. Do not use the algorithm's own prediction as truth. Do not define contact truth solely by the IMU pulse you are evaluating. Include no-contact positives in every split and report their recall separately. Do not intentionally drive into damage to collect positives.

## Sidecars

`examples/telemetry.synthetic.json` demonstrates the schema, not real measurements. `t` is seconds; aligned video time is `t + time_offset_s`; SI units; `az` is gravity-removed acceleration along vertical in m/s², gyro is rad/s, speed m/s. GPS lat/lon must be paired. Sensor times must increase strictly. An uncertain sync greater than .25 s disables IMU use; usable axle windows need at least four samples with at most .15 s gaps and boundary coverage. Starting two independent recordings simultaneously does not guarantee synchronization.

`examples/calibration.template.json` is intentionally disabled (`straight_path_valid=false`). It is not your vehicle's calibration. Measure track width, tyre width and wheelbase; measure at least six well-spread ground-point correspondences relative to the front axle using the same rectified camera image. From the backend directory:

```bash
python -m tools.calibrate --points measured-points.json --output calibration.json
```

Input point schema is shown in `examples/points.synthetic.json`. Verify independent held-out points, timing and mounting stability before manually enabling `straight_path_valid`. Fit residual alone does not validate a calibration. Do not reuse synthetic homography values on real roads.

## Build a labeled training dataset

1. Run analysis and export Evidence JSON for each drive. Retain the source recording, provenance, calibration, source kind and software version.
2. Annotate candidate events with the schema in `examples/labels.template.json`. Use finding UUID as event_id. Assign true drive/site IDs and frozen splits. Set `source_kind=real` only for real recordings.
3. Join labels to observed features:

```bash
python -m tools.prepare_events --exports drive1.json drive2.json --labels candidate-labels.json --output events.json
python -m app.ml.train --data events.json --output models/fusion.json
```

The trainer refuses synthetic records by default. `--allow-synthetic` is for pipeline tests only. Model output includes held-out metrics, no-contact recall, algorithm choice, dataset hash and limitations. Keep a separate model card naming collection dates, cameras/vehicles, dataset rights, group split counts and errors. No target accuracy is assumed.

4. Configure `ROADREPORT_MODEL` to the absolute JSON path and restart the backend. PowerShell: `$env:ROADREPORT_MODEL="C:\path\fusion.json"`. Without this variable the app uses the explicit baseline. Training is not performed at request time.

## End-to-end evaluation

Candidate classification metrics omit undetected defects. Independently label every real pothole in the full videos, including blurred/skipped/missed events, and match predictions to those events once using location/time evidence. Unmatched predictions get `event_id:null`; duplicate predictions of the same event count false positive after the first. Use the same frozen event universe for visual-only, shock-only and fusion decisions.

```bash
python -m eval.eval --labels all-ground-truth.json --predictions matched-predictions.json --output metrics.json
```

See `examples/benchmark.*.json`. The evaluator counts missed events, false positives, unresolved-review count and no-contact recall. Freeze thresholds before test evaluation. Report review rate, failures, latency, coverage, and held-out vehicles/sites alongside precision/recall/F1. A tiny test set or synthetic test passing is not evidence of real accuracy. Turn, mount and weather subgroups should be evaluated separately; no automated confidence intervals or vehicle-domain validation is included yet.
