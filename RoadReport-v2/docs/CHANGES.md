# Changes from original snapshot

The starting snapshot and 134-file original inventory are in ORIGINAL-AUDIT.md. This is a new clean working copy, not a pushed GitHub change.

- Replaced pipeline/agent integration with explicit contracts; fixed nested tracker inputs, best-frame lookup, maximum linking gap, finding IDs and trace linkage.
- Added native motion, calibrated tyre geometry, timed IMU evidence, strict sidecars, source labels, training/export/inference and decision policy.
- Added calibration/label preparation tools and event-level evaluation including misses and duplicates.
- Rebuilt dashboard around actual evidence; removed invented telemetry/confidence and automatic sample substitution. Added recording, history, real review persistence, synchronized charts and clear missing-data states.
- Replaced report adapter and PDF content; now filters unsupported findings, downloads evidence correctly and embeds licensed DejaVu fonts for stable rendering.
- Added storage containment, keyed API, signed local evidence, limits, queue bounds and restart recovery.
- Replaced stale deployment paths with root Docker/Compose and HTTPS instructions; pinned dependencies and retained npm lockfile.
- Replaced mocked integration tests with real-code video pipeline tests and focused contact/training/security regressions.
- Removed obsolete unused frontend components, default framework icons, scratch asset scripts and unverified/composited demo media from this ZIP. Original materials remain in the original repository; the audit documents them.
- Included original audit, architecture, training/calibration, verification, deployment and Windows setup instructions.

Real-data training, AWS execution and optional fleet/repair/budget features are not represented as completed work. No award or accuracy outcome is promised.
