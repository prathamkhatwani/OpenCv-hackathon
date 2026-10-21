"""Comprehensive unit and integration tests for backend core components."""

import io
import json
import os
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

# Configure environment for testing
test_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["ROADREPORT_DB"] = test_db_file
os.environ["USE_LOCAL_STORAGE"] = "true"

from app.db import (
    add_agent_step,
    create_finding,
    create_job,
    get_agent_steps,
    get_db,
    get_finding,
    get_findings,
    get_job,
    init_db,
    update_finding,
    update_job,
)
from app.main import app
from app.pipeline import run_pipeline
from app.storage import download_file, presign_get, upload_bytes, upload_file


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    init_db()
    yield
    with get_db() as conn:
        conn.execute("DELETE FROM agent_steps;")
        conn.execute("DELETE FROM findings;")
        conn.execute("DELETE FROM jobs;")


def test_db_crud_operations():
    # 1. Job CRUD
    job = create_job(
        video_key="jobs/job1/video.mp4",
        location_text="123 Main St",
        lat=12.9716,
        lon=77.5946,
    )
    job_id = job["id"]
    assert job["status"] == "uploaded"
    assert job["location_text"] == "123 Main St"
    assert job["lat"] == 12.9716

    # Update job
    updated_job = update_job(
        job_id,
        status="processing",
        frames_total=100,
        frames_done=50,
    )
    assert updated_job["status"] == "processing"
    assert updated_job["frames_total"] == 100
    assert updated_job["frames_done"] == 50

    # 2. Finding CRUD
    finding = create_finding(
        job_id=job_id,
        type="pothole",
        severity_label="high",
        severity_score=0.88,
        confidence=0.95,
        frame_index=15,
        timestamp_s=3.75,
        bbox_json=json.dumps([100, 200, 50, 60]),
    )
    finding_id = finding["id"]
    assert finding["type"] == "pothole"
    assert finding["status"] == "confirmed"

    # Update finding
    updated_finding = update_finding(
        finding_id,
        status="rejected",
        image_key=f"jobs/{job_id}/findings/finding_{finding_id}_annotated.jpg",
    )
    assert updated_finding["status"] == "rejected"
    assert "annotated.jpg" in updated_finding["image_key"]

    # Retrieve findings list
    findings = get_findings(job_id)
    assert len(findings) == 1
    assert findings[0]["id"] == finding_id

    # 3. Agent steps CRUD
    step1 = add_agent_step(
        job_id=job_id,
        finding_ref=finding_id,
        step_no=1,
        action="shadow_verification",
        reason="Intensity gradient indicates real depth rather than surface stain",
        confidence_before=0.80,
        confidence_after=0.95,
    )
    assert step1["id"] is not None
    assert step1["step_no"] == 1

    steps = get_agent_steps(job_id, finding_ref=finding_id)
    assert len(steps) == 1
    assert steps[0]["action"] == "shadow_verification"


def test_storage_helpers():
    temp_dir = tempfile.mkdtemp()
    try:
        sample_path = os.path.join(temp_dir, "test.txt")
        Path(sample_path).write_text("road damage report payload")

        # Test upload_file
        key = upload_file(sample_path, "reports/test.txt")
        assert key == "reports/test.txt"

        # Test presign_get
        url = presign_get("reports/test.txt")
        assert url.startswith("file://")

        # Test download_file
        download_target = os.path.join(temp_dir, "downloaded.txt")
        download_file("reports/test.txt", download_target)
        assert Path(download_target).read_text() == "road damage report payload"

        # Test upload_bytes
        byte_key = upload_bytes(b"binary payload", "bytes/sample.bin", "application/octet-stream")
        assert byte_key == "bytes/sample.bin"
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_api_jobs_upload_and_validation():
    client = TestClient(app)

    # 1. Invalid extension test
    fake_txt = io.BytesIO(b"fake text content")
    resp = client.post(
        "/api/jobs",
        files={"video": ("sample.txt", fake_txt, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Invalid file extension" in resp.json()["detail"]

    # 2. Valid video upload
    fake_video = io.BytesIO(b"fake mp4 video bytes")
    resp = client.post(
        "/api/jobs",
        files={"video": ("test_road.mp4", fake_video, "video/mp4")},
        data={"location_text": "MG Road", "lat": "12.97", "lon": "77.59"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "uploaded"
    job_id = data["job_id"]

    # 3. GET /api/jobs/{job_id}
    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["id"] == job_id
    assert job_data["location_text"] == "MG Road"
    assert job_data["status"] in ("uploaded", "processing", "done", "failed")


def test_api_findings_and_review():
    client = TestClient(app)

    # Create job and finding directly in DB
    job = create_job(video_key="jobs/j1/video.mp4")
    f = create_finding(
        job_id=job["id"],
        type="crack",
        severity_label="medium",
        severity_score=0.6,
        confidence=0.85,
        status="needs_review",
        bbox_json=json.dumps({"x": 10, "y": 20, "w": 30, "h": 40}),
        image_key=f"jobs/{job['id']}/finding_annotated.jpg",
    )
    add_agent_step(
        job_id=job["id"],
        finding_ref=f["id"],
        step_no=1,
        action="crack_width_analysis",
        reason="Crack width > 5mm classified as medium severity",
    )

    # Test GET /api/jobs/{job_id}/findings
    findings_resp = client.get(f"/api/jobs/{job['id']}/findings")
    assert findings_resp.status_code == 200
    f_data = findings_resp.json()
    assert f_data["total"] == 1
    finding_item = f_data["findings"][0]
    assert finding_item["id"] == f["id"]
    assert finding_item["image_url"] is not None
    assert len(finding_item["agent_steps"]) == 1

    # Test POST /api/findings/{finding_id}/review - approve
    review_resp = client.post(
        f"/api/findings/{f['id']}/review",
        json={"decision": "approve"},
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["status"] == "confirmed"

    # Test POST /api/findings/{finding_id}/review - reject
    review_resp2 = client.post(
        f"/api/findings/{f['id']}/review",
        json={"decision": "reject"},
    )
    assert review_resp2.status_code == 200
    assert review_resp2.json()["status"] == "rejected"


def test_api_report_endpoint(monkeypatch):
    client = TestClient(app)
    job = create_job(video_key="jobs/j2/video.mp4")

    # Mock app.report module
    import sys
    import types
    fake_report_module = types.ModuleType("app.report")

    def mock_generate_report(job, findings, output_path=None):
        content = b"%PDF-1.4 Mock PDF Report Content"
        if output_path:
            Path(output_path).write_bytes(content)
            return output_path
        return content

    fake_report_module.generate_report = mock_generate_report
    monkeypatch.setitem(sys.modules, "app.report", fake_report_module)

    resp = client.post(f"/api/jobs/{job['id']}/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job["id"]
    assert "report_url" in data
    assert data["report_key"] == f"jobs/{job['id']}/report.pdf"


def test_pipeline_e2e(monkeypatch):
    import cv2
    import numpy as np
    import sys
    import types

    # 1. Create a dummy test video
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

    # Write 15 synthetic frames with high Laplacian variance (sharp road texture)
    np.random.seed(42)
    for i in range(15):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Add random noise so is_blurry is False
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        # Draw a simulated pothole in lower half
        cv2.circle(frame, (320, 360), 30, (40, 40, 40), -1)
        out.write(frame)
    out.release()

    # Upload video to local storage
    s3_key = "jobs/test_e2e_job/video.mp4"
    upload_file(video_path, s3_key)

    # Create job
    job = create_job(video_key=s3_key, status="uploaded")
    job_id = job["id"]

    # Mock pothole, crack, track, and agent modules
    mock_pothole_mod = types.ModuleType("app.cv.pothole")
    mock_crack_mod = types.ModuleType("app.cv.crack")
    mock_track_mod = types.ModuleType("app.cv.track")
    mock_agent_mod = types.ModuleType("app.agent")

    def mock_detect_potholes(frame_dict):
        return [{
            "type": "pothole",
            "bbox": [300, 340, 60, 50],
            "confidence": 0.92,
            "severity_score": 0.85,
            "severity_label": "high",
            "measurements": {"area_px": 2500, "depth_est_cm": 6.5},
        }]

    def mock_detect_cracks(frame_dict):
        return []

    def mock_build_tracks(frames, detections):
        return [{
            "type": "pothole",
            "bbox": [300, 340, 60, 50],
            "confidence": 0.92,
            "severity_score": 0.85,
            "severity_label": "high",
            "frame_index": 0,
            "timestamp_s": 0.0,
            "measurements": {"area_px": 2500, "depth_est_cm": 6.5},
            "status": "confirmed",
        }]

    def mock_run_agent(job_id, tracks, frames):
        add_agent_step(
            job_id=job_id,
            finding_ref=None,
            step_no=1,
            action="multiframe_pothole_verification",
            reason="Pothole persisted across frames with consistent shadow depression",
            confidence_before=0.85,
            confidence_after=0.92,
        )
        return tracks

    mock_pothole_mod.detect_potholes = mock_detect_potholes
    mock_crack_mod.detect_cracks = mock_detect_cracks
    mock_track_mod.build_tracks = mock_build_tracks
    mock_agent_mod.run_agent = mock_run_agent

    monkeypatch.setitem(sys.modules, "app.cv.pothole", mock_pothole_mod)
    monkeypatch.setitem(sys.modules, "app.cv.crack", mock_crack_mod)
    monkeypatch.setitem(sys.modules, "app.cv.track", mock_track_mod)
    monkeypatch.setitem(sys.modules, "app.agent", mock_agent_mod)

    try:
        # Run orchestrator
        run_pipeline(job_id)

        # Assert job state
        finished_job = get_job(job_id)
        assert finished_job["status"] == "done"
        assert finished_job["frames_total"] > 0
        assert finished_job["frames_done"] == finished_job["frames_total"]
        assert finished_job["summary_json"] is not None

        summary = json.loads(finished_job["summary_json"])
        assert summary["total_findings"] >= 1
        assert summary["by_type"]["pothole"] >= 1

        # Assert findings state
        findings = get_findings(job_id)
        assert len(findings) >= 1
        f = findings[0]
        assert f["image_key"] is not None
        assert f["crop_key"] is not None

        # Assert agent steps
        steps = get_agent_steps(job_id)
        assert len(steps) >= 1
        assert steps[0]["action"] == "multiframe_pothole_verification"

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

