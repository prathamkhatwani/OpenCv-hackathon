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


@pytest.mark.parametrize("confidence, expected", [(0.92, 1), (0.20, 0)])
def test_pipeline_e2e(monkeypatch, tmp_path, confidence, expected):
    import cv2
    import numpy as np
    from app.cv import detector
    # Only model inference is stubbed; decode, tracking, decisions, database,
    # annotations and API serialization run for real.
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320,240))
    rng = np.random.default_rng(42)
    for _ in range(18):
        writer.write(rng.integers(0,255,(240,320,3),dtype=np.uint8))
    writer.release()
    class Detector:
        def detect(self, image, roi):
            return [{"type":"pothole", "bbox":(300,400,100,60),
                     "confidence":confidence,"measurements":{"area_frac":.01}}]
        def info(self):
            return {"detector":"test_stub"}
    monkeypatch.setattr(detector, 'get_detector', lambda:Detector())
    key = 'jobs/pipeline_regression/video.mp4'
    upload_file(str(path), key)
    job = create_job(video_key=key)
    run_pipeline(job['id'])
    result = get_job(job['id'])
    assert result['status']=='done', result.get('error_message')
    findings = get_findings(job['id'])
    assert len(findings)==expected
    summary = json.loads(result['summary_json'])
    assert summary['total_findings']==expected
    assert summary['video']['width']==320
    if expected:
        assert findings[0]['image_key'] and findings[0]['crop_key']
        metadata=json.loads(findings[0]['measurements_json'])
        assert metadata['track_id']==1
        assert metadata['track_observations']>=3
        client=TestClient(app)
        payload=client.get(f"/api/jobs/{job['id']}/findings").json()['findings']
        assert payload[0]['measurements']['track'][0]['box']==[300/960,400/720,100/960,60/720]
