"""FastAPI Application for Pothole & Road Damage Reporter."""

import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import uuid

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.db import (
    create_job,
    get_agent_steps,
    get_finding,
    get_findings,
    get_job,
    init_db,
    list_jobs,
    update_finding,
)
from app.pipeline import run_pipeline
from app.storage import is_local_storage, presign_get, upload_file

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Ensure database schema is created on application startup."""
    logger.info("Initializing database schema on startup...")
    init_db()
    yield


app = FastAPI(
    title="Pothole & Road Damage Reporter API",
    description="Backend API for automated road damage detection, verification, and reporting.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized JSON format for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler to log and return clean error responses."""
    logger.exception("Unhandled server error processing request to %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


class ReviewRequest(BaseModel):
    """Payload schema for human review of road damage findings."""
    decision: str = Field(..., description="'approve' or 'reject'")


@app.get("/health")
@app.get("/api/health")
def health_endpoint() -> Dict[str, Any]:
    """Health check endpoint providing runtime and detector status."""
    import cv2
    storage_type = "local" if is_local_storage() else "s3"
    return {
        "status": "ok",
        "opencv": cv2.__version__,
        "detector": "YOLOv8n ONNX (RDD2022)",
        "authentication_required": False,
        "storage": storage_type,
    }


@app.get("/api/jobs")
def list_jobs_endpoint(limit: int = 50) -> Dict[str, Any]:
    """List recent road damage inspection jobs."""
    jobs = list_jobs(limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


@app.post("/api/jobs", status_code=201)
async def create_job_endpoint(
    background_tasks: BackgroundTasks,
    request: Request,
    video: UploadFile = File(...),
    location_text: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """Upload a video and launch the road damage detection pipeline.
    
    Validates file extension and ensures file does not exceed 100MB limit.
    """
    filename = video.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Supported extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check Content-Length header if present
    content_length_header = request.headers.get("content-length")
    if content_length_header:
        try:
            if int(content_length_header) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of 100MB (Header: {content_length_header} bytes).",
                )
        except ValueError:
            pass

    job_id = uuid.uuid4().hex
    s3_key = f"jobs/{job_id}/video{ext}"

    temp_dir = tempfile.mkdtemp(prefix=f"roadreport_upload_{job_id}_")
    temp_video_path = os.path.join(temp_dir, f"video{ext}")

    try:
        total_bytes = 0
        with open(temp_video_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of 100MB (Read {total_bytes} bytes).",
                    )
                buffer.write(chunk)

        # Upload video to storage
        upload_file(temp_video_path, s3_key)

        # Create record in DB
        create_job(
            job_id=job_id,
            video_key=s3_key,
            location_text=location_text,
            lat=lat,
            lon=lon,
            status="uploaded",
        )

        # Enqueue pipeline in background
        background_tasks.add_task(run_pipeline, job_id)

        return {"job_id": job_id, "status": "uploaded"}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/jobs/{job_id}")
def get_job_endpoint(job_id: str) -> Dict[str, Any]:
    """Retrieve job status, progress, and damage summary."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    frames_total = job.get("frames_total") or 0
    frames_done = job.get("frames_done") or 0
    progress = round(frames_done / frames_total, 3) if frames_total > 0 else 0.0

    summary = None
    if job.get("summary_json"):
        try:
            summary = json.loads(job["summary_json"])
        except Exception:
            summary = job["summary_json"]

    video_key = job.get("video_key")
    v_url = presign_get(video_key) if video_key else None
    video_url = f"/api/storage/{video_key.strip('/')}" if (v_url and v_url.startswith("file://") and not os.getenv("FORCE_FILE_URI")) else v_url

    return {
        "id": job["id"],
        "status": job["status"],
        "video_key": job["video_key"],
        "video_url": video_url,
        "location_text": job["location_text"],
        "lat": job["lat"],
        "lon": job["lon"],
        "created_at": job["created_at"],
        "frames_total": frames_total,
        "frames_used": job.get("frames_used"),
        "frames_done": frames_done,
        "progress": progress,
        "summary": summary,
        "evidence": {"video": summary.get("video", {}), "detector": summary.get("detector", {})} if isinstance(summary, dict) else {},
        "summary_json": job.get("summary_json"),
        "error_message": job.get("error_message"),
    }


@app.get("/api/jobs/{job_id}/findings")
def get_job_findings_endpoint(job_id: str) -> Dict[str, Any]:
    """Retrieve all findings for a job with presigned URLs and agent verification steps."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    raw_findings = get_findings(job_id)
    enriched_findings: List[Dict[str, Any]] = []

    for f in raw_findings:
        item = dict(f)

        # Presign image and crop URLs
        image_key = item.get("image_key")
        crop_key = item.get("crop_key")
        img_url = presign_get(image_key) if image_key else None
        crp_url = presign_get(crop_key) if crop_key else None
        
        # If local file URI, convert to accessible /api/storage HTTP path for browsers
        if img_url and img_url.startswith("file://") and not os.getenv("FORCE_FILE_URI"):
            item["image_url"] = f"/api/storage/{image_key.strip('/')}"
        else:
            item["image_url"] = img_url
            
        if crp_url and crp_url.startswith("file://") and not os.getenv("FORCE_FILE_URI"):
            item["crop_url"] = f"/api/storage/{crop_key.strip('/')}"
        else:
            item["crop_url"] = crp_url

        # Parse JSON fields
        if item.get("bbox_json"):
            try:
                item["bbox"] = json.loads(item["bbox_json"])
            except Exception:
                item["bbox"] = item["bbox_json"]
        else:
            item["bbox"] = None

        if item.get("measurements_json"):
            try:
                item["measurements"] = json.loads(item["measurements_json"])
            except Exception:
                item["measurements"] = item["measurements_json"]
        else:
            item["measurements"] = None

        # Presign zoom crop URL
        zoom_key = None
        if isinstance(item.get("measurements"), dict):
            zoom_key = item["measurements"].get("zoom_key")
        if not zoom_key:
            zoom_key = f"jobs/{job_id}/findings/finding_{item['id']}_zoom.jpg"

        zm_url = presign_get(zoom_key) if zoom_key else None
        if zm_url and zm_url.startswith("file://") and not os.getenv("FORCE_FILE_URI"):
            item["zoom_url"] = f"/api/storage/{zoom_key.strip('/')}"
        else:
            item["zoom_url"] = zm_url

        # Retrieve agent reasoning steps for this finding
        steps = get_agent_steps(job_id, finding_ref=item["id"])
        item["agent_steps"] = [dict(s) for s in steps]

        enriched_findings.append(item)

    return {
        "job_id": job_id,
        "findings": enriched_findings,
        "total": len(enriched_findings),
    }


@app.post("/api/findings/{finding_id}/review")
def review_finding_endpoint(finding_id: str, payload: ReviewRequest) -> Dict[str, Any]:
    """Approve or reject a finding."""
    decision = payload.decision.strip().lower()
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'reject'")

    finding = get_finding(finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail=f"Finding '{finding_id}' not found")

    new_status = "confirmed" if decision == "approve" else "rejected"
    updated = update_finding(finding_id, status=new_status)
    return {"id": finding_id, "status": new_status, "finding": updated}


def _execute_report_generator(
    job_id: str,
    job: Dict[str, Any],
    findings: List[Dict[str, Any]],
    output_pdf_path: str,
) -> str:
    """Helper to dynamically invoke the generate_report function from report module."""
    try:
        from app.report import generate_report
    except ImportError:
        try:
            from report import generate_report
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Report generator module (app.report or report) is not installed.",
            )

    sig = inspect.signature(generate_report)
    params = list(sig.parameters.keys())

    kwargs = {}
    if "job_id" in params:
        kwargs["job_id"] = job_id
    elif "job" in params:
        kwargs["job"] = job

    if "findings" in params:
        kwargs["findings"] = findings

    target_param = next(
        (p for p in ("output_path", "pdf_path", "dest_path", "path") if p in params),
        None,
    )
    if target_param:
        kwargs[target_param] = output_pdf_path

    if len(kwargs) == len(params):
        result = generate_report(**kwargs)
    elif len(params) == 1 and params[0] == "job_id":
        result = generate_report(job_id)
    elif len(params) == 1 and params[0] == "job":
        result = generate_report(job)
    elif len(params) == 2 and "job" in params and "findings" in params:
        result = generate_report(job, findings)
    elif len(params) == 2 and "job_id" in params and "findings" in params:
        result = generate_report(job_id, findings)
    else:
        try:
            result = generate_report(job_id=job_id)
        except TypeError:
            try:
                result = generate_report(job=job, findings=findings)
            except TypeError:
                result = generate_report(output_pdf_path)

    if isinstance(result, bytes):
        with open(output_pdf_path, "wb") as f:
            f.write(result)
        return output_pdf_path
    elif isinstance(result, (str, Path)) and os.path.exists(str(result)):
        return str(result)
    elif os.path.exists(output_pdf_path):
        return output_pdf_path
    else:
        raise RuntimeError("Report generator did not generate output file at expected path.")


@app.post("/api/jobs/{job_id}/report")
def generate_job_report_endpoint(job_id: str) -> Dict[str, Any]:
    """Generate a PDF report for a job, upload to storage, and return presigned URL."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    findings = get_findings(job_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        temp_pdf_path = tmp_pdf.name

    try:
        final_pdf_path = _execute_report_generator(job_id, job, findings, temp_pdf_path)
        report_key = f"jobs/{job_id}/report.pdf"
        upload_file(final_pdf_path, report_key)
        presigned_url = presign_get(report_key)

        if presigned_url and presigned_url.startswith("file://") and not os.getenv("FORCE_FILE_URI"):
            presigned_url = f"/api/storage/{report_key.strip('/')}"

        return {
            "job_id": job_id,
            "report_key": report_key,
            "report_url": presigned_url,
        }
    finally:
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass


@app.get("/api/storage/{file_path:path}")
def serve_local_storage_file(file_path: str):
    """Serve files directly from local storage for browser preview and downloads."""
    from app.storage import _get_local_dest_path
    local_path = _get_local_dest_path(file_path)
    if not local_path.exists():
        raise HTTPException(status_code=404, detail="File not found in local storage")
    return FileResponse(str(local_path))

