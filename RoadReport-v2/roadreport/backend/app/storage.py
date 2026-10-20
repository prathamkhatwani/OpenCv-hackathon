"""S3 and local storage management helpers for Pothole & Road Damage Reporter."""

import logging
import os
from pathlib import Path
import shutil
from typing import Any, Optional
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None  # type: ignore
    BotoCoreError = Exception  # type: ignore
    ClientError = Exception  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "roadreport-dev-2026"
DEFAULT_REGION = "ap-south-1"
DEFAULT_LOCAL_DIR = "./storage"

_s3_client: Optional[boto3.client] = None


def get_bucket_name() -> str:
    """Return the configured S3 bucket name from environment."""
    return os.getenv("S3_BUCKET", DEFAULT_BUCKET)


def get_region_name() -> str:
    """Return the configured AWS region from environment."""
    return os.getenv("AWS_REGION", DEFAULT_REGION)


def is_local_storage() -> bool:
    """Check if local storage mode is enabled via environment variable."""
    return os.getenv("USE_LOCAL_STORAGE", "true").strip().lower() in ("true", "1", "yes")


def get_local_storage_dir() -> Path:
    """Return the resolved Path to the local storage directory."""
    path_str = os.getenv("LOCAL_STORAGE_DIR", DEFAULT_LOCAL_DIR)
    path = Path(path_str).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_s3_client() -> Any:
    """Return or initialize a cached boto3 S3 client."""
    global _s3_client
    if boto3 is None:
        raise RuntimeError("boto3 is not installed. Install boto3 to enable S3 storage operations.")
    if _s3_client is None:
        region = get_region_name()
        _s3_client = boto3.client("s3", region_name=region)
    return _s3_client


def _get_local_dest_path(s3_key: str) -> Path:
    """Map an S3 key path to a safe local filesystem path inside the storage directory."""
    root = get_local_storage_dir().resolve()
    if not s3_key or "\\" in s3_key or Path(s3_key).is_absolute():
        raise ValueError("Invalid storage key")
    result = (root / s3_key).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Storage key escapes storage directory")
    return result



def upload_file(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3 or local storage directory.
    
    Args:
        local_path: Path to the local file to upload.
        s3_key: S3 object key / destination relative path.

    Returns:
        The uploaded S3 key string.
    """
    if is_local_storage():
        dest_path = _get_local_dest_path(s3_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest_path)
        logger.info("Saved file locally at %s for key %s", dest_path, s3_key)
        return s3_key

    bucket = get_bucket_name()
    try:
        client = get_s3_client()
        client.upload_file(local_path, bucket, s3_key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, s3_key)
        return s3_key
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, s3_key, e)
        raise


def upload_bytes(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload in-memory bytes to S3 or local storage directory.
    
    Args:
        data: The binary payload.
        s3_key: S3 object key / destination relative path.
        content_type: MIME type of the payload.

    Returns:
        The uploaded S3 key string.
    """
    if is_local_storage():
        dest_path = _get_local_dest_path(s3_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        logger.info("Saved %d bytes locally at %s for key %s", len(data), dest_path, s3_key)
        return s3_key

    bucket = get_bucket_name()
    try:
        client = get_s3_client()
        client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("Uploaded %d bytes to s3://%s/%s", len(data), bucket, s3_key)
        return s3_key
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to upload bytes to s3://%s/%s: %s", bucket, s3_key, e)
        raise


def presign_get(s3_key: str, expires: int = 3600) -> str:
    """Generate a presigned GET URL for an S3 object or file URI for local storage.
    
    Args:
        s3_key: S3 object key.
        expires: Expiration duration in seconds (default 1 hour).

    Returns:
        Presigned GET URL or file:// URI string.
    """
    if is_local_storage():
        dest_path = _get_local_dest_path(s3_key).resolve()
        return dest_path.as_uri()

    bucket = get_bucket_name()
    try:
        client = get_s3_client()
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=expires,
        )
        return str(url)
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to generate presigned GET URL for s3://%s/%s: %s", bucket, s3_key, e)
        raise


def download_file(s3_key: str, local_path: str) -> None:
    """Download an object from S3 or local storage to a local file path.
    
    Args:
        s3_key: S3 object key to download.
        local_path: Destination path on the local filesystem.
    """
    dest = Path(local_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if is_local_storage():
        src_path = _get_local_dest_path(s3_key)
        if not src_path.exists():
            raise FileNotFoundError(f"Local storage file not found for key '{s3_key}' at '{src_path}'")
        shutil.copyfile(src_path, dest)
        logger.info("Copied local storage file from %s to %s", src_path, dest)
        return

    bucket = get_bucket_name()
    try:
        client = get_s3_client()
        client.download_file(bucket, s3_key, str(dest))
        logger.info("Downloaded s3://%s/%s to %s", bucket, s3_key, dest)
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to download s3://%s/%s to %s: %s", bucket, s3_key, dest, e)
        raise
