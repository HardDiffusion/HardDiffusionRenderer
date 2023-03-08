"""Establishes a connection to S3 or DigitalOcean Spaces."""
import os

import boto3
import botocore

USE_VIRTUAL_ADDRESSING = os.getenv("USE_VIRTUAL_ADDRESSING", "false").lower() == "true"
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION_NAME = os.getenv("S3_REGION_NAME")
S3_ACCESS_KEY_ID = os.getenv(
    "S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", os.getenv("SPACES_KEY"))
)
S3_SECRET_ACCESS_KEY = os.getenv(
    "S3_SECRET_ACCESS_KEY",
    os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("SPACES_SECRET")),
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", os.getenv("SPACES_BUCKET"))

S3_KWARGS = {
    "config": botocore.config.Config(s3={"addressing_style": "virtual"}),
    "region_name": S3_REGION_NAME,
    "endpoint_url": S3_ENDPOINT_URL,
    "aws_access_key_id": S3_ACCESS_KEY_ID,
    "aws_secret_access_key": S3_SECRET_ACCESS_KEY,
}
if not USE_VIRTUAL_ADDRESSING:
    S3_KWARGS.pop("config")

if USE_S3:
    if not S3_ENDPOINT_URL:
        raise ValueError("S3_ENDPOINT_URL must be set if USE_S3 is true.")
    if not S3_REGION_NAME:
        raise ValueError("S3_REGION_NAME must be set if USE_S3 is true.")
    if not S3_ACCESS_KEY_ID:
        raise ValueError("S3_ACCESS_KEY_ID must be set if USE_S3 is true.")
    if not S3_SECRET_ACCESS_KEY:
        raise ValueError("S3_SECRET_ACCESS_KEY must be set if USE_S3 is true.")
    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME must be set if USE_S3 is true.")
    _BOTO_SESSION = boto3.session.Session()
    _S3_CLIENT = _BOTO_SESSION.client("s3", **S3_KWARGS)
    S3_HOSTNAME = f"{S3_BUCKET_NAME}.{S3_ENDPOINT_URL.replace('https://', '')}"
else:
    _S3_CLIENT = None
    S3_HOSTNAME = None


def upload_file(filename, contents, acl="private", metadata=None):
    """Upload a file to S3 or DigitalOcean Spaces.

    Args:
        filename (str): The filename.
        contents (bytes): The contents of the file.
        acl (str, optional): The ACL to use. Defaults to "private". "public-read" is
            also supported.
        metadata (Optional[Dict[str, str]], optional): The metadata to use.
            Defaults to None.

    Raises: ValueError: If S3 is not configured.

    Returns:
        None
    """
    if not _S3_CLIENT:
        raise ValueError("S3 is not configured.")
    if not metadata:
        metadata = {}
    _S3_CLIENT.put_object(
        Bucket=S3_BUCKET_NAME, Key=filename, Body=contents, ACL=acl, Metadata=metadata
    )
