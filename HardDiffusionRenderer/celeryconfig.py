import os

import orjson

# from pathlib import Path
from dotenv import load_dotenv
from kombu.serialization import register

load_dotenv()


REDIS_CONNECTION_STRING = os.getenv(
    "REDIS_CONNECTION_STRING", "redis://127.0.0.1:6379/5"
)


def dumps(obj):
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)


def loads(obj):
    return orjson.loads(obj)


register(
    "orjson",
    dumps,
    loads,
    content_type="application/x-orjson",
    content_encoding="utf-8",
)

result_accept_content = ["application/x-orjson"]
accept_content = ["application/x-orjson"]
task_serializer = "orjson"
result_serializer = "orjson"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
cache_backend = REDIS_CONNECTION_STRING
CELERY_BROKER_URL = REDIS_CONNECTION_STRING
result_backend = REDIS_CONNECTION_STRING
timezone = "UTC"

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "/tmp")
DEFAULT_TEXT_TO_IMAGE_MODEL = os.getenv(
    "DEFAULT_TEXT_TO_IMAGE_MODEL", "CompVis/stable-diffusion-v1-4"
)
