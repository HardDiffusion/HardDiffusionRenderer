"""Celery tasks for generating images."""
import io
import os
import platform
from datetime import datetime
from typing import Optional, Tuple

from celery import Task, shared_task

from HardDiffusionRenderer import celeryconfig
from HardDiffusionRenderer.image import generate_image_status, render_image
from HardDiffusionRenderer.logs import logger
from HardDiffusionRenderer.s3_upload import S3_HOSTNAME, USE_S3, upload_file

hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node()))
if "." in hostname:
    hostname = hostname.split(".")[0]


@shared_task(name="generate_image_status", queue="image_progress")
def generate_image_status_task(event):
    raise NotImplementedError("This task shouldn't run here...")


@shared_task(name="generate_image_completed", queue="image_progress")
def generate_image_completed_task(image_id, task_id, hostname, start, end, seed):
    raise NotImplementedError("This task shouldn't run here...")


@shared_task(name="generate_image_error", queue="image_progress")
def generate_image_error_task(image_id, task_id, hostname):
    raise NotImplementedError("This task shouldn't run here...")


def ensure_model_name(model_path_or_name: str) -> str:
    """Ensure the model name is set.

    If the model name is not set, use the default model name.

    Args:
        model_path_or_name (str): The model padoth or name.

    Returns:
        str: The model name.
    """
    return model_path_or_name or celeryconfig.DEFAULT_TEXT_TO_IMAGE_MODEL


@shared_task(bind=True, name="generate_image", queue="render", max_concurrency=1)
def generate_image(
    self,
    image_id: int,
    prompt: str = "An astronaut riding a horse on the moon.",
    negative_prompt: Optional[str] = None,
    model_path_or_name: Optional[str] = None,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
    nsfw: bool = False,
    callback_steps: int = 1,
    preview_image: bool = False,
) -> Tuple[str, str]:
    """Generate an image.

    Args:
        image_id (int): The image ID.
        prompt (str, optional): The prompt to use.
            Defaults to "An astronaut riding a horse on the moon.".
        negative_prompt (Optional[str], optional):
            The negative prompt to use. Defaults to None.
        model_path_or_name (Optional[str], optional): The model path or name.
            Defaults to None.
        seed (Optional[int], optional): The seed to use. Defaults to None.
        guidance_scale (float, optional): The guidance scale to use. Defaults to 7.5.
        num_inference_steps (int, optional): The number of inference steps to use.
            Defaults to 50.
        height (int, optional): The height of the image. Defaults to 512.
        width (int, optional): The width of the image. Defaults to 512.
        nsfw (bool, optional): Whether to use the NSFW model. Defaults to False.
        callback_steps (int, optional): The number of steps between callbacks.
            Defaults to 1.
        preview_image (bool, optional): Whether to preview the image. Defaults to False.

    Returns:
        Tuple[str, str]: The image filename and seed.
    """
    task_id = self.request.id
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width,
    }
    _model_path_or_name = ensure_model_name(model_path_or_name)
    try:
        start = datetime.now()
        image, seed = render_image(
            _model_path_or_name,
            nsfw,
            seed,
            params,
            callback=generate_image_status,
            callback_steps=callback_steps,
            callback_args=[task_id],
            callback_kwargs={
                "total_steps": num_inference_steps,
                "preview_image": preview_image,
            },
        )
        end = datetime.now()
        filename = f"{task_id}.png"
        if not USE_S3:
            image.save(os.path.join(celeryconfig.MEDIA_ROOT, filename))
        else:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            upload_file(filename, image_bytes.getvalue(), acl="public-read")
            hostname = S3_HOSTNAME
        # Task
        generate_image_completed_task.apply_async(
            args=[image_id, task_id, hostname, start, end, seed], queue="image_progress"
        )
    except Exception as e:
        logger.error("%s", e)
        generate_image_error_task.apply_async(
            args=[image_id, task_id, hostname], queue="image_progress"
        )
        raise e
    return filename, hostname
