"""Image utilities."""

import base64
import json
from io import BytesIO
from random import randint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image

from HardDiffusionRenderer.noise import decode_latents
from HardDiffusionRenderer.pipeline import get_pipeline


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Get the generator for the given seed.

    Args:
        seed (int): The seed to use.

    Returns:
        torch.Generator: The generator with specified seed or a random seed.
    """
    generator = torch.Generator("cuda")
    return generator.manual_seed(seed or randint(0, 2**32))


def render_image(
    model_path_or_name: str,
    nsfw: bool,
    seed: Optional[int],
    params: dict[str, Any],
    callback: Callable,
    callback_steps: int,
    callback_args: List[Any],
    callback_kwargs: Dict[str, Any],
) -> Tuple[Image.Image, int]:
    """Render the image.

    Args:
        model_path_or_name (str): The path or name of the model to use.
        nsfw (bool): Whether to use the NSFW model.
        seed (int): The seed to use.
        params (dict): The parameters to use.
        callback (callable): The callback to use.
        callback_steps (int): The number of steps between callbacks.
        callback_args (list): The arguments to pass to the callback.
        callback_kwargs (dict): The keyword arguments to pass to the callback.

    Returns:
        tuple: The generated image and the seed used to generate it.
    """
    try:
        pipe = get_pipeline(model_path_or_name, nsfw)
    except OSError as ex:
        raise RuntimeError("Error loading model") from ex
    pipe = pipe.to("cuda")
    generator = get_generator(seed)
    if isinstance(model_path_or_name, list):
        merged_pipe = pipe.merge(
            model_path_or_name[1:],
            interp="sigmoid",
            alpha=0.4,
        )
    else:
        merged_pipe = pipe

    merged_pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    image = merged_pipe(
        generator=generator,
        callback=callback,
        callback_steps=callback_steps,
        callback_args=callback_args,
        callback_kwargs=callback_kwargs,
        **params,
    ).images[0]
    seed = generator.initial_seed()
    return image, seed


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (np.ndarray): The image or batch of images to convert.

    Returns:
        PIL.Image.Image: The converted image or batch of images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return (
        [Image.fromarray(image.squeeze(), mode="L") for image in images]
        if images.shape[-1] == 1
        else [Image.fromarray(image) for image in images]
    )


def generate_image_status(
    step: int,
    timestep: Optional[torch.FloatTensor],
    latents: Optional[torch.Tensor],
    task_id,
    session_id=None,
    total_steps=None,
    vae=None,
    message=None,
    preview_image=False,
    *args,
    **kwargs,
) -> None:
    """Status callback for image generation.

    Preview images are only sent if the VAE is provided.

    Preview images reduce the performance of the generation process, by half of more.
    Args:
        step (int): The current step.
        timestep (torch.FloatTensor): The current timestep, unused currently.
        latents (torch.Tensor): The current latents.
        task_id (str): The task ID.
        session_id (str): The session ID.
        message (str): The message to send.
        preview_image (bool): Whether to send a preview image.

    Returns:
        None
    """
    if latents is not None and vae and preview_image:
        image = decode_latents(vae, latents)
    else:
        image = None

    if image is not None and preview_image:
        buffered = BytesIO()
        image = numpy_to_pil(image)
        image[0].save(buffered, format="PNG")
        img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img = None

    event = {
        "task_id": task_id,
        "session_id": session_id,
        "step": step + 1,
        "total_steps": total_steps,
        "message": message,
        "image": img,
    }
    from HardDiffusionRenderer.tasks import generate_image_status_task

    generate_image_status_task.delay(session_id, json.dumps(event))
    """
    await CHANNEL_LAYER.group_send(
        "generate",
        {
            "type": "event_message",
            "event": "image_generating",
            "message": json.dumps(event),
        },
    )

    """
