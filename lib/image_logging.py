import os
import tempfile
from typing import Literal, Sequence

import numpy as np
import wandb
from PIL import Image as PILImage

ImageFileType = Literal["png", "jpg"]


def discrete_object_to_wandb_image(
    obj: np.ndarray,
    vocab_size: int,
    n_channels: int,
    file_type: ImageFileType,
) -> tuple[wandb.Image, str]:
    """Save a discrete image object and return a W&B image plus temp path."""
    image = _to_pil_image(obj, vocab_size, n_channels, file_type)
    with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as f:
        image_path = f.name

    if file_type == "png":
        image.save(
            image_path,
            format="PNG",
            optimize=True,
            **_png_palette_kwargs(image, vocab_size, n_channels),
        )
    elif file_type == "jpg":
        image.save(image_path, format="JPEG", quality=75, optimize=True)
    else:
        raise ValueError(f"Unsupported image file type: {file_type!r}")

    return wandb.Image(image_path), image_path


def discrete_object_to_uint8_image(
    obj: np.ndarray,
    vocab_size: int,
    n_channels: int,
) -> np.ndarray:
    image = _token_values(vocab_size)[_clipped_tokens(obj, vocab_size)]
    if n_channels == 1:
        return image[..., 0]
    return image[..., :3]


def cleanup_temp_files(paths: Sequence[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _to_pil_image(
    obj: np.ndarray,
    vocab_size: int,
    n_channels: int,
    file_type: ImageFileType,
) -> PILImage.Image:
    if file_type == "png":
        palette_image = _to_palette_image(obj, vocab_size, n_channels)
        if palette_image is not None:
            return palette_image

    image = discrete_object_to_uint8_image(obj, vocab_size, n_channels)
    mode = "L" if n_channels == 1 else "RGB"
    return PILImage.fromarray(image, mode=mode)


def _to_palette_image(
    obj: np.ndarray,
    vocab_size: int,
    n_channels: int,
) -> PILImage.Image | None:
    n_values = vocab_size**n_channels
    if n_channels not in (1, 3) or n_values > 256:
        return None

    tokens = _clipped_tokens(obj, vocab_size).astype(np.uint16, copy=False)
    if n_channels == 1:
        packed = tokens[..., 0]
    else:
        packed = (
            tokens[..., 0] * (vocab_size**2)
            + tokens[..., 1] * vocab_size
            + tokens[..., 2]
        )

    image = PILImage.fromarray(packed.astype(np.uint8), mode="P")
    image.putpalette(_palette(vocab_size, n_channels))
    return image


def _palette(vocab_size: int, n_channels: int) -> list[int]:
    values = _token_values(vocab_size)
    n_values = vocab_size**n_channels
    palette = np.zeros((256, 3), dtype=np.uint8)

    if n_channels == 1:
        palette[:n_values] = values[:, None]
    else:
        indexes = np.arange(n_values)
        palette[:n_values, 0] = values[indexes // (vocab_size**2)]
        palette[:n_values, 1] = values[(indexes // vocab_size) % vocab_size]
        palette[:n_values, 2] = values[indexes % vocab_size]

    return palette.ravel().tolist()


def _png_palette_kwargs(
    image: PILImage.Image, vocab_size: int, n_channels: int
) -> dict[str, int]:
    if image.mode != "P":
        return {}
    n_values = vocab_size**n_channels
    if n_values <= 2:
        return {"bits": 1}
    if n_values <= 4:
        return {"bits": 2}
    if n_values <= 16:
        return {"bits": 4}
    return {"bits": 8}


def _clipped_tokens(obj: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.clip(obj, 0, vocab_size - 1).astype(np.int64, copy=False)


def _token_values(vocab_size: int) -> np.ndarray:
    scale = 255.0 / max(vocab_size - 1, 1)
    return np.rint(np.arange(vocab_size) * scale).astype(np.uint8)
