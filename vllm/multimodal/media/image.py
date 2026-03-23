# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
import os
from pathlib import Path
import threading
import time

import pybase64
import torch
from PIL import Image

from vllm import envs
from vllm.utils.serial_utils import tensor2base64

from ..image import IMAGE_LOADER_REGISTRY, convert_image_mode, rgba_to_rgb
from .base import MediaIO, MediaWithBytes


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    _thread_local = threading.local()
    _cache_first_result_enabled = bool(
        int(os.getenv("VLLM_IMAGE_DECODE_CACHE_ENABLED", "0"))
    )
    _cache_sleep_ms = int(os.getenv("VLLM_IMAGE_DECODE_SLEEP_MS", "0"))

    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        super().__init__()

        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        image_loader_backend = (
            kwargs.pop("image_backend", None) or envs.VLLM_IMAGE_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.image_loader = IMAGE_LOADER_REGISTRY.load(image_loader_backend)

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _to_pil_image(self, image: Image.Image | object) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        # Handle torch.Tensor (from GPU-resident decoders)
        if isinstance(image, torch.Tensor):
            import numpy as np
            if image.is_cuda:
                image = image.cpu()
            arr = image.numpy()
            return Image.fromarray(arr)
        # Keep loader outputs flexible while preserving PIL-based behavior.
        # Backends can return ndarray-like objects and we normalize here.
        if hasattr(image, "__array_interface__"):
            return Image.fromarray(image)
        # numpy arrays don't always have __array_interface__ with newer numpy
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
        except ImportError:
            pass

        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(image, self.image_mode)

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image] | torch.Tensor:
        image = self.image_loader.load_bytes(data, **self.kwargs)
        # GPU-resident loaders return torch.Tensor on CUDA — pass through
        # without PIL conversion to preserve zero-copy GPU decode path.
        if isinstance(image, torch.Tensor) and image.is_cuda:
            return image
        # PIL bypass: if backend returned an ndarray that is already in the
        # target mode (3-channel for RGB), skip the expensive PIL round-trip.
        # HF's is_valid_image() accepts np.ndarray, so downstream is safe.
        if (
            hasattr(image, "__array_interface__")
            and self.image_mode == "RGB"
            and hasattr(image, "ndim")
            and image.ndim == 3
            and image.shape[2] == 3
        ):
            return MediaWithBytes(image, data)
        image = self._to_pil_image(image)
        return MediaWithBytes(self._convert_image_mode(image), data)

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
        with open(filepath, "rb") as f:
            data = f.read()
        return self.load_bytes(data)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        # Enable sparse tensor integrity checks to prevent out-of-bounds
        # writes from maliciously crafted tensors
        with torch.sparse.check_sparse_tensor_invariants():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
