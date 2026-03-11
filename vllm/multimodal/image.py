# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from io import BytesIO
import threading

import numpy as np
from PIL import Image

from vllm.utils.registry import ExtensionManager


class ImageLoader:
    @classmethod
    @abstractmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image | np.ndarray:
        raise NotImplementedError


IMAGE_LOADER_REGISTRY = ExtensionManager()


@IMAGE_LOADER_REGISTRY.register("pil")
class PILImageLoader(ImageLoader):
    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image:
        return Image.open(BytesIO(data))


@IMAGE_LOADER_REGISTRY.register("nvimagecodec")
class NVImageCodecLoader(ImageLoader):
    _thread_local = threading.local()

    @classmethod
    def _get_decoder(cls):
        try:
            from nvidia import nvimgcodec
        except ImportError as exc:
            raise ImportError(
                "nvimagecodec is not available. Please install the NVIDIA "
                "nvImageCodec Python package to use image_backend='nvimagecodec'."
            ) from exc

        decoder = getattr(cls._thread_local, "decoder", None)
        if decoder is None:
            decoder = nvimgcodec.Decoder()
            cls._thread_local.decoder = decoder

        return decoder

    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> np.ndarray:
        try:
            decoder = cls._get_decoder()
            from nvidia import nvimgcodec

            # Prefer CodeStream for broader nvimgcodec API compatibility.
            try:
                decoded = decoder.decode(nvimgcodec.CodeStream(data))
            except Exception:
                # Fallback for environments where raw bytes are accepted.
                decoded = decoder.decode(data)
            if decoded is None:
                raise ValueError("nvimagecodec failed to decode image bytes.")

            decoded_cpu = decoded.cpu()
            if decoded_cpu is None:
                raise ValueError("nvimagecodec failed to copy decoded image to CPU.")

            return np.asarray(decoded_cpu)
        except Exception as exc:
            raise ValueError(f"Failed to decode image with nvimagecodec: {exc}") from exc


def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    image = image.resize((new_width, new_height))
    if transpose >= 0:
        image = image.transpose(Image.Transpose(transpose))
    return image


def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(image: Image.Image, to_mode: str):
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)
