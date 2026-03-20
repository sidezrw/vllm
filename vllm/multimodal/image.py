# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from io import BytesIO
import os
import threading

import numpy as np
import torch
from PIL import Image

from vllm.utils.registry import ExtensionManager


class ImageLoader:
    @classmethod
    @abstractmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image | np.ndarray:
        raise NotImplementedError


class BatchImageLoader(ImageLoader):
    """Extended loader interface supporting batch decode."""

    @classmethod
    def load_bytes_batch(
        cls, data_list: list[bytes], **kwargs
    ) -> list[np.ndarray]:
        """Decode a batch of images. Default: sequential fallback."""
        return [cls.load_bytes(d, **kwargs) for d in data_list]


IMAGE_LOADER_REGISTRY = ExtensionManager()


@IMAGE_LOADER_REGISTRY.register("pil")
class PILImageLoader(ImageLoader):
    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image:
        return Image.open(BytesIO(data))


@IMAGE_LOADER_REGISTRY.register("nvimagecodec")
class NVImageCodecLoader(ImageLoader):
    """nvImageCodec CPU-output decoder.

    Decodes on GPU HW engine, transfers result to CPU as numpy array.
    Usage: VLLM_IMAGE_LOADER_BACKEND=nvimagecodec
    """

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


@IMAGE_LOADER_REGISTRY.register("nvimagecodec_gpu_resident")
class NVImageCodecGPUResidentLoader(BatchImageLoader):
    """GPU-resident decode loader — images stay on GPU as torch.Tensor.

    Decodes JPEG on GPU HW engine and returns torch.Tensor (H, W, 3) uint8
    on CUDA. Combined with Qwen2VLGPUPreprocessor, this eliminates all
    CPU<->GPU transfers in the vision preprocessing pipeline.

    Performance (RTX PRO 6000 Blackwell):
      GPU-resident batch=128, 2048²: ~90% JPEG HW utilization, 780 img/s

    Usage: VLLM_IMAGE_LOADER_BACKEND=nvimagecodec_gpu_resident
    Config:
      VLLM_NVIMGCODEC_BATCH_SIZE=128   (batch accumulation size)
      VLLM_NVIMGCODEC_GPU_RESIDENT=1   (keep tensors on GPU; default)
    """

    _thread_local = threading.local()
    _batch_lock = threading.Lock()
    _pending_items: list[tuple[bytes, threading.Event, list]] = []
    _batch_size = int(os.environ.get("VLLM_NVIMGCODEC_BATCH_SIZE", "128"))
    _batch_timeout_s = (
        int(os.environ.get("VLLM_NVIMGCODEC_BATCH_TIMEOUT_MS", "10"))
        / 1000.0
    )
    _gpu_resident = os.environ.get(
        "VLLM_NVIMGCODEC_GPU_RESIDENT", "1"
    ).lower() in ("1", "true", "yes")

    @classmethod
    def _get_decoder(cls):
        try:
            from nvidia import nvimgcodec
        except ImportError as exc:
            raise ImportError(
                "nvimagecodec is not available."
            ) from exc

        decoder = getattr(cls._thread_local, "decoder", None)
        if decoder is None:
            decoder = nvimgcodec.Decoder()
            cls._thread_local.decoder = decoder
        return decoder

    @classmethod
    def _get_streams(cls):
        streams = getattr(cls._thread_local, "decode_streams", None)
        if streams is None:
            streams = (torch.cuda.Stream(), torch.cuda.Stream())
            cls._thread_local.decode_streams = streams
        return streams

    @classmethod
    def _decode_batch_gpu(
        cls, data_list: list[bytes]
    ) -> list[torch.Tensor]:
        """Decode a batch on GPU, return GPU tensors."""
        decoder = cls._get_decoder()
        stream_a, stream_b = cls._get_streams()
        from nvidia import nvimgcodec

        code_streams = []
        for data in data_list:
            try:
                code_streams.append(nvimgcodec.CodeStream(data))
            except Exception:
                code_streams.append(data)

        with torch.cuda.stream(stream_a):
            decoded_list = decoder.decode(
                code_streams,
                cuda_stream=stream_a.cuda_stream,
            )

        stream_a.synchronize()

        results = []
        for decoded in decoded_list:
            if decoded is None:
                raise ValueError(
                    "nvimagecodec GPU-resident batch decode failed."
                )
            gpu_tensor = torch.from_dlpack(decoded).clone()
            results.append(gpu_tensor)

        return results

    @classmethod
    def load_bytes(cls, data: bytes, **kwargs):
        """Submit a single image for batch decode."""
        try:
            done_event = threading.Event()
            result_holder: list = []

            with cls._batch_lock:
                cls._pending_items.append(
                    (data, done_event, result_holder)
                )
                batch_full = (
                    len(cls._pending_items) >= cls._batch_size
                )

            if batch_full:
                cls._flush_batch()
            else:
                if not done_event.wait(
                    timeout=cls._batch_timeout_s + 0.01
                ):
                    cls._flush_batch()
                    done_event.wait(timeout=5.0)

            if not result_holder:
                raise ValueError("Batch decode produced no result.")
            if isinstance(result_holder[0], Exception):
                raise result_holder[0]
            return result_holder[0]
        except Exception as exc:
            raise ValueError(
                f"Failed GPU-resident decode: {exc}"
            ) from exc

    @classmethod
    def _flush_batch(cls):
        """Flush pending images as a single batch decode."""
        with cls._batch_lock:
            if not cls._pending_items:
                return
            items = list(cls._pending_items)
            cls._pending_items.clear()

        data_list = [item[0] for item in items]

        try:
            results = cls._decode_batch_gpu(data_list)
            for (_, event, holder), result in zip(items, results):
                holder.append(result)
                event.set()
        except Exception as e:
            for _, event, holder in items:
                holder.append(e)
                event.set()

    @classmethod
    def load_bytes_batch(
        cls, data_list: list[bytes], **kwargs
    ) -> list:
        """Directly decode a batch."""
        return cls._decode_batch_gpu(data_list)


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
