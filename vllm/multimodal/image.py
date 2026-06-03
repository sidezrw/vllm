# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from abc import abstractmethod
from io import BytesIO

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

            try:
                decoded = decoder.decode(nvimgcodec.CodeStream(data))
            except Exception:
                decoded = decoder.decode(data)
            if decoded is None:
                raise ValueError("nvimagecodec failed to decode image bytes.")

            decoded_cpu = decoded.cpu()
            if decoded_cpu is None:
                raise ValueError(
                    "nvimagecodec failed to copy decoded image to CPU."
                )

            return np.asarray(decoded_cpu)
        except Exception as exc:
            raise ValueError(
                f"Failed to decode image with nvimagecodec: {exc}"
            ) from exc


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
    # batch_size/timeout defaults tuned for the v0.21 inner-loop pass on H100
    # fast profile: smaller batches + sub-ms wait beat the original 128/10ms
    # values when the upstream rate is well below the batch fill rate (offline
    # mode + max_num_seqs=1024 ⇒ requests arrive in bursts but the per-image
    # decode wait dominated at the prior 10ms timeout). Override via env var
    # at the docker run boundary for sweep experiments.
    _batch_size = int(os.environ.get("VLLM_NVIMGCODEC_BATCH_SIZE", "8"))
    _batch_timeout_s = (
        int(os.environ.get("VLLM_NVIMGCODEC_BATCH_TIMEOUT_MS", "1"))
        / 1000.0
    )
    _gpu_resident = os.environ.get(
        "VLLM_NVIMGCODEC_GPU_RESIDENT", "1"
    ).lower() in ("1", "true", "yes")
    _logged_once = False

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

        # REFLEAK_FIX_DECODE_BATCH — explicit dlpack-view drop +
        # immediate code_streams del + post-loop decoded del +
        # post-clone synchronize so nvimgcodec pool slots release
        # before this function returns. See patches/refleak-fix/
        # apply_patch.py docstring for the full rationale.
        # GPURES_MINIMAL_DECODE_GATE
        from vllm.multimodal.gpures_minimal import reserve_decode_batch
        _gpures_decode_token = reserve_decode_batch(
            len(code_streams), "nvimagecodec.decode"
        )
        try:
            with torch.cuda.stream(stream_a):
                decoded_list = decoder.decode(
                    code_streams,
                    cuda_stream=stream_a.cuda_stream,
                )

            # Drop the per-image CodeStream descriptor list immediately;
            # the decoder has consumed them and any pinned per-stream
            # GPU scratch can release now.
            del code_streams

            stream_a.synchronize()

            results = []
            for i, decoded in enumerate(decoded_list):
                if decoded is None:
                    raise ValueError(
                        "nvimagecodec GPU-resident batch decode failed."
                    )
                # Explicit dlpack view binding so we can `del` it at a
                # deterministic point. If nvimgcodec's __dlpack__
                # deleter is responsible for releasing the pool slot,
                # this is what triggers it. ``view.clone()`` produces
                # a brand-new PyTorch-owned buffer.
                view = torch.from_dlpack(decoded)
                gpu_tensor = view.clone()
                del view
                # GPURES_MINIMAL_DECODE_RESIDENCY
                from vllm.multimodal.gpures_minimal import attach_decode_residency
                attach_decode_residency(gpu_tensor, f"nvimagecodec.decoded.{i}")
                results.append(gpu_tensor)
                # Free nvimgcodec buffer reference (t465: prevent decoder
                # buffers from being held via the decoded_list)
                decoded_list[i] = None

            # The for-loop binds `decoded` to the last item; drop it so
            # the last nvimgcodec Image object decrefs before return.
            del decoded
            del decoded_list
            # Make sure the clones and any in-flight deleters finish
            # before we hand `results` back to the caller — otherwise
            # the dlpack deleter may still be queued and the underlying
            # pool slot stays pinned.
            torch.cuda.synchronize()
            return results
        finally:
            _gpures_decode_token.release()

    @classmethod
    def load_bytes(cls, data: bytes, **kwargs):
        """Submit a single image for batch decode."""
        if not cls._logged_once:
            cls._logged_once = True
            print(
                "[NVImageCodecGPUResidentLoader] active: "
                f"backend=nvimagecodec_gpu_resident batch_size={cls._batch_size} "
                f"gpu_resident={cls._gpu_resident}",
                flush=True,
            )
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
                    # GPURES_MINIMAL_DECODE_WAIT
                    _gpures_wait_s = float(os.environ.get(
                        "VLLM_GPU_DECODE_RESULT_WAIT_TIMEOUT_S", "0"
                    ) or "0")
                    if _gpures_wait_s > 0:
                        done_event.wait(timeout=_gpures_wait_s)
                    else:
                        done_event.wait()

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

        # GPURES_MINIMAL_FLUSH_CHUNKS
        from vllm.multimodal.gpures_minimal import decode_batch_limit
        limit = decode_batch_limit(cls._batch_size)
        try:
            for start in range(0, len(items), limit):
                chunk = items[start:start + limit]
                data_list = [item[0] for item in chunk]
                results = cls._decode_batch_gpu(data_list)
                for (_, event, holder), result in zip(chunk, results):
                    holder.append(result)
                    event.set()
                del results
                del data_list
                del chunk
        except Exception as e:
            for _, event, holder in items:
                if not event.is_set():
                    holder.append(e)
                    event.set()
        finally:
            del items

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
