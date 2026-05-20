"""
GPU-Accelerated Preprocessing for Qwen2VL Vision Transformer.

Replaces the CPU-bound Qwen2VL image preprocessor with GPU-native operations.
Uses PyTorch + torchvision on CUDA for resize and normalize, eliminating
PIL/numpy intermediaries and achieving zero-copy GPU preprocessing.

Pipeline: GPU tensor (CHW, uint8) -> GPU resize -> GPU rescale+normalize -> GPU reshape/flatten

Author: Task t203 (GPU preprocessing prototype)
Fixed: Task t464 (GPU memory leak fix)
  - Added torch.no_grad() to all preprocessing methods
  - Added explicit del of intermediate tensors
  - Added in-place operations where possible
  - Added optional concurrency semaphore
"""

import math
import threading
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

# CLIP normalization constants (same as OPENAI_CLIP_MEAN / OPENAI_CLIP_STD)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Maximum number of concurrent GPU preprocessing operations.
# Prevents GPU OOM from too many simultaneous tensor allocations.
_GPU_PREPROCESS_SEMAPHORE = threading.Semaphore(
    int(__import__('os').environ.get("VLLM_GPU_PREPROCESS_CONCURRENCY", "8"))
)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Compute target resize dimensions for Qwen2VL dynamic resolution.

    Matches the HuggingFace transformers implementation exactly:
    1. Both dimensions divisible by `factor`
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio preserved as closely as possible

    Args:
        height: Original image height.
        width: Original image width.
        factor: Divisibility factor (patch_size * merge_size).
        min_pixels: Minimum total pixels after resize.
        max_pixels: Maximum total pixels after resize.

    Returns:
        (resized_height, resized_width) tuple.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLGPUPreprocessor:
    """GPU-native image preprocessor for Qwen2VL.

    Accepts GPU tensors directly (e.g., from nvImageCodec decode) and performs
    all preprocessing on GPU without CPU round-trips.

    All tensor operations run under torch.no_grad() to prevent autograd
    graph construction, which would retain intermediate tensors and cause
    GPU memory leaks under sustained load.

    Attributes:
        image_mean: Per-channel mean for normalization (CLIP default).
        image_std: Per-channel std for normalization (CLIP default).
        rescale_factor: Scale factor (1/255 for uint8->float).
        min_pixels: Minimum total pixels for dynamic resolution.
        max_pixels: Maximum total pixels for dynamic resolution.
        patch_size: Spatial patch size of ViT (default 14).
        temporal_patch_size: Temporal patch size (default 2).
        merge_size: Merge factor for ViT->LLM (default 2).
        interpolation_mode: PyTorch interpolation mode (default 'bicubic').
    """

    def __init__(
        self,
        image_mean: Tuple[float, ...] = CLIP_MEAN,
        image_std: Tuple[float, ...] = CLIP_STD,
        rescale_factor: float = 1.0 / 255.0,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        interpolation_mode: str = "bicubic",
        device: Optional[torch.device] = None,
    ):
        self.rescale_factor = rescale_factor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.factor = patch_size * merge_size
        self.interpolation_mode = interpolation_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-compute normalization tensors on GPU for efficiency.
        # Shape: (3, 1, 1) for broadcasting over spatial dims.
        self._mean = torch.tensor(image_mean, dtype=torch.float32, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(image_std, dtype=torch.float32, device=self.device).view(3, 1, 1)

    @torch.no_grad()
    def _resize_gpu(
        self,
        image: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Resize a GPU tensor using torch.nn.functional.interpolate.

        Args:
            image: (C, H, W) float32 tensor on GPU.
            target_h: Target height.
            target_w: Target width.

        Returns:
            (C, target_h, target_w) float32 tensor on GPU.
        """
        # interpolate expects (N, C, H, W)
        img_4d = image.unsqueeze(0)
        resized = F.interpolate(
            img_4d,
            size=(target_h, target_w),
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode == "bicubic" else None,
            antialias=True,  # Match PIL BICUBIC antialiasing behavior
        )
        result = resized.squeeze(0)
        # Explicitly free the 4D intermediate
        del img_4d, resized
        return result

    @torch.no_grad()
    def _normalize_gpu(self, image: torch.Tensor) -> torch.Tensor:
        """Rescale and normalize a GPU tensor using in-place ops.

        Combined rescale (1/255) + normalize ((x - mean) / std) using
        in-place operations to minimize intermediate tensor allocations.

        Args:
            image: (C, H, W) float32 tensor on GPU, pixel values [0, 255] or [0, 1].

        Returns:
            Normalized (C, H, W) float32 tensor (modified in-place).
        """
        # In-place operations to avoid creating intermediate tensors
        image.mul_(self.rescale_factor)
        image.sub_(self._mean)
        image.div_(self._std)
        return image

    @torch.no_grad()
    def _reshape_to_patches(
        self,
        patches: torch.Tensor,
        resized_height: int,
        resized_width: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Reshape preprocessed images into flattened patch representation.

        Mirrors the exact reshape/transpose logic from HuggingFace's
        Qwen2VLImageProcessor._preprocess method.

        Args:
            patches: (T, C, H, W) tensor, where T = number of temporal frames
                     (1 for images), C=3, H=resized_height, W=resized_width.
            resized_height: Height after smart_resize.
            resized_width: Width after smart_resize.

        Returns:
            (flatten_patches, grid_thw) where:
                flatten_patches: (grid_t*grid_h*grid_w, C*temporal_patch_size*patch_size*patch_size) tensor
                grid_thw: (grid_t, grid_h, grid_w) tuple
        """
        temporal_patch_size = self.temporal_patch_size
        patch_size = self.patch_size
        merge_size = self.merge_size

        # Pad temporal dimension if not divisible by temporal_patch_size
        T = patches.shape[0]
        if T % temporal_patch_size != 0:
            pad_count = temporal_patch_size - (T % temporal_patch_size)
            repeats = patches[-1:].expand(pad_count, -1, -1, -1)
            patches = torch.cat([patches, repeats], dim=0)
            del repeats

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        # Reshape: (T, C, H, W) -> (grid_t, temp_ps, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Transpose to match HF: (0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # Flatten to (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        # Make contiguous to allow reshape views to be freed
        flatten_patches = flatten_patches.contiguous()
        del patches

        return flatten_patches, (grid_t, grid_h, grid_w)

    @torch.no_grad()
    def preprocess_single(
        self,
        image: torch.Tensor,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Preprocess a single image entirely on GPU.

        All operations run under torch.no_grad() to prevent autograd graph
        construction. Intermediate tensors are explicitly freed.

        Args:
            image: Input tensor. Supported formats:
                - (C, H, W) uint8 or float32 on GPU (e.g., from nvImageCodec)
                - (H, W, C) uint8 or float32 on GPU (will be permuted)
                - (C, H, W) uint8 or float32 on CPU (will be moved to GPU)
            do_resize: Whether to apply smart_resize.
            do_rescale: Whether to rescale by rescale_factor.
            do_normalize: Whether to normalize with mean/std.

        Returns:
            (flatten_patches, grid_thw) matching CPU preprocessor output format.
        """
        # Ensure GPU
        if image.device != self.device:
            image = image.to(self.device, non_blocking=True)

        # Handle HWC -> CHW
        if image.ndim == 3 and image.shape[2] in (1, 3, 4):
            # Heuristic: if last dim is small channel count, it's HWC
            if image.shape[0] not in (1, 3, 4) or image.shape[2] <= 4:
                image = image.permute(2, 0, 1).contiguous()

        # Convert to float32 if needed
        if image.dtype != torch.float32:
            float_image = image.float()
            del image
            image = float_image

        C, H, W = image.shape
        assert C in (1, 3), f"Expected 1 or 3 channels, got {C}"

        # Convert grayscale to RGB
        if C == 1:
            image = image.expand(3, -1, -1).contiguous()

        resized_height, resized_width = H, W

        if do_resize:
            resized_height, resized_width = smart_resize(
                H, W,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            if resized_height != H or resized_width != W:
                pre_resize = image
                image = self._resize_gpu(image, resized_height, resized_width)
                del pre_resize

        if do_rescale and do_normalize:
            # Fused in-place rescale + normalize
            image = self._normalize_gpu(image)
        elif do_rescale:
            image.mul_(self.rescale_factor)
        elif do_normalize:
            # Assume already in [0, 1] range
            image.sub_(self._mean)
            image.div_(self._std)

        # Add temporal dimension: (C, H, W) -> (1, C, H, W)
        patches = image.unsqueeze(0)
        del image

        return self._reshape_to_patches(patches, resized_height, resized_width)

    @torch.no_grad()
    def preprocess_batch(
        self,
        images: list,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> Tuple[torch.Tensor, list]:
        """Preprocess a batch of images on GPU.

        Args:
            images: List of tensors (C, H, W) or (H, W, C) on GPU or CPU.
            do_resize: Whether to apply smart_resize.
            do_rescale: Whether to rescale.
            do_normalize: Whether to normalize.

        Returns:
            (all_patches, grid_thws) where:
                all_patches: Concatenated flattened patches tensor on GPU.
                grid_thws: List of (grid_t, grid_h, grid_w) tuples per image.
        """
        all_patches = []
        grid_thws = []

        for image in images:
            patches, grid_thw = self.preprocess_single(
                image,
                do_resize=do_resize,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
            )
            all_patches.append(patches)
            grid_thws.append(grid_thw)

        # Concatenate all patches along the first dimension
        all_patches_cat = torch.cat(all_patches, dim=0)
        # Free individual patch tensors
        del all_patches

        return all_patches_cat, grid_thws

    @staticmethod
    def from_cpu_preprocessor(cpu_processor) -> "Qwen2VLGPUPreprocessor":
        """Create a GPU preprocessor from an existing HF Qwen2VLImageProcessor.

        Handles both Qwen2-VL (max_pixels/min_pixels as direct attrs) and
        Qwen3-VL (values stored in size["longest_edge"]/size["shortest_edge"]
        with max_pixels/min_pixels set to None).

        Args:
            cpu_processor: A Qwen2VLImageProcessor instance.

        Returns:
            Configured Qwen2VLGPUPreprocessor matching the CPU processor's settings.
        """
        # Qwen3-VL stores pixel limits in size dict, not as direct attributes.
        # In some transformers versions Qwen2VLImageProcessor.min_pixels/max_pixels
        # are removed entirely (raises AttributeError on direct access), so use
        # getattr so we cleanly fall through to the size dict.
        min_pixels = getattr(cpu_processor, "min_pixels", None)
        max_pixels = getattr(cpu_processor, "max_pixels", None)
        size = getattr(cpu_processor, "size", None)
        if min_pixels is None and size is not None:
            min_pixels = size.get("shortest_edge") if hasattr(size, "get") else getattr(size, "shortest_edge", None)
        if max_pixels is None and size is not None:
            max_pixels = size.get("longest_edge") if hasattr(size, "get") else getattr(size, "longest_edge", None)
        # Final fallback to Qwen2-VL defaults
        if min_pixels is None:
            min_pixels = 56 * 56
        if max_pixels is None:
            max_pixels = 14 * 14 * 4 * 1280

        return Qwen2VLGPUPreprocessor(
            image_mean=tuple(cpu_processor.image_mean),
            image_std=tuple(cpu_processor.image_std),
            rescale_factor=cpu_processor.rescale_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=cpu_processor.patch_size,
            temporal_patch_size=cpu_processor.temporal_patch_size,
            merge_size=cpu_processor.merge_size,
        )


# ---------------------------------------------------------------------------
# Convenience functions for integration with nvImageCodec decode pathway
# ---------------------------------------------------------------------------

@torch.no_grad()
def gpu_preprocess_from_decoded(
    decoded_tensor: torch.Tensor,
    preprocessor: Optional[Qwen2VLGPUPreprocessor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """One-shot GPU preprocessing from a decoded GPU tensor.

    Designed to be called directly after nvImageCodec GPU decode:
        decoded = nvimgcodec_decoder.decode(jpeg_bytes)  # GPU tensor
        patches, grid_thw = gpu_preprocess_from_decoded(decoded)

    Args:
        decoded_tensor: (C, H, W) or (H, W, C) GPU tensor from decoder.
        preprocessor: Optional pre-configured preprocessor. If None, creates
                      one with default Qwen2VL settings.
        **kwargs: Additional kwargs passed to Qwen2VLGPUPreprocessor.__init__.

    Returns:
        (flatten_patches, grid_thw) ready for Qwen2VL model input.
    """
    if preprocessor is None:
        preprocessor = Qwen2VLGPUPreprocessor(**kwargs)
    return preprocessor.preprocess_single(decoded_tensor)


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

def benchmark_gpu_vs_cpu(
    images_pil: list,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cuda",
) -> dict:
    """Benchmark GPU vs CPU preprocessing and return timing comparison.

    Args:
        images_pil: List of PIL Images to benchmark.
        num_warmup: Warm-up iterations before timing.
        num_runs: Number of timed iterations.
        device: CUDA device string.

    Returns:
        Dict with 'cpu_mean_ms', 'gpu_mean_ms', 'speedup', 'num_images'.
    """
    import time
    import numpy as np
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    cpu_proc = Qwen2VLImageProcessor()
    gpu_proc = Qwen2VLGPUPreprocessor.from_cpu_preprocessor(cpu_proc)

    # Prepare GPU tensors
    gpu_tensors = []
    for pil_img in images_pil:
        arr = np.array(pil_img.convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
        gpu_tensors.append(t)

    # CPU benchmark
    for _ in range(num_warmup):
        cpu_proc.preprocess(images=images_pil, return_tensors="pt")

    cpu_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cpu_proc.preprocess(images=images_pil, return_tensors="pt")
        cpu_times.append(time.perf_counter() - start)

    # GPU benchmark
    for _ in range(num_warmup):
        gpu_proc.preprocess_batch(gpu_tensors)
        torch.cuda.synchronize()

    gpu_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_proc.preprocess_batch(gpu_tensors)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)

    cpu_mean = np.mean(cpu_times) * 1000
    gpu_mean = np.mean(gpu_times) * 1000

    return {
        "cpu_mean_ms": round(cpu_mean, 2),
        "gpu_mean_ms": round(gpu_mean, 2),
        "speedup": round(cpu_mean / gpu_mean, 2) if gpu_mean > 0 else float("inf"),
        "num_images": len(images_pil),
        "num_runs": num_runs,
    }
