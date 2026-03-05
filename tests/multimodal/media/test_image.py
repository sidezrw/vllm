# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.image import IMAGE_LOADER_REGISTRY, ImageLoader
from vllm.multimodal.media import ImageMediaIO

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


def test_image_media_io_rgba_custom_background(tmp_path):
    """Test RGBA to RGB conversion with custom background colors."""
    # Create a simple RGBA image with transparent and opaque pixels
    rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # Red with full opacity

    # Make top-left quadrant transparent
    for i in range(5):
        for j in range(5):
            rgba_image.putpixel((i, j), (0, 0, 0, 0))  # Fully transparent

    # Save the test image to tmp_path
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Test 1: Default white background (backward compatibility)
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    default_numpy = np.array(converted_default)

    # Check transparent pixels are white
    assert default_numpy[0][0][0] == 255  # R
    assert default_numpy[0][0][1] == 255  # G
    assert default_numpy[0][0][2] == 255  # B
    # Check opaque pixels remain red
    assert default_numpy[5][5][0] == 255  # R
    assert default_numpy[5][5][1] == 0  # G
    assert default_numpy[5][5][2] == 0  # B

    # Test 2: Custom black background via kwargs
    image_io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
    converted_black = image_io_black.load_file(test_image_path)
    black_numpy = np.array(converted_black)

    # Check transparent pixels are black
    assert black_numpy[0][0][0] == 0  # R
    assert black_numpy[0][0][1] == 0  # G
    assert black_numpy[0][0][2] == 0  # B
    # Check opaque pixels remain red
    assert black_numpy[5][5][0] == 255  # R
    assert black_numpy[5][5][1] == 0  # G
    assert black_numpy[5][5][2] == 0  # B

    # Test 3: Custom blue background via kwargs (as list)
    image_io_blue = ImageMediaIO(rgba_background_color=[0, 0, 255])
    converted_blue = image_io_blue.load_file(test_image_path)
    blue_numpy = np.array(converted_blue)

    # Check transparent pixels are blue
    assert blue_numpy[0][0][0] == 0  # R
    assert blue_numpy[0][0][1] == 0  # G
    assert blue_numpy[0][0][2] == 255  # B

    # Test 4: Test with load_bytes method
    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))
    converted_green = image_io_green.load_bytes(image_data)
    green_numpy = np.array(converted_green)

    # Check transparent pixels are green
    assert green_numpy[0][0][0] == 0  # R
    assert green_numpy[0][0][1] == 255  # G
    assert green_numpy[0][0][2] == 0  # B


def test_image_media_io_rgba_background_color_validation():
    """Test that invalid rgba_background_color values are properly rejected."""

    # Test invalid types
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color="255,255,255")

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=255)

    # Test wrong number of elements
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255, 255, 255))

    # Test non-integer values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255.0, 255.0, 255.0))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, "255", 255))

    # Test out of range values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(256, 255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, -1, 255))

    # Test that valid values work
    ImageMediaIO(rgba_background_color=(0, 0, 0))  # Should not raise
    ImageMediaIO(rgba_background_color=[255, 255, 255])  # Should not raise
    ImageMediaIO(rgba_background_color=(128, 128, 128))  # Should not raise


FAKE_IMAGE_1 = Image.fromarray(np.full((8, 8, 3), 10, dtype=np.uint8))
FAKE_IMAGE_2 = Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8))


@IMAGE_LOADER_REGISTRY.register("test_image_backend_override_1")
class TestImageBackendOverride1(ImageLoader):
    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image:
        return FAKE_IMAGE_1.copy()


@IMAGE_LOADER_REGISTRY.register("test_image_backend_override_2")
class TestImageBackendOverride2(ImageLoader):
    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> Image.Image:
        return FAKE_IMAGE_2.copy()


@IMAGE_LOADER_REGISTRY.register("test_image_backend_ndarray")
class TestImageBackendNdarray(ImageLoader):
    @classmethod
    def load_bytes(cls, data: bytes, **kwargs) -> np.ndarray:
        return np.asarray(FAKE_IMAGE_2)


def test_image_media_io_backend_kwarg_override(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_IMAGE_LOADER_BACKEND", "test_image_backend_override_1")

        imageio_default = ImageMediaIO()
        image_default = imageio_default.load_bytes(b"test")
        np.testing.assert_array_equal(np.asarray(image_default), np.asarray(FAKE_IMAGE_1))

        imageio_override = ImageMediaIO(image_backend="test_image_backend_override_2")
        image_override = imageio_override.load_bytes(b"test")
        np.testing.assert_array_equal(np.asarray(image_override), np.asarray(FAKE_IMAGE_2))


def test_image_media_io_backend_kwarg_not_passed_to_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    @IMAGE_LOADER_REGISTRY.register("test_reject_image_backend_kwarg")
    class RejectImageBackendKwargLoader(ImageLoader):
        @classmethod
        def load_bytes(cls, data: bytes, **kwargs) -> Image.Image:
            if "image_backend" in kwargs:
                raise AssertionError(
                    "image_backend should be consumed by ImageMediaIO, "
                    "not passed to loader"
                )
            if kwargs.get("other_kwarg") != "should_pass_through":
                raise AssertionError("Expected other_kwarg to pass through")
            return FAKE_IMAGE_1.copy()

    with monkeypatch.context() as m:
        m.setenv("VLLM_IMAGE_LOADER_BACKEND", "test_reject_image_backend_kwarg")
        imageio = ImageMediaIO(
            image_backend="test_reject_image_backend_kwarg",
            other_kwarg="should_pass_through",
        )
        image = imageio.load_bytes(b"test")
        np.testing.assert_array_equal(np.asarray(image), np.asarray(FAKE_IMAGE_1))


def test_image_media_io_backend_env_var_fallback(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_IMAGE_LOADER_BACKEND", "test_image_backend_override_2")

        imageio_none = ImageMediaIO(image_backend=None)
        image_none = imageio_none.load_bytes(b"test")
        np.testing.assert_array_equal(np.asarray(image_none), np.asarray(FAKE_IMAGE_2))

        imageio_missing = ImageMediaIO()
        image_missing = imageio_missing.load_bytes(b"test")
        np.testing.assert_array_equal(np.asarray(image_missing), np.asarray(FAKE_IMAGE_2))


def test_image_media_io_converts_ndarray_loader_output():
    imageio = ImageMediaIO(image_backend="test_image_backend_ndarray")
    image = imageio.load_bytes(b"test")
    assert isinstance(image.media, Image.Image)
    np.testing.assert_array_equal(np.asarray(image), np.asarray(FAKE_IMAGE_2))


def test_image_media_io_default_backend_is_pil(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with monkeypatch.context() as m:
        m.delenv("VLLM_IMAGE_LOADER_BACKEND", raising=False)

        raw = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8
        )
        image = Image.fromarray(raw, mode="RGB")
        image_path = tmp_path / "test_default_pil.png"
        image.save(image_path)
        image_data = image_path.read_bytes()

        imageio = ImageMediaIO()
        decoded = imageio.load_bytes(image_data)
        np.testing.assert_array_equal(np.asarray(decoded), raw)


def test_image_media_io_default_backend_is_nvimagecodec_if_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pytest.importorskip("nvidia.nvimgcodec")

    with monkeypatch.context() as m:
        m.setenv("VLLM_IMAGE_LOADER_BACKEND", "nvimagecodec")

        raw = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8
        )
        image = Image.fromarray(raw, mode="RGB")
        image_path = tmp_path / "test_default_nvimagecodec.png"
        image.save(image_path)
        image_data = image_path.read_bytes()

        imageio = ImageMediaIO()
        decoded = imageio.load_bytes(image_data)
        np.testing.assert_array_equal(np.asarray(decoded), raw)
