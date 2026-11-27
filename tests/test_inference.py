import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from inference_service import CeramicInferenceService


@pytest.fixture
def sample_image_bytes():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format="JPEG")
    return byte_buffer.getvalue()


@pytest.fixture
def sample_rgba_image_bytes():
    image = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format="PNG")
    return byte_buffer.getvalue()


def test_prepare_image_shape_and_range(sample_image_bytes):
    service = CeramicInferenceService(model_file_path="dummy_path.h5")
    image_array = service.prepare_image(sample_image_bytes)

    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (1, 224, 224, 3)
    assert 0.0 <= image_array.min() <= image_array.max() <= 1.0


def test_prepare_image_converts_rgba_to_rgb(sample_rgba_image_bytes):
    service = CeramicInferenceService(model_file_path="dummy_path.h5")
    image_array = service.prepare_image(sample_rgba_image_bytes)

    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (1, 224, 224, 3)


def test_predict_format_with_mocked_model(sample_image_bytes):
    service = CeramicInferenceService(model_file_path="dummy_path.h5")
    mock_model = MagicMock()
    # Mocking probabilities for 4 classes: ["crack", "normal", "scratch", "stain"]
    # Predict "normal" (index 1) with 0.85 confidence
    mock_model.predict.return_value = np.array([[0.05, 0.85, 0.08, 0.02]])
    service.model = mock_model

    result = service.predict(sample_image_bytes)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence_score" in result
    assert result["prediction"] == "normal"
    assert result["confidence_score"] == 85.0
    mock_model.predict.assert_called_once()


def test_predict_raises_runtime_error_when_model_is_none(sample_image_bytes):
    service = CeramicInferenceService(model_file_path="dummy_path.h5")
    service.model = None

    with pytest.raises(RuntimeError) as exc_info:
        service.predict(sample_image_bytes)

    assert "Model is not initialized" in str(exc_info.value)
