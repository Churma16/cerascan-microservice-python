import io
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


class CeramicInferenceService:
    def __init__(self, model_file_path: str = None):
        base_directory = os.path.dirname(os.path.abspath(__file__))
        default_relative_path = os.path.join("models", "ceramic_defect_vgg19.h5")
        legacy_relative_path = "ceramic_tiles_4class_bs16_lr0.01_epoch50_do0.1.h5"

        if model_file_path is None:
            configured_path = os.getenv("MODEL_PATH", default_relative_path)
            resolved_configured_path = (
                os.path.join(base_directory, configured_path)
                if not os.path.isabs(configured_path)
                else configured_path
            )
            resolved_legacy_path = os.path.join(base_directory, legacy_relative_path)

            if not os.path.exists(resolved_configured_path) and os.path.exists(resolved_legacy_path):
                self.model_file_path = legacy_relative_path
            else:
                self.model_file_path = configured_path
        else:
            self.model_file_path = model_file_path
        self.class_labels = ["crack", "normal", "scratch", "stain"]
        self.model = self._load_model()

    def _load_model(self):
        resolved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_file_path)
        if not os.path.exists(resolved_path):
            print(f"[warn] Model weights file not found at: {resolved_path}")
            return None

        try:
            print("[info] Rebuilding Sequential architecture for VGG19...")
            base_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights=None)
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.1),
                Dense(4, activation='softmax')
            ])
            print(f"[info] Loading model weights from: {resolved_path}...")
            model.load_weights(resolved_path)
            print("[info] Model loaded successfully!")
            return model
        except Exception as load_exception:
            print(f"[error] Failed to load model weights: {load_exception}")
            return None

    def prepare_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        resized_image = image.resize((224, 224))
        normalized_array = np.array(resized_image) / 255.0
        return np.expand_dims(normalized_array, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not initialized or failed to load weights.")

        prepared_image_array = self.prepare_image(image_bytes)
        prediction_probabilities = self.model.predict(prepared_image_array, verbose=0)[0]

        predicted_class_index = int(np.argmax(prediction_probabilities))
        raw_confidence = float(prediction_probabilities[predicted_class_index])
        confidence_score_percentage = round(raw_confidence * 100, 2)
        predicted_label = self.class_labels[predicted_class_index]

        return {
            "prediction": predicted_label,
            "confidence_score": confidence_score_percentage
        }


# Singleton instance shared across the application
inference_service = CeramicInferenceService()
