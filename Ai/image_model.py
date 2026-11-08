"""Research-only hemoglobin estimation from conjunctival images.

The serialized model is a transparent linear regression trained on the eye
datasets supplied with this project. It is not a clinical diagnostic device.
"""

from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_PATH = Path(__file__).parent / "artifacts" / "hemoglobin_model.json"
MAX_IMAGE_PIXELS = 20_000_000


class ImageModelError(ValueError):
    """Raised when a scan cannot be safely scored by the research model."""


class ModelUnavailableError(ImageModelError):
    """Raised when the trained model artifact is missing or incomplete."""


@lru_cache(maxsize=1)
def load_models() -> dict[str, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            "The research model is not installed. Run Ai/train_image_model.py first."
        )
    with MODEL_PATH.open("r", encoding="utf-8") as model_file:
        artifact = json.load(model_file)
    models = artifact.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelUnavailableError("The saved research model is incomplete.")
    return models


def _region_features(rgb: np.ndarray, cohort: str) -> list[float]:
    height, width, _ = rgb.shape
    if cohort == "adult":
        regions = [
            rgb,
            rgb[int(height * 0.08):int(height * 0.92), int(width * 0.35):],
            rgb[int(height * 0.15):int(height * 0.85), int(width * 0.15):int(width * 0.85)],
        ]
    else:
        regions = [
            rgb,
            rgb[int(height * 0.12):int(height * 0.88), int(width * 0.12):int(width * 0.88)],
        ]

    values: list[float] = []
    for region in regions:
        if region.size == 0:
            raise ImageModelError("The scan is too small for analysis.")
        channels = [region[:, :, i] for i in range(3)]
        means = [float(channel.mean()) for channel in channels]
        deviations = [float(channel.std()) for channel in channels]
        values.extend(means)
        values.extend(deviations)
        values.extend([
            means[0] - means[1],
            means[0] - means[2],
            means[0] / max(sum(means), 1e-6),
            means[1] / max(sum(means), 1e-6),
        ])
    return values


def image_features(image_bytes: bytes, cohort: str) -> list[float]:
    if cohort not in {"adult", "child"}:
        raise ImageModelError("This age group is outside the model's supported cohorts.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageModelError("Please upload a JPG, PNG, or WEBP eye image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ImageModelError("The uploaded image is too large to process.")
            if source.width < 160 or source.height < 120:
                raise ImageModelError("Use a clear eye photo at least 160 by 120 pixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((128, 96), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ImageModelError):
            raise
        raise ImageModelError("Please upload a valid JPG, PNG, or WEBP eye image.") from None
    return _region_features(rgb, cohort)


def estimate_hemoglobin(
    image_bytes: bytes,
    *,
    cohort: str,
    age_years: float,
    sex: str,
) -> dict[str, float | str | int | list[float]]:
    models = load_models()
    model = models.get(cohort)
    if model is None:
        raise ImageModelError("There is no image model for this age group.")

    features = image_features(image_bytes, cohort)
    age_feature = age_years * 12.0 if cohort == "child" else age_years
    sex_feature = 1.0 if sex == "male" else 0.0 if sex == "female" else 0.5
    features.extend([age_feature, sex_feature])
    means = model["feature_means"]
    scales = model["feature_scales"]
    coefficients = model["coefficients"]
    if len(features) != len(coefficients):
        raise ImageModelError("The model feature schema does not match this app version.")

    standardized = [
        (value - center) / scale if scale > 1e-12 else 0.0
        for value, center, scale in zip(features, means, scales)
    ]
    estimated_hb = float(model["intercept"]) + sum(
        value * coefficient
        for value, coefficient in zip(standardized, coefficients)
    )
    if not math.isfinite(estimated_hb):
        raise ImageModelError("The scan could not be scored.")
    return {
        "estimated_hb_g_dl": round(estimated_hb, 1),
        "heldout_mae_g_dl": float(model["heldout_mae_g_dl"]),
        "heldout_test_images": int(model["test_images"]),
        "heldout_residuals_g_dl": [float(value) for value in model["heldout_residuals_g_dl"]],
        "cohort": cohort,
        "training_images": int(model["training_images"]),
    }

