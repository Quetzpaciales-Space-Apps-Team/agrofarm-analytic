"""Small CPU classifier over color, spatial and texture features.

This baseline uses no generative diagnosis and supplies no pesticide prescription.
Scores are uncalibrated. A confident result is still a candidate requiring review.
"""

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES = ("common_rust", "corn_leaf_blight", "gray_leaf_spot", "healthy", "insect_damage")
FEATURE_VERSION = "rgb-hsv-texture-1"
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
SOURCE = "https://data.mendeley.com/datasets/hmkd6nbngr/1"
LIMITS = [
    "Candidate image classification, not a confirmed diagnosis or a measure of crop growth.",
    "Only the five listed classes are supported; other plants and disorders may be misclassified.",
    "A healthy label does not rule out disease or pests elsewhere on the plant or field.",
    "Generic insect damage does not identify the insect species or prove fall armyworm.",
    "No external field validation in Guatemala; background and lighting can affect the result.",
    "Treatment choice requires local diagnosis, crop stage and a reviewed management guide.",
]
GUIDE = "https://extension.umn.edu/agriculture/crop-production/corn/"
SOURCES = {
    "common_rust": GUIDE + "common-rust-on-corn",
    "gray_leaf_spot": GUIDE + "gray-leaf-spot-on-corn",
    "corn_leaf_blight": "https://www.vegetables.cornell.edu/pest-management/disease-factsheets/"
    "sweet-corn-diseases-and-control-measures/",
    "insect_damage": "https://www.fao.org/fileadmin/templates/fcc/Fall_Armyworm/"
    "web_guidance-note-2.pdf",
    "healthy": "https://extension.umn.edu/corn/corn-pest-management",
}


class Predictor(Protocol):
    def predict(self, data: NDArray[np.float64]) -> Any: ...


def decoded_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise OverflowError("Image exceeds the 5 MiB limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported")
            if image.width * image.height > MAX_PIXELS:
                raise OverflowError("Image exceeds the decoded pixel limit")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or corrupt image") from exc


def image_features(data: bytes) -> NDArray[np.float64]:
    image = decoded_image(data).resize((64, 64))
    spatial = np.asarray(image.resize((16, 16)), dtype=float).ravel() / 255
    hsv = np.asarray(image.convert("HSV"), dtype=float)
    histograms = [
        np.histogram(hsv[:, :, channel], bins=16, range=(0, 256))[0] / 4096 for channel in range(3)
    ]
    gray = np.asarray(image.convert("L"), dtype=float) / 255
    texture = np.array([np.abs(np.diff(gray, axis=axis)).mean() for axis in (0, 1)])
    return np.concatenate([spatial, *histograms, texture])


def image_group(data: bytes) -> str:
    """Group perceptually identical/resized/flipped images; not a verified physical leaf ID."""
    image = decoded_image(data).convert("L").resize((8, 8))
    variants = [image, ImageOps.mirror(image), ImageOps.flip(image), image.rotate(180)]
    hashes = []
    for variant in variants:
        pixels = np.asarray(variant, dtype=float)
        hashes.append(np.packbits(pixels > pixels.mean()).tobytes())
    return hashlib.sha256(min(hashes)).hexdigest()


class LeafModel:
    def __init__(self, predictor: Predictor, card: dict[str, Any]) -> None:
        self.predictor = predictor
        self.card = card

    @classmethod
    def load(cls, directory: Path) -> "LeafModel":
        card = json.loads((directory / "model_card.json").read_text())
        if (
            tuple(card.get("classes", [])) != CLASSES
            or card.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError("Incompatible leaf model class order or feature version")
        if card.get("internal_gate_passed") is not True:
            raise ValueError("Leaf model did not pass the internal evaluation gate")
        model_file = directory / "model.lgb.txt"
        if hashlib.sha256(model_file.read_bytes()).hexdigest() != card.get("model_sha256"):
            raise ValueError("Leaf model checksum mismatch")
        return cls(lgb.Booster(model_file=str(model_file)), card)

    def assess(self, data: bytes) -> dict[str, Any]:
        features = image_features(data)
        scores = np.asarray(self.predictor.predict(features.reshape(1, -1)))[0]
        if scores.shape != (len(CLASSES),) or not np.isfinite(scores).all():
            raise ValueError("Invalid leaf model output")
        if (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=1e-5):
            raise ValueError("Invalid leaf model scores")
        order = np.argsort(scores)
        best = CLASSES[int(order[-1])]
        uncertain = scores[order[-1]] < 0.8 or scores[order[-1]] - scores[order[-2]] < 0.2
        return {
            "assessment": "uncertain" if uncertain else "possible_condition",
            "candidate": None if uncertain else best,
            "scores": {
                label: round(float(score), 6) for label, score in zip(CLASSES, scores, strict=True)
            },
            "score_interpretation": (
                "Model scores are not calibrated probabilities or diagnostic certainty."
            ),
            "needs_local_review": True,
            "guidance": {
                "status": "needs_local_review",
                "next_steps": [
                    "Inspect several plants and photograph the whole plant and both leaf surfaces.",
                    (
                        "Record crop stage and the spread of symptoms; seek local "
                        "agronomy confirmation."
                    ),
                    (
                        "Discuss management only after diagnosis; an image score does not "
                        "select a treatment."
                    ),
                ],
                "sources": []
                if uncertain
                else [
                    {
                        "url": SOURCES[best],
                        "scope": (
                            "Reference for local review, not a Guatemala treatment prescription"
                        ),
                    }
                ],
            },
            "limits": LIMITS,
            "provenance": {
                "model_version": self.card["model_version"],
                "feature_version": FEATURE_VERSION,
                "data_versions": self.card.get("data_versions", {}),
                "validation_status": "candidate_not_field_validated",
                "dataset": SOURCE,
            },
        }
