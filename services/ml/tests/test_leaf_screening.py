"""Image screening never fabricates a model or a species-specific pest diagnosis."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agrofarm_ml.leaves.model import (
    CLASSES,
    FEATURE_VERSION,
    LeafModel,
    image_features,
    image_group,
)
from agrofarm_ml.leaves.training import (
    FOLDERS,
    metrics,
    normalize_source_image,
    original_name,
    split_for,
    train,
)
from agrofarm_ml.main import create_app


def image_bytes(color: str = "green", size: tuple[int, int] = (64, 64)) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return data.getvalue()


def test_image_features_are_finite_deterministic_and_reject_non_images() -> None:
    image = image_bytes()
    first = image_features(image)
    assert np.isfinite(first).all()
    assert np.array_equal(first, image_features(image))
    assert not np.array_equal(first, image_features(image_bytes("brown")))
    with pytest.raises(ValueError, match="image"):
        image_features(b"not an image")


def test_unconfigured_model_returns_503_without_a_fake_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEAF_MODEL_DIR", raising=False)
    client = TestClient(create_app())
    response = client.post(
        "/v1/leaves/assess", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 503
    assert "scores" not in response.json()


def test_endpoint_rejects_wrong_type_and_oversized_body() -> None:
    client = TestClient(create_app())
    assert client.post("/v1/leaves/assess", content=b"text").status_code == 415
    response = client.post(
        "/v1/leaves/assess",
        content=b"0" * (5 * 1024 * 1024 + 1),
        headers={"Content-Type": "image/jpeg"},
    )
    assert response.status_code == 413


class Scores:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.array([self.scores])


def test_generic_insect_damage_has_no_species_or_pesticide_prescription() -> None:
    model = LeafModel(
        Scores([0.01, 0.01, 0.01, 0.01, 0.96]), {"model_version": "test-only", "data_versions": {}}
    )
    result = model.assess(image_bytes())
    assert result["candidate"] == "insect_damage"
    assert result["needs_local_review"] is True
    assert "not calibrated" in result["score_interpretation"]
    assert result["guidance"]["status"] == "needs_local_review"
    assert "species" in " ".join(result["limits"])
    assert "pesticide" not in result["guidance"]


def test_ambiguous_prediction_returns_uncertain_instead_of_diagnosis() -> None:
    model = LeafModel(Scores([0.3, 0.25, 0.2, 0.15, 0.1]), {"model_version": "test-only"})
    result = model.assess(image_bytes())
    assert result["assessment"] == "uncertain"
    assert result["candidate"] is None
    assert set(result["scores"]) == set(CLASSES)


def test_loading_requires_artifact_and_compatible_class_order(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        LeafModel.load(tmp_path)


def test_augmentation_copies_keep_the_source_name_and_fold() -> None:
    original = original_name("Gray Leaf Spot/Corn_Gray_Spot (25).jpg")
    assert original_name("Gray Leaf Spot/Corn_Gray_Spot (25)_aug1.jpg") == original
    assert original_name("Gray Leaf Spot/Corn_Gray_Spot (25)_enhanced.jpg") == original
    assert split_for(original) == split_for(original)
    assert split_for(original) in {"train", "validation", "test"}


def test_metrics_expose_each_class_and_do_not_hide_a_failed_class() -> None:
    actual = np.array([0, 1, 2, 3, 4])
    perfect = metrics(actual, actual)
    assert perfect["macro_f1"] == perfect["balanced_accuracy"] == 1
    broken = metrics(actual, np.array([0, 1, 2, 3, 3]))
    assert broken["recall_by_class"]["insect_damage"] == 0
    assert broken["macro_f1"] < 1


def test_perceptual_group_is_invariant_to_flipped_image() -> None:
    data = io.BytesIO()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    image.save(data, format="PNG")
    flipped = io.BytesIO()
    image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flipped, format="PNG")
    assert image_group(data.getvalue()) == image_group(flipped.getvalue())


def test_failed_gate_artifact_cannot_be_loaded(tmp_path: Path) -> None:
    (tmp_path / "model_card.json").write_text(
        json.dumps(
            {
                "classes": list(CLASSES),
                "feature_version": FEATURE_VERSION,
                "internal_gate_passed": False,
            }
        )
    )
    with pytest.raises(ValueError, match="evaluation gate"):
        LeafModel.load(tmp_path)


def test_training_round_trip_keeps_augmentations_out_of_other_splits(tmp_path: Path) -> None:
    archive = tmp_path / "tiny.zip"
    rng = np.random.default_rng(0)
    colors = [(210, 40, 30), (50, 210, 40), (50, 40, 210), (200, 200, 30), (190, 30, 190)]
    with zipfile.ZipFile(archive, "w") as bundle:
        for label, folder in enumerate(FOLDERS):
            for index in range(40):
                # Spatial patches survive JPEG normalization and represent distinct leaves;
                # pixel-level noise alone is deliberately collapsed by perceptual grouping.
                texture = rng.integers(-25, 25, (8, 8, 1)).repeat(4, 0).repeat(4, 1)
                pixels = np.clip(np.array(colors[label]) + texture, 0, 255).astype(np.uint8)
                data = io.BytesIO()
                Image.fromarray(pixels).save(data, format="PNG")
                name = f"{folder}/class{label}_leaf{index}"
                bundle.writestr(name + ".png", data.getvalue())
                if index == 0:
                    bundle.writestr(name + "_aug1.png", data.getvalue())
    out = tmp_path / "model"
    card = train(archive, out)
    assert card["source_rows_by_class"] == dict.fromkeys(CLASSES, 41)
    assert card["internal_gate_passed"] is True
    manifest = json.loads((out / "split_manifest.json").read_text())
    by_group: dict[str, set[str]] = {}
    for row in manifest:
        by_group.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    loaded = LeafModel.load(out)
    assert set(loaded.assess(image_bytes())["scores"]) == set(CLASSES)
    (out / "model.lgb.txt").write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        LeafModel.load(out)


def test_configured_endpoint_handles_images_and_rejects_corrupt_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LeafModel(Scores([0.96, 0.01, 0.01, 0.01, 0.01]), {"model_version": "test-only"})
    monkeypatch.setenv("LEAF_MODEL_DIR", "/test-artifact")
    monkeypatch.setattr(LeafModel, "load", lambda _: model)
    client = TestClient(create_app())
    loaded = client.get("/v1/models").json()["items"]
    assert any(item["version"] == "test-only" and item["status"] == "trained" for item in loaded)
    result = client.post(
        "/v1/leaves/assess", content=image_bytes(), headers={"Content-Type": "image/png"}
    )
    assert result.status_code == 200
    assert result.json()["candidate"] == "common_rust"
    assert (
        client.post(
            "/v1/leaves/assess", content=b"invalid", headers={"Content-Type": "image/png"}
        ).status_code
        == 400
    )
    oversized = image_bytes(size=(4001, 4001))
    assert (
        client.post(
            "/v1/leaves/assess", content=oversized, headers={"Content-Type": "image/png"}
        ).status_code
        == 413
    )


@pytest.mark.parametrize("scores", [[1, 0], [float("nan")] * 5, [2, 0, 0, 0, 0]])
def test_invalid_model_scores_are_rejected(scores: list[float]) -> None:
    model = LeafModel(Scores(scores), {"model_version": "test-only"})
    with pytest.raises(ValueError, match="Invalid leaf model"):
        model.assess(image_bytes())


def test_training_normalizes_large_source_images_without_relaxing_upload_limit() -> None:
    rng = np.random.default_rng(2)
    data = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (1536, 1536, 3), dtype=np.uint8)).save(data, format="PNG")
    raw = data.getvalue()
    assert len(raw) > 5 * 1024 * 1024
    with pytest.raises(OverflowError):
        image_features(raw)
    normalized = normalize_source_image(raw)
    assert len(normalized) < 5 * 1024 * 1024
    assert np.isfinite(image_features(normalized)).all()


def test_training_and_inference_use_identical_image_features() -> None:
    rng = np.random.default_rng(5)
    data = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (384, 512, 3), dtype=np.uint8)).save(data, format="PNG")
    raw = data.getvalue()
    assert np.array_equal(image_features(raw), image_features(normalize_source_image(raw)))
