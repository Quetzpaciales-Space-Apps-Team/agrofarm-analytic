"""Train a CPU LightGBM baseline from the versioned five-class maize-leaf ZIP.

Usage: python -m agrofarm_ml.leaves.training --archive DATA.zip --out MODEL_DIR.
No image or fitted model is checked into git. Keep file groups together before splitting.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from agrofarm_ml.leaves.model import CLASSES, FEATURE_VERSION, LIMITS, image_features, image_group

FOLDERS = {
    "Common Rust": "common_rust",
    "Corn Leaf Blight": "corn_leaf_blight",
    "Gray Leaf Spot": "gray_leaf_spot",
    "Healthy": "healthy",
    "Insects damage": "insect_damage",
}
# Fixed before the first training run, not a field-validation claim.
ACCEPTANCE_RULE = {
    "macro_f1_min": 0.75,
    "balanced_accuracy_min": 0.75,
    "each_class_recall_min": 0.5,
}


def original_name(name: str) -> str:
    stem = Path(name).stem
    # The archive has augmented/enhanced copies. They must stay with their source image.
    return re.sub(r"(?:_aug\d+|_enhanced|_original)+$", "", stem, flags=re.IGNORECASE)


def normalize_source_image(data: bytes) -> bytes:
    """Downsample the known training archive; public upload bounds remain unchanged."""
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("Training image exceeds 25 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 64_000_000:
            raise ValueError("Training image exceeds 64 million pixels")
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Training image must be JPEG or PNG")
        image.draft("RGB", (1024, 1024))
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.thumbnail((1024, 1024))
        output = io.BytesIO()
        rgb.save(output, format="PNG")
    return output.getvalue()


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def metrics(actual: NDArray[np.int64], predicted: NDArray[np.int64]) -> dict[str, Any]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(confusion, (actual, predicted), 1)
    tp = confusion.diagonal()
    recall = tp / np.maximum(confusion.sum(axis=1), 1)
    precision = tp / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "rows": int(len(actual)),
        "accuracy": float(tp.sum() / max(len(actual), 1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "recall_by_class": dict(zip(CLASSES, recall.tolist(), strict=True)),
        "confusion_matrix": confusion.tolist(),
        "class_order": list(CLASSES),
    }


def train(archive: Path, out: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(
            n for n in bundle.namelist() if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        unknown = sorted({Path(n).parts[0] for n in names} - set(FOLDERS))
        if unknown:
            raise ValueError(f"Unexpected class folders: {unknown}")

        def read(name: str) -> tuple[str, int, str, str, NDArray[np.float64]]:
            if bundle.getinfo(name).file_size > 25 * 1024 * 1024:
                raise ValueError("Training image exceeds 25 MiB")
            data = normalize_source_image(bundle.read(name))
            label = CLASSES.index(FOLDERS[Path(name).parts[0]])
            # Perceptual hashes also group identical/resized/flipped copies across names.
            return name, label, original_name(name), image_group(data), image_features(data)

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(read, names))

    # Union by source filename OR perceptual hash, before any split. Conflicting labels
    # within a group are excluded and counted instead of choosing a convenient label.
    parents: dict[str, str] = {}

    def root(value: str) -> str:
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for _, _, original, perceptual, _ in rows:
        first, second = root("name:" + original), root("hash:" + perceptual)
        if first != second:
            parents[max(first, second)] = min(first, second)
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for _, label, original, _, _ in rows:
        labels_by_group[root("name:" + original)].add(label)
    conflicts = {group for group, labels in labels_by_group.items() if len(labels) > 1}
    selected = [r for r in rows if root("name:" + r[2]) not in conflicts]
    groups = [root("name:" + r[2]) for r in selected]
    folds = np.array([split_for(group) for group in groups])
    matrix = np.stack([row[4] for row in selected])
    labels = np.array([row[1] for row in selected], dtype=np.int64)
    for split in ("train", "validation", "test"):
        if set(labels[folds == split]) != set(range(len(CLASSES))):
            raise ValueError(f"Every class needs independent groups in {split}")
    train_mask, validation_mask, test_mask = [
        folds == split for split in ("train", "validation", "test")
    ]
    model = lgb.train(
        {
            "objective": "multiclass",
            "num_class": len(CLASSES),
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "num_threads": 4,
            "seed": 0,
            "deterministic": True,
            "force_col_wise": True,
        },
        lgb.Dataset(matrix[train_mask], label=labels[train_mask]),
        num_boost_round=200,
        valid_sets=[lgb.Dataset(matrix[validation_mask], label=labels[validation_mask])],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    held_out = metrics(labels[test_mask], np.argmax(model.predict(matrix[test_mask]), axis=1))
    passed = (
        held_out["macro_f1"] >= ACCEPTANCE_RULE["macro_f1_min"]
        and held_out["balanced_accuracy"] >= ACCEPTANCE_RULE["balanced_accuracy_min"]
        and min(held_out["recall_by_class"].values()) >= ACCEPTANCE_RULE["each_class_recall_min"]
    )
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / "model.lgb.txt"
    model.save_model(str(model_file))
    manifest = [
        {"file": row[0], "class": CLASSES[row[1]], "group": group, "split": split_for(group)}
        for row, group in zip(selected, groups, strict=True)
    ]
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    card = {
        "model_version": "maize-leaf-lightgbm-0.1.1",
        "status": "candidate_not_field_validated",
        "algorithm": "LightGBM over RGB spatial, HSV histogram and texture features",
        "classes": list(CLASSES),
        "source_preprocessing": (
            "EXIF orientation, at most 1024 pixels per side, lossless PNG staging; "
            "public uploads retain the 5 MiB / 16 million pixel limit"
        ),
        "feature_version": FEATURE_VERSION,
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "data_versions": {"mendeley": "10.17632/hmkd6nbngr.1", "archive_sha256": archive_hash},
        "license": "Dataset declared CC BY 4.0; preserve third-party provenance caveats",
        "dataset_attribution": (
            "Md Shajedur Rahman, Corn Leaf Disease Classification Dataset "
            "(2026), DOI 10.17632/hmkd6nbngr.1"
        ),
        "source_rows_by_class": dict(Counter(CLASSES[row[1]] for row in rows)),
        "rows_by_split": dict(Counter(folds.tolist())),
        "groups_by_split": dict(Counter(split_for(group) for group in set(groups))),
        "excluded_conflicting_groups": len(conflicts),
        "excluded_rows": len(rows) - len(selected),
        "split": (
            "Deterministic 70/15/15 by unions of original filename and "
            "perceptual image hash; augmentation suffixes stripped before splitting"
        ),
        "validation_limits": [
            (
                "No verified physical leaf IDs or acquisition-session metadata; "
                "near-duplicates may remain."
            ),
            (
                "Same-dataset held-out evaluation, not external field validation "
                "or diagnostic probability calibration."
            ),
        ],
        "acceptance_rule_fixed_before_training": ACCEPTANCE_RULE,
        "internal_gate_passed": passed,
        "test_metrics": held_out,
        "limits": LIMITS,
        "generated_by": (
            "Training code assisted by OpenAI Codex (GPT-6); source images "
            "not AI-generated by this project"
        ),
        "reviewed_by": None,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    card = train(args.archive, args.out)
    sys.stdout.write(
        json.dumps(
            {
                "test_metrics": card["test_metrics"],
                "internal_gate_passed": card["internal_gate_passed"],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
