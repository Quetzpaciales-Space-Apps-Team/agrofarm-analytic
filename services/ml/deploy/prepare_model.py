"""Verify a pinned model release before copying its files into a deployment image."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def prepare(source: Path, output: Path, manifest: dict[str, str]) -> None:
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Model archive checksum mismatch")
    prefix = manifest["directory"]
    expected = {
        f"{prefix}/{name}" for name in ("model.lgb.txt", "model_card.json", "split_manifest.json")
    }
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if (
            {member.name for member in members} != expected
            or len(members) != 3
            or not all(member.isfile() and member.size < 10 * 1024 * 1024 for member in members)
        ):
            raise ValueError("Unexpected model archive members")
        card_file = archive.extractfile(f"{prefix}/model_card.json")
        weights_file = archive.extractfile(f"{prefix}/model.lgb.txt")
        if card_file is None or weights_file is None:
            raise ValueError("Missing model metadata")
        card = json.load(card_file)
        if (
            card.get("model_version") != manifest["model_version"]
            or card.get("feature_version") != manifest["feature_version"]
            or card.get("internal_gate_passed") is not True
        ):
            raise ValueError("Model version, features or evaluation gate mismatch")
        if hashlib.sha256(weights_file.read()).hexdigest() != card.get("model_sha256"):
            raise ValueError("Model weights checksum mismatch")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output, filter="data")  # noqa: S202 -- exact regular members checked above


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name("model-release.json")
    prepare(args.archive, args.out, json.loads(manifest_path.read_text()))
