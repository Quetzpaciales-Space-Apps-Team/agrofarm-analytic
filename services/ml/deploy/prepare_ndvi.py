"""Verify the accepted limited-use NDVI release before building the image."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

FILES = {
    "p10.lgb.txt",
    "p50.lgb.txt",
    "p90.lgb.txt",
    "feature_ranges.json",
    "reference_cells.json",
    "model_card.json",
    "serving_card.json",
}


def prepare(source: Path, output: Path, manifest: dict[str, str]) -> None:
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("NDVI archive checksum mismatch")
    prefix = manifest["directory"]
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if (
            len(members) != len(FILES)
            or {m.name for m in members} != {f"{prefix}/{n}" for n in FILES}
            or not all(m.isfile() and m.size < 10 * 1024 * 1024 for m in members)
        ):
            raise ValueError("Unexpected NDVI archive members")
        data = {}
        for name in FILES:
            member = archive.extractfile(f"{prefix}/{name}")
            if member is None:
                raise ValueError("Missing NDVI file")
            data[name] = member.read()
        card = json.loads(data["serving_card.json"])
        if (
            card["model_version"] != manifest["model_version"]
            or card["owner_accepted_limited_use"] is not True
            or set(card["files_sha256"]) != FILES - {"serving_card.json"}
        ):
            raise ValueError("NDVI acceptance or version mismatch")
        for name, checksum in card["files_sha256"].items():
            if hashlib.sha256(data[name]).hexdigest() != checksum:
                raise ValueError("NDVI file checksum mismatch")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output, filter="data")  # noqa: S202 -- exact regular members checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.archive,
        args.out,
        json.loads(Path(__file__).with_name("ndvi-release.json").read_text()),
    )
