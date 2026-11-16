"""AGENTS.md section 6: the two services in this repo never import each other."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
OTHER_SERVICE = "agrofarm_data"


def find_cross_imports(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            found += [
                f"{path.name} imports {name}"
                for name in names
                if name == OTHER_SERVICE or name.startswith(OTHER_SERVICE + ".")
            ]
    return found


def test_this_service_does_not_import_the_other_service() -> None:
    assert find_cross_imports(SRC) == []


def test_canary_a_cross_import_is_caught(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(f"from {OTHER_SERVICE}.x import y\n", encoding="utf-8")

    assert find_cross_imports(tmp_path) == [f"bad.py imports {OTHER_SERVICE}.x"]
