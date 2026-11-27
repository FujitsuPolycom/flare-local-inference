# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Enforce the core-engine boundary between Flare (`ignition`, `helios`, `heliopause`) and `hyperflare_recipes`."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RETAINED_IGNITION_COMPONENTS = frozenset({
    "__init__",
    "capacity",
    "completion",
    "engram",
    "flashinfer_sampling",
    "logprobs",
    "ngram_delivery",
    "readiness",
    "sampling",
    "state",
    "state_transfer",
    "vision_delivery",
    "vision_state",
})


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_ignition_never_imports_recipes_or_hyperflare_recipes():
    assert not (ROOT / "hyperflare_recipes").exists()
    assert not (ROOT / "ignition" / "recipes").exists()
    violations = []
    for py in sorted((ROOT / "ignition").rglob("*.py")):
        for mod in sorted(_imported_modules(py)):
            if mod in ("ignition.recipes", "hyperflare_recipes") or mod.startswith(("ignition.recipes.", "hyperflare_recipes.")):
                violations.append(f"{py.relative_to(ROOT)} imports {mod}")
    assert not violations, "\n".join(violations)


def test_core_tests_do_not_import_extracted_recipes_or_qualify():
    violations = []
    for py in sorted((ROOT / "tests").rglob("*.py")):
        if py.name in ("test_core_boundary.py", "test_source_identity.py"):
            continue
        for mod in sorted(_imported_modules(py)):
            if mod in ("hyperflare_recipes", "qualify") or mod.startswith(("hyperflare_recipes.", "qualify.")):
                violations.append(f"{py.relative_to(ROOT)} imports {mod}")
    assert not violations, "\n".join(violations)


def test_ignition_components_keeps_only_helios_abi_modules():
    actual = {py.stem for py in (ROOT / "ignition/components").glob("*.py")}
    assert actual == RETAINED_IGNITION_COMPONENTS


def test_rust_core_and_heliopause_do_not_reference_recipe_paths():
    forbidden = ("ignition/recipes", "hyperflare_recipes.recipes", "hyperflare_recipes")
    violations = []
    scan_dirs = [ROOT / "helios/src", ROOT / "helios/tests", ROOT / "heliopause/src"]
    scan_files = [ROOT / "heliopause/Cargo.toml", ROOT / "heliopause/pyproject.toml"]
    for directory in scan_dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in (".rs", ".py", ".toml"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden:
                if term in text:
                    violations.append(f"{path.relative_to(ROOT)} mentions {term!r}")
    for path in scan_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)} mentions {term!r}")
    assert not violations, "\n".join(violations)
