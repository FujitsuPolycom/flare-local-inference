#!/usr/bin/env python3
"""Set or check versions for Flare's independently packaged release crates."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tomllib

CRATES = (
    (Path("flare-cli"), "flare-cli"),
    (Path("flare-winui/bridge"), "flare-winui-bridge"),
    (Path("flare-winui/launcher"), "flare-launcher"),
)
STABLE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
PACKAGE_SECTION = re.compile(r"(?ms)^\[package\]\n(.*?)(?=^\[|\Z)")
MANIFEST_VERSION = re.compile(r'(?m)^version = "[^"]+"')


def manifest(path: Path, package: str) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    document = tomllib.loads(text)
    if document["package"]["name"] != package:
        raise ValueError(f"{path} does not define {package}")
    return text, document["package"]["version"]


def lock_version(path: Path, package: str) -> str:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    versions = [entry["version"] for entry in document["package"] if entry["name"] == package]
    if len(versions) != 1:
        raise ValueError(f"{path} has {len(versions)} entries for {package}; expected one")
    return versions[0]


def replace_manifest_version(path: Path, package: str, version: str) -> tuple[str, str, str]:
    text, current = manifest(path, package)
    section = PACKAGE_SECTION.search(text)
    if section is None:
        raise ValueError(f"{path} has no package section")
    field = MANIFEST_VERSION.search(section.group(1))
    if field is None:
        raise ValueError(f"{path} has no package version field")
    start = section.start(1) + field.start()
    end = section.start(1) + field.end()
    return text[:start] + f'version = "{version}"' + text[end:], current, lock_version(
        path.with_name("Cargo.lock"), package
    )


def check(root: Path, version: str) -> list[str]:
    errors = []
    for directory, package in CRATES:
        manifest_path = root / directory / "Cargo.toml"
        _, actual_manifest = manifest(manifest_path, package)
        actual_lock = lock_version(manifest_path.with_name("Cargo.lock"), package)
        if actual_manifest != version:
            errors.append(f"{manifest_path}: version is {actual_manifest}; expected {version}")
        if actual_lock != version:
            errors.append(
                f"{manifest_path.with_name('Cargo.lock')}: {package} is {actual_lock}; expected {version}"
            )
    return errors


def set_version(root: Path, version: str) -> None:
    updates = []
    for directory, package in CRATES:
        path = root / directory / "Cargo.toml"
        text, current, current_lock = replace_manifest_version(path, package, version)
        updates.append((path, text, current, current_lock))

    for path, text, _, _ in updates:
        path.write_text(text, encoding="utf-8")

    for (directory, package), (_, _, current, current_lock) in zip(CRATES, updates):
        path = root / directory / "Cargo.toml"
        if current_lock != version:
            subprocess.run(
                [
                    "cargo",
                    "update",
                    "--manifest-path",
                    str(path),
                    "--package",
                    package,
                    "--precise",
                    version,
                ],
                cwd=root,
                check=True,
            )
        print(f"{path.relative_to(root)}: {current} -> {version}")

    errors = check(root, version)
    if errors:
        raise ValueError("\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="stable MAJOR.MINOR.PATCH version")
    parser.add_argument("--check", action="store_true", help="check without changing files")
    args = parser.parse_args()
    if not STABLE_VERSION.fullmatch(args.version):
        parser.error("version must be a stable MAJOR.MINOR.PATCH version")

    root = Path(__file__).resolve().parent.parent
    try:
        if args.check:
            errors = check(root, args.version)
            if errors:
                raise ValueError("\n".join(errors))
            print(f"Release manifests and lockfiles are {args.version}.")
        else:
            set_version(root, args.version)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
