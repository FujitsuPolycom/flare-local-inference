# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolve checkpoint aliases, Hugging Face repositories, and local directories against explicit recipes.

A recipe serves any checkpoint of its architecture. Its declared checkpoints are
the pinned defaults that aliases name; a local directory or another Hugging Face
repository replaces them for one invocation.
"""

from collections.abc import Sequence
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, replace
from difflib import get_close_matches
import hashlib
import json
from pathlib import Path

from ignition.declaration import Checkpoint, Recipe
from ignition.errors import FetchError, UsageError

_ACTIVE_RECIPES: contextvars.ContextVar[tuple[Recipe, ...]] = contextvars.ContextVar(
    "flare_active_recipes", default=()
)


def current_recipes() -> tuple[Recipe, ...]:
    """Return the recipes active in the current CLI invocation context, or an empty tuple."""
    return _ACTIVE_RECIPES.get()


@contextmanager
def active_recipes(recipes: Sequence[Recipe]):
    """Bind `recipes` for the duration of a CLI invocation or test scope."""
    token = _ACTIVE_RECIPES.set(tuple(recipes))
    try:
        yield
    finally:
        _ACTIVE_RECIPES.reset(token)


def snapshot_directory(hub_cache: Path, checkpoint: Checkpoint) -> Path:
    """Resolve a checkpoint's local source or pinned hub-cache directory."""
    if Path(checkpoint.repository).is_absolute():
        return Path(checkpoint.repository)
    hub_cache = Path(hub_cache)
    name = f"models--{checkpoint.repository.replace('/', '--')}"
    exact = hub_cache / name / "snapshots" / checkpoint.revision
    if _snapshot_cached(exact) or not hub_cache.is_dir():
        return exact
    for repository in sorted(hub_cache.iterdir()):
        if repository.name != name and repository.name.casefold() == name.casefold():
            snapshot = repository / "snapshots" / checkpoint.revision
            if _snapshot_cached(snapshot):
                return snapshot
    return exact


def _csf_container(snapshot: Path):
    path = snapshot / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) and value.get("schema") == "lil-mxfp4-csf-checkpoint/1" else None


def _csf_stored_bytes(container, snapshot):
    stored = container.get("compressed_weight_file_bytes")
    if type(stored) is not int or stored <= 0:
        raise UsageError(f"{snapshot} has an invalid compressed checkpoint size: {stored!r}")
    return stored


def _snapshot_cached(snapshot: Path) -> bool:
    """Report whether the index and every tensor file it names are present."""
    try:
        if _csf_container(snapshot) is not None:
            from ignition.preparation.checkpoint import inspect_indexed_checkpoint

            inspect_indexed_checkpoint(snapshot)
            return True
        index_path = snapshot / "model.safetensors.index.json"
        if not index_path.exists() and (snapshot / "model.safetensors").is_file():
            return (snapshot / "model.safetensors").stat().st_size > 0
        if not index_path.exists() and any(p.is_file() and p.stat().st_size > 0 for p in snapshot.glob("*.gguf")):
            return True
        index = json.loads(index_path.read_text())
        return all((snapshot / shard).is_file() for shard in set(index["weight_map"].values()))
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return False


@dataclass(frozen=True)
class ResolvedModel:
    """A recipe, the checkpoint it serves, and that checkpoint's local directory when the user named one."""

    recipe: Recipe
    checkpoint: Checkpoint
    directory: Path | None = None

    @property
    def source(self) -> str:
        """Name the checkpoint as the user chose it: a pinned revision, repository and revision, or directory."""
        if self.directory is not None:
            return str(self.directory)
        if self.checkpoint in self.recipe.checkpoints:
            return self.checkpoint.revision[:8]
        return f"{self.checkpoint.repository}@{self.checkpoint.revision[:8]}"

    @property
    def model_id(self) -> str:
        return self.directory.name if self.directory is not None else self.checkpoint.repository

    def snapshot(self, hub_cache: Path) -> Path:
        """Return the checkpoint directory, which exists only once the revision is cached."""
        if self.directory is not None:
            return self.directory
        return snapshot_directory(hub_cache, self.checkpoint)

    def verify(self, hub_cache: Path) -> None:
        """Check index metadata; recipes validate tensor storage and geometry."""
        snapshot = self.snapshot(hub_cache)
        try:
            container = _csf_container(snapshot)
            if container is not None:
                _csf_stored_bytes(container, snapshot)
                return
            index_path = snapshot / "model.safetensors.index.json"
            if not index_path.exists() and (snapshot / "model.safetensors").is_file():
                if (snapshot / "model.safetensors").stat().st_size <= 0:
                    raise UsageError(f"{snapshot} has an invalid safetensors file size")
                return
            index = json.loads(index_path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            raise UsageError(f"{snapshot} has an unreadable safetensors index: {error}") from error
        stored = index.get("metadata", {}).get("total_size") if isinstance(index, dict) else None
        if type(stored) is not int or stored <= 0:
            raise UsageError(f"{snapshot} has an invalid safetensors total_size: {stored!r}")

    def cached(self, hub_cache: Path) -> bool:
        """Report whether the index and every tensor file it names are present."""
        return _snapshot_cached(self.snapshot(hub_cache))


def aliases(recipes: Sequence[Recipe]) -> tuple[str, ...]:
    return tuple(checkpoint.alias for recipe in recipes for checkpoint in recipe.checkpoints)


def _unsupported(reason: str, recipes: Sequence[Recipe]) -> UsageError:
    supported = aliases(recipes)
    if not supported:
        return UsageError(f"{reason}; no model recipes are registered in this CLI")
    return UsageError(f"{reason}; supported models: {', '.join(supported)}")


def _supporting(config_path: Path, source: str, recipes: Sequence[Recipe]) -> tuple[Recipe, Checkpoint]:
    try:
        architectures = json.loads(config_path.read_text())["architectures"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise _unsupported(f"{source} has no readable config.json architecture", recipes) from error
    for recipe in recipes:
        for checkpoint in recipe.checkpoints:
            if isinstance(architectures, list) and checkpoint.architecture in architectures:
                return recipe, checkpoint
    raise _unsupported(f"no recipe supports architecture {architectures!r} in {source}", recipes)


def _stored_bytes(index_path: Path, source: str) -> int:
    try:
        stored = json.loads(index_path.read_text())["metadata"]["total_size"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UsageError(f"{source} has no readable model.safetensors.index.json total_size") from error
    if type(stored) is not int or stored <= 0:
        raise UsageError(f"{source} has an invalid safetensors total_size: {stored!r}")
    return stored


def _replacement(qualified: Checkpoint, repository: str, revision: str, stored_bytes: int) -> Checkpoint:
    """Describe another checkpoint of a recipe's model, scaling placement from the qualified checkpoint."""
    scale = stored_bytes / qualified.stored_bytes
    return replace(qualified, repository=repository, revision=revision, stored_bytes=stored_bytes,
                   device_bytes=round(qualified.device_bytes * scale),
                   scale_compressed_device_bytes={encoding: round(size * scale)
                       for encoding, size in qualified.scale_compressed_device_bytes.items()},
                   mapped_host_bytes=round(qualified.mapped_host_bytes * scale),
                   expansion_bytes=round(qualified.expansion_bytes * scale))


def _resolve_replacement(recipe: Recipe, qualified: Checkpoint, metadata: Path, repository: str,
                         revision: str, stored_bytes: int, source: str) -> Checkpoint:
    """Describe an unregistered checkpoint, letting the recipe inspect alternate weight containers."""
    replacement = _replacement(qualified, repository, revision, stored_bytes)
    if recipe.inspect_replacement is None:
        return replacement
    try:
        inspected = recipe.inspect_replacement(metadata, replacement)
    except UsageError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UsageError(f"{source} has unreadable checkpoint metadata: {error}") from error
    if inspected is None:
        return replacement
    if not isinstance(inspected, Checkpoint):
        raise UsageError(f"{source}: recipe replacement inspection returned {type(inspected).__name__}")
    return inspected


def _hub_metadata(repository: str, revision: str | None, hub_cache: Path | None,
                  recipes: Sequence[Recipe]) -> tuple[Path, Path]:
    """Fetch a repository's config and safetensors index, which name its commit, without its tensors."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        return tuple(Path(hf_hub_download(repository, name, revision=revision, cache_dir=hub_cache))
                     for name in ("config.json", "model.safetensors.index.json"))
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as error:
        raise _unsupported(f"{repository}{'@' + revision if revision else ''} is not a safetensors checkpoint"
                           f" on Hugging Face: {type(error).__name__}", recipes) from error
    except Exception as error:  # noqa: BLE001 - Network and authorization failures share one exit code.
        raise FetchError(f"cannot read {repository} from Hugging Face: {error}") from error


def resolve(model: str, hub_cache: Path | None = None, *, recipes: Sequence[Recipe] | None = None) -> ResolvedModel:
    """Resolve an alias, a Hugging Face repository id with an optional `@revision`, or a checkpoint directory.

    A directory or repository that no recipe declares is served by the recipe
    whose architecture its config.json names.
    """
    if recipes is None:
        recipes = current_recipes()
    directory = Path(model).expanduser()
    if directory.is_dir():
        directory = directory.resolve()
        try:
            container = _csf_container(directory)
        except (OSError, ValueError) as error:
            raise UsageError(f"{directory} has an unreadable checkpoint manifest: {error}") from error
        metadata = directory if container is None else directory / "metadata"
        for recipe in recipes:
            for checkpoint in recipe.checkpoints:
                # A directory that a recipe declares keeps its declared placement.
                if Path(checkpoint.repository).is_absolute() and Path(checkpoint.repository).resolve() == directory:
                    return ResolvedModel(recipe, checkpoint, directory)
        recipe, qualified = _supporting(metadata / "config.json", str(directory), recipes)
        # The directory's path identifies its measurements on this host.
        revision = hashlib.sha256(str(directory).encode()).hexdigest()[:40]
        index = directory / "model.safetensors.index.json"
        single = directory / "model.safetensors"
        stored = (_csf_stored_bytes(container, directory) if container is not None else
                  _stored_bytes(index, str(directory)) if index.exists() else
                  single.stat().st_size if single.is_file() else qualified.stored_bytes)
        replacement = _resolve_replacement(recipe, qualified, metadata, str(directory), revision, stored, str(directory))
        return ResolvedModel(recipe, replacement, directory)
    repository, _, revision = model.partition("@")
    if "/" not in repository:
        for recipe in recipes:
            for checkpoint in recipe.checkpoints:
                if checkpoint.alias == model:
                    if Path(checkpoint.repository).is_absolute():
                        if not Path(checkpoint.repository).is_dir():
                            raise UsageError(f"checkpoint directory is unavailable: {checkpoint.repository}")
                        return replace(resolve(checkpoint.repository, hub_cache, recipes=recipes),
                                       recipe=recipe, checkpoint=checkpoint)
                    return ResolvedModel(recipe, checkpoint)
        suggestions = get_close_matches(model, aliases(recipes), n=1, cutoff=0.6)
        hint = f"; did you mean {suggestions[0]!r}?" if suggestions else ""
        raise _unsupported(f"unknown model {model!r}{hint}", recipes)
    for recipe in recipes:
        for checkpoint in recipe.checkpoints:
            matches = (checkpoint.repository == repository if Path(checkpoint.repository).is_absolute()
                       else checkpoint.repository.casefold() == repository.casefold())
            if matches and revision in ("", checkpoint.revision):
                return ResolvedModel(recipe, checkpoint)
    config_path, index_path = _hub_metadata(repository, revision or None, hub_cache, recipes)
    recipe, qualified = _supporting(config_path, model, recipes)
    # The hub cache stores each file under its commit hash.
    replacement = _resolve_replacement(recipe, qualified, config_path.parent, repository, config_path.parent.name,
                                       _stored_bytes(index_path, model), model)
    return ResolvedModel(recipe, replacement)
