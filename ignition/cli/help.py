# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline command help epilogs drawn from recipe declarations."""


def catalog_epilog(*, recipes=(), prog="ignition") -> str:
    """Format the available-recipes table for top-level and bare preparation `--help`."""
    catalog = [(recipe, checkpoint) for recipe in recipes for checkpoint in recipe.checkpoints]
    if not catalog:
        return (
            "No model recipes are registered in this CLI.\n"
            "Core publishing and rank-join commands (keygen, key-password, sign, edit, combine, convert, join)\n"
            "work without recipes. Build a recipe CLI with ignition.cli.main(recipes=...)."
        )
    model_width = max(len("MODEL"), *(len(checkpoint.alias) for _, checkpoint in catalog)) + 2
    gpu_width = max(len("GPUs"), *(len(",".join(map(str, recipe.checkpoint_tensor_parallel(checkpoint))))
                                   for recipe, checkpoint in catalog)) + 2
    lines = [
        "Available model recipes:",
        f"  {'MODEL':<{model_width}}{'GPUs':<{gpu_width}}Drafts (default/max)",
    ]
    for recipe, checkpoint in catalog:
        gpus = ",".join(map(str, recipe.checkpoint_tensor_parallel(checkpoint)))
        spec = recipe.speculation
        drafts = "none" if spec is None else f"{spec.draft_tokens}/{recipe.max_draft_tokens}"
        lines.append(f"  {checkpoint.alias:<{model_width}}{gpus:<{gpu_width}}{drafts}")
    lines.extend((
        "",
        "Show a recipe's supported options, defaults, and limits:",
        f"  {prog} export MODEL --help",
        "MODEL also accepts org/repository[@revision] or a checkpoint directory.",
        "GPU counts are supported tensor-parallel sizes; memory must still fit.",
    ))
    return "\n".join(lines)


def model_epilog(recipe, checkpoint, *, prog="ignition") -> str:
    """Format the recipe summary appended to dynamic `MODEL --help` output."""
    gpus = ", ".join(map(str, recipe.checkpoint_tensor_parallel(checkpoint)))
    lines = [
        f"Model recipe: {checkpoint.alias}",
        f"  Checkpoint: {checkpoint.repository}",
        f"  --gpus: select {gpus} GPUs",
        f"  Context: {recipe.declared_context(checkpoint):,} tokens (page size: {recipe.page_tokens})",
    ]
    spec = recipe.speculation
    if spec is not None:
        layout = "packed" if spec.packed_verification else "rectangular"
        lines.append(
            f"  Prepared verification: {layout} rows with {spec.draft_prefix_selection.upper()} prefix selection."
        )
    else:
        lines.append("  Multi-token speculation: unavailable.")
    lines.append(
        "  Images per request: bounded by the prepared image-token budget."
        if recipe.max_images
        else "  Images: unavailable."
    )
    lines.extend((
        "  Defaults shown above come from the recipe; configuration and explicit flags override them.",
        "  Supported GPU counts do not guarantee that the selected capacities fit in memory.",
    ))
    return "\n".join(lines)
