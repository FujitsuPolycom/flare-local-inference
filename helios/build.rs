/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

fn main() {
    #[cfg(feature = "runtime-core")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_core.proto");
        println!("cargo:rerun-if-changed=proto/core.proto");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // The execution schema is the single definition of
        // contracts, graph selection, frontend settings, and engine placement.
        let mut runtime = prost_build::Config::new();
        runtime
            .protoc_arg("--experimental_allow_proto3_optional")
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .file_descriptor_set_path(out.join("runtime_core_descriptor.bin"))
            .field_attribute(".flare.runtime.ModelIndex.preparation", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.source_identity", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.checkpoint_repository", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.checkpoint_revision", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.local_checkpoint", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.reproduction_args", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationManifest.parameters", "#[serde(default)]")
            .field_attribute(".flare.runtime.PreparationValueList.values", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildSpeculation.vocabulary_parallel_sampling", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildSpeculation.mtp_only", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.online_mxfp8_scope", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.context_parallel_size", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.prompt_logprobs", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.autotune_effort", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.native_dense_activations", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.lock_gpu_clocks", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.reproduction_args", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.row_shard_prefill", "#[serde(default)]")
            .field_attribute(
                ".flare.runtime.ExecutionGraph.packed_token_capacity",
                "#[serde(default)]",
            )
            .field_attribute(
                ".flare.runtime.Execution.persistent_workers",
                "#[serde(default)]",
            )
            .field_attribute(
                ".flare.runtime.PersistentWorker.metrics",
                "#[serde(default)]",
            )
            .field_attribute(
                ".flare.runtime.PersistentWorker.metrics_count",
                "#[serde(default)]",
            )
            .compile_protos(&["proto/runtime_core.proto"], &["proto"])
            .expect("compile runtime core protobuf schema");
        // The payload schema imports the execution schema and refers to the generated types
        // instead of restating them.
        let mut core = prost_build::Config::new();
        core
            .protoc_arg("--experimental_allow_proto3_optional")
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .field_attribute(".flare.core.DeploymentBody.preparation", "#[serde(default)]")
            .extern_path(".flare.runtime", "crate::runtime_core::wire")
            .compile_protos(&["proto/core.proto"], &["proto"])
            .expect("compile payload core protobuf schema");
    }
    #[cfg(feature = "roce")]
    {
        println!("cargo:rerun-if-changed=native/roce_proxy.c");
        cc::Build::new()
            .file("native/roce_proxy.c")
            .std("gnu11")
            .compile("flare_roce_proxy");
        println!("cargo:rustc-link-lib=ibverbs");
        println!("cargo:rustc-link-lib=pthread");
    }
}
