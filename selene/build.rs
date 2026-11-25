/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

fn main() {
    if std::env::var_os("CARGO_FEATURE_CODEC").is_some() {
        println!("cargo:rerun-if-changed=native/kv_codec");
        let mut build = cc::Build::new();
        build.cpp(true).std("c++17");
        if build.get_compiler().is_like_msvc() {
            build.flag("/fp:strict");
        } else {
            build.flag("-fno-fast-math").flag("-ffp-contract=off");
        }
        build
            .file("native/kv_codec/cpu.cpp")
            .compile("flare_kv_cpu");
    }
    let proto = "proto/selene.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&[proto], &["proto"])
        .expect("compile the selene pool protocol");
}
