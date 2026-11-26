// SPDX-License-Identifier: Apache-2.0

use super::codec::{Encoding, Field, Plan};
use crate::geometry::{CacheGeometry, Plane};
use blake2::{Blake2b, Digest};

pub const REVISION_CPU: &str = "ZKV1/v1/64-pivots";
pub const REVISION_GPU: &str = "GKV1/v2/64-pivots-rans32";

pub fn revision_for(placement: crate::offload::KvCompression) -> &'static str {
    match placement {
        crate::offload::KvCompression::Gpu => REVISION_GPU,
        crate::offload::KvCompression::Cpu | crate::offload::KvCompression::Off => REVISION_CPU,
    }
}

fn encoding(id: u32) -> Option<Encoding> {
    Some(match id {
        0 => Encoding::Bytes,
        1 => Encoding::Float32Matrix128,
        2 => Encoding::Bfloat16,
        20 => Encoding::Convolution20,
        32 => Encoding::Convolution32,
        40 => Encoding::Latent528,
        41 => Encoding::Index132,
        42 => Encoding::Fp8Page512,
        45 => Encoding::WindowFp4,
        47 => Encoding::LatentFp4,
        48 => Encoding::Latent576,
        _ => return None,
    })
}

pub fn profile(g: &CacheGeometry, revision: &str) -> Result<[u8; 32], String> {
    plans_and_profile(g, revision)
        .map(|(_, digest)| digest)
        .map_err(|error| error.to_string())
}

pub fn plans(g: &CacheGeometry) -> std::io::Result<[Plan; 2]> {
    plans_and_profile(g, revision_for(g.placement)).map(|(plans, _)| plans)
}

pub fn plans_and_profile(
    g: &CacheGeometry,
    revision: &str,
) -> std::io::Result<([Plan; 2], [u8; 32])> {
    use std::io::Error;
    if revision != REVISION_CPU && revision != REVISION_GPU {
        return Err(Error::other("unsupported codec layout revision"));
    }
    let mut hash = Blake2b::<blake2::digest::consts::U32>::new();
    hash.update(revision.as_bytes());
    hash.update(g.page_tokens.to_le_bytes());
    let mut build_plan = |planes: &[Plane]| {
        let mut fields = Vec::new();
        let mut base = 0u64;
        for plane in planes {
            let mut matched = false;
            let padded = crate::geometry::padded_bytes(plane.unit_bytes)
                .and_then(|size| base.checked_add(size))
                .ok_or_else(|| Error::other("codec unit size overflow"))?;
            for field in g.codec_fields.iter().filter(|f| f.buffer == plane.name) {
                matched = true;
                if field
                    .offset
                    .checked_add(field.length)
                    .is_none_or(|end| end > plane.unit_bytes)
                {
                    return Err(Error::other(format!(
                        "codec field exceeds plane {}",
                        plane.name
                    )));
                }
                fields.push(Field {
                    offset: usize::try_from(base + field.offset).map_err(Error::other)?,
                    length: usize::try_from(field.length).map_err(Error::other)?,
                    encoding: encoding(field.encoding)
                        .ok_or_else(|| Error::other("unsupported codec field encoding"))?,
                });
                hash.update((plane.name.len() as u64).to_le_bytes());
                hash.update(plane.name.as_bytes());
                hash.update(field.offset.to_le_bytes());
                hash.update(field.length.to_le_bytes());
                hash.update(field.encoding.to_le_bytes());
            }
            if !matched {
                return Err(Error::other(format!(
                    "codec layout has no field for plane {}",
                    plane.name
                )));
            }
            hash.update(plane.unit_bytes.to_le_bytes());
            base = padded;
        }
        Plan::new(usize::try_from(base).map_err(Error::other)?, &fields)
    };
    let plans = [build_plan(&g.planes)?, build_plan(std::slice::from_ref(&g.row))?];
    Ok((plans, hash.finalize().into()))
}
