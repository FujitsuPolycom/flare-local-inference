/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! RDMA handshake and libibverbs device and queue-pair lifecycle (experimental).
use std::io;
#[cfg(any(feature = "rdma", test))]
use std::net::IpAddr;

/// Version, qpn, psn, gid, lid, staging rkey and base address.
pub const HANDSHAKE_BYTES: usize = 4 + 4 + 4 + 16 + 2 + 4 + 8;
pub const HANDSHAKE_VERSION: u32 = 2;

/// One end of a queue pair and the registered memory the other end may address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handshake {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub lid: u16,
    pub rkey_staging: u32,
    pub staging_base: u64,
}

impl Handshake {
    pub fn encode(&self) -> [u8; HANDSHAKE_BYTES] {
        let mut out = [0u8; HANDSHAKE_BYTES];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            out[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };
        put(&HANDSHAKE_VERSION.to_le_bytes());
        put(&self.qpn.to_le_bytes());
        put(&self.psn.to_le_bytes());
        put(&self.gid);
        put(&self.lid.to_le_bytes());
        put(&self.rkey_staging.to_le_bytes());
        put(&self.staging_base.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() != HANDSHAKE_BYTES {
            return Err(invalid("an RDMA handshake has the wrong length"));
        }
        if u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != HANDSHAKE_VERSION {
            return Err(invalid("an unsupported RDMA handshake version"));
        }
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&bytes[12..28]);
        Ok(Self {
            qpn: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            psn: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            gid,
            lid: u16::from_le_bytes(bytes[28..30].try_into().unwrap()),
            rkey_staging: u32::from_le_bytes(bytes[30..34].try_into().unwrap()),
            staging_base: u64::from_le_bytes(bytes[34..42].try_into().unwrap()),
        })
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(any(feature = "rdma", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GidEntry {
    device: usize,
    port: u8,
    gid_index: u32,
    gid: [u8; 16],
    roce_v2: bool,
    lid: u16,
}

#[cfg(any(feature = "rdma", test))]
fn no_gid(local_address: IpAddr) -> io::Error {
    invalid(&format!(
        "no RDMA device has a RoCE v2 GID for {local_address}; RDMA needs the data connection on a RoCE interface"
    ))
}

#[cfg(any(feature = "rdma", test))]
fn select_gid(local_address: IpAddr, entries: &[GidEntry]) -> io::Result<GidEntry> {
    let gid = match local_address {
        IpAddr::V4(address) => address.to_ipv6_mapped().octets(),
        IpAddr::V6(address) => address.octets(),
    };
    entries
        .iter()
        .find(|entry| entry.roce_v2 && entry.gid == gid)
        .copied()
        .ok_or_else(|| no_gid(local_address))
}

#[cfg(feature = "rdma")]
mod verbs {
    use super::*;
    pub use ibverbs_sys as ffi;
    use std::{ptr, sync::Arc};

    const PSN: u32 = 0x1234_5678;

    /// An active device and its protection domain.
    pub struct Device {
        pub context: *mut ffi::ibv_context,
        pub pd: *mut ffi::ibv_pd,
        port: u8,
        lid: u16,
        gid: [u8; 16],
        gid_index: u32,
    }

    // SAFETY: libibverbs handles allow concurrent use and stay live until the last owner drops.
    unsafe impl Send for Device {}
    // SAFETY: the handles are immutable after open.
    unsafe impl Sync for Device {}

    impl Device {
        pub fn open(local_address: IpAddr) -> io::Result<Arc<Self>> {
            let mut count = 0;
            // SAFETY: libibverbs owns the returned list until it is freed below.
            let list = unsafe { ffi::ibv_get_device_list(&mut count) };
            if list.is_null() {
                return Err(io::Error::last_os_error());
            }
            let result = (|| {
                for index in 0..count as usize {
                    // SAFETY: the list has `count` entries.
                    let device = unsafe { *list.add(index) };
                    if device.is_null() {
                        continue;
                    }
                    // SAFETY: device belongs to the live libibverbs list.
                    let context = unsafe { ffi::ibv_open_device(device) };
                    if context.is_null() {
                        continue;
                    }
                    let mut attr = ffi::ibv_device_attr::default();
                    // SAFETY: the context is open and attr is writable.
                    let status = unsafe { ffi::ibv_query_device(context, &mut attr) };
                    let mut entries = Vec::new();
                    if status == 0 {
                        for port in 1..=attr.phys_port_cnt {
                            let mut attr = ffi::ibv_port_attr::default();
                            // SAFETY: the compat cast matches libibverbs' inline wrapper.
                            let status = unsafe {
                                ffi::ibv_query_port(
                                    context,
                                    port,
                                    &mut attr as *mut ffi::ibv_port_attr
                                        as *mut ffi::_compat_ibv_port_attr,
                                )
                            };
                            if status != 0 || attr.state != ffi::ibv_port_state::IBV_PORT_ACTIVE {
                                continue;
                            }
                            for gid_index in 0..attr.gid_tbl_len as u32 {
                                let mut entry = unsafe { std::mem::zeroed::<ffi::ibv_gid_entry>() };
                                // SAFETY: context is open and entry has the supplied size.
                                let status = unsafe {
                                    ffi::_ibv_query_gid_ex(
                                        context,
                                        u32::from(port),
                                        gid_index,
                                        &mut entry,
                                        0,
                                        std::mem::size_of::<ffi::ibv_gid_entry>(),
                                    )
                                };
                                if status == 0 {
                                    entries.push(GidEntry {
                                        device: index,
                                        port,
                                        gid_index,
                                        // SAFETY: libibverbs filled the union.
                                        gid: unsafe { entry.gid.raw },
                                        roce_v2: entry.gid_type == ffi::IBV_GID_TYPE_ROCE_V2,
                                        lid: attr.lid,
                                    });
                                }
                            }
                        }
                    }
                    if let Ok(entry) = select_gid(local_address, &entries) {
                        // SAFETY: the context is open.
                        let pd = unsafe { ffi::ibv_alloc_pd(context) };
                        if pd.is_null() {
                            let error = io::Error::last_os_error();
                            unsafe { ffi::ibv_close_device(context) };
                            return Err(error);
                        }
                        return Ok(Arc::new(Self {
                            context,
                            pd,
                            port: entry.port,
                            lid: entry.lid,
                            gid: entry.gid,
                            gid_index: entry.gid_index,
                        }));
                    }
                    // SAFETY: no PD or registration uses this context.
                    unsafe { ffi::ibv_close_device(context) };
                }
                Err(no_gid(local_address))
            })();
            // SAFETY: all contexts were opened before freeing the list.
            unsafe { ffi::ibv_free_device_list(list) };
            result
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: registrations and pairs release their handles before their device owner.
            unsafe {
                ffi::ibv_dealloc_pd(self.pd);
                ffi::ibv_close_device(self.context);
            }
        }
    }

    /// A queue pair for one transfer, in INIT until connected.
    pub struct Qp {
        pub cq: *mut ffi::ibv_cq,
        pub qp: *mut ffi::ibv_qp,
        device: Arc<Device>,
    }

    // SAFETY: a pair is moved into one task and used there.
    unsafe impl Send for Qp {}
    // SAFETY: its methods take `&self` and libibverbs allows concurrent use of a queue pair.
    unsafe impl Sync for Qp {}

    impl Qp {
        pub fn new(
            device: &Arc<Device>,
            send_wr: u32,
            cq_entries: i32,
            access: u32,
        ) -> io::Result<Self> {
            // SAFETY: the context is open; no completion channel is used.
            let cq = unsafe {
                ffi::ibv_create_cq(
                    device.context,
                    cq_entries,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                )
            };
            if cq.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut init = ffi::ibv_qp_init_attr {
                send_cq: cq,
                recv_cq: cq,
                cap: ffi::ibv_qp_cap {
                    max_send_wr: send_wr,
                    max_recv_wr: 1,
                    max_send_sge: 1,
                    max_recv_sge: 1,
                    ..Default::default()
                },
                qp_type: ffi::ibv_qp_type::IBV_QPT_RC,
                sq_sig_all: 0,
                ..Default::default()
            };
            // SAFETY: the PD is live and `init` describes a valid RC queue pair.
            let qp = unsafe { ffi::ibv_create_qp(device.pd, &mut init) };
            if qp.is_null() {
                let error = io::Error::last_os_error();
                let mut pair = Self {
                    cq,
                    qp,
                    device: device.clone(),
                };
                let _ = pair.close();
                return Err(error);
            }
            let pair = Self {
                cq,
                qp,
                device: device.clone(),
            };
            let mut attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_INIT,
                pkey_index: 0,
                port_num: device.port,
                qp_access_flags: access,
                ..Default::default()
            };
            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
                | ffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX.0
                | ffi::ibv_qp_attr_mask::IBV_QP_PORT.0
                | ffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS.0;
            // SAFETY: the pair is in RESET and the attributes are valid for INIT.
            if unsafe { ffi::ibv_modify_qp(pair.qp, &mut attr, mask as i32) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(pair)
        }

        /// This end's description: the queue pair and the registered staging.
        pub fn handshake(&self, rkey_staging: u32, staging_base: u64) -> Handshake {
            Handshake {
                qpn: unsafe { (*self.qp).qp_num },
                psn: PSN,
                gid: self.device.gid,
                lid: self.device.lid,
                rkey_staging,
                staging_base,
            }
        }

        /// Bring the pair up against the manager's description.
        pub fn connect(&self, peer: &Handshake) -> io::Result<()> {
            let mut attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTR,
                path_mtu: ffi::IBV_MTU_1024,
                dest_qp_num: peer.qpn,
                rq_psn: peer.psn,
                max_dest_rd_atomic: 1,
                min_rnr_timer: 12,
                ah_attr: ffi::ibv_ah_attr {
                    dlid: peer.lid,
                    port_num: self.device.port,
                    is_global: u8::from(peer.gid != [0; 16]),
                    grh: ffi::ibv_global_route {
                        dgid: ffi::ibv_gid { raw: peer.gid },
                        sgid_index: self.device.gid_index as u8,
                        hop_limit: 64,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
                | ffi::ibv_qp_attr_mask::IBV_QP_AV.0
                | ffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU.0
                | ffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN.0
                | ffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN.0
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0
                | ffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER.0;
            // SAFETY: the pair is in INIT and the RTR attributes are valid.
            if unsafe { ffi::ibv_modify_qp(self.qp, &mut attr, mask as i32) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut attr = ffi::ibv_qp_attr {
                qp_state: ffi::ibv_qp_state::IBV_QPS_RTS,
                timeout: 14,
                retry_cnt: 7,
                rnr_retry: 7,
                sq_psn: PSN,
                max_rd_atomic: 1,
                ..Default::default()
            };
            let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
                | ffi::ibv_qp_attr_mask::IBV_QP_TIMEOUT.0
                | ffi::ibv_qp_attr_mask::IBV_QP_RETRY_CNT.0
                | ffi::ibv_qp_attr_mask::IBV_QP_RNR_RETRY.0
                | ffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN.0
                | ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0;
            // SAFETY: the pair is in RTR and the RTS attributes are valid.
            if unsafe { ffi::ibv_modify_qp(self.qp, &mut attr, mask as i32) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    impl Qp {
        /// Destroy the QP before its credited slots can be reused.
        pub fn close(&mut self) -> io::Result<()> {
            if !self.qp.is_null() {
                let mut attr = ffi::ibv_qp_attr {
                    qp_state: ffi::ibv_qp_state::IBV_QPS_ERR,
                    ..Default::default()
                };
                // SAFETY: the QP is live and owned by this pair.
                let status = unsafe {
                    ffi::ibv_modify_qp(
                        self.qp,
                        &mut attr,
                        ffi::ibv_qp_attr_mask::IBV_QP_STATE.0 as i32,
                    )
                };
                if status != 0 {
                    return Err(io::Error::from_raw_os_error(status));
                }
                // SAFETY: owned here; success fences all remote access through this QP.
                let status = unsafe { ffi::ibv_destroy_qp(self.qp) };
                if status != 0 {
                    return Err(io::Error::from_raw_os_error(status));
                }
                self.qp = ptr::null_mut();
            }
            if !self.cq.is_null() {
                // SAFETY: its QP has been destroyed.
                let status = unsafe { ffi::ibv_destroy_cq(self.cq) };
                if status != 0 {
                    return Err(io::Error::from_raw_os_error(status));
                }
                self.cq = ptr::null_mut();
            }
            Ok(())
        }
    }

    impl Drop for Qp {
        fn drop(&mut self) {
            if let Err(error) = self.close() {
                eprintln!("warning: quarantining an unfenced RDMA pair: {error}");
                std::mem::forget(self.device.clone());
            }
        }
    }
}

#[cfg(feature = "rdma")]
pub use verbs::{Device, Qp, ffi};
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gid_selection_follows_the_local_address_across_devices_and_ports() {
        let ipv4: IpAddr = "172.16.1.6".parse().unwrap();
        let ipv6: IpAddr = "2001:db8::6".parse().unwrap();
        let mapped = "::ffff:172.16.1.6"
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets();
        let entries = [
            GidEntry {
                device: 0,
                port: 1,
                gid_index: 0,
                gid: mapped,
                roce_v2: false,
                lid: 0,
            },
            GidEntry {
                device: 0,
                port: 1,
                gid_index: 1,
                gid: "2001:db8::6"
                    .parse::<std::net::Ipv6Addr>()
                    .unwrap()
                    .octets(),
                roce_v2: true,
                lid: 0,
            },
            GidEntry {
                device: 1,
                port: 2,
                gid_index: 3,
                gid: mapped,
                roce_v2: true,
                lid: 0,
            },
        ];
        assert_eq!(select_gid(ipv4, &entries).unwrap(), entries[2]);
        assert_eq!(select_gid(ipv6, &entries).unwrap(), entries[1]);
        assert_eq!(
            select_gid("::ffff:172.16.1.6".parse().unwrap(), &entries).unwrap(),
            entries[2]
        );
        let error = select_gid(ipv4, &entries[..2]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "no RDMA device has a RoCE v2 GID for 172.16.1.6; RDMA needs the data connection on a RoCE interface"
        );
        assert!(select_gid("172.16.1.7".parse().unwrap(), &entries).is_err());
    }

    #[test]
    fn a_handshake_round_trips_and_refuses_the_wrong_length_or_version() {
        let value = Handshake {
            qpn: 17,
            psn: 0x1234_5678,
            gid: [0xab; 16],
            lid: 0x0102,
            rkey_staging: 0x3333_4444,
            staging_base: 0x7f00_1000_0000,
        };
        let bytes = value.encode();
        assert_eq!(Handshake::decode(&bytes).unwrap(), value);
        assert!(Handshake::decode(&bytes[..HANDSHAKE_BYTES - 1]).is_err());
        let mut other = bytes;
        other[..4].copy_from_slice(&1u32.to_le_bytes());
        assert!(Handshake::decode(&other).is_err());
    }
}
