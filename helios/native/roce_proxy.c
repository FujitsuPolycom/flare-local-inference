/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Vendored from FlashInfer (flashinfer/experimental/b12x/comm/roce/_roce_proxy.c,
// proxy ABI 15) from local-inference-lab/flashinfer #106 and #107, with the
// four-rank ring partner relay. The protocol tag, parity-based routing and
// relay-plan handling differ from FlashInfer's source.

// RDMA proxy for the b12x RoCE one-shot all-reduce.
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[src][slot][rail] sequence number written on each rail after that rail's
//                         payload stripe
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on flag[peer][seq & 1][rail] for every peer and rail.  The
// proxy thread
// below spins on ctrl.seq and, for every peer, stripes the payload across every
// rail.  A rail is one (local HCA, peer HCA) link chosen per peer at connect time:
// on a switched fabric rail h is HCA h on both ends; on a switchless ring each
// peer is reached through the HCAs cabled to it.  Each stripe is followed by its own 4-byte seq
// write on the same reliable QP, so its flag cannot become visible before its payload.  The GPU
// waits for every stripe flag before consuming the receive slot.  Nothing on the receive path
// involves the host.
//
// Each rail is n_lane reliable QPs between the same two HCAs (lanes).  A stripe and its seq
// write always share one lane, so the flag-after-payload guarantee holds per stripe.  One-shot
// stripes and ring relays use lane 0; two-shot and variable-length chunk c uses lane
// c % n_lane, so consecutive chunks of one peer and rail proceed on separate QPs.  No buffer
// lifetime argument depends on ordering between chunks or between operations on one QP:
// every one rests on flags, the kernel's waits and pipe_done, which counts every lane.
//
// Messages too large for one-shot latency to matter take the pipelined two-shot
// path instead (ctrl kind[slot] = 1).  The region then also holds
//
//   pipe_send          (pipe_bytes)                 staged input; our reduced shard
//   pipe_recv[src][2]  (world * 2 * shard_cap)       peers' reduce-scatter and
//                                                   all-gather chunks
//   pipe_flag[2][src][chunk][rail]                   seq after each chunk stripe
//
// Shard j covers packs [j * P / world, (j + 1) * P / world) of a P-pack message and
// is split into chunks of chunk_packs.  The kernel publishes ready_rs[c] = seq
// once every block staged chunk c of every peer's shard and ready_ag[c] = seq once
// chunk c of its own shard is reduced; the proxy posts each chunk as its word
// turns, so staging, the wire and the reduction overlap.  The two-shot buffers
// need one copy only: a rank leaves op N after every peer's all-gather of N
// arrived, which every peer sent after reducing all of our N chunks, and a peer
// sends its N + 1 all-gather only after our N + 1 reduce-scatter, which we post
// after our N kernel finished reading.  pipe_send is the exception: our own
// all-gather of N reads our reduced shard from it, and that send is not part of
// what lets our kernel N finish.  So once every write of op N completed, this
// thread sets ctrl pipe_done = seq, and the kernel publishes its epoch only after
// seeing it; otherwise our next two-shot op could stage over the shard while it
// was still to be sent, and peers would gather that input as our reduced values.
//
// The two halves of the two-shot path are also separate operations over the same
// buffers and flags: a reduce-scatter (ctrl kind[slot] = 3) stages and posts the
// phase-0 chunks only, and its kernel keeps the reduced shard; a shard gather
// (ctrl kind[slot] = 4) stages this rank's shard of the message into pipe_send
// and posts the phase-1 chunks only.  Each releases pipe_send through pipe_done
// like a complete two-shot op.  The single-copy argument above covers two-shot ops following any
// kind, a reduce-scatter following anything but a reduce-scatter, and a shard gather following a
// one-shot, variable-length or reduce-scatter op: in each case some write of op N + 1 that a peer
// needs before finishing N + 1 (or the predecessor's own completion) depends on that peer having
// started N + 1.  Two transitions lack that dependency.  After a reduce-scatter, our next
// reduce-scatter or two-shot posts phase-0 chunks as soon as our kernel staged them, while a peer's
// kernel may still be reducing the previous op from the same rows; after a two-shot or shard
// gather, our next shard gather posts phase-1 chunks while a peer may still be gathering.  Either
// write could replace a flag before the peer saw it (a wait timeout) or the rows it is reading
// (silent corruption).  For those transitions every rank writes a start-of-op credit, the op's
// seq, into ctrl credit[rank] of every peer as it begins the op (our kernel N + 1 rang, so our
// kernel N finished), and holds the hazardous phase until every peer's credit reached the op.  No
// other op posts or waits for credits, so one-shot and back-to-back two-shot traffic is unchanged.
// On the ring transports a credit to a rank that is not a neighbor travels as an empty mailbox
// write, whose flag lands on every rail; only rail 0's credit word is read.
//
// A cabled fabric can relay the all-gather chunks (phase 1 of a two-shot op or a shard gather)
// when its ranks form pairs joined by direct cables (roce_set_relay).  A rank then sends its
// reduced shard to its partner and to one rank in every other pair, which forwards each chunk to
// its own partner after the flags of every rail of that chunk arrived: from its pipe_recv row for
// the source to the same row and flags at the partner, each stripe again followed by its flag on
// the same queue pair.  The receiver waits on the same flags as for a direct write.  A forwarder
// reads the row between the chunk's arrival and its own pipe_done, which counts the forwarded
// writes, so the row's reuse rules are those of its kernel's reads.  The bytes are copied, not
// combined, so results are unchanged bit for bit.  With two pairs, each rank's two-shot traffic is
// half cable and half switch instead of one third cable.  A four-rank closed ring uses the same
// mechanism: rank r reaches rank r + 2 through r + 1 and forwards rank r - 1's chunks to r + 1, so
// its shard leaves as two direct writes and one forward instead of two direct writes, one mailbox
// packet and one CPU forward.  Both of its neighbors forward for it, the clockwise one the even
// chunks and the counter-clockwise one the odd chunks, because a forwarded chunk is a direct write
// whose lane has no per-chunk order.
//
// A variable-length all-gather (ctrl kind[slot] = 2) streams each rank's contribution, whose size
// the kernel reads from device memory, through
//
//   varlen_recv[slot][src]  (2 * world * varlen_bytes)  src's contribution of the slot's op;
//                                                       our own row is the posting source
//   varlen_flag[slot][src][chunk][rail]                 seq after each chunk stripe
//
// The kernel stages chunk c of its contribution into its own row and publishes
// ready_vl[c] = seq; the proxy posts the chunk to the same row of every peer.  Ops alternate
// slots: a kernel finishes on its peers' rows alone, so our op N + 1 may stage (and overwrite
// ready_vl) before this thread posted our op N.  Its rows and flags are the other slot's, a
// ready word of N + 1 also releases chunk c of N, and no peer can begin op N + 2, which would
// reuse N's slot, before receiving our N.
//
// This file is compiled by b12x.comm.roce._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 4
#define ROCE_MAX_RAILS 2
#define ROCE_MAX_LANES 4
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 512
// A relay packet addressed to one destination carries its descriptor body and
// its payload in one RDMA write, so every queue pair is created with two send
// SGEs, on the direct transport as well. open_hca rejects a device whose
// max_sge is lower, so such a device fails at setup rather than at the first
// relayed packet; the CPU tests check the same limit.
#define ROCE_MAX_SEND_SGE 2
#define ROCE_ABI_VERSION 15
#define ROCE_MAX_CHUNKS 64
#define ROCE_VARLEN_MAX_CHUNKS 512
#define ROCE_CTRL_BYTES 4096
// ctrl words (u32): the one-shot record, then per-slot op descriptors, the
// two-shot chunk-ready words and the variable-length all-gather's.
#define CTRL_SEQ 0
#define CTRL_ERROR 2
#define CTRL_SLOT_NBYTES 4
#define CTRL_SLOT_KIND 8
#define CTRL_SLOT_CHUNK_PACKS 10
#define CTRL_SLOT_CHUNKS 12
#define CTRL_PIPE_DONE 48
#define CTRL_READY_RS 64
#define CTRL_READY_AG 128
#define CTRL_READY_VL 256
// Start-of-op credits written by peers (one word per source rank), after the variable-length
// all-gather's ready words.
#define CTRL_CREDIT 768
_Static_assert(CTRL_CREDIT >= CTRL_READY_VL + ROCE_VARLEN_MAX_CHUNKS &&
                   (CTRL_CREDIT + ROCE_MAX_PEERS) * 4 + (ROCE_MAX_RAILS - 1) * ROCE_FLAG_STRIDE <=
                       ROCE_CTRL_BYTES,
               "credit words (one per rail for relayed credits) must fit the ctrl page after the "
               "variable-length ready words");
// Phases of a pipelined op that must wait for every peer's start-of-op credit.
#define CREDIT_RS 1u
#define CREDIT_AG 2u
#define KIND_ONESHOT 0
#define KIND_TWOSHOT 1
#define KIND_VARLEN 2
#define KIND_REDUCE_SCATTER 3
#define KIND_GATHER_SHARDS 4
// run_pipe_op phase mask: reduce-scatter chunks, all-gather chunks.
#define PIPE_PHASE_RS 1u
#define PIPE_PHASE_AG 2u
// Model graphs leave sub-millisecond gaps between collectives.  Keep the
// proxy hot across those gaps; sleeping there adds one scheduler wakeup to
// every collective on the graph's critical path.
#define ROCE_IDLE_SPINS 20000000

// Experimental neighbor transport in rank order. Only the two cable-neighbor QPs
// connect. Non-neighbor writes cross CPU-driven, bounded RDMA mailboxes.
// No GPU program or captured address depends on these startup-owned buffers.
//
// Every original sender owns a mailbox window per rail at every rank. On a ring
// that broadcasts it is dense, one window per destination, so every packet the
// origin sends toward that destination takes the next serial, whichever hop
// currently carries it. Anywhere else one window per (origin, rail) is shared by
// the origin's destinations in posting order. The startup region, sized for the
// world and the topology, holds per (origin, rail, slot): mailboxes, then 8-byte
// acknowledgment words, then outgoing descriptors.
//
// A pipelined all-gather chunk is broadcast once per direction on an open chain
// (ring_broadcast, fanout = 1). Every other transfer, one-shot payloads of any
// size included, is delivered per destination. Each broadcast packet names the
// chain endpoint as its target; its payload is written straight into the next
// hop's final receive row and its descriptor plus commit serial into that hop's
// mailbox. Every host the packet passes publishes the flag for its own row and
// forwards the bytes from that row, so a shard is read from host memory once
// per direction instead of once per destination. An intermediate may forward
// only the exact next serial of a dense destination lane (ring_order + 1); an
// endpoint follows the source's fanout_previous chain, because a destination
// lane also carries ordinary unicast serials that endpoint never consumes. A
// stripe's ready flag rides its last packet and every hop posts payload before
// flag on one queue pair, so a flag cannot overtake an earlier packet of its
// lane and no scan has to sort arrivals.
//
// Fanout never wraps a cycle: on a line the two endpoint branches cover every
// rank without overlap. A forwarded write is causally downstream of the origin's
// posting, so it inherits the origin's start-of-op credit gate and the pipelined
// single-copy argument. A fast rank therefore cannot overwrite a slow peer's
// pipelined row. For a two-shot all-gather the peer's own phase-0 chunk of the
// next op precedes this rank's reduction, so that peer's kernel finished the
// previous op before any fanout payload reaches its row. For a shard gather the
// origin holds the hazardous phase until every peer's credit proves its kernel
// finished the previous op. Mailbox slots are reused only after their own
// end-to-end acknowledgment, so a forwarded serial is never rewritten in place.
//
// Direct unicast between cabled neighbors and the cable-partner relay keep their
// ABI 14 behavior. ABI 15 changes this proxy-private geometry, the ring
// protocol and the connection blob: the region layout, the doorbell words and
// every GPU-visible address are exactly as ABI 14 defined them.
//
// Invariants the fanout ordering argument relies on. Violating any of them voids
// the safety claim, so a native integrator must preserve them:
//   1. Every rank executes the same serialized collective sequence, and only the
//      proxy advances a destination lane's serial.
//   2. A kernel publishes a ready word only after it staged or reduced valid
//      data; an empty chunk tail still follows the same publication rule.
//   3. Payload writes become visible before their ready flag: payload before
//      flag on one queue pair, a release store for the local fanout flag, and
//      the GPU's system-scope acquire for the flag load.
//   4. A failed or partially posted context is terminal. It is destroyed and a
//      fresh group is set up; it is never restarted. Restarting would leave a
//      missing serial that every downstream lane predecessor check enforces.
//   5. A fanout packet is posted only after the origin's start-of-op credit
//      gate opens. An intermediate's receive row is also an outgoing NIC
//      source, so its reuse additionally waits on the downstream consumer's
//      evidence (a next phase-0 contribution or a start credit), not merely the
//      intermediate's local completion.
//   6. Mailbox and outgoing-descriptor slots are reused only after their own
//      end-to-end acknowledgment; a forwarded serial is never rewritten in
//      place. A local CQE releases the original source, not the row.
//   7. Rails are independent: serials and acknowledgments are per rail, the GPU
//      waits on every rail's flag, and a credit proves the previous kernel's
//      completion on all rails.
// Per-destination relay mailboxes use 64 KiB packets and an eight-slot window.
// Open chains of four or more ranks use 256 KiB packets and one 32-slot window
// per destination for broadcast. Two- and three-rank chains keep the smaller
// mailbox and per-destination delivery.
#define RING_CHUNK_BYTES 65536
#define RING_WINDOW 8
#define RING_OPEN_CHUNK_BYTES 262144
#define RING_OPEN_WINDOW 32
#define RING_MAX_SLOTS (RING_OPEN_WINDOW * (ROCE_MAX_PEERS - 1))
#define RING_PROTOCOL 0x45504f37u  // EPO7: parity routing, fanout and four-rank partner relay
#define RING_ACK_READY 3
#define RING_COMPLETION (UINT64_C(1) << 63)
typedef struct {
  uint64_t serial, target, offset, flag_offset, flag_value, nbytes, previous;
  uint64_t fanout;           // 1: deliver at every hop through the chain endpoint
  uint64_t fanout_previous;  // previous fanout serial on this destination/rail lane
  uint8_t padding[56];
} ring_descriptor_t;
_Static_assert(sizeof(ring_descriptor_t) == 128, "ring descriptor wire size");
typedef struct {
  ring_descriptor_t descriptor;
  uint8_t data[];  // one packet: RING_CHUNK_BYTES, or RING_OPEN_CHUNK_BYTES when dense
} ring_mailbox_t;
_Static_assert(sizeof(ring_mailbox_t) == sizeof(ring_descriptor_t), "mailbox header size");

// Only an open chain of four ranks or more broadcasts, and only there does the
// mailbox need a 256 KiB packet and one 32-slot window per destination.
static int ring_dense_at(int world, int open) { return open && world > 3; }
static size_t ring_chunk_at(int world, int open) {
  return (size_t)(ring_dense_at(world, open) ? RING_OPEN_CHUNK_BYTES : RING_CHUNK_BYTES);
}
// Descriptor plus one packet: the stride between mailboxes.
static size_t ring_box_at(int world, int open) {
  return sizeof(ring_descriptor_t) + ring_chunk_at(world, open);
}
// Slots per (origin, rail): a dense window per destination, or one shared window.
static size_t ring_lane_slots_at(int world, int open) {
  return ring_dense_at(world, open) ? (size_t)RING_OPEN_WINDOW * ((size_t)world - 1)
                                    : (size_t)RING_WINDOW;
}
// Startup-owned ring memory, exposed so a runtime can budget it before creating
// the context. `open` is the topology the runtime will enable: a ring that never
// broadcasts keeps the smaller block. Zero means the world size is invalid or
// the product does not fit.
uint64_t roce_ring_bytes(int world, int open) {
  if (world < 2 || world > ROCE_MAX_PEERS) return 0;
  uint64_t slots = (uint64_t)world * ROCE_MAX_RAILS * ring_lane_slots_at(world, open);
  uint64_t per = (uint64_t)ring_box_at(world, open) + sizeof(uint64_t) + sizeof(ring_descriptor_t);
  if (slots > UINT64_MAX / per) return 0;
  return slots * per;
}

typedef struct {
  uint64_t region_addr;
  uint64_t ring_addr;
  // ABI 15 ring geometry. Every rank must agree; a mismatch is rejected before
  // any mailbox address is used. Zero when this rank has no ring.
  uint32_t ring_window, ring_chunk_bytes, ring_protocol;
  uint32_t ring_lane_window, ring_reserved;
  uint32_t ring_rkey[ROCE_MAX_HCAS];
  uint32_t rkey[ROCE_MAX_HCAS];
  uint16_t lid[ROCE_MAX_HCAS];
  uint8_t gid[ROCE_MAX_HCAS][16];
  uint32_t mtu[ROCE_MAX_HCAS];
  uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS][ROCE_MAX_LANES];
  uint32_t lanes;
} roce_blob_t;

typedef struct {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_mr* mr;
  struct ibv_cq* cq;
  struct ibv_qp* qp[ROCE_MAX_PEERS][ROCE_MAX_LANES];
  uint32_t outstanding[ROCE_MAX_PEERS][ROCE_MAX_LANES];
  uint64_t writes_completed;
  uint64_t bytes_posted;
  union ibv_gid gid;
  uint16_t lid;
  enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
  int world;
  int rank;
  int n_hca;
  int n_rail;
  int n_lane;
  int gid_index;
  int traffic_class;
  roce_hca_t hca[ROCE_MAX_HCAS];
  uint8_t* region;
  size_t region_bytes;
  size_t slot_bytes;
  size_t recv_off;
  size_t flag_off;
  size_t send_off;
  size_t ctrl_off;
  size_t pipe_bytes;
  size_t pipe_send_off;
  size_t pipe_recv_off;
  size_t pipe_flag_off;
  size_t shard_cap;
  size_t varlen_bytes;
  size_t varlen_recv_off;
  size_t varlen_flag_off;
  int started;
  // Set while one doorbell operation is being posted. A stop that interrupts it
  // makes the context terminal: a partially accepted work request may already
  // own a descriptor or payload, and retrying would leave a missing serial.
  int in_op;
  uint64_t peer_addr[ROCE_MAX_PEERS];
  // Per peer and rail: the local HCA that carries it and the peer's rkey on the
  // remote HCA at the other end of that link.
  int rail_hca[ROCE_MAX_PEERS][ROCE_MAX_RAILS];
  uint32_t peer_rkey[ROCE_MAX_RAILS][ROCE_MAX_PEERS];
  pthread_t thread;
  atomic_int running;
  atomic_int failed;
  uint32_t last_seq;
  uint32_t last_kind;  // kind of op last_seq (KIND_ONESHOT before the first op)
  uint64_t ops_posted;
  uint64_t writes_completed;
  uint8_t* ring;
  uint32_t ring_sources;
  uint32_t ring_destinations[ROCE_MAX_PEERS];
  int ring_open;  // the ring is a line: no link between rank 0 and rank world - 1
  struct ibv_mr* ring_mr[ROCE_MAX_HCAS];
  uint64_t ring_addr[ROCE_MAX_PEERS];
  uint32_t ring_rkey[ROCE_MAX_RAILS][ROCE_MAX_PEERS];
  uint64_t ring_serial[ROCE_MAX_PEERS][ROCE_MAX_RAILS];
  uint64_t ring_fanout_serial[ROCE_MAX_PEERS][ROCE_MAX_RAILS];
  uint64_t ring_seen[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_MAX_SLOTS];
  int ring_pending[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_MAX_SLOTS];
  // Validated copies of each in-flight mailbox's target and previous hop. The
  // mailbox is remote-writable, so completions index by these, never by the
  // descriptor as it reads later.
  uint8_t ring_target[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_MAX_SLOTS];
  uint8_t ring_previous[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_MAX_SLOTS];
  uint64_t ring_inflight[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_MAX_SLOTS];
  // Exact next serial accepted per destination lane at an intermediate, and the
  // last fanout serial accepted at a chain endpoint.
  uint64_t ring_order[ROCE_MAX_PEERS][ROCE_MAX_PEERS][ROCE_MAX_RAILS];
  uint64_t ring_forwarded;
  // All-gather relay (roce_set_relay): per peer, the rank that forwards our chunks to it (-1: we
  // send them); per source, the rank we forward its chunks to (-1: none).
  int relay;
  int8_t relay_via[ROCE_MAX_PEERS];
  int8_t relay_to[ROCE_MAX_PEERS];
  uint64_t relay_forwarded;
  char err[512];
} roce_ctx_t;

static int drain_all(roce_ctx_t* c);
static int ring_complete(roce_ctx_t* c, int src, int rail, int slot);
static int ring_dense(const roce_ctx_t* c) { return ring_dense_at(c->world, c->ring_open); }
static size_t ring_chunk(const roce_ctx_t* c) { return ring_chunk_at(c->world, c->ring_open); }
static int ring_window(const roce_ctx_t* c) {
  return ring_dense(c) ? RING_OPEN_WINDOW : RING_WINDOW;
}
static size_t ring_slot(const roce_ctx_t* c, int origin, int rail, int slot) {
  return ((size_t)origin * ROCE_MAX_RAILS + (size_t)rail) *
             ring_lane_slots_at(c->world, c->ring_open) +
         (size_t)slot;
}
static size_t ring_slots(const roce_ctx_t* c) {
  return (size_t)c->world * ROCE_MAX_RAILS * ring_lane_slots_at(c->world, c->ring_open);
}
static size_t ring_inbox_offset(const roce_ctx_t* c, size_t i) {
  return i * ring_box_at(c->world, c->ring_open);
}
static size_t ring_ack_offset(const roce_ctx_t* c, size_t i) {
  return ring_slots(c) * ring_box_at(c->world, c->ring_open) + i * sizeof(uint64_t);
}
static size_t ring_outgoing_offset(const roce_ctx_t* c, size_t i) {
  return ring_slots(c) * (ring_box_at(c->world, c->ring_open) + sizeof(uint64_t)) +
         i * sizeof(ring_descriptor_t);
}
static ring_mailbox_t* ring_inbox(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (ring_mailbox_t*)(c->ring + ring_inbox_offset(c, ring_slot(c, origin, rail, slot)));
}
static uint64_t* ring_ack(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (uint64_t*)(c->ring + ring_ack_offset(c, ring_slot(c, origin, rail, slot)));
}
static ring_descriptor_t* ring_outgoing(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (ring_descriptor_t*)(c->ring + ring_outgoing_offset(c, ring_slot(c, origin, rail, slot)));
}
// Next hop from `rank` toward target: on a line always toward it; on a cycle the
// shorter way, and on a tie the way its rank parity names, so an even rank sends
// a two-hop peer clockwise and an odd rank counter-clockwise. A tie only exists
// where the two arcs are equal, so no rank has two first hops toward one target
// and every (origin, target) lane keeps one fixed route.
static int ring_next_at(const roce_ctx_t* c, int rank, int target) {
  if (c->ring_open) return target > rank ? rank + 1 : rank - 1;
  int clockwise = (target - rank + c->world) % c->world;
  int counter = (rank - target + c->world) % c->world;
  int clockwise_first = clockwise < counter || (clockwise == counter && rank % 2 == 0);
  return clockwise_first ? (rank + 1) % c->world : (rank + c->world - 1) % c->world;
}
static int ring_next(const roce_ctx_t* c, int target) { return ring_next_at(c, c->rank, target); }
// Relay mailboxes are never addressed at a path endpoint or at the final target.
// Derive exactly which origins can traverse this rank, once before posting starts.
static uint32_t ring_source_mask(const roce_ctx_t* c) {
  uint32_t sources = 0;
  for (int origin = 0; origin < c->world; origin++) {
    if (origin == c->rank) continue;
    if (c->ring_open) {
      sources |= UINT32_C(1) << origin;
      continue;
    }
    for (int target = 0; target < c->world; target++) {
      if (target == origin || target == c->rank) continue;
      int hop = origin;
      for (int step = 0; step < c->world && hop != target; step++) {
        hop = ring_next_at(c, hop, target);
        if (hop == c->rank) {
          sources |= UINT32_C(1) << origin;
          break;
        }
      }
    }
  }
  return sources;
}
// A fanout endpoint consumes its mailbox too. Include both intermediate unicast
// routes and every chain branch through its endpoint in this startup mask, so a
// sparse poll never skips the last consumer or its upstream acknowledgment.
static uint32_t ring_destination_mask(const roce_ctx_t* c, int origin) {
  uint32_t destinations = 0;
  if (origin == c->rank) return 0;
  for (int target = 0; target < c->world; target++) {
    if (target == origin ||
        (target == c->rank && !(c->ring_open && (target == 0 || target == c->world - 1))))
      continue;
    int hop = origin;
    for (int step = 0; step < c->world && hop != target; step++) {
      hop = ring_next_at(c, hop, target);
      if (hop == c->rank) {
        destinations |= UINT32_C(1) << target;
        break;
      }
    }
  }
  return destinations;
}
// One dense slot window per destination. Destination indices skip the origin, so
// the lane base is the target renumbered over the other ranks.
static int ring_lane_base(int origin, int target) {
  return (target < origin ? target : target - 1) * RING_OPEN_WINDOW;
}
// Slot holding `serial` of origin -> target. A dense window serves one
// destination; a shared window serves every destination of the origin in the
// order it posted them, so its slot follows the serial alone.
static int ring_slot_of(const roce_ctx_t* c, int origin, int target, uint64_t serial) {
  if (ring_dense(c)) return ring_lane_base(origin, target) + (int)((serial - 1) % RING_OPEN_WINDOW);
  return (int)((serial - 1) % RING_WINDOW);
}
static int ring_relayed(const roce_ctx_t* c, int peer) {
  if (c->ring == NULL || peer == c->rank) return 0;
  if (c->ring_open) return peer != c->rank + 1 && peer != c->rank - 1;
  return peer != (c->rank + 1) % c->world && peer != (c->rank + c->world - 1) % c->world;
}
// All-gather fanout sends one packet per direction instead of one per destination.
// It is enabled only on open chains of four or more ranks. Two- and three-rank
// chains, including the non-neighbor endpoint pair of a three-rank chain, keep
// per-destination delivery. Cable-partner relay plans use per-destination delivery.
static int ring_fanout(const roce_ctx_t* c) {
  return c->ring != NULL && c->ring_open && c->world > 3 && !c->relay;
}

void roce_destroy(roce_ctx_t* c);

static void set_err(roce_ctx_t* c, const char* what, int e) {
  snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t pipe_bytes, uint64_t varlen_bytes,
                uint64_t* out) {
  // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots,
  //        pipe_send_off, pipe_recv_off, pipe_flag_off, shard_cap, max_chunks, pipe_bytes,
  //        varlen_recv_off, varlen_flag_off, varlen_bytes, varlen_max_chunks}
  if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0 ||
      (pipe_bytes % 4096) != 0 || (varlen_bytes % 4096) != 0 ||
      varlen_bytes > ((uint64_t)1 << 32)) {
    return -1;
  }
  // Reject a layout whose arithmetic would wrap; the caller sizes slots from
  // configuration, so a wrapped region must fail here rather than at the NIC.
  uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, pipe_send_off, pipe_recv_off,
      pipe_flag_off, pipe_recv_bytes, total;
  if (slot_bytes > ((uint64_t)1 << 40) || pipe_bytes > ((uint64_t)1 << 32) ||
      __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
      __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS * ROCE_MAX_RAILS,
                             (uint64_t)ROCE_FLAG_STRIDE, &flag_bytes) ||
      __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
      __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
      __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
      __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_CTRL_BYTES, &pipe_send_off)) {
    return -1;
  }
  // A shard holds at most ceil(packs / world) packs.
  uint64_t shard_cap = (pipe_bytes / 16 + (uint64_t)world - 1) / (uint64_t)world * 16;
  shard_cap = (shard_cap + 4095) / 4096 * 4096;
  pipe_recv_off = pipe_send_off + pipe_bytes;
  pipe_recv_bytes = (uint64_t)world * 2 * shard_cap;
  pipe_flag_off = pipe_recv_off + pipe_recv_bytes;
  uint64_t pipe_flag_bytes =
      pipe_bytes == 0 ? 0
                      : (uint64_t)2 * world * ROCE_MAX_CHUNKS * ROCE_MAX_RAILS * ROCE_FLAG_STRIDE;
  // The variable-length area follows; without one the region is unchanged.
  uint64_t varlen_recv_off = pipe_flag_off + pipe_flag_bytes;
  uint64_t varlen_flag_off = varlen_recv_off + (uint64_t)ROCE_SLOTS * world * varlen_bytes;
  uint64_t varlen_flag_bytes = varlen_bytes == 0
                                   ? 0
                                   : (uint64_t)ROCE_SLOTS * world * ROCE_VARLEN_MAX_CHUNKS *
                                         ROCE_MAX_RAILS * ROCE_FLAG_STRIDE;
  total = varlen_flag_off + varlen_flag_bytes;
  uint64_t recv_off = 0;
  uint64_t flag_off = recv_off + recv_bytes;
  out[0] = recv_off;
  out[1] = flag_off;
  out[2] = send_off;
  out[3] = ctrl_off;
  out[4] = total;
  out[5] = ROCE_FLAG_STRIDE;
  out[6] = ROCE_SLOTS;
  out[7] = pipe_send_off;
  out[8] = pipe_recv_off;
  out[9] = pipe_flag_off;
  out[10] = shard_cap;
  out[11] = ROCE_MAX_CHUNKS;
  out[12] = pipe_bytes;
  out[13] = varlen_recv_off;
  out[14] = varlen_flag_off;
  out[15] = varlen_bytes;
  out[16] = ROCE_VARLEN_MAX_CHUNKS;
  return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

// The single place that defines a queue pair's work-request capacity. Tests call
// this directly so the requested SGE count cannot drift from the packets posted.
static void roce_qp_cap(struct ibv_qp_cap* cap) {
  memset(cap, 0, sizeof(*cap));
  cap->max_send_wr = ROCE_SEND_DEPTH;
  cap->max_recv_wr = 1;
  cap->max_send_sge = ROCE_MAX_SEND_SGE;
  cap->max_recv_sge = 1;
  cap->max_inline_data = 16;
}

static int open_hca(roce_ctx_t* c, int h, const char* name) {
  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (list == NULL) {
    set_err(c, "ibv_get_device_list", errno);
    return -1;
  }
  struct ibv_device* dev = NULL;
  for (int i = 0; i < num; i++) {
    if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
      dev = list[i];
      break;
    }
  }
  if (dev == NULL) {
    ibv_free_device_list(list);
    snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
    return -1;
  }
  roce_hca_t* hca = &c->hca[h];
  hca->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (hca->ctx == NULL) {
    set_err(c, "ibv_open_device", errno);
    return -1;
  }
  struct ibv_port_attr port;
  if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
    set_err(c, "ibv_query_port", errno);
    return -1;
  }
  if (port.state != IBV_PORT_ACTIVE) {
    snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
    return -1;
  }
  struct ibv_device_attr attrs;
  if (ibv_query_device(hca->ctx, &attrs) != 0) {
    set_err(c, "ibv_query_device", errno);
    return -1;
  }
  if (attrs.max_sge < ROCE_MAX_SEND_SGE) {
    snprintf(c->err, sizeof(c->err),
             "RDMA device %s grants %d send SGEs; the proxy's queue pairs require %d", name,
             (int)attrs.max_sge, ROCE_MAX_SEND_SGE);
    return -1;
  }
  hca->lid = port.lid;
  hca->mtu = port.active_mtu;
  if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
    set_err(c, "ibv_query_gid", errno);
    return -1;
  }
  hca->pd = ibv_alloc_pd(hca->ctx);
  if (hca->pd == NULL) {
    set_err(c, "ibv_alloc_pd", errno);
    return -1;
  }
  hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (hca->mr == NULL) {
    set_err(c, "ibv_reg_mr(pinned region)", errno);
    return -1;
  }
  hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS * c->n_lane, NULL, NULL, 0);
  if (hca->cq == NULL) {
    set_err(c, "ibv_create_cq", errno);
    return -1;
  }
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = hca->cq;
    attr.recv_cq = hca->cq;
    attr.qp_type = IBV_QPT_RC;
    roce_qp_cap(&attr.cap);
    for (int l = 0; l < c->n_lane; l++) {
      hca->qp[p][l] = ibv_create_qp(hca->pd, &attr);
      if (hca->qp[p][l] == NULL) {
        set_err(c, "ibv_create_qp", errno);
        return -1;
      }
      struct ibv_qp_attr init;
      memset(&init, 0, sizeof(init));
      init.qp_state = IBV_QPS_INIT;
      init.pkey_index = 0;
      init.port_num = ROCE_PORT;
      init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
      int rc = ibv_modify_qp(hca->qp[p][l], &init,
                             IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      if (rc != 0) {
        set_err(c, "ibv_modify_qp(INIT)", rc);
        return -1;
      }
    }
  }
  return 0;
}

roce_ctx_t* roce_create(int world, int rank, const char* const* hca_names, int n_hca, int gid_index,
                        int traffic_class, int n_lane, void* region, uint64_t region_bytes,
                        uint64_t slot_bytes, uint64_t pipe_bytes, uint64_t varlen_bytes, char* err,
                        uint64_t err_len) {
  uint64_t layout[17];
  if (roce_layout(world, slot_bytes, pipe_bytes, varlen_bytes, layout) != 0 ||
      layout[4] > region_bytes || rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_HCAS ||
      traffic_class < 0 || traffic_class > 255 || n_lane < 1 || n_lane > ROCE_MAX_LANES) {
    snprintf(err, err_len, "invalid roce runtime geometry");
    return NULL;
  }
  roce_ctx_t* c = calloc(1, sizeof(*c));
  if (c == NULL) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  c->world = world;
  c->rank = rank;
  c->n_hca = n_hca;
  c->gid_index = gid_index;
  c->traffic_class = traffic_class;
  c->n_lane = n_lane;
  c->region = region;
  c->region_bytes = region_bytes;
  c->slot_bytes = slot_bytes;
  c->recv_off = layout[0];
  c->flag_off = layout[1];
  c->send_off = layout[2];
  c->ctrl_off = layout[3];
  c->pipe_send_off = layout[7];
  c->pipe_recv_off = layout[8];
  c->pipe_flag_off = layout[9];
  c->shard_cap = layout[10];
  c->pipe_bytes = layout[12];
  c->varlen_recv_off = layout[13];
  c->varlen_flag_off = layout[14];
  c->varlen_bytes = layout[15];
  for (int h = 0; h < n_hca; h++) {
    if (open_hca(c, h, hca_names[h]) != 0) {
      snprintf(err, err_len, "%s", c->err);
      roce_destroy(c);
      return NULL;
    }
  }
  return c;
}

// Enable before exchanging blobs; startup allocation, never during replay. The
// caller's choice of entry point is the topology, so the block is sized for it
// before anything is allocated or registered.
static int ring_enable(roce_ctx_t* c, int open) {
  if ((c->world < 2 || c->world > ROCE_MAX_PEERS) || c->started || c->ring != NULL) {
    snprintf(c->err, sizeof(c->err), "experimental ring requires 2..16 ranks before connect");
    return -1;
  }
  size_t bytes = (size_t)roce_ring_bytes(c->world, open);
  if (bytes == 0) {
    snprintf(c->err, sizeof(c->err), "ring mailbox size overflows for %d ranks", c->world);
    return -1;
  }
  void* memory = NULL;
  int rc = posix_memalign(&memory, 4096, bytes);
  if (rc != 0) {
    set_err(c, "ring mailbox allocation", rc);
    return -1;
  }
  c->ring = memory;
  c->ring_open = open;
  memset(c->ring, 0, bytes);
  for (int h = 0; h < c->n_hca; h++) {
    c->ring_mr[h] =
        ibv_reg_mr(c->hca[h].pd, c->ring, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (c->ring_mr[h] == NULL) {
      set_err(c, "register ring mailboxes", errno);
      return -1;
    }
  }
  return 0;
}

int roce_enable_ring(roce_ctx_t* c) { return ring_enable(c, 0); }

// The ring as a line in rank order: ranks 0 and world - 1 have one neighbor each and share no link.
int roce_enable_ring_open(roce_ctx_t* c) { return ring_enable(c, 1); }

int roce_local_blob(roce_ctx_t* c, void* out, uint64_t out_len) {
  if (out_len < sizeof(roce_blob_t)) {
    return -1;
  }
  roce_blob_t blob;
  memset(&blob, 0, sizeof(blob));
  blob.region_addr = (uint64_t)(uintptr_t)c->region;
  blob.ring_addr = (uint64_t)(uintptr_t)c->ring;
  blob.ring_window = c->ring == NULL ? 0 : (uint32_t)ring_window(c);
  blob.ring_chunk_bytes = c->ring == NULL ? 0 : (uint32_t)ring_chunk(c);
  blob.ring_protocol = c->ring == NULL ? 0 : RING_PROTOCOL;
  blob.ring_lane_window = c->ring == NULL ? 0 : (uint32_t)ring_window(c);
  blob.lanes = (uint32_t)c->n_lane;
  for (int h = 0; h < c->n_hca; h++) {
    blob.rkey[h] = c->hca[h].mr->rkey;
    blob.ring_rkey[h] = c->ring == NULL ? 0 : c->ring_mr[h]->rkey;
    blob.lid[h] = c->hca[h].lid;
    blob.mtu[h] = (uint32_t)c->hca[h].mtu;
    memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
    for (int p = 0; p < c->world; p++) {
      for (int l = 0; l < c->n_lane; l++) {
        blob.qp_num[h][p][l] = (p == c->rank) ? 0 : c->hca[h].qp[p][l]->qp_num;
      }
    }
  }
  memcpy(out, &blob, sizeof(blob));
  return 0;
}

// Move local HCA h's queue pair `lane` for rank p to RTS, connected to the same lane of the
// peer's HCA r.
static int connect_qp(roce_ctx_t* c, int h, int r, int p, int lane, const roce_blob_t* peer) {
  roce_hca_t* hca = &c->hca[h];
  struct ibv_qp_attr rtr;
  memset(&rtr, 0, sizeof(rtr));
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu =
      (enum ibv_mtu)(peer->mtu[r] < (uint32_t)hca->mtu ? peer->mtu[r] : (uint32_t)hca->mtu);
  rtr.dest_qp_num = peer->qp_num[r][c->rank][lane];
  rtr.rq_psn = 0;
  rtr.max_dest_rd_atomic = 1;
  rtr.min_rnr_timer = 12;
  rtr.ah_attr.is_global = 1;
  rtr.ah_attr.dlid = peer->lid[r];
  rtr.ah_attr.sl = 0;
  rtr.ah_attr.src_path_bits = 0;
  rtr.ah_attr.port_num = ROCE_PORT;
  memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[r], 16);
  rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
  rtr.ah_attr.grh.hop_limit = 64;
  rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
  rtr.ah_attr.grh.flow_label = 0;
  int rc = ibv_modify_qp(hca->qp[p][lane], &rtr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTR)", rc);
    return -1;
  }
  struct ibv_qp_attr rts;
  memset(&rts, 0, sizeof(rts));
  rts.qp_state = IBV_QPS_RTS;
  rts.timeout = 14;
  rts.retry_cnt = 7;
  rts.rnr_retry = 7;
  rts.sq_psn = 0;
  rts.max_rd_atomic = 1;
  rc = ibv_modify_qp(hca->qp[p][lane], &rts,
                     IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTS)", rc);
    return -1;
  }
  return 0;
}

// routes holds world * n_rail * 2 ints: for peer p and rail k, the local HCA
// index then the peer's HCA index of the link that carries the rail (the
// entries for this rank are ignored).  Both ends must pass mirrored routes.
int roce_connect(roce_ctx_t* c, const void* blobs, uint64_t blobs_len, const int32_t* routes,
                 int n_rail) {
  if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
    snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
    return -1;
  }
  if (n_rail < 1 || n_rail > ROCE_MAX_RAILS) {
    snprintf(c->err, sizeof(c->err), "invalid rail count %d", n_rail);
    return -1;
  }
  const roce_blob_t* all = (const roce_blob_t*)blobs;
  c->ring_sources = c->ring ? ring_source_mask(c) : 0;
  for (int origin = 0; origin < c->world; origin++)
    c->ring_destinations[origin] = c->ring ? ring_destination_mask(c, origin) : 0;
  for (int p = 0; p < c->world; p++) {
    if ((all[p].ring_addr != 0) != (c->ring != NULL)) {
      snprintf(c->err, sizeof(c->err), "ranks disagree on experimental ring transport");
      return -1;
    }
    if (all[p].ring_protocol != (uint32_t)(c->ring ? RING_PROTOCOL : 0) ||
        all[p].ring_window != (uint32_t)(c->ring ? ring_window(c) : 0) ||
        all[p].ring_lane_window != (uint32_t)(c->ring ? ring_window(c) : 0) ||
        all[p].ring_chunk_bytes != (uint32_t)(c->ring ? ring_chunk(c) : 0) ||
        all[p].ring_reserved != 0) {
      snprintf(c->err, sizeof(c->err), "ranks disagree on CPU relay protocol or geometry");
      return -1;
    }
    if (all[p].lanes != (uint32_t)c->n_lane) {
      snprintf(c->err, sizeof(c->err), "rank %d uses %u queue-pair lanes where this rank uses %d",
               p, all[p].lanes, c->n_lane);
      return -1;
    }
    c->peer_addr[p] = all[p].region_addr;
    c->ring_addr[p] = all[p].ring_addr;
  }
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank || ring_relayed(c, p)) {
      continue;
    }
    for (int k = 0; k < n_rail; k++) {
      int h = routes[(p * n_rail + k) * 2];
      int r = routes[(p * n_rail + k) * 2 + 1];
      if (h < 0 || h >= c->n_hca || r < 0 || r >= ROCE_MAX_HCAS) {
        snprintf(c->err, sizeof(c->err), "invalid route to rank %d rail %d: %d -> %d", p, k, h, r);
        return -1;
      }
      for (int j = 0; j < k; j++) {
        if (c->rail_hca[p][j] == h) {
          snprintf(c->err, sizeof(c->err), "rank %d rails %d and %d share HCA %d", p, j, k, h);
          return -1;
        }
      }
      c->rail_hca[p][k] = h;
    }
  }
  c->n_rail = n_rail;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank || ring_relayed(c, p)) {
      continue;
    }
    c->peer_addr[p] = all[p].region_addr;
    for (int k = 0; k < n_rail; k++) {
      int h = c->rail_hca[p][k];
      int r = routes[(p * n_rail + k) * 2 + 1];
      c->peer_rkey[k][p] = all[p].rkey[r];
      c->ring_rkey[k][p] = all[p].ring_rkey[r];
      for (int l = 0; l < c->n_lane; l++) {
        if (connect_qp(c, h, r, p, l, &all[p]) != 0) {
          return -1;
        }
      }
    }
  }
  return 0;
}

// plan holds 2 * world ints: plan[p] is the rank that forwards our all-gather chunks to rank p
// (-1: we send them ourselves) and plan[world + s] the rank we forward rank s's chunks to (-1:
// none).  Call after roce_connect and before roce_start.  Every rank must pass the plan derived
// from the same topology: a chunk nobody forwards stops its receiver's kernel at its wait timeout.
// A closed ring accepts the plan too, and both neighbors carry parts of the far chunk: rank r
// reaches rank r + 2 through r + 1 and forwards rank r - 1's chunks to r + 1, while rank r + 2
// forwards the remaining chunks on the other side (see relay_forwards_chunk).
int roce_set_relay(roce_ctx_t* c, const int32_t* plan, int n) {
  // A four-rank ring forwards every chunk from one neighbor to the other, so both entries must be
  // present: one of them alone would leave far chunks unforwarded.
  const int ring = c->ring != NULL && !c->ring_open && c->world == 4;
  int clockwise = ring ? (c->rank + 1) % c->world : -1;
  int counter = ring ? (c->rank + c->world - 1) % c->world : -1;
  // Every forward is a direct write to a cable neighbor: between cable partners on the mesh, and
  // between the two neighbors of a r + 2 peer on a four-rank ring. Any other ring shape reaches
  // those peers through the ring mailbox, so a plan built for direct writes does not describe it.
  // A line is the sharpest case: the plan is accepted, ring_fanout stays off, and every chunk the
  // plan forwards would leave by mailbox instead.
  if (c->ring != NULL && !ring) {
    snprintf(c->err, sizeof(c->err),
             "an all-gather relay needs the mesh transport or a closed ring of four ranks, this "
             "is %s with %d ranks",
             c->ring_open ? "an open line" : "a closed ring", c->world);
    return -1;
  }
  if (c->started || c->n_rail == 0 || n != 2 * c->world) {
    snprintf(c->err, sizeof(c->err),
             "all-gather relay needs a connected transport that has not started and a plan "
             "of %d entries, got %d",
             2 * c->world, n);
    return -1;
  }
  // On a four-rank ring the far peer is two hops away, so it must be forwarded, and each of our
  // neighbors forwards the other one's chunks to the rank across the ring.
  if (ring &&
      (plan[(c->rank + 2) % c->world] < 0 || plan[c->world + clockwise] != counter ||
       plan[c->world + counter] != clockwise || plan[c->world + (c->rank + 2) % c->world] != -1)) {
    snprintf(
        c->err, sizeof(c->err),
        "a four-rank ring needs via[%d] >= 0 for rank %d, forward[%d] = %d and forward[%d] = %d",
        (c->rank + 2) % c->world, c->rank, clockwise, counter, counter, clockwise);
    return -1;
  }
  for (int p = 0; p < c->world; p++) {
    int via = plan[p], to = plan[c->world + p];
    int bad = (p == c->rank && (via != -1 || to != -1)) ||
              (via != -1 &&
               (via < 0 || via >= c->world || via == c->rank || via == p || plan[via] != -1)) ||
              (to != -1 && (to < 0 || to >= c->world || to == c->rank || to == p));
    if (bad) {
      snprintf(c->err, sizeof(c->err), "invalid all-gather relay plan entry for rank %d: %d, %d", p,
               via, to);
      return -1;
    }
  }
  for (int p = 0; p < c->world; p++) {
    c->relay_via[p] = (int8_t)plan[p];
    c->relay_to[p] = (int8_t)plan[c->world + p];
  }
  c->relay = 1;
  return 0;
}

static int drain_cq(roce_ctx_t* c, int h) {
  struct ibv_wc wc[32];
  int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
  if (n < 0) {
    set_err(c, "ibv_poll_cq", errno);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    int relayed = (wc[i].wr_id & RING_COMPLETION) != 0;
    int src = (int)((wc[i].wr_id >> 24) & 255);
    int slot = (int)((wc[i].wr_id >> 8) & 65535);
    int rail = (int)(wc[i].wr_id & 255);
    int peer = relayed ? (int)c->ring_target[src][rail][slot] : (int)(wc[i].wr_id & 255);
    int lane = relayed ? 0 : (int)((wc[i].wr_id >> 8) & 255);
    if (wc[i].status != IBV_WC_SUCCESS) {
      snprintf(c->err, sizeof(c->err), "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
               (unsigned)peer, ibv_wc_status_str(wc[i].status), wc[i].vendor_err, c->last_seq);
      return -1;
    }
    c->hca[h].outstanding[peer][lane] -= 1;
    c->hca[h].writes_completed += 1;
    c->writes_completed += 1;
    if (relayed && ring_complete(c, src, rail, slot) != 0) return -1;
  }
  return 0;
}

// Wait until lane `lane` of HCA h can take one more stripe to peer p. Direct stripes need two
// requests and relay packets need three. A quarter of the queue leaves room
// for provider overhead and per-origin acks posted while draining completions.
static int reserve(roce_ctx_t* c, int h, int p, int lane) {
  roce_hca_t* hca = &c->hca[h];
  while (hca->outstanding[p][lane] >= ROCE_SEND_DEPTH / 4) {
    if (drain_all(c) != 0) {
      return -1;
    }
    // A peer that stopped acknowledging keeps the QP retrying for
    // a long time; honour a stop request instead of blocking
    // roce_stop (and so teardown) behind it.
    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
      snprintf(c->err, sizeof(c->err), "RoCE proxy stopped with %u writes outstanding to rank %d",
               hca->outstanding[p][lane], p);
      return -1;
    }
  }
  return 0;
}

// Each original sender has a private mailbox window per rail at every relay. It
// reuses a slot only after the end-to-end ack of the packet that last used it
// returns. Different senders never hold each other's mailbox credit, so
// a cycle of active senders cannot deadlock. Credit waits still progress
// incoming relay work and shared-QP completions; ACK posting relies on the
// queue slack left by reserve. A packet's remaining shortest-path distance
// decreases at every hop. Its header and payload remain live until the
// downstream ack.
static int ring_complete(roce_ctx_t* c, int origin, int rail, int slot) {
  int previous = (int)c->ring_previous[origin][rail][slot];
  int h = c->rail_hca[previous][rail];
  // A larger destination window can retire many downstream ACKs in one scan.
  // Keep this mailbox owned until the upstream ACK fits, without recursively
  // progressing the CQ from inside its completion callback. Each counted send
  // uses at most three WQEs, so this conservative bound also leaves hardware SQ
  // slack when data and ACK traffic share the same neighbor queue pair.
  if (c->hca[h].outstanding[previous][0] >= ROCE_SEND_DEPTH / 4) {
    c->ring_pending[origin][rail][slot] = RING_ACK_READY;
    return 0;
  }
  uint64_t serial = c->ring_inflight[origin][rail][slot];
  struct ibv_sge sge = {.addr = (uintptr_t)&serial, .length = 8};
  struct ibv_send_wr wr = {0}, *bad = NULL;
  wr.wr_id = (uint64_t)previous;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr =
      c->ring_addr[previous] + ring_ack_offset(c, ring_slot(c, origin, rail, slot));
  wr.wr.rdma.rkey = c->ring_rkey[rail][previous];
  int rc = ibv_post_send(c->hca[h].qp[previous][0], &wr, &bad);
  if (rc != 0) {
    set_err(c, "ring ack", rc);
    return -1;
  }
  c->hca[h].outstanding[previous][0]++;
  c->ring_seen[origin][rail][slot] = serial;
  c->ring_pending[origin][rail][slot] = 0;
  c->ring_forwarded++;
  return 0;
}

// Source packet batches stay on the caller's stack until ibv_post_send has
// consumed their WQEs and inline serials. Registered descriptors remain in
// ring_outgoing until the ordinary end-to-end ACK. Each QP keeps source
// submission order.
#define RING_POST_BATCH 8
typedef struct {
  struct ibv_sge body[2], bell;
  struct ibv_send_wr data, payload, commit;
  struct ibv_send_wr* bytes_wr;
  uint64_t serial;
  uint32_t bytes;
  int next, h;
} ring_packet_t;

static void ring_prepare(roce_ctx_t* c, ring_packet_t* packet, int origin, int rail, int slot,
                         int next, const uint8_t* payload, uint32_t lkey,
                         const ring_descriptor_t* descriptor) {
  memset(packet, 0, sizeof(*packet));
  packet->next = next;
  packet->h = c->rail_hca[next][rail];
  ring_descriptor_t* d = ring_outgoing(c, origin, rail, slot);
  *d = *descriptor;
  d->previous = (uint64_t)c->rank;
  packet->serial = d->serial;
  packet->bytes = (uint32_t)d->nbytes;
  uint64_t mailbox = c->ring_addr[next] + ring_inbox_offset(c, ring_slot(c, origin, rail, slot));
  packet->body[0] = (struct ibv_sge){
      .addr = (uintptr_t)d + 8, .length = sizeof(*d) - 8, .lkey = c->ring_mr[packet->h]->lkey};
  packet->body[1] =
      (struct ibv_sge){.addr = (uintptr_t)payload, .length = packet->bytes, .lkey = lkey};
  packet->bell = (struct ibv_sge){.addr = (uintptr_t)&packet->serial, .length = 8};
  packet->data.wr_id = packet->payload.wr_id = packet->commit.wr_id = (uint64_t)next;
  packet->data.sg_list = packet->body;
  packet->data.num_sge = !d->fanout && d->nbytes ? 2 : 1;
  packet->data.opcode = IBV_WR_RDMA_WRITE;
  packet->data.wr.rdma.remote_addr = mailbox + 8;
  packet->data.wr.rdma.rkey = c->ring_rkey[rail][next];
  packet->data.next = &packet->commit;
  if (d->fanout && d->nbytes) {
    // The region and the private descriptor mailbox use different MRs/rkeys.
    // One QP orders descriptor, final receive payload, then the commit serial.
    packet->payload.sg_list = &packet->body[1];
    packet->payload.num_sge = 1;
    packet->payload.opcode = IBV_WR_RDMA_WRITE;
    packet->payload.wr.rdma.remote_addr = c->peer_addr[next] + d->offset;
    packet->payload.wr.rdma.rkey = c->peer_rkey[rail][next];
    packet->payload.next = &packet->commit;
    packet->data.next = &packet->payload;
    packet->bytes_wr = &packet->payload;
  } else if (d->nbytes) {
    packet->bytes_wr = &packet->data;
  }
  packet->commit.sg_list = &packet->bell;
  packet->commit.num_sge = 1;
  packet->commit.opcode = IBV_WR_RDMA_WRITE;
  packet->commit.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  packet->commit.wr.rdma.remote_addr = mailbox;
  packet->commit.wr.rdma.rkey = c->ring_rkey[rail][next];
}

static int ring_flush(roce_ctx_t* c, ring_packet_t* packets, int count) {
  unsigned handled = 0;
  for (int i = 0; i < count; i++) {
    if (handled & (1u << i)) continue;
    int h = packets[i].h, next = packets[i].next;
    struct ibv_send_wr* first = &packets[i].data;
    struct ibv_send_wr* tail = NULL;
    for (int j = i; j < count; j++) {
      ring_packet_t* p = &packets[j];
      if (p->h != h || p->next != next) continue;
      if (tail) tail->next = &p->data;
      tail = &p->commit;
      handled |= 1u << j;
    }
    tail->next = NULL;
    struct ibv_send_wr* bad = NULL;
    int rc = ibv_post_send(c->hca[h].qp[next][0], first, &bad);
    // Verbs can accept a prefix before returning an error. Account only that
    // prefix; never retry or reuse any of this failed operation's source slots.
    // A body accepted without its commit has no CQE, so failure deliberately
    // does not establish a successful drain or permit releasing its owners.
    for (struct ibv_send_wr* wr = first; wr && (!rc || (bad && wr != bad)); wr = wr->next) {
      if (wr->send_flags & IBV_SEND_SIGNALED)
        c->hca[h].outstanding[next][0]++;
      else {
        // Count payload only when its particular WQE was accepted. Header,
        // direct-row payload and commit can fail at distinct prefix points.
        for (int j = i; j < count; j++) {
          if (packets[j].h == h && packets[j].next == next && packets[j].bytes_wr == wr)
            c->hca[h].bytes_posted += packets[j].bytes;
        }
      }
    }
    if (rc != 0) {
      set_err(c, "ring batch submit", rc);
      return -1;
    }
  }
  return 0;
}

// The caller reserves queue capacity; ring_submit never progresses another
// packet recursively. In particular a forwarding mailbox is marked pending
// before any subsequent progress call can inspect it again.
static int ring_submit(roce_ctx_t* c, int origin, int rail, int slot, int next,
                       const uint8_t* payload, uint32_t lkey, const ring_descriptor_t* descriptor) {
  ring_packet_t packet;
  ring_prepare(c, &packet, origin, rail, slot, next, payload, lkey, descriptor);
  return ring_flush(c, &packet, 1);
}

static int ring_distance(const roce_ctx_t* c, int a, int b) {
  if (c->ring_open) return a > b ? a - b : b - a;
  int cw = (b - a + c->world) % c->world, ccw = (a - b + c->world) % c->world;
  return cw < ccw ? cw : ccw;
}

// C thread-order fences do not by themselves express DMA ordering scope on
// ARM. After observing a fresh NIC commit, order descriptor/data visibility
// across the outer-shareable domain before using it or handing it to the GPU.
static inline void ring_dma_order(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dmb osh" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static int ring_service_slot(roce_ctx_t* c, int origin, int rail, int slot) {
  ring_mailbox_t* box = ring_inbox(c, origin, rail, slot);
  uint64_t serial = __atomic_load_n(&box->descriptor.serial, __ATOMIC_ACQUIRE);
  int* pending = &c->ring_pending[origin][rail][slot];
  if (*pending == RING_ACK_READY) return ring_complete(c, origin, rail, slot);
  if (*pending == 2) {
    if (__atomic_load_n(ring_ack(c, origin, rail, slot), __ATOMIC_ACQUIRE) == serial &&
        ring_complete(c, origin, rail, slot) != 0)
      return -1;
    return 0;
  }
  uint64_t seen = c->ring_seen[origin][rail][slot];
  if (serial == seen) return 1;
  if (*pending) return 0;
  ring_dma_order();
  // The mailbox stays writable by remote peers. Validate one private copy and
  // use only that copy, so a rewrite after validation cannot redirect a store or
  // a post.
  const ring_descriptor_t local = box->descriptor;
  const ring_descriptor_t* d = &local;
  int target = (int)d->target, previous = (int)d->previous;
  if (d->serial != serial || d->target >= (uint64_t)c->world || target == origin || serial < seen ||
      serial == 0 || ring_slot_of(c, origin, target, serial) != slot ||
      d->previous >= (uint64_t)c->world || previous == c->rank || ring_relayed(c, previous) ||
      (!d->fanout && target == c->rank) || d->fanout > 1 ||
      (d->fanout && (!ring_dense(c) || (target != 0 && target != c->world - 1))) ||
      (d->fanout ? d->fanout_previous >= serial : d->fanout_previous != 0) ||
      ring_distance(c, previous, target) != ring_distance(c, c->rank, target) + 1 ||
      d->nbytes > ring_chunk(c) || d->offset > c->region_bytes ||
      d->nbytes > c->region_bytes - d->offset ||
      (d->fanout && (d->target == (uint64_t)origin ||
                     (d->flag_offset != UINT64_MAX && d->flag_offset % sizeof(uint32_t)))) ||
      (d->flag_offset != UINT64_MAX &&
       (d->flag_offset > c->region_bytes - 4 || d->flag_value > UINT32_MAX))) {
    snprintf(c->err, sizeof(c->err), "invalid ring mailbox from origin %d rail %d slot %d", origin,
             rail, slot);
    return -1;
  }
  c->ring_target[origin][rail][slot] = (uint8_t)target;
  c->ring_previous[origin][rail][slot] = (uint8_t)previous;
  c->ring_inflight[origin][rail][slot] = serial;
  if (d->fanout) {
    // The payload landed in this rank's final receive row before the NIC
    // commit. Fanout is causally downstream of the origin's posting, so the
    // origin's credit gate and single-copy epoch argument keep that row live
    // for both the GPU and downstream NIC reads. Mailbox ACKs protect
    // descriptors, not GPU consumption. Queue admission still precedes flag
    // publication so a retry is inert.
    int next = target == c->rank ? previous : ring_next(c, target);
    int h = c->rail_hca[next][rail];
    if (c->hca[h].outstanding[next][0] >= ROCE_SEND_DEPTH / 4) return 1;
    ring_dma_order();
    // Registered host memory is the existing CPU/GPU control substrate.
    // Publish its payload before the existing GPU acquire.sys flag load; no GPU
    // reduction, captured address, or region geometry changes.
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (d->flag_offset != UINT64_MAX)
      __atomic_store_n((uint32_t*)(c->region + d->flag_offset), (uint32_t)d->flag_value,
                       __ATOMIC_RELEASE);
    if (target == c->rank) return ring_complete(c, origin, rail, slot);
    if (ring_submit(c, origin, rail, slot, next, c->region + d->offset, c->hca[h].mr->lkey, d) != 0)
      return -1;
    *pending = 2;
    return 0;
  }
  int next = ring_next(c, target);
  int h = c->rail_hca[next][rail];
  if (c->hca[h].outstanding[next][0] >= ROCE_SEND_DEPTH / 4) return 1;
  if (next != target) {
    if (ring_submit(c, origin, rail, slot, next, box->data, c->ring_mr[h]->lkey, d) != 0) return -1;
    *pending = 2;
    return 0;
  }
  struct ibv_sge data = {
      .addr = (uintptr_t)box->data, .length = (uint32_t)d->nbytes, .lkey = c->ring_mr[h]->lkey};
  uint32_t value = (uint32_t)d->flag_value;
  struct ibv_sge flag = {.addr = (uintptr_t)&value, .length = 4};
  struct ibv_send_wr write = {0}, mark = {0}, *bad = NULL;
  uint64_t id = RING_COMPLETION | ((uint64_t)origin << 24) | ((uint64_t)slot << 8) | (uint64_t)rail;
  write.wr_id = id;
  write.sg_list = &data;
  write.num_sge = 1;
  write.opcode = IBV_WR_RDMA_WRITE;
  write.send_flags = IBV_SEND_SIGNALED;
  write.wr.rdma.remote_addr = c->peer_addr[target] + d->offset;
  write.wr.rdma.rkey = c->peer_rkey[rail][target];
  struct ibv_send_wr* first = &write;
  if (d->flag_offset != UINT64_MAX) {
    mark.wr_id = id;
    mark.sg_list = &flag;
    mark.num_sge = 1;
    mark.opcode = IBV_WR_RDMA_WRITE;
    mark.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    mark.wr.rdma.remote_addr = c->peer_addr[target] + d->flag_offset;
    mark.wr.rdma.rkey = c->peer_rkey[rail][target];
    if (d->nbytes) {
      write.next = &mark;
      write.send_flags = 0;
    } else
      first = &mark;
  } else if (!d->nbytes) {
    snprintf(c->err, sizeof(c->err), "empty ring write without a flag");
    return -1;
  }
  int rc = ibv_post_send(c->hca[h].qp[target][0], first, &bad);
  if (rc != 0) {
    set_err(c, "ring forward", rc);
    return -1;
  }
  *pending = 1;
  c->hca[h].outstanding[target][0]++;
  c->hca[h].bytes_posted += d->nbytes;
  return 0;
}

// Forwards each sender's packets on a rail in serial order. A stripe's readiness
// flag rides its last chunk, and writes on one queue pair land in posting order,
// so chunks whose serials straddle the window boundary must be forwarded in the
// order the sender posted them. A relay sees only the serials whose route passes
// through it, so the order comes from the serials that have arrived.
//
// Retire ACK ownership separately from accepting new data. A dense window
// belongs to one destination, so only its exact next serial may be posted; a
// shared window interleaves the origin's destinations, so its arrived packets are
// forwarded in the order the origin posted them. A source can reuse an ACKed slot
// during this scan; expected-slot lookup keeps that new arrival ordered before a
// later packet or its readiness flag.
//
// The shared window's reads are not a snapshot: one slot's serial can land
// between the reads of two others, so a pass that forwarded everything it saw
// would send a stripe's flag ahead of that stripe's earlier chunk, and a receiver
// that trusts the flag reads a row the earlier chunk has not reached. Writes of
// one queue pair land in posting order, so every serial below the highest one a
// pass saw reached the mailbox no later than that one; the second pass therefore
// holds them all. Each call forwards the second pass's candidates up to that
// highest serial and leaves the rest to the next one. A dense lane needs no such
// watermark: it posts only its exact next serial, or the fanout predecessor, so
// it cannot put a flag ahead of its own stripe.
static int ring_service(roce_ctx_t* c) {
  if (c->ring == NULL || c->ring_sources == 0) return 0;
  for (int origin = 0; origin < c->world; origin++) {
    if (!(c->ring_sources & (UINT32_C(1) << origin))) continue;
    for (int rail = 0; rail < c->n_rail; rail++) {
      if (!ring_dense(c)) {
        uint64_t top = 0;
        for (int slot = 0; slot < RING_WINDOW; slot++) {
          if (c->ring_pending[origin][rail][slot]) {
            if (ring_service_slot(c, origin, rail, slot) < 0) return -1;
            continue;
          }
          uint64_t serial = __atomic_load_n(&ring_inbox(c, origin, rail, slot)->descriptor.serial,
                                            __ATOMIC_ACQUIRE);
          if (serial > top) top = serial;
        }
        int ready[RING_WINDOW], count = 0;
        uint64_t serials[RING_WINDOW];
        for (int slot = 0; slot < RING_WINDOW; slot++) {
          if (c->ring_pending[origin][rail][slot]) continue;
          uint64_t serial = __atomic_load_n(&ring_inbox(c, origin, rail, slot)->descriptor.serial,
                                            __ATOMIC_ACQUIRE);
          if (serial == c->ring_seen[origin][rail][slot] || serial > top) continue;
          int at = count++;
          while (at > 0 && serials[at - 1] > serial) {
            serials[at] = serials[at - 1];
            ready[at] = ready[at - 1];
            at--;
          }
          serials[at] = serial;
          ready[at] = slot;
        }
        for (int k = 0; k < count; k++) {
          int rc = ring_service_slot(c, origin, rail, ready[k]);
          if (rc < 0) return -1;
          if (rc > 0) break;
        }
        continue;
      }
      for (int target = 0; target < c->world; target++) {
        if (!(c->ring_destinations[origin] & (UINT32_C(1) << target))) continue;
        int base = ring_lane_base(origin, target);
        int width = RING_OPEN_WINDOW;
        for (int slot = base; slot < base + width; slot++) {
          if (c->ring_pending[origin][rail][slot] && ring_service_slot(c, origin, rail, slot) < 0)
            return -1;
        }

        if (c->ring_open && target == c->rank) {
          // Endpoints do not receive unicast packets from this lane: the last
          // relay writes those directly. Follow the source's fanout predecessor
          // chain instead of expecting every credit serial. Scan at most one
          // window per accepted packet; a missing predecessor stops publication.
          for (int accepted = 0; accepted < width; accepted++) {
            int advanced = 0;
            for (int slot = base; slot < base + width; slot++) {
              if (c->ring_pending[origin][rail][slot]) continue;
              ring_descriptor_t* d = &ring_inbox(c, origin, rail, slot)->descriptor;
              uint64_t serial = __atomic_load_n(&d->serial, __ATOMIC_ACQUIRE);
              if (serial == c->ring_seen[origin][rail][slot]) continue;
              ring_dma_order();
              if (d->fanout_previous != c->ring_order[origin][target][rail]) continue;
              int rc = ring_service_slot(c, origin, rail, slot);
              if (rc < 0) return -1;
              if (rc == 0) {
                c->ring_order[origin][target][rail] = serial;
                advanced = 1;
              }
              break;
            }
            if (!advanced) break;
          }
          continue;
        }

        for (int accepted = 0; accepted < width; accepted++) {
          uint64_t serial = c->ring_order[origin][target][rail] + 1;
          int slot = base + (int)((serial - 1) % (uint64_t)width);
          if (c->ring_pending[origin][rail][slot]) break;
          uint64_t arrived = __atomic_load_n(&ring_inbox(c, origin, rail, slot)->descriptor.serial,
                                             __ATOMIC_ACQUIRE);
          if (arrived != serial) break;
          int rc = ring_service_slot(c, origin, rail, slot);
          if (rc < 0) return -1;
          if (rc > 0) break;
          c->ring_order[origin][target][rail] = serial;
        }
      }
    }
  }
  return 0;
}

// Post without waiting for delivery. A slot is reused only after its previous
// packet's end-to-end ack, which also releases its outgoing descriptor. The
// payload source is read by the first-hop write alone; its lifetime follows
// the same completion rules as a direct write.
typedef struct {
  int peer;
  const uint8_t* send;
  uint64_t offset, flag_offset;
  uint32_t packs, value;
  int lane;  // direct writes only; ring packets share lane 0 of the neighbor QP
} roce_write_t;

// One packet per peer, rail and pass. A distant peer waiting for mailbox credit
// cannot prevent ready work for another peer or rail from reaching its first
// hop. A fanout packet writes its payload into the next hop's final receive row;
// an ordinary relayed packet keeps it in the mailbox until the final hop.
static int ring_post_many_mode(roce_ctx_t* c, const roce_write_t* writes, int count, int fanout) {
  struct cursor {
    uint32_t bytes, posted;
    uint64_t stripe;
    int done;
  } cursor[ROCE_MAX_PEERS][ROCE_MAX_RAILS] = {0};
  ring_packet_t packets[RING_POST_BATCH];
  unsigned staged[ROCE_MAX_HCAS][ROCE_MAX_PEERS] = {{0}};
  int packet_count = 0;
  int remaining = count * c->n_rail;
  if (fanout && !ring_dense(c)) {
    snprintf(c->err, sizeof(c->err), "ring broadcast needs an open chain of four ranks or more");
    return -1;
  }
  for (int i = 0; i < count; i++) {
    uint32_t offset = 0;
    for (int k = 0; k < c->n_rail; k++) {
      uint32_t packs = writes[i].packs / (uint32_t)c->n_rail +
                       ((uint32_t)k < writes[i].packs % (uint32_t)c->n_rail);
      cursor[i][k].bytes = packs * 16;
      cursor[i][k].stripe = (uint64_t)offset * 16;
      offset += packs;
    }
  }
  while (remaining) {
    int progress = 0;
    for (int k = 0; k < c->n_rail; k++) {
      for (int i = 0; i < count; i++) {
        struct cursor* at = &cursor[i][k];
        if (at->done) continue;
        const roce_write_t* write = &writes[i];
        int next = ring_next(c, write->peer), h = c->rail_hca[next][k];
        // A dense window carries one serial stream per destination; a shared window
        // carries one per (origin, rail), which is indexed at 0.
        int stream = ring_dense(c) ? write->peer : 0;
        uint64_t serial = c->ring_serial[stream][k] + 1;
        int slot = ring_slot_of(c, c->rank, write->peer, serial);
        uint64_t prior = serial > (uint64_t)ring_window(c) ? serial - (uint64_t)ring_window(c) : 0;
        if (__atomic_load_n(ring_ack(c, c->rank, k, slot), __ATOMIC_ACQUIRE) != prior ||
            c->hca[h].outstanding[next][0] + staged[h][next] >= ROCE_SEND_DEPTH / 4)
          continue;
        uint32_t bytes = at->bytes - at->posted;
        if (bytes > ring_chunk(c)) bytes = (uint32_t)ring_chunk(c);
        int last = at->posted + bytes == at->bytes;
        ring_descriptor_t descriptor = {
            .serial = serial,
            .target = (uint64_t)write->peer,
            .offset = write->offset + at->stripe + at->posted,
            .nbytes = bytes,
            .flag_offset = last ? write->flag_offset + (uint64_t)k * ROCE_FLAG_STRIDE : UINT64_MAX,
            .flag_value = write->value,
            .fanout = (uint64_t)fanout,
            .fanout_previous = fanout ? c->ring_fanout_serial[write->peer][k] : 0};
        ring_prepare(c, &packets[packet_count++], c->rank, k, slot, next,
                     write->send + at->stripe + at->posted, c->hca[h].mr->lkey, &descriptor);
        staged[h][next]++;
        c->ring_serial[stream][k] = serial;
        if (fanout) c->ring_fanout_serial[write->peer][k] = serial;
        at->posted += bytes;
        if (last) {
          at->done = 1;
          remaining--;
        }
        progress = 1;
        if (packet_count == RING_POST_BATCH) {
          if (ring_flush(c, packets, packet_count) != 0) return -1;
          packet_count = 0;
          memset(staged, 0, sizeof(staged));
        }
      }
    }
    if (!progress) {
      if (packet_count) {
        if (ring_flush(c, packets, packet_count) != 0) return -1;
        packet_count = 0;
        memset(staged, 0, sizeof(staged));
      }
      if (drain_all(c) != 0) return -1;
      uint32_t* ctrl = (uint32_t*)(c->region + c->ctrl_off);
      if (!atomic_load_explicit(&c->running, memory_order_relaxed) ||
          __atomic_load_n(&ctrl[CTRL_ERROR], __ATOMIC_ACQUIRE)) {
        snprintf(c->err, sizeof(c->err), "ring stopped waiting for a peer's mailbox credit");
        return -1;
      }
    }
  }
  return packet_count ? ring_flush(c, packets, packet_count) : 0;
}

// Every host between origin and endpoint consumes the same bytes. Fanout never
// wraps a cycle: two endpoint branches cover a chain without overlap.
static int ring_broadcast(roce_ctx_t* c, const uint8_t* send, uint64_t offset, uint32_t packs,
                          uint64_t flag_offset, uint32_t value) {
  roce_write_t writes[2];
  int count = 0;
  if (c->rank > 0) writes[count++] = (roce_write_t){0, send, offset, flag_offset, packs, value, 0};
  if (c->rank + 1 < c->world)
    writes[count++] = (roce_write_t){c->world - 1, send, offset, flag_offset, packs, value, 0};
  return ring_post_many_mode(c, writes, count, 1);
}

static int ring_post(roce_ctx_t* c, int target, const uint8_t* send, uint64_t offset,
                     uint32_t packs, uint64_t flag_offset, uint32_t value) {
  roce_write_t write = {target, send, offset, flag_offset, packs, value, 0};
  return ring_post_many_mode(c, &write, 1, 0);
}

// Write `packs` 16-byte packs from local `send` to peer p at region offset
// `remote_off`, striped across every rail, each stripe followed on its QP by a
// 4-byte seq write to `flag_off + rail * ROCE_FLAG_STRIDE`.  An empty payload
// still writes every flag, so receivers wait the same way for every chunk.
static int post_striped(roce_ctx_t* c, int p, const uint8_t* send, uint64_t remote_off,
                        uint32_t packs, uint64_t flag_off, uint32_t seq, int lane) {
  if (ring_relayed(c, p)) return ring_post(c, p, send, remote_off, packs, flag_off, seq);
  uint32_t seq_copy = seq;
  uint32_t pack_offset = 0;
  uint64_t remote = c->peer_addr[p];
  for (int k = 0; k < c->n_rail; k++) {
    int h = c->rail_hca[p][k];
    roce_hca_t* hca = &c->hca[h];
    if (reserve(c, h, p, lane) != 0) {
      return -1;
    }
    uint32_t stripe_packs = packs / (uint32_t)c->n_rail;
    if ((uint32_t)k < packs % (uint32_t)c->n_rail) {
      stripe_packs += 1;
    }
    uint32_t stripe_bytes = stripe_packs * 16;
    uint64_t byte_offset = (uint64_t)pack_offset * 16;
    struct ibv_sge flag_sge = {
        .addr = (uint64_t)(uintptr_t)&seq_copy,
        .length = 4,
        .lkey = 0,
    };
    struct ibv_send_wr flag_wr;
    memset(&flag_wr, 0, sizeof(flag_wr));
    flag_wr.wr_id = (uint64_t)p | ((uint64_t)lane << 8);
    flag_wr.sg_list = &flag_sge;
    flag_wr.num_sge = 1;
    flag_wr.opcode = IBV_WR_RDMA_WRITE;
    flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    flag_wr.wr.rdma.remote_addr = remote + flag_off + (uint64_t)k * ROCE_FLAG_STRIDE;
    flag_wr.wr.rdma.rkey = c->peer_rkey[k][p];

    struct ibv_send_wr data_wr;
    struct ibv_sge data_sge;
    struct ibv_send_wr* first_wr = &flag_wr;
    if (stripe_bytes != 0) {
      data_sge = (struct ibv_sge){
          .addr = (uint64_t)(uintptr_t)(send + byte_offset),
          .length = stripe_bytes,
          .lkey = hca->mr->lkey,
      };
      memset(&data_wr, 0, sizeof(data_wr));
      data_wr.wr_id = (uint64_t)p | ((uint64_t)lane << 8);
      data_wr.next = &flag_wr;
      data_wr.sg_list = &data_sge;
      data_wr.num_sge = 1;
      data_wr.opcode = IBV_WR_RDMA_WRITE;
      data_wr.wr.rdma.remote_addr = remote + remote_off + byte_offset;
      data_wr.wr.rdma.rkey = c->peer_rkey[k][p];
      first_wr = &data_wr;
    }
    struct ibv_send_wr* bad = NULL;
    int rc = ibv_post_send(hca->qp[p][lane], first_wr, &bad);
    if (rc != 0) {
      set_err(c, "ibv_post_send", rc);
      return -1;
    }
    hca->outstanding[p][lane] += 1;
    hca->bytes_posted += stripe_bytes;
    pack_offset += stripe_packs;
  }
  return 0;
}

static int post_many(roce_ctx_t* c, const roce_write_t* writes, int count) {
  roce_write_t relayed[ROCE_MAX_PEERS];
  int n = 0;
  for (int i = 0; i < count; i++) {
    const roce_write_t* w = &writes[i];
    if (ring_relayed(c, w->peer))
      relayed[n++] = *w;
    else if (post_striped(c, w->peer, w->send, w->offset, w->packs, w->flag_offset, w->value,
                          w->lane) != 0)
      return -1;
  }
  return n ? ring_post_many_mode(c, relayed, n, 0) : 0;
}

static int drain_all(roce_ctx_t* c) {
  for (int h = 0; h < c->n_hca; h++) {
    if (drain_cq(c, h) != 0) {
      return -1;
    }
  }
  return ring_service(c);
}

// Post one-shot op seq to every peer: the whole payload, one stripe plus its flag
// write per rail, each on the HCA routed to that peer for the rail.
static int post_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes) {
  if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->slot_bytes) {
    snprintf(c->err, sizeof(c->err),
             "RoCE payload size must be a positive multiple of 16 bytes within the slot, got %u",
             nbytes);
    return -1;
  }
  uint32_t slot = seq & 1u;
  const uint8_t* send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
  uint64_t remote_off = c->recv_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes;
  uint64_t flag_off = c->flag_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_rail *
                                        ROCE_FLAG_STRIDE;
  // A one-shot payload is delivered per destination at every size, on the ring
  // transports as well: only pipelined all-gather chunks are broadcast.
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    if (post_striped(c, p, send, remote_off, nbytes / 16, flag_off, seq, 0) != 0) {
      return -1;
    }
  }
  c->ops_posted += 1;
  return drain_all(c);
}

// First pack of shard j of a total-pack message (the kernel uses the same split).
static inline uint32_t shard_start(const roce_ctx_t* c, uint32_t total, int j) {
  return (uint32_t)(((uint64_t)j * total) / (uint64_t)c->world);
}

// Post chunk `chunk` of two-shot phase `phase` (0 reduce-scatter: peer p's shard
// of our staged input; 1 all-gather: our reduced shard) to every peer.
static int post_pipe_chunk(roce_ctx_t* c, uint32_t seq, int phase, uint32_t chunk, uint32_t total,
                           uint32_t chunk_packs) {
  const uint8_t* send = c->region + c->pipe_send_off;
  roce_write_t writes[ROCE_MAX_PEERS];
  int count = 0;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank || (phase == 1 && c->relay && c->relay_via[p] >= 0)) {
      continue;
    }
    int shard = phase == 0 ? p : c->rank;
    uint32_t first = shard_start(c, total, shard);
    uint32_t end = shard_start(c, total, shard + 1);
    uint64_t lo = (uint64_t)first + (uint64_t)chunk * chunk_packs;
    uint64_t hi = lo + chunk_packs;
    if (lo > end) {
      lo = end;
    }
    if (hi > end) {
      hi = end;
    }
    uint64_t remote_off = c->pipe_recv_off +
                          ((uint64_t)c->rank * 2 + (uint64_t)phase) * c->shard_cap +
                          (lo - first) * 16;
    uint64_t flag_off =
        c->pipe_flag_off +
        ((((uint64_t)phase * c->world + c->rank) * ROCE_MAX_CHUNKS + chunk) * (uint64_t)c->n_rail) *
            ROCE_FLAG_STRIDE;
    // The all-gather phase is the same shard for every destination, so an open
    // chain of four or more ranks without a cable relay fans it out once per
    // direction. The reduce-scatter phase is destination-specific, and a
    // cabled-pair plan keeps ABI 13's partner forwarding, so both retain the
    // per-peer path.
    if (phase == 1 && ring_fanout(c))
      return ring_broadcast(c, send + lo * 16, remote_off, (uint32_t)(hi - lo), flag_off, seq);
    writes[count++] = (roce_write_t){p,
                                     send + lo * 16,
                                     remote_off,
                                     flag_off,
                                     (uint32_t)(hi - lo),
                                     seq,
                                     (int)(chunk % (uint32_t)c->n_lane)};
  }
  return post_many(c, writes, count);
}

// True once every rail's flag of all-gather chunk `chunk` from rank s holds seq.  Each flag follows
// its stripe on one queue pair, so the chunk's bytes are in our pipe_recv row for s.
static int pipe_chunk_arrived(const roce_ctx_t* c, uint32_t seq, int s, uint32_t chunk) {
  uint64_t base = c->pipe_flag_off +
                  ((((uint64_t)c->world + s) * ROCE_MAX_CHUNKS + chunk) * (uint64_t)c->n_rail) *
                      ROCE_FLAG_STRIDE;
  for (int k = 0; k < c->n_rail; k++) {
    const uint32_t* flag = (const uint32_t*)(c->region + base + (uint64_t)k * ROCE_FLAG_STRIDE);
    if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != seq) {
      return 0;
    }
  }
  return 1;
}

// Forward all-gather chunk `chunk` of rank s, which arrived in our pipe_recv row for s, to the same
// row and flags at rank relay_to[s].
static int forward_pipe_chunk(roce_ctx_t* c, uint32_t seq, int s, uint32_t chunk, uint32_t total,
                              uint32_t chunk_packs) {
  uint32_t first = shard_start(c, total, s);
  uint32_t end = shard_start(c, total, s + 1);
  uint64_t lo = (uint64_t)first + (uint64_t)chunk * chunk_packs;
  uint64_t hi = lo + chunk_packs;
  if (lo > end) {
    lo = end;
  }
  if (hi > end) {
    hi = end;
  }
  uint64_t row = c->pipe_recv_off + ((uint64_t)s * 2 + 1) * c->shard_cap + (lo - first) * 16;
  uint64_t flag_off = c->pipe_flag_off +
                      ((((uint64_t)c->world + s) * ROCE_MAX_CHUNKS + chunk) * (uint64_t)c->n_rail) *
                          ROCE_FLAG_STRIDE;
  if (post_striped(c, c->relay_to[s], c->region + row, row, (uint32_t)(hi - lo), flag_off, seq,
                   (int)(chunk % (uint32_t)c->n_lane)) != 0) {
    return -1;
  }
  c->relay_forwarded += 1;
  return 0;
}

// On a four-rank ring the clockwise neighbor forwards even-numbered far-shard chunks and the
// counter-clockwise neighbor forwards odd-numbered chunks. Each chunk has exactly one carrier;
// odd chunk counts and partial tails need not split bytes equally. A forwarded chunk's lane has no
// per-chunk order, so splitting by chunk costs nothing.
static int relay_forwards_chunk(const roce_ctx_t* c, int s, uint32_t chunk) {
  if (c->ring == NULL || c->ring_open || c->world != 4) return 1;
  return (chunk & 1u) == ((c->rank + c->world - s) % c->world == 1 ? 0u : 1u);
}

// The phases of an op of `kind` following an op of `prev` that must wait for peers' credits (see
// the header): phase 0 after a reduce-scatter, phase 1 of a shard gather after an op that gathers.
static uint32_t credit_phases(uint32_t prev, uint32_t kind) {
  uint32_t need = 0;
  if (prev == KIND_REDUCE_SCATTER && (kind == KIND_REDUCE_SCATTER || kind == KIND_TWOSHOT)) {
    need |= CREDIT_RS;
  }
  if (kind == KIND_GATHER_SHARDS && (prev == KIND_TWOSHOT || prev == KIND_GATHER_SHARDS)) {
    need |= CREDIT_AG;
  }
  return need;
}

// Tell every peer that we began op seq, so our kernel finished the previous op: one inline 4-byte
// write of seq into the peer's ctrl credit[our rank], on rail 0 and lane 0, or an empty mailbox
// write through the ring for a rank that is not a neighbor.
static int post_credits(roce_ctx_t* c, uint32_t seq) {
  uint32_t value = seq;
  uint64_t credit_off = c->ctrl_off + 4ull * (CTRL_CREDIT + c->rank);
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    if (ring_relayed(c, p)) {
      if (ring_post(c, p, c->region, 0, 0, credit_off, seq) != 0) {
        return -1;
      }
      continue;
    }
    int h = c->rail_hca[p][0];
    if (reserve(c, h, p, 0) != 0) {
      return -1;
    }
    struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)p;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    wr.wr.rdma.remote_addr = c->peer_addr[p] + credit_off;
    wr.wr.rdma.rkey = c->peer_rkey[0][p];
    int rc = ibv_post_send(c->hca[h].qp[p][0], &wr, &bad);
    if (rc != 0) {
      set_err(c, "ibv_post_send(credit)", rc);
      return -1;
    }
    c->hca[h].outstanding[p][0] += 1;
  }
  return 0;
}

// True once every peer's credit reached seq.
static int credits_arrived(const roce_ctx_t* c, uint32_t seq) {
  const uint32_t* credit = (const uint32_t*)(c->region + c->ctrl_off) + CTRL_CREDIT;
  for (int p = 0; p < c->world; p++) {
    if (p != c->rank && (int32_t)(__atomic_load_n(&credit[p], __ATOMIC_ACQUIRE) - seq) < 0) {
      return 0;
    }
  }
  return 1;
}

// Run pipelined op seq over the phases in `phases`: post each reduce-scatter
// chunk once the kernel staged it and each all-gather chunk once the kernel
// reduced (or, for a shard gather, staged) it.  With the relay, also forward each relayed source's
// all-gather chunk as it arrives.
static int run_pipe_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes, uint32_t chunk_packs,
                       uint32_t chunks, uint32_t phases, uint32_t credit) {
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  uint32_t total = nbytes / 16;
  uint32_t widest = (total + (uint32_t)c->world - 1) / (uint32_t)c->world;
  if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->pipe_bytes || chunks == 0 ||
      chunks > ROCE_MAX_CHUNKS || chunk_packs == 0 || (uint64_t)chunks * chunk_packs < widest) {
    snprintf(c->err, sizeof(c->err),
             "invalid pipelined op %u: %u bytes in %u chunks of %u packs (capacity %zu)", seq,
             nbytes, chunks, chunk_packs, c->pipe_bytes);
    return -1;
  }
  uint32_t rs = (phases & PIPE_PHASE_RS) ? 0 : chunks;
  uint32_t ag = (phases & PIPE_PHASE_AG) ? 0 : chunks;
  uint32_t forwarded[ROCE_MAX_PEERS];
  int forwarding = 0;  // relayed sources with chunks still to forward
  for (int s = 0; s < c->world; s++) {
    forwarded[s] = chunks;
    if (c->relay && (phases & PIPE_PHASE_AG) && c->relay_to[s] >= 0) {
      forwarded[s] = 0;
      forwarding++;
    }
  }
  uint64_t idle = 0;
  while (rs < chunks || ag < chunks || forwarding > 0) {
    int posted = 0;
    if (credit != 0 && credits_arrived(c, seq)) {
      credit = 0;  // every peer began this op: no phase can overrun a peer's previous op
    }
    if (rs < chunks && !(credit & CREDIT_RS) &&
        __atomic_load_n(&ctrl[CTRL_READY_RS + rs], __ATOMIC_ACQUIRE) == seq) {
      if (post_pipe_chunk(c, seq, 0, rs, total, chunk_packs) != 0) {
        return -1;
      }
      rs++;
      posted = 1;
    }
    if (ag < chunks && !(credit & CREDIT_AG) &&
        __atomic_load_n(&ctrl[CTRL_READY_AG + ag], __ATOMIC_ACQUIRE) == seq) {
      if (post_pipe_chunk(c, seq, 1, ag, total, chunk_packs) != 0) {
        return -1;
      }
      ag++;
      posted = 1;
    }
    for (int s = 0; forwarding > 0 && !(credit & CREDIT_AG) && s < c->world; s++) {
      if (forwarded[s] < chunks && !relay_forwards_chunk(c, s, forwarded[s])) {
        // The other neighbor carries this chunk, so its row here is never read.
        if (++forwarded[s] == chunks) {
          forwarding--;
        }
        posted = 1;
        continue;
      }
      if (forwarded[s] < chunks && pipe_chunk_arrived(c, seq, s, forwarded[s])) {
        if (forward_pipe_chunk(c, seq, s, forwarded[s], total, chunk_packs) != 0) {
          return -1;
        }
        if (++forwarded[s] == chunks) {
          forwarding--;
        }
        posted = 1;
      }
    }
    if (posted) {
      idle = 0;
      continue;
    }
    if (++idle % 64 == 0) {
      if (drain_all(c) != 0) {
        return -1;
      }
      // A kernel whose peer wait timed out publishes no further chunks.
      if (ctrl[CTRL_ERROR] != 0) {
        snprintf(c->err, sizeof(c->err), "two-shot op %u abandoned: kernel timed out at seq %u",
                 seq, ctrl[CTRL_ERROR]);
        return -1;
      }
      if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
        snprintf(c->err, sizeof(c->err), "RoCE proxy stopped inside two-shot op %u", seq);
        return -1;
      }
    }
  }
  c->ops_posted += 1;
  // Release pipe_send: wait until every posted write, our all-gather included,
  // completed, so the NIC no longer reads the shard the next op will overwrite.
  for (;;) {
    if (drain_all(c) != 0) {
      return -1;
    }
    uint32_t pending = 0;
    for (int h = 0; h < c->n_hca; h++) {
      for (int p = 0; p < c->world; p++) {
        for (int l = 0; l < c->n_lane; l++) {
          pending += c->hca[h].outstanding[p][l];
        }
      }
    }
    if (pending == 0) {
      break;
    }
    if (ctrl[CTRL_ERROR] != 0) {
      snprintf(c->err, sizeof(c->err), "two-shot op %u abandoned: kernel timed out at seq %u", seq,
               ctrl[CTRL_ERROR]);
      return -1;
    }
    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
      snprintf(c->err, sizeof(c->err), "RoCE proxy stopped releasing two-shot op %u", seq);
      return -1;
    }
  }
  __atomic_store_n(&ctrl[CTRL_PIPE_DONE], seq, __ATOMIC_RELEASE);
  return 0;
}

// Run variable-length all-gather op seq: post each chunk of our contribution, from our row of
// the op's slot to the same row of every peer, once the kernel staged it.
static int run_varlen_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes, uint32_t chunk_packs,
                         uint32_t chunks) {
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  uint32_t total = nbytes / 16;
  uint32_t expected = chunk_packs == 0 ? 0 : (total + chunk_packs - 1) / chunk_packs;
  if (expected == 0) {
    expected = 1;  // an empty contribution still sends one chunk of flags
  }
  if (c->varlen_bytes == 0 || nbytes % 16 != 0 || nbytes > c->varlen_bytes || chunk_packs == 0 ||
      chunks != expected || chunks > ROCE_VARLEN_MAX_CHUNKS) {
    snprintf(c->err, sizeof(c->err),
             "invalid variable-length op %u: %u bytes in %u chunks of %u packs (capacity %zu)", seq,
             nbytes, chunks, chunk_packs, c->varlen_bytes);
    return -1;
  }
  uint32_t slot = seq & 1u;
  uint64_t row_off =
      c->varlen_recv_off + ((uint64_t)slot * c->world + (uint64_t)c->rank) * c->varlen_bytes;
  uint32_t posted = 0;
  uint64_t idle = 0;
  while (posted < chunks) {
    // A ready word of a later op releases this one's chunk too (see the header).
    uint32_t ready = __atomic_load_n(&ctrl[CTRL_READY_VL + posted], __ATOMIC_ACQUIRE);
    if ((int32_t)(ready - seq) >= 0) {
      uint64_t lo = (uint64_t)posted * chunk_packs;
      uint64_t hi = lo + chunk_packs;
      if (lo > total) {
        lo = total;
      }
      if (hi > total) {
        hi = total;
      }
      for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
          continue;
        }
        uint64_t flag_off =
            c->varlen_flag_off +
            ((((uint64_t)slot * c->world + c->rank) * ROCE_VARLEN_MAX_CHUNKS + posted) *
             (uint64_t)c->n_rail) *
                ROCE_FLAG_STRIDE;
        if (post_striped(c, p, c->region + row_off + lo * 16, row_off + lo * 16,
                         (uint32_t)(hi - lo), flag_off, seq,
                         (int)(posted % (uint32_t)c->n_lane)) != 0) {
          return -1;
        }
      }
      posted++;
      idle = 0;
      continue;
    }
    if (++idle % 64 == 0) {
      if (drain_all(c) != 0) {
        return -1;
      }
      if (ctrl[CTRL_ERROR] != 0) {
        snprintf(c->err, sizeof(c->err),
                 "variable-length op %u abandoned: kernel stopped at seq %u", seq,
                 ctrl[CTRL_ERROR]);
        return -1;
      }
      if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
        snprintf(c->err, sizeof(c->err), "RoCE proxy stopped inside variable-length op %u", seq);
        return -1;
      }
    }
  }
  c->ops_posted += 1;
  return drain_all(c);
}

static void* proxy_main(void* arg) {
  roce_ctx_t* c = (roce_ctx_t*)arg;
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
  // doorbell, request a short nanosleep between polls (the OS decides the
  // actual delay) so an idle runtime does not hold a core next to the
  // serving process.  The missed-doorbell catch-up below keeps the protocol
  // correct however long the thread is away.
  uint64_t idle = 0;
  const struct timespec nap = {0, 20000};
  while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
    uint32_t seq = __atomic_load_n(&ctrl[CTRL_SEQ], __ATOMIC_ACQUIRE);
    // Dormant chunk-ready words remain zero. Signed ready/sequence comparisons
    // cease to distinguish them after INT32_MAX, and wrapping to zero would
    // also collide with the unpoisoned error sentinel. Fail before dispatch.
    if (seq > INT32_MAX) {
      snprintf(c->err, sizeof(c->err),
               "RoCE sequence space exhausted at %u; recreate the transport", seq);
      atomic_store(&c->failed, 1);
      return NULL;
    }
    if (seq == c->last_seq) {
      idle++;
      if (idle % 64 == 0) {
        if (drain_all(c) != 0) {
          atomic_store(&c->failed, 1);
          return NULL;
        }
      }
      if (idle >= ROCE_IDLE_SPINS) {
        nanosleep(&nap, NULL);
      }
      continue;
    }
    idle = 0;
    // The doorbell holds only the newest sequence.  Our kernel for op N
    // completes on the peers' payloads alone, so op N+1 can ring before
    // this thread has seen op N (it slept, or the scheduler moved it).
    // Peers cannot get further than one op ahead of us, so at most
    // ROCE_SLOTS doorbells are pending and every slot descriptor is intact:
    // post each missed sequence in order using its per-slot record.
    uint32_t pending = seq - c->last_seq;
    if (pending > ROCE_SLOTS) {
      snprintf(c->err, sizeof(c->err), "doorbell skipped %u ops (last %u, now %u)", pending,
               c->last_seq, seq);
      atomic_store(&c->failed, 1);
      return NULL;
    }
    for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
      uint32_t slot = s & 1u;
      uint32_t nbytes = ctrl[CTRL_SLOT_NBYTES + slot];
      uint32_t kind = ctrl[CTRL_SLOT_KIND + slot];
      int rc;
      c->in_op = 1;
      if (kind == KIND_ONESHOT) {
        rc = post_op(c, s, nbytes);
      } else if (kind == KIND_TWOSHOT || kind == KIND_REDUCE_SCATTER ||
                 kind == KIND_GATHER_SHARDS) {
        uint32_t phases = kind == KIND_TWOSHOT          ? (PIPE_PHASE_RS | PIPE_PHASE_AG)
                          : kind == KIND_REDUCE_SCATTER ? PIPE_PHASE_RS
                                                        : PIPE_PHASE_AG;
        // Every rank runs the same op sequence, so all of them post and await the same credits.
        uint32_t credit = credit_phases(c->last_kind, kind);
        rc = credit != 0 ? post_credits(c, s) : 0;
        if (rc == 0) {
          rc = run_pipe_op(c, s, nbytes, ctrl[CTRL_SLOT_CHUNK_PACKS + slot],
                           ctrl[CTRL_SLOT_CHUNKS + slot], phases, credit);
        }
      } else if (kind == KIND_VARLEN) {
        rc = run_varlen_op(c, s, nbytes, ctrl[CTRL_SLOT_CHUNK_PACKS + slot],
                           ctrl[CTRL_SLOT_CHUNKS + slot]);
      } else {
        snprintf(c->err, sizeof(c->err), "unknown op kind %u at seq %u", kind, s);
        rc = -1;
      }
      if (rc != 0) {
        atomic_store(&c->failed, 1);
        return NULL;
      }
      c->last_seq = s;
      c->last_kind = kind;
      // Clear the in-operation flag only after the sequence is committed, so a
      // stop racing this update errs toward a terminal context rather than a
      // restart that could skip a serial.
      c->in_op = 0;
    }
  }
  return NULL;
}

int roce_start(roce_ctx_t* c) {
  if (atomic_load(&c->running)) {
    return 0;
  }
  // A failed context is terminal, and so is one stopped inside an operation.
  // Verbs may have accepted a prefix before failing; a serial or descriptor
  // cannot be rolled back safely, so the only recovery is a fresh group.
  if (atomic_load(&c->failed) || c->in_op) {
    if (c->err[0] == '\0')
      snprintf(c->err, sizeof(c->err),
               "RoCE context is failed or was stopped inside an operation; destroy it and create "
               "a fresh transport");
    return -1;
  }
  if (!c->started) {
    // A quiescent restart continues from the last posted sequence so ops that
    // rang the doorbell while the thread was stopped are still posted.
    volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
    c->last_seq = ctrl[0];
    c->started = 1;
  }
  atomic_store(&c->running, 1);
  int rc = pthread_create(&c->thread, NULL, proxy_main, c);
  if (rc != 0) {
    atomic_store(&c->running, 0);
    set_err(c, "pthread_create", rc);
    return -1;
  }
  return 0;
}

void roce_stop(roce_ctx_t* c) {
  if (atomic_exchange(&c->running, 0)) {
    pthread_join(c->thread, NULL);
  }
  // Teardown may stop a busy proxy, but the same context must not start again:
  // the interrupted operation can own a serial or a partially posted request.
  if (c->in_op && !atomic_load(&c->failed)) {
    snprintf(c->err, sizeof(c->err),
             "RoCE proxy was stopped during an operation; destroy it and create a fresh transport");
    atomic_store(&c->failed, 1);
  }
}

int roce_failed(roce_ctx_t* c) { return atomic_load(&c->failed); }

const char* roce_error(roce_ctx_t* c) { return c->err; }

uint64_t roce_stat(roce_ctx_t* c, int which) {
  switch (which) {
    case 0:
      return c->ops_posted;
    case 1:
      return c->writes_completed;
    case 2:
      return c->last_seq;
    case 3:
      return c->ring_forwarded;
    case 4:
      if (c->ring == NULL) return c->world - 1;
      if (c->world == 2 || (c->ring_open && (c->rank == 0 || c->rank == c->world - 1))) return 1;
      return 2;
    case 5:
      return c->relay_forwarded;
    case 6:
      return (uint64_t)c->relay;
    default:
      return 0;
  }
}

uint64_t roce_hca_stat(roce_ctx_t* c, int hca, int which) {
  if (hca < 0 || hca >= c->n_hca) {
    return -1;
  }
  switch (which) {
    case 0:
      return c->hca[hca].writes_completed;
    case 1:
      return c->hca[hca].bytes_posted;
    default:
      return 0;
  }
}

void roce_destroy(roce_ctx_t* c) {
  if (c == NULL) {
    return;
  }
  roce_stop(c);
  for (int h = 0; h < ROCE_MAX_HCAS; h++) {
    roce_hca_t* hca = &c->hca[h];
    for (int p = 0; p < ROCE_MAX_PEERS; p++) {
      for (int l = 0; l < ROCE_MAX_LANES; l++) {
        if (hca->qp[p][l] != NULL) {
          ibv_destroy_qp(hca->qp[p][l]);
        }
      }
    }
    if (hca->cq != NULL) {
      ibv_destroy_cq(hca->cq);
    }
    if (hca->mr != NULL) {
      ibv_dereg_mr(hca->mr);
    }
    if (c->ring_mr[h] != NULL) ibv_dereg_mr(c->ring_mr[h]);
    if (hca->pd != NULL) {
      ibv_dealloc_pd(hca->pd);
    }
    if (hca->ctx != NULL) {
      ibv_close_device(hca->ctx);
    }
  }
  free(c->ring);
  free(c);
}
