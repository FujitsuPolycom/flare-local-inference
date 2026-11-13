// A relay group that disagrees on its wire geometry is refused at connect, before a
// queue pair leaves INIT, an all-gather relay plan is refused on a transport where a
// forward would leave by mailbox, and a device that cannot post two SGEs is refused at
// create. All run against the production proxy with a fake verbs provider.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <stdint.h>

// Both header wrappers route to the fakes, so `roce_create` sees this provider.
#undef ibv_query_port
#undef ibv_reg_mr
static int fake_query_port(struct ibv_context*, uint8_t, struct ibv_port_attr*);
#define ibv_query_port(context, port, attr) fake_query_port(context, port, attr)
#include ROCE_PROXY_SOURCE

#define WORLD 4
#define RANK 1
#define HCAS 2
#define RAILS 2
#define LANES 4
#define SLOT_BYTES 4096
#define PIPE_BYTES 4096
#define GEOMETRY "disagree on CPU relay protocol or geometry"
#define EXPECTED_RTR ((WORLD - 2) * RAILS * LANES)  // a two-HCA chain links ranks 0 and 2

static int failures;
static void require(int condition, const char* what) {
  if (!condition) fprintf(stderr, "failed: %s\n", what), failures++;
}

// --- fake verbs provider ----------------------------------------------------

static struct ibv_device devices[HCAS], *device_list[HCAS + 1];
static struct ibv_context contexts[HCAS];
static struct ibv_pd pds[HCAS];
static struct ibv_mr mrs[HCAS];
static struct ibv_cq cqs[HCAS];
static struct ibv_qp qps[HCAS][WORLD][LANES];
static const char* names[HCAS] = {"rail0", "rail1"};
static int rtr_transitions, provider_max_sge = ROCE_MAX_SEND_SGE, qps_created[HCAS];

struct ibv_device** ibv_get_device_list(int* num_devices) {
  for (int i = 0; i < HCAS; i++) device_list[i] = &devices[i];
  if (num_devices) *num_devices = HCAS;
  return device_list;
}
void ibv_free_device_list(struct ibv_device** list) { (void)list; }
const char* ibv_get_device_name(struct ibv_device* device) { return names[device - devices]; }
struct ibv_context* ibv_open_device(struct ibv_device* device) {
  contexts[device - devices].device = device;
  return &contexts[device - devices];
}
int ibv_close_device(struct ibv_context* context) { (void)context; return 0; }
static int fake_query_port(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attr) {
  (void)context; (void)port;
  *attr = (struct ibv_port_attr){.state = IBV_PORT_ACTIVE, .lid = 1, .active_mtu = IBV_MTU_4096};
  return 0;
}
int ibv_query_gid(struct ibv_context* context, uint8_t port, int index, union ibv_gid* gid) {
  (void)context; (void)port; (void)index;
  memset(gid, 0, sizeof(*gid)); return 0;
}
int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* attr) {
  (void)context;
  *attr = (struct ibv_device_attr){.max_sge = (uint32_t)provider_max_sge}; return 0;
}
struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) {
  pds[context - contexts].context = context;
  return &pds[context - contexts];
}
int ibv_dealloc_pd(struct ibv_pd* pd) { (void)pd; return 0; }
struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  (void)access; struct ibv_mr* mr = &mrs[pd - pds];
  *mr = (struct ibv_mr){.pd = pd, .addr = addr, .length = length, .rkey = 0x100u + (uint32_t)(pd - pds)};
  return mr;
}
int ibv_dereg_mr(struct ibv_mr* mr) { (void)mr; return 0; }
struct ibv_cq* ibv_create_cq(struct ibv_context* context, int cqe, void* channel,
                             struct ibv_comp_channel* comp, int comp_vector) {
  (void)channel; (void)comp; (void)comp_vector;
  cqs[context - contexts] = (struct ibv_cq){.context = context, .cqe = cqe};
  return &cqs[context - contexts];
}
int ibv_destroy_cq(struct ibv_cq* cq) { (void)cq; return 0; }
struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* attr) {
  int h = (int)(pd - pds), n = qps_created[h]++;
  (void)attr; struct ibv_qp* qp = &qps[h][(n / LANES) % WORLD][n % LANES];
  *qp = (struct ibv_qp){
      .context = pd->context, .pd = pd, .qp_num = 0x200u + (uint32_t)n, .state = IBV_QPS_RESET};
  return qp;
}
int ibv_destroy_qp(struct ibv_qp* qp) { (void)qp; return 0; }
const char* ibv_wc_status_str(enum ibv_wc_status status) { (void)status; return "fake status"; }
int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  if (attr_mask & IBV_QP_STATE) {
    if (attr->qp_state == IBV_QPS_RTR) rtr_transitions++;
    qp->state = attr->qp_state;
  }
  return 0;
}

static void* region;
static uint64_t layout[17];
static roce_ctx_t* context;
static roce_blob_t peers[WORLD];
static int32_t routes[WORLD * RAILS * 2];

// A fresh first-connect group: a context connects once, so each case needs one.
static roce_ctx_t* group(void) {
  char err[512] = {0};
  roce_ctx_t* fresh = roce_create(WORLD, RANK, names, HCAS, 3, 0, LANES, region, layout[4],
                                  SLOT_BYTES, PIPE_BYTES, 0, err, sizeof err);
  require(fresh != NULL, err[0] != '\0' ? err : "the proxy creates a context");
  if (fresh == NULL) return NULL;
  require(roce_enable_ring_open(fresh) == 0, "the chain allocates its mailboxes");
  for (int p = 0; p < WORLD; p++)
    require(roce_local_blob(fresh, &peers[p], sizeof peers[p]) == 0, "local record");
  fresh->err[0] = '\0';
  return fresh;
}

// One case: a fresh group with one peer field perturbed; the RTR count may not move.
static void refuses(const char* what, void (*mutate)(roce_blob_t*), const char* named) {
  roce_destroy(context);
  if ((context = group()) == NULL) return;
  roce_blob_t saved = peers[2];
  mutate(&peers[2]);
  int before = rtr_transitions;
  require(roce_connect(context, peers, sizeof peers, routes, RAILS) == -1, what);
  require(strstr(context->err, named) != NULL, what);
  require(rtr_transitions == before, what);
  peers[2] = saved;
}

static void drop_ring_presence(roce_blob_t* b) { b->ring_addr = 0; }
static void change_protocol(roce_blob_t* b) { b->ring_protocol ^= 1u; }
// The ABI 12 geometry this runtime replaces must not be accepted from a peer.
static void old_window(roce_blob_t* b) { b->ring_window = 8; }
static void old_lane_window(roce_blob_t* b) { b->ring_lane_window = 8; }
static void old_chunk(roce_blob_t* b) { b->ring_chunk_bytes = 65536; }
static void nonzero_reserved(roce_blob_t* b) { b->ring_reserved = 1; }
static void fewer_lanes(roce_blob_t* b) { b->lanes = LANES / 2; }

static int connect_refusals(void) {
  char err[512] = {0};
  require(roce_layout(WORLD, SLOT_BYTES, PIPE_BYTES, 0, layout) == 0, "layout for the group");
  region = calloc(1, layout[4]);
  if (region == NULL) abort();
  // Rank 1 holds a link to ranks 0 and 2 and none to rank 3.
  if ((context = group()) == NULL) return 1;
  for (int p = 0; p < WORLD; p++)
    for (int k = 0; k < RAILS; k++)
      routes[(p * RAILS + k) * 2] = routes[(p * RAILS + k) * 2 + 1] = k;

  // Positive control: an agreeing group connects, which moves its lanes to RTR.
  context->err[0] = '\0';
  int rc = roce_connect(context, peers, sizeof peers, routes, RAILS);
  require(rc == 0, "an agreeing group connects");
  if (rc != 0) fprintf(stderr, "connect failed: %s\n", context->err);
  require(rtr_transitions == EXPECTED_RTR, "every linked lane reaches RTR exactly once");

  refuses("a peer without the ring is refused", drop_ring_presence,
          "disagree on experimental ring transport");
  refuses("a peer with another relay protocol is refused", change_protocol, GEOMETRY);
  refuses("a peer with the 8-slot window is refused", old_window, GEOMETRY);
  refuses("a peer with the 8-slot lane window is refused", old_lane_window, GEOMETRY);
  refuses("a peer with 64 KiB chunks is refused", old_chunk, GEOMETRY);
  refuses("a peer with a reserved word set is refused", nonzero_reserved, GEOMETRY);
  refuses("a peer with another lane count is refused", fewer_lanes, "queue-pair lanes");
  // Nothing above touched the region the group posts into.
  require(((uint8_t*)region)[0] == 0, "a refused connect posts nothing");
  roce_destroy(context);

  // A provider that cannot do two SGEs is refused at create, by name.
  provider_max_sge = 1;
  require(roce_create(WORLD, RANK, names, HCAS, 3, 0, LANES, region, layout[4], SLOT_BYTES,
                      PIPE_BYTES, 0, err, sizeof err) == NULL &&
              strstr(err, "send SGEs") != NULL,
          "a one-SGE provider is refused by name");
  if (!failures) puts("roce connect refusal: ok");
  return failures != 0;
}

// --- relay-plan admission --------------------------------------------------

enum { MESH, RING_SHAPE, LINE };
#define PEERS 6  // the largest group an admission case builds

// A connected group of `world` ranks on `shape` for rank 1.
static roce_ctx_t* connected(int world, int shape) {
  char err[512] = {0};
  roce_ctx_t* c = roce_create(world, RANK, names, HCAS, 3, 0, LANES, region, layout[4],
                              SLOT_BYTES, PIPE_BYTES, 0, err, sizeof err);
  require(c != NULL, err[0] != '\0' ? err : "the proxy creates the admission group");
  if (c == NULL) return NULL;
  if (shape == RING_SHAPE) require(roce_enable_ring(c) == 0, "the ring allocates mailboxes");
  if (shape == LINE) require(roce_enable_ring_open(c) == 0, "the line allocates mailboxes");
  roce_blob_t blobs[PEERS]; int32_t table[PEERS * RAILS * 2];
  for (int p = 0; p < world; p++) {
    require(roce_local_blob(c, &blobs[p], sizeof blobs[p]) == 0, "local record");
    for (int k = 0; k < RAILS; k++)
      table[(p * RAILS + k) * 2] = table[(p * RAILS + k) * 2 + 1] = k;
  }
  require(roce_connect(c, blobs, sizeof(roce_blob_t) * (uint64_t)world, table, RAILS) == 0,
          "the admission group connects");
  c->err[0] = '\0';
  return c;
}

// One case: `named` is the text a refusal must carry, NULL when the plan is accepted.
static void admits(const char* what, int world, int shape, const int32_t* plan, int n, const char* named) {
  roce_ctx_t* c = connected(world, shape);
  if (c == NULL) return;
  require(roce_set_relay(c, plan, n) == (named == NULL ? 0 : -1), what);
  require(c->relay == (named == NULL), what);
  if (named != NULL) require(strstr(c->err, named) != NULL, what);
  roce_destroy(c);
}

static int relay_admission(void) {
  // The region is sized for the largest group; `roce_create` accepts a larger one.
  require(roce_layout(PEERS, SLOT_BYTES, PIPE_BYTES, 0, layout) == 0, "layout for the groups");
  region = calloc(1, layout[4]);
  if (region == NULL) abort();
  // Rank 1 on a closed four-rank ring, where rank 3 is two hops away through rank 2.
  int32_t ring[2 * WORLD] = {-1, -1, -1, 2, 2, -1, 0, -1};
  admits("a four-rank ring accepts its plan", WORLD, RING_SHAPE, ring, 2 * WORLD, NULL);
  // A plan missing the far shard leaves half of every far chunk unforwarded.
  int32_t partial[2 * WORLD] = {-1, -1, -1, -1, 2, -1, 0, -1};
  admits("a four-rank ring needs the far shard", WORLD, RING_SHAPE, partial, 2 * WORLD,
         "a four-rank ring needs via[3] >= 0 for rank 1");
  // A mesh forwards between cable partners, so its plan is accepted unchanged.
  int32_t mesh[2 * WORLD] = {-1, -1, -1, 0, -1, -1, -1, -1};
  admits("the mesh accepts a relay plan", WORLD, MESH, mesh, 2 * WORLD, NULL);
  // Every other ring or line reaches the far peer by mailbox, not by direct write.
  int32_t none[2 * PEERS] = {[0 ... 2 * PEERS - 1] = -1};
  admits("an open line refuses the plan", WORLD, LINE, none, 2 * WORLD,
         "this is an open line with 4 ranks");
  admits("a ring of three ranks refuses the plan", 3, RING_SHAPE, none, 6,
         "this is a closed ring with 3 ranks");
  admits("a ring of five ranks refuses the plan", 5, RING_SHAPE, none, 10,
         "this is a closed ring with 5 ranks");
  if (!failures) puts("roce relay admission: ok");
  return failures != 0;
}

int main(int argc, char** argv) {
  return (argc == 2 && strcmp(argv[1], "relay") == 0) ? relay_admission() : connect_refusals();
}
