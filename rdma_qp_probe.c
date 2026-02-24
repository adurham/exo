#include <errno.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Probe the Apple Thunderbolt RDMA driver's QP and CQ limits.
 *
 * Tests:
 * 1. ibv_query_device — what does the driver report?
 * 2. CQ creation — what's the max CQ depth?
 * 3. QP creation — what max_send_wr / max_recv_wr does the driver accept?
 * 4. Post work requests — how many can we actually post before failure?
 */

int main() {
  struct ibv_device **dev_list;
  struct ibv_context *context = NULL;
  struct ibv_pd *pd = NULL;
  int num_devices;

  printf("=== Apple Thunderbolt RDMA QP/CQ Depth Probe ===\n\n");

  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0) {
    fprintf(stderr, "No RDMA devices found.\n");
    return 1;
  }

  for (int d = 0; d < num_devices; d++) {
    context = ibv_open_device(dev_list[d]);
    if (!context)
      continue;
    pd = ibv_alloc_pd(context);
    if (pd) {
      printf("Using device: %s\n\n", ibv_get_device_name(dev_list[d]));
      break;
    }
    ibv_close_device(context);
    context = NULL;
  }
  if (!pd) {
    fprintf(stderr, "No usable RDMA device.\n");
    return 1;
  }

  // === TEST 1: Query device capabilities ===
  printf("[Test 1] ibv_query_device\n");
  struct ibv_device_attr dev_attr;
  if (ibv_query_device(context, &dev_attr) == 0) {
    printf("  max_qp:           %d\n", dev_attr.max_qp);
    printf("  max_qp_wr:        %d\n", dev_attr.max_qp_wr);
    printf("  max_sge:          %d\n", dev_attr.max_sge);
    printf("  max_cq:           %d\n", dev_attr.max_cq);
    printf("  max_cqe:          %d\n", dev_attr.max_cqe);
    printf("  max_mr:           %d\n", dev_attr.max_mr);
    printf("  max_pd:           %d\n", dev_attr.max_pd);
    printf("  max_qp_rd_atom:   %d\n", dev_attr.max_qp_rd_atom);
    printf("  max_qp_init_rd_atom: %d\n", dev_attr.max_qp_init_rd_atom);
    printf("  max_srq:          %d\n", dev_attr.max_srq);
    printf("  max_srq_wr:       %d\n", dev_attr.max_srq_wr);
  } else {
    printf("  ibv_query_device failed (errno=%d)\n", errno);
  }
  printf("\n");

  // === TEST 2: CQ creation — probe max depth ===
  printf("[Test 2] CQ max depth\n");
  int cq_sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  int max_cq_depth = 0;
  for (int i = 0; i < 8; i++) {
    struct ibv_cq *cq = ibv_create_cq(context, cq_sizes[i], NULL, NULL, 0);
    if (cq) {
      printf("  CQ depth %4d: OK\n", cq_sizes[i]);
      max_cq_depth = cq_sizes[i];
      ibv_destroy_cq(cq);
    } else {
      printf("  CQ depth %4d: FAIL (errno=%d)\n", cq_sizes[i], errno);
    }
  }
  printf("  Max CQ depth: %d\n\n", max_cq_depth);

  // === TEST 3: QP creation — probe max send/recv WR ===
  printf("[Test 3] QP max send/recv WR\n");
  int wr_sizes[] = {8, 16, 32, 64, 128, 256, 512};

  // We need a CQ for QP creation
  struct ibv_cq *test_cq = ibv_create_cq(
      context, max_cq_depth > 0 ? max_cq_depth : 64, NULL, NULL, 0);
  if (!test_cq) {
    printf("  Failed to create test CQ\n");
  } else {
    for (int i = 0; i < 7; i++) {
      struct ibv_qp_init_attr init_attr;
      memset(&init_attr, 0, sizeof(init_attr));
      init_attr.qp_context = context;
      init_attr.send_cq = test_cq;
      init_attr.recv_cq = test_cq;
      init_attr.srq = NULL;
      init_attr.cap.max_send_wr = wr_sizes[i];
      init_attr.cap.max_recv_wr = wr_sizes[i];
      init_attr.cap.max_send_sge = 1;
      init_attr.cap.max_recv_sge = 1;
      init_attr.cap.max_inline_data = 0;
      init_attr.qp_type = IBV_QPT_UC;
      init_attr.sq_sig_all = 0;

      struct ibv_qp *qp = ibv_create_qp(pd, &init_attr);
      if (qp) {
        printf(
            "  max_send_wr=%3d max_recv_wr=%3d: OK (actual: send=%d recv=%d)\n",
            wr_sizes[i], wr_sizes[i], init_attr.cap.max_send_wr,
            init_attr.cap.max_recv_wr);
        ibv_destroy_qp(qp);
      } else {
        printf("  max_send_wr=%3d max_recv_wr=%3d: FAIL (errno=%d)\n",
               wr_sizes[i], wr_sizes[i], errno);
      }
    }
    ibv_destroy_cq(test_cq);
  }
  printf("\n");

  // === TEST 4: Post recv WRs — how many can we post before failure? ===
  printf("[Test 4] Max outstanding recv WRs (actual post test)\n");

  // Create a fresh CQ + QP for posting
  struct ibv_cq *post_cq = ibv_create_cq(
      context, max_cq_depth > 0 ? max_cq_depth : 64, NULL, NULL, 0);
  if (!post_cq) {
    printf("  Failed to create CQ\n");
    goto cleanup;
  }

  {
    struct ibv_qp_init_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));
    init_attr.qp_context = context;
    init_attr.send_cq = post_cq;
    init_attr.recv_cq = post_cq;
    init_attr.srq = NULL;
    init_attr.cap.max_send_wr = 128; // request high
    init_attr.cap.max_recv_wr = 128;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 0;
    init_attr.qp_type = IBV_QPT_UC;
    init_attr.sq_sig_all = 0;

    struct ibv_qp *post_qp = ibv_create_qp(pd, &init_attr);
    if (!post_qp) {
      printf("  Failed to create QP for posting (errno=%d)\n", errno);
      printf("  (Actual caps: send=%d recv=%d)\n", init_attr.cap.max_send_wr,
             init_attr.cap.max_recv_wr);
      ibv_destroy_cq(post_cq);
      goto cleanup;
    }
    printf("  QP created with actual caps: send=%d recv=%d\n",
           init_attr.cap.max_send_wr, init_attr.cap.max_recv_wr);

    // Transition QP to INIT state so we can post recvs
    struct ibv_qp_attr qp_attr = {};
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = 1;
    qp_attr.pkey_index = 0;
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE;
    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(post_qp, &qp_attr, mask) != 0) {
      printf("  Failed to transition QP to INIT (errno=%d)\n", errno);
      ibv_destroy_qp(post_qp);
      ibv_destroy_cq(post_cq);
      goto cleanup;
    }

    // Allocate a small buffer and register it
    void *buf;
    posix_memalign(&buf, 4096, 4096);
    memset(buf, 0, 4096);
    struct ibv_mr *mr =
        ibv_reg_mr(pd, buf, 4096,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
    if (!mr) {
      printf("  Failed to register MR\n");
      ibv_destroy_qp(post_qp);
      ibv_destroy_cq(post_cq);
      free(buf);
      goto cleanup;
    }

    // Post recv WRs one at a time until failure
    int posted = 0;
    for (int i = 0; i < 512; i++) {
      struct ibv_sge sge;
      sge.addr = (uintptr_t)buf;
      sge.length = 64;
      sge.lkey = mr->lkey;

      struct ibv_recv_wr wr, *bad_wr;
      wr.wr_id = i;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.next = NULL;

      int ret = ibv_post_recv(post_qp, &wr, &bad_wr);
      if (ret != 0) {
        printf("  ibv_post_recv failed at WR #%d (errno=%d)\n", i, ret);
        break;
      }
      posted++;
    }
    printf("  Successfully posted %d recv WRs before failure/limit\n", posted);

    ibv_dereg_mr(mr);
    free(buf);
    ibv_destroy_qp(post_qp);
  }
  ibv_destroy_cq(post_cq);

  printf("\n============ SUMMARY ============\n");
  printf("JACCL uses MAX_SEND_WR=%d MAX_RECV_WR=%d\n", 32, 32);
  printf("JACCL MeshGroup PIPELINE = NUM_BUFFERS\n");
  printf("With PIPELINE=8, prefill posts 8 * num_peers * 2 WRs\n");
  printf("  2 peers: 8 * 1 * 2 = 16 outstanding\n");
  printf("  3 peers: 8 * 2 * 2 = 32 outstanding\n");
  printf("=================================\n");

cleanup:
  ibv_dealloc_pd(pd);
  ibv_close_device(context);
  ibv_free_device_list(dev_list);
  return 0;
}
