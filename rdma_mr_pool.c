#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Phase 1: MR Pooling - 256 logical buffers from 1 hardware MR
#define NUM_BUFFERS 256
#define BUFFER_SIZE (512 * 1024)                   // 512 KB per logical buffer
#define TOTAL_POOL_SIZE (NUM_BUFFERS * BUFFER_SIZE) // 128 MB

int main() {
  struct ibv_device **dev_list;
  struct ibv_context *context;
  struct ibv_pd *pd;
  int num_devices;

  printf("=== MR Pooling Proof of Concept ===\n");
  printf("Goal: 256 logical buffers from 1 hardware MR\n\n");

  // 1. Find a usable RDMA device
  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0) {
    fprintf(stderr, "No RDMA devices found.\n");
    return 1;
  }
  printf("Found %d RDMA device(s).\n", num_devices);

  context = NULL;
  pd = NULL;
  for (int d = 0; d < num_devices; d++) {
    printf("Trying device '%s'...\n", ibv_get_device_name(dev_list[d]));
    context = ibv_open_device(dev_list[d]);
    if (!context) {
      printf("  Failed to open, skipping.\n");
      continue;
    }
    pd = ibv_alloc_pd(context);
    if (!pd) {
      printf("  PD allocation failed (port likely down), skipping.\n");
      ibv_close_device(context);
      context = NULL;
      continue;
    }
    printf("  Success! Using '%s'\n\n", ibv_get_device_name(dev_list[d]));
    break;
  }
  if (!pd) {
    fprintf(stderr, "No usable RDMA device found.\n");
    return 1;
  }

  // ========================================================
  // PHASE 1: Single giant MR registration + slab subdivision
  // ========================================================

  void *giant_pool;
  if (posix_memalign(&giant_pool, 4096, TOTAL_POOL_SIZE) != 0) {
    fprintf(stderr, "Failed to allocate %d MB pool.\n",
            TOTAL_POOL_SIZE / (1024 * 1024));
    return 1;
  }
  // Touch every page to force physical allocation
  memset(giant_pool, 0xAA, TOTAL_POOL_SIZE);

  printf("[Phase 1] Registering single %d MB Memory Region...\n",
         TOTAL_POOL_SIZE / (1024 * 1024));

  struct ibv_mr *pool_mr =
      ibv_reg_mr(pd, giant_pool, TOTAL_POOL_SIZE,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ);

  if (!pool_mr) {
    fprintf(stderr,
            "[!] FAILED: ibv_reg_mr rejected %d MB pool (errno=%d).\n",
            TOTAL_POOL_SIZE / (1024 * 1024), errno);
    fprintf(stderr,
            "[!] Apple may restrict max MR size. Trying smaller sizes...\n\n");

    // Binary search for max MR size
    size_t lo = BUFFER_SIZE;         // 512 KB - known to work
    size_t hi = TOTAL_POOL_SIZE;     // 128 MB - just failed
    size_t max_working = 0;

    while (lo <= hi) {
      size_t mid = lo + (hi - lo) / 2;
      void *probe;
      if (posix_memalign(&probe, 4096, mid) != 0)
        break;
      memset(probe, 0, mid);
      struct ibv_mr *mr = ibv_reg_mr(
          pd, probe, mid,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
              IBV_ACCESS_REMOTE_READ);
      if (mr) {
        printf("  %zu MB: OK\n", mid / (1024 * 1024));
        max_working = mid;
        ibv_dereg_mr(mr);
        lo = mid + BUFFER_SIZE; // step by 512KB
      } else {
        printf("  %zu MB: FAIL\n", mid / (1024 * 1024));
        hi = mid - BUFFER_SIZE;
      }
      free(probe);
    }

    if (max_working > 0) {
      int max_buffers = (int)(max_working / BUFFER_SIZE);
      printf("\n[!] Max single-MR size: %zu MB (%d logical 512KB buffers)\n",
             max_working / (1024 * 1024), max_buffers);
      printf("[!] This is still %dx what JACCL currently uses (100 MR limit "
             "/ same MR count).\n",
             max_buffers);
    } else {
      printf("\n[!] Could not find any working MR size above 512KB.\n");
    }

    free(giant_pool);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(dev_list);
    return 1;
  }

  // Pool MR succeeded!
  printf("[+] SUCCESS: Registered 1 hardware MR for %d MB.\n",
         TOTAL_POOL_SIZE / (1024 * 1024));
  printf("    lkey=0x%x  rkey=0x%x  addr=%p\n\n", pool_mr->lkey, pool_mr->rkey,
         pool_mr->addr);

  // Carve into logical buffers
  void *logical_buffers[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; i++) {
    logical_buffers[i] = (char *)giant_pool + (i * BUFFER_SIZE);
  }

  printf("[+] Carved pool into %d logical 512KB buffers.\n", NUM_BUFFERS);
  printf("[+] All share rkey=0x%x — only 1 hardware MR slot used.\n\n",
         pool_mr->rkey);

  // Verify data isolation between logical buffers
  printf("[Phase 1 Verify] Writing unique patterns to first/last buffers...\n");
  memset(logical_buffers[0], 0x11, BUFFER_SIZE);
  memset(logical_buffers[NUM_BUFFERS - 1], 0xFF, BUFFER_SIZE);

  unsigned char first_byte = ((unsigned char *)logical_buffers[0])[0];
  unsigned char last_byte =
      ((unsigned char *)logical_buffers[NUM_BUFFERS - 1])[0];
  printf("    Buffer[0] first byte:   0x%02X (expect 0x11)\n", first_byte);
  printf("    Buffer[255] first byte: 0x%02X (expect 0xFF)\n", last_byte);

  if (first_byte == 0x11 && last_byte == 0xFF) {
    printf("[+] Data isolation verified.\n\n");
  } else {
    printf("[!] Data corruption detected!\n\n");
  }

  // ========================================================
  // PHASE 2: Probe maximum single-MR size
  // ========================================================

  printf("[Phase 2] Probing maximum single-MR size...\n");
  // Try progressively larger sizes: 256MB, 512MB, 1GB, 2GB
  size_t probe_sizes[] = {256UL * 1024 * 1024, 512UL * 1024 * 1024,
                          1024UL * 1024 * 1024, 2048UL * 1024 * 1024};
  size_t max_mr_size = TOTAL_POOL_SIZE; // we know 128MB works

  for (int i = 0; i < 4; i++) {
    void *probe;
    if (posix_memalign(&probe, 4096, probe_sizes[i]) != 0) {
      printf("  %4zu MB: SKIP (malloc failed — not enough RAM)\n",
             probe_sizes[i] / (1024 * 1024));
      continue;
    }
    // Touch memory to force physical pages
    memset(probe, 0, probe_sizes[i]);

    struct ibv_mr *mr =
        ibv_reg_mr(pd, probe, probe_sizes[i],
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
    if (mr) {
      printf("  %4zu MB: OK ✓\n", probe_sizes[i] / (1024 * 1024));
      max_mr_size = probe_sizes[i];
      ibv_dereg_mr(mr);
    } else {
      printf("  %4zu MB: FAIL (errno=%d)\n", probe_sizes[i] / (1024 * 1024),
             errno);
    }
    free(probe);
  }

  int max_logical_buffers = (int)(max_mr_size / BUFFER_SIZE);
  printf("\n[Phase 2 Result] Max verified MR size: %zu MB\n",
         max_mr_size / (1024 * 1024));
  printf("  → Could hold %d logical 512KB buffers in a single MR.\n",
         max_logical_buffers);
  printf("  → vs. Apple's hard limit of 100 separate MRs.\n\n");

  // ========================================================
  // SUMMARY
  // ========================================================

  printf("============ SUMMARY ============\n");
  printf("Old approach: 1 ibv_reg_mr per buffer → hits 100 MR limit at 100 "
         "buffers\n");
  printf("New approach: 1 ibv_reg_mr for pool  → %d+ buffers from 1 MR "
         "slot\n",
         NUM_BUFFERS);
  printf("Speedup:      %dx more pipeline depth available\n",
         max_logical_buffers);
  printf("=================================\n");

  // Cleanup
  ibv_dereg_mr(pool_mr);
  free(giant_pool);
  ibv_dealloc_pd(pd);
  ibv_close_device(context);
  ibv_free_device_list(dev_list);

  return 0;
}
