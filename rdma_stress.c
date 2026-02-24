#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE (512 * 1024) // 512 KB buffers
#define MAX_MRS 256              // Try to push past the suspected 128 limit

int main() {
  struct ibv_device **dev_list;
  struct ibv_context *context;
  struct ibv_pd *pd;
  int num_devices;

  printf("Starting AppleThunderboltRDMA Driver Stress Test...\n");

  // 1. Get RDMA devices
  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0) {
    fprintf(stderr,
            "Error: No RDMA devices found. Is Thunderbolt connected?\n");
    return 1;
  }

  printf("Found %d RDMA device(s).\n", num_devices);

  // 2. Try each device until we find one that works
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
    printf("  Success! Using '%s'\n", ibv_get_device_name(dev_list[d]));
    break;
  }
  if (!pd) {
    fprintf(stderr, "Error: No usable RDMA device found.\n");
    return 1;
  }

  printf("Protection Domain allocated. Beginning MR registration loop...\n\n");

  struct ibv_mr *mrs[MAX_MRS];
  void *buffers[MAX_MRS];

  // 4. The Stress Test: Register Memory Regions until failure or limit
  for (int i = 0; i < MAX_MRS; i++) {
    // Allocate page-aligned memory for the driver
    if (posix_memalign(&buffers[i], 4096, BUFFER_SIZE) != 0) {
      fprintf(stderr, "Memory allocation failed at index %d\n", i);
      return 1;
    }

    // Fill buffer with dummy data to force physical memory allocation
    memset(buffers[i], 0xAA, BUFFER_SIZE);

    // Attempt to register the Memory Region
    mrs[i] = ibv_reg_mr(pd, buffers[i], BUFFER_SIZE,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ);

    if (!mrs[i]) {
      // A well-behaved driver will return NULL and set errno here.
      fprintf(stderr,
              "\n[!] ibv_reg_mr gracefully failed at iteration %d (Total MRs: "
              "%d)\n",
              i, i);
      fprintf(
          stderr,
          "[!] This indicates a hard table limit, but avoided a segfault.\n");
      break;
    }

    printf("\rSuccessfully registered MR %d / %d...", i + 1, MAX_MRS);
    fflush(stdout);
  }

  printf("\n\nTest concluded. Cleaning up...\n");

  // Cleanup (If it survived without segfaulting)
  for (int i = 0; i < MAX_MRS; i++) {
    if (mrs[i])
      ibv_dereg_mr(mrs[i]);
    if (buffers[i])
      free(buffers[i]);
  }

  ibv_dealloc_pd(pd);
  ibv_close_device(context);
  ibv_free_device_list(dev_list);

  return 0;
}