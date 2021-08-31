/**
 * A simple test stencil computation that computes the sum of each point
 * and its four direct neighbours in a 2D grid.
 * The implementation is very naive and optimised.
 *
 * Author: Paul Metzger
 */

#include <NoCL.h>

#define DEBUG false

void populate_in_buf(int *in_buf, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x)
      in_buf[y * y_size + x] = 1;
  }
}

// Generate a 'golden output' to check if the output computed
// by the GPU kernel is correct.
void generate_golden_output(int *in_buf, int *golden_out, int x_size,
                            int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x) {
      const int ind = y * y_size + x;

      int result = in_buf[ind];
      if (x < x_size - 1)
        result += in_buf[y * y_size + x + 1];
      if (x > 0)
        result += in_buf[y * y_size + x - 1];
      if (y < y_size - 1)
        result += in_buf[(y + 1) * y_size + x];
      if (y > 0)
        result += in_buf[(y - 1) * y_size + x];
      golden_out[ind] = result;
    }
  }
}

// Check if the results computed by the GPU kernel match
// the golden output.
bool check_output(int *out_buf, int *golden_buf, int buf_size) {
  for (int i = 0; i < buf_size; ++i) {
    if (out_buf[i] != golden_buf[i]) {
      puts("Detected an error at index: ");
      puthex(i);
      putchar('\n');
      puts("Expected value: ");
      puthex(golden_buf[i]);
      putchar('\n');
      puts("Computed value: ");
      puthex(out_buf[i]);
      putchar('\n');
      return false;
    }
  }
  return true;
}

struct SimpleStencil : Kernel {
  int x_size = 0;
  int y_size = 0;
  int *out_buf, *in_buf;

  void kernel() {
    int x          = threadIdx.x;
    const int y    = blockIdx.y * blockDim.y + threadIdx.y;
    int global_ind = y * y_size + x;

    const int shared_mem_x_size = SIMTLanes * 3;
    auto c = shared.array<int, shared_mem_x_size, SIMTWarps>();
    for (int i = 0; i < x_size; i += SIMTLanes) {
      // Load values into local memory
      c[threadIdx.y][x % shared_mem_x_size] = in_buf[global_ind];
      if (i + SIMTLanes < x_size) c[threadIdx.y][(x + SIMTLanes) % shared_mem_x_size] = in_buf[global_ind];
      noclConverge();
      __syncthreads();

      int result = in_buf[global_ind];
      if (x < x_size - 1) result += c[threadIdx.y][(x + 1) % shared_mem_x_size];
      noclConverge();

      if (x > 0)          result += c[threadIdx.y][(x - 1) % shared_mem_x_size];
      noclConverge();

      if (y < y_size - 1) {
        if (threadIdx.y == blockDim.y - 1) result += in_buf[(y + 1) * y_size + x];
        else result += c[threadIdx.y + 1][x % shared_mem_x_size]; // in_buf[(y + 1) * y_size + x];
      }
      noclConverge();

      if (y > 0) {
        if (threadIdx.y == 0) result += in_buf[(y - 1) * y_size + x];
        else result += c[threadIdx.y - 1][x % shared_mem_x_size]; // in_buf[(y - 1) * y_size + x];
      }
      noclConverge();

      out_buf[global_ind] = result;
      x          += SIMTLanes;
      global_ind += SIMTLanes;
    }

    // Code that generates wrong results sometimes.
    // if (x < x_size - 1) out_buf[ind] += in_buf[y * y_size + x + 1];
    // if (x > 0)          out_buf[ind] += in_buf[y * y_size + x - 1];
    // if (y < y_size - 1) out_buf[ind] += in_buf[(y + 1) * y_size + x];
    // if (y > 0)          out_buf[ind] += in_buf[(y - 1) * y_size + x];
  }
};

int main() {
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int buf_size_x = 1024;
  int buf_size_y = 1024;
  if (isSim) {
    buf_size_x = 64;
    buf_size_y = 64;
  }

  const int buf_size = buf_size_x * buf_size_y;
  simt_aligned int in_buf[buf_size];
  simt_aligned int out_buf[buf_size];
  int golden_out_buf[buf_size];

  // Prepare buffers
  // Zero out the ouput buffers
  /*for (int i = 0; i < buf_size; ++i) {
    out_buf[i] = 0;
    golden_out_buf[i] = 0;
  }*/
  populate_in_buf(in_buf, buf_size_x, buf_size_y);

  // Generate the golden output to check if
  // the results computed by the GPU kernel are correct (see below).
  generate_golden_output(in_buf, golden_out_buf, buf_size_x, buf_size_y);

  // Do computation on the GPU
  SimpleStencil k;
  k.blockDim.x =
      SIMTLanes; // FIXME: Ensure that buf_size_x is a multiple of SIMTLanes.
  k.blockDim.y =
      SIMTWarps; // FIXME: Ensure that buf_size_y is a multiple of SIMTWarps.
  k.gridDim.x = SIMTLanes;
  k.gridDim.y = buf_size_y / SIMTWarps;
  k.x_size = buf_size_x;
  k.y_size = buf_size_y;
  k.out_buf = out_buf;
  k.in_buf = in_buf;
  if (DEBUG)
    puts("Kernel running... ");
  noclRunKernelAndDumpStats(&k);
  if (DEBUG)
    puts("Done\n");

  // Check result
  bool ok = check_output(out_buf, golden_out_buf, buf_size_x * buf_size_y);
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}