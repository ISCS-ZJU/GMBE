#ifndef __UTILITY_H
#define __UTILITY_H

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

/**
 *
 *   Utility for CPU codes (Host)
 *
 */

double get_cur_time();
int seq_intersect_cnt(const int* base_0, int size_0, const int* base_1,
                      int size_1);
std::vector<int> seq_intersect(const int* base_0, int size_0, const int* base_1,
                               int size_1);

/**
 *
 *   Utility for GPU codes (Device)
 *
 */

constexpr int kMaxThread = 32;
constexpr int kMaxBlock = 2048;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
__global__ void freemem(unsigned* ptr);
__global__ void generate_bitset(const int* base, const int size, const int vc,
                                const int vc_id, const int* r_offset_ptr,
                                const int* r_value_ptr, unsigned int* bitsets,
                                const int size_r, int* bitsets_size,
                                int* ismaximal, int* vvc);
void seq_intersect_cnt_gpu_wrapper(const int* base_0, int size_0,
                                   const int* base_1, int size_1, int* res_sum);

#endif
