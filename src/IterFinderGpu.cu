
#include "IterFinderGpu.h"
#include "WorkList.cuh"
#include <queue>
#include <algorithm>

extern bool printSMTime;

#define SHARED_CACHE_SIZE 0x1000
#define BUFFER_PER_WARP (2 * MAX_DEGREE_BOUND + 2 * MAX_2_H_DEGREE_BOUND)

__device__ __forceinline__ void IterProcess(CSRBiGraph &graph, int *L_vertices,
                                            int *L_level, int L_size,
                                            int *C_vertices,
                                            int *C_level_neighbors, int C_size,
                                            int *exe_stack, int first_id,
                                            int &local_mb_counter,
                                            unsigned long long *non_maximal = nullptr) {
  int level = 1;
  if (get_lane_id() == 0) exe_stack[0] = first_id;
  bool is_maximal;

  while (level != 0) {  // stack is not empty

    int cur_cid = -1;
    int last_cid = exe_stack[level - 1];
    for (int i = last_cid; i < C_size; i++) {
      int c_level_neighbors = C_level_neighbors[i];
      if ((c_level_neighbors >> 16) > level - 1 &&
          (c_level_neighbors & 0xffff)) {
        cur_cid = i;
        break;
      }
    }

    if (cur_cid != -1) {  // next candidate id is found
      is_maximal = finder_push(graph, cur_cid, level, L_vertices, L_level,
                               L_size, C_vertices, C_level_neighbors, C_size);

      if (is_maximal) local_mb_counter++;
      else {
        if (get_lane_id() == 0 && non_maximal != nullptr) {
          atomicAdd(non_maximal, 1);
        }
      }
      if (is_maximal && (C_level_neighbors[cur_cid] & 0xffff) > 1) {
        if (get_lane_id() == 0) {
          exe_stack[level - 1] = cur_cid;
          exe_stack[level] = cur_cid + 1;
        }
        level++;
      } else {
        if (get_lane_id() == 0) exe_stack[level - 1] = cur_cid + 1;

        finder_pop(graph, cur_cid, level, L_vertices, L_level, L_size,
                   C_vertices, C_level_neighbors, C_size);
      }
    } else {  // not found
      level--;

      if (level > 0) {
        cur_cid = exe_stack[level - 1];
        finder_pop(graph, cur_cid, level, L_vertices, L_level, L_size,
                   C_vertices, C_level_neighbors, C_size);
        if (get_lane_id() == 0) exe_stack[level - 1] = cur_cid + 1;
      }
    }
  }
}

__launch_bounds__(32 * WARP_PER_SM, 1) __global__
    void IterFinderKernel(CSRBiGraph graph, int *global_buffer,
                          int *maximal_bicliques, int *processing_vertex,
                          double clock_rate = 0
                          ) {
  auto sm_start = clock64();
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  // int num_warps = WARP_PER_BLOCK * gridDim.x;
  int *warp_buffer = global_buffer + BUFFER_PER_WARP * warp_id;
  //int warp_cache_size = SHARED_CACHE_SIZE / WARP_PER_BLOCK;
  //int *warp_cache = cache + threadIdx.x / warpSize * warp_cache_size;
  int local_mb_counter = 0;

  int *L_vertices;
  int *L_level = warp_buffer;
  int L_size;

  int *exe_stack = warp_buffer + MAX_DEGREE_BOUND;

  int *C_vertices = warp_buffer + 2 * MAX_DEGREE_BOUND;
  int *C_level_neighbors =
      warp_buffer + 2 * MAX_DEGREE_BOUND + MAX_2_H_DEGREE_BOUND;
  int C_size;

  int cur_vertex;

  while (true) {
    if (get_lane_id() == 0)
      cur_vertex = graph.V_size - 1 - atomicAdd(processing_vertex, 1);
    cur_vertex = get_value_from_lane_x(cur_vertex);
    if (cur_vertex < 0) break;

    // step 1: initialize L set
    L_size = graph.row_offset[cur_vertex + 1] - graph.row_offset[cur_vertex];
    L_vertices = graph.column_indices + graph.row_offset[cur_vertex];
    if (L_size > MAX_DEGREE_BOUND && get_lane_id() == 0)
      printf("L_size is out of memory %d/%d\n", L_size, MAX_DEGREE_BOUND);
    if (L_size == 1) {
      if (get_lane_id() == 0) {
        int u0 = L_vertices[0];
        int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
        if (base_0[0] == cur_vertex) local_mb_counter++;
      }
    } else {
      // step 2: initialize R set and maximality check
      int *R_vertices = exe_stack;
      int R_size = NeighborsIntersectL(graph, L_vertices, L_size, C_vertices,
                                       R_vertices);

      if (R_vertices[0] == cur_vertex) {
        local_mb_counter++;

        // step 3: initialize C set
        C_size = NeighborsUnionL(graph, L_vertices, L_size, C_level_neighbors,
                                 C_vertices);
        C_size = seq_diff_warp(C_vertices, C_size, R_vertices, R_size);

        if (C_size > MAX_2_H_DEGREE_BOUND && get_lane_id() == 0)
          printf("C_size is out of memory %d/%d\n", C_size,
                 MAX_2_H_DEGREE_BOUND);

        if (C_size > 0 && C_vertices[C_size - 1] > cur_vertex) {
          int first_level_cand_id = C_size - 1;
          for (int i = get_lane_id(); i < C_size; i += warpSize) {
            if (C_vertices[i] > cur_vertex) {
              first_level_cand_id = i;
              break;
            }
          }
          first_level_cand_id = warp_min(first_level_cand_id);
          if (get_lane_id() == 0) exe_stack[0] = first_level_cand_id;

          for (int i = get_lane_id(); i < L_size; i += warpSize) {
            L_level[i] = 0;
          }

          for (int i = get_lane_id(); i < C_size; i += warpSize) {
            C_level_neighbors[i] = 0x7fff0000;
          }

          for (int i = 0; i < L_size; i++) {
            int u0 = L_vertices[i];
            int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
            int size_0 =
                graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

            seq_intersect_warp_for_iter_finder(C_vertices, C_level_neighbors,
                                               C_size, base_0, size_0);
          }

          IterProcess(graph, L_vertices, L_level, L_size, C_vertices,
                      C_level_neighbors, C_size, exe_stack, first_level_cand_id,
                      local_mb_counter);
        }
      }
    }
  }
  if (get_lane_id() == 0 && local_mb_counter != 0)
    atomicAdd(maximal_bicliques, local_mb_counter);
  // if (get_lane_id() == 0) printf("%4d\t", warp_id);
  __syncthreads();
  unsigned long long sm_end = clock64();
  if (threadIdx.x == 0 && clock_rate != 0) {
    printf("SM exit: %lf\n", (sm_end - sm_start) / 1000.0 / clock_rate);
  }
  if (threadIdx.x == 0) {
    //printf("sm end. now maximal_bicliques is %d\n", *maximal_bicliques);
  }
}

IterFinderGpu::IterFinderGpu(CSRBiGraph *graph_in) : BicliqueFinder(graph_in) {
  graph_gpu_ = new CSRBiGraph();
  gpuErrchk(cudaSetDevice(0));
  graph_gpu_->CopyToGpu(*graph_in);
  gpuErrchk(cudaMalloc((void **)&dev_mb_, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&dev_processing_vertex_, sizeof(int)));
  size_t g_size = (size_t)MAX_BLOCKS * WARP_PER_BLOCK * BUFFER_PER_WARP;
  gpuErrchk(cudaMalloc((void **)&dev_global_buffer_, g_size * sizeof(int)));
  maximal_nodes_ = 0;
  gpuErrchk(cudaMemset(dev_mb_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_processing_vertex_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_global_buffer_, 0, g_size * sizeof(int)));

  cudaDeviceProp dev;
  cudaGetDeviceProperties(&dev, 0);
  if (printSMTime) {
    clock_rate = dev.clockRate;
  } else {
    clock_rate = 0;
  }
}

IterFinderGpu::~IterFinderGpu() {
  graph_gpu_->Reset();
  delete graph_gpu_;
  gpuErrchk(cudaFree(dev_global_buffer_));
  gpuErrchk(cudaFree(dev_mb_));
  gpuErrchk(cudaFree(dev_processing_vertex_));
}

void IterFinderGpu::Execute() {
  start_time_ = get_cur_time();
  IterFinderKernel<<<MAX_BLOCKS, WARP_PER_BLOCK * 32>>>(
      *graph_gpu_, dev_global_buffer_, dev_mb_, dev_processing_vertex_, clock_rate);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(&maximal_nodes_, dev_mb_, sizeof(int),
                       cudaMemcpyDeviceToHost));
  exe_time_ = get_cur_time() - start_time_;
}

#define SD_BUFFER_PER_BLOCK (MAX_2_H_DEGREE_BOUND)  // for common C vertices
#define BUFFER_PER_WARP_2 (3 * MAX_DEGREE_BOUND + 2 * MAX_2_H_DEGREE_BOUND)
#define BUFFER_PER_BLOCK_2 \
  (SD_BUFFER_PER_BLOCK + BUFFER_PER_WARP_2 * WARP_PER_BLOCK)
// L_vertices, L_level, exe_stack, C, C_level_neighbors

__launch_bounds__(32 * WARP_PER_SM, 1) __global__
    void IterFinderKernel_2(CSRBiGraph graph, int *global_buffer,
                            int *maximal_bicliques, int *processing_vertex,
                            double clock_rate = 0
                            ) {
  auto sm_start = clock64();
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int *warp_buffer = global_buffer + BUFFER_PER_WARP_2 * warp_id +
                     SD_BUFFER_PER_BLOCK * (blockIdx.x + 1);
  int local_mb_counter = 0;
  // shared C for each block
  int *C_vertices_sd = global_buffer + BUFFER_PER_BLOCK_2 * blockIdx.x;
  __shared__ int C_size_sd;
  __shared__ int C_scanner_id_sd;

  // private L,C for each warp
  int *L_vertices = warp_buffer;
  int *L_level = warp_buffer + MAX_DEGREE_BOUND;
  int L_size;

  int *exe_stack = warp_buffer + 2 * MAX_DEGREE_BOUND;

  int *C_vertices = warp_buffer + 3 * MAX_DEGREE_BOUND;
  int *C_level_neighbors =
      warp_buffer + 3 * MAX_DEGREE_BOUND + MAX_2_H_DEGREE_BOUND;
  int C_size;
  __shared__ int first_vertex;
  int second_id;
  int second_vertex;

  while (true) {
    if (threadIdx.x == 0) {
      first_vertex = graph.V_size - 1 - atomicAdd(processing_vertex, 1);
      C_size_sd = 0;
    }
    if (threadIdx.x < 32) {
      __syncwarp();
      if(first_vertex >= 0){
        // init L
        L_size =
            graph.row_offset[first_vertex + 1] - graph.row_offset[first_vertex];
        int *L_vertices_sd =
            graph.column_indices + graph.row_offset[first_vertex];
        if (L_size > MAX_DEGREE_BOUND && get_lane_id() == 0)
          printf("L_size is out of memory %d/%d\n", L_size, MAX_DEGREE_BOUND);

        if (L_size == 1) {
          if (get_lane_id() == 0) {
            int u0 = L_vertices_sd[0];
            int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
            if (base_0[0] == first_vertex) local_mb_counter++;
          }
        } else {
          // init R
          int *R_vertices = exe_stack;
          int R_size = NeighborsIntersectL(graph, L_vertices_sd, L_size,
                                          C_vertices_sd, R_vertices);
          if (R_vertices[0] == first_vertex) {
            local_mb_counter++;
            // init C

            C_size = NeighborsUnionL(graph, L_vertices_sd, L_size,
                                    C_level_neighbors, C_vertices_sd);
            C_size = seq_diff_warp(C_vertices_sd, C_size, R_vertices, R_size);

            if (C_size > MAX_2_H_DEGREE_BOUND && get_lane_id() == 0)
              printf("C_size is out of memory %d/%d\n", C_size,
                    MAX_2_H_DEGREE_BOUND);
            if (C_size > 0 && C_vertices_sd[C_size - 1] > first_vertex) {
              int scanner_id = C_size - 1;
              for (int i = get_lane_id(); i < C_size; i += warpSize) {
                if (C_vertices_sd[i] > first_vertex) {
                  scanner_id = i;
                  break;
                }
              }
              scanner_id = warp_min(scanner_id);
              if (get_lane_id() == 0) {
                C_size_sd = C_size;
                C_scanner_id_sd = scanner_id;
              }
            }
          }
        }
      }
    }

    __syncthreads();
    if (first_vertex < 0) break;
    if (C_size_sd == 0) continue;

    while (true) {
      if (get_lane_id() == 0) second_id = atomicAdd(&C_scanner_id_sd, 1);
      second_id = get_value_from_lane_x(second_id);
      if (second_id >= C_size_sd) break;
      second_vertex = C_vertices_sd[second_id];
  

      // gen L
      int *base_0, size_0, *base_1, size_1;
      base_0 = graph.column_indices + graph.row_offset[first_vertex];
      size_0 =
          graph.row_offset[first_vertex + 1] - graph.row_offset[first_vertex];
      base_1 = graph.column_indices + graph.row_offset[second_vertex];
      size_1 =
          graph.row_offset[second_vertex + 1] - graph.row_offset[second_vertex];
      L_size = seq_intersect_warp(base_0, size_0, base_1, size_1, L_vertices);

      // gen C
      for (int i = get_lane_id(); i < C_size_sd; i += warpSize) {
        C_level_neighbors[i] = 0x7fff0000;
      }

      for (int i = 0; i < L_size; i++) {
        int u0 = L_vertices[i];
        base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
        size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

        seq_intersect_warp_for_iter_finder(C_vertices_sd, C_level_neighbors,
                                           C_size_sd, base_0, size_0);
      }
      int next_id = second_id + 1;

      C_vertices = C_vertices_sd;
      C_size = C_size_sd;
      // maximality check
      bool is_maximal = true;
      for (int i = get_lane_id(); i < C_size; i += warpSize) {
        int cur_size_l = C_level_neighbors[i] & 0xffff;
        if (cur_size_l == L_size) {
          if (i < second_id) {
            is_maximal = false;
            break;
          }
          C_level_neighbors[i] = cur_size_l;
        }
      }
      is_maximal = __all_sync(FULL_MASK, is_maximal);
      if (!is_maximal) {
        continue;
      }
      local_mb_counter++;

      for (int i = get_lane_id(); i < L_size; i += warpSize) L_level[i] = 0;

      auto iterate_start = clock64();
      IterProcess(graph, L_vertices, L_level, L_size, C_vertices,
                  C_level_neighbors, C_size, exe_stack, next_id,
                  local_mb_counter);
    }
    __syncthreads();
  }

  if (get_lane_id() == 0 && local_mb_counter != 0)
    atomicAdd(maximal_bicliques, local_mb_counter);
  // if (get_lane_id() == 0) printf("%4d\t", warp_id);
  __syncthreads();
  unsigned long long sm_end = clock64();
  if (threadIdx.x == 0 && clock_rate != 0) {
    printf("SM exit: %lf\n", (sm_end - sm_start) / 1000.0 / clock_rate);
  }
}

IterFinderGpu2::IterFinderGpu2(CSRBiGraph *graph_in) : IterFinderGpu(graph_in) {
  graph_gpu_ = new CSRBiGraph();
  gpuErrchk(cudaSetDevice(0));
  graph_gpu_->CopyToGpu(*graph_in);
  gpuErrchk(cudaMalloc((void **)&dev_mb_, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&dev_processing_vertex_, sizeof(int)));
  size_t g_size = (size_t)MAX_BLOCKS * BUFFER_PER_BLOCK_2;
  gpuErrchk(cudaMalloc((void **)&dev_global_buffer_, g_size * sizeof(int)));
  maximal_nodes_ = 0;
  gpuErrchk(cudaMemset(dev_mb_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_processing_vertex_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_global_buffer_, 0, g_size * sizeof(int)));
}

IterFinderGpu2::~IterFinderGpu2() {
  graph_gpu_->Reset();
  delete graph_gpu_;
  gpuErrchk(cudaFree(dev_global_buffer_));
  gpuErrchk(cudaFree(dev_mb_));
  gpuErrchk(cudaFree(dev_processing_vertex_));
}

void IterFinderGpu2::Execute() {
  start_time_ = get_cur_time();
  unsigned long long clock_initialize, clock_iterate, host_total_clock;
  IterFinderKernel_2<<<MAX_BLOCKS, WARP_PER_BLOCK * 32>>>(
    *graph_gpu_, dev_global_buffer_, dev_mb_, dev_processing_vertex_, 
    clock_rate);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(&maximal_nodes_, dev_mb_, sizeof(int),
                       cudaMemcpyDeviceToHost));
  cudaMemcpy(&clock_initialize, total_clock_initialize, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&clock_iterate, total_clock_iterate, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_total_clock, dev_total_clock, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  printf("initialize clock: %lld\n", clock_initialize);
  printf("iterate clock: %lld\n", clock_iterate);
  printf("other clock: %lld\n", host_total_clock - clock_initialize - clock_iterate);
  exe_time_ = get_cur_time() - start_time_;
}

__launch_bounds__(32 * WARP_PER_SM, 1) __global__
    void ChildKernel(CSRBiGraph graph, int *global_buffer,
                     int *maximal_bicliques, int block_id, int first_vertex,
                     int next_id, int C_size_sd) {
  __shared__ int C_scanner_id_sd;
  int warp_id = threadIdx.x / warpSize;

  int *C_vertices_sd = global_buffer + BUFFER_PER_BLOCK_2 * block_id;
  int *warp_buffer =
      C_vertices_sd + SD_BUFFER_PER_BLOCK + warp_id * BUFFER_PER_WARP_2;

  int *L_vertices = warp_buffer;
  int *L_level = warp_buffer + MAX_DEGREE_BOUND;
  int L_size;

  int *exe_stack = warp_buffer + 2 * MAX_DEGREE_BOUND;

  int *C_vertices = warp_buffer + 3 * MAX_DEGREE_BOUND;
  int *C_level_neighbors =
      warp_buffer + 3 * MAX_DEGREE_BOUND + MAX_2_H_DEGREE_BOUND;
  int C_size;

  int second_id, second_vertex;
  int local_mb_counter = 0;

  if (threadIdx.x == 0) {
    C_scanner_id_sd = next_id;
  }

  __syncthreads();

  while (true) {
    if (get_lane_id() == 0) {
      second_id = atomicAdd(&C_scanner_id_sd, 1);
    }
    second_id = get_value_from_lane_x(second_id);
    if (second_id >= C_size_sd) {
      break;
    }
    second_vertex = C_vertices_sd[second_id];

    // gen L
    int *base_0, size_0, *base_1, size_1;
    base_0 = graph.column_indices + graph.row_offset[first_vertex];
    size_0 =
        graph.row_offset[first_vertex + 1] - graph.row_offset[first_vertex];
    base_1 = graph.column_indices + graph.row_offset[second_vertex];
    size_1 =
        graph.row_offset[second_vertex + 1] - graph.row_offset[second_vertex];

    L_size = seq_intersect_warp(base_0, size_0, base_1, size_1, L_vertices);

    // gen C
    for (int i = get_lane_id(); i < C_size_sd; i += warpSize) {
      C_level_neighbors[i] = 0x7fff0000;
    }

    for (int i = 0; i < L_size; i++) {
      int u0 = L_vertices[i];
      base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
      size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

      seq_intersect_warp_for_iter_finder(C_vertices_sd, C_level_neighbors,
                                         C_size_sd, base_0, size_0);
    }
    int next_id = second_id + 1;

    C_vertices = C_vertices_sd;
    C_size = C_size_sd;
    // maximality check
    bool is_maximal = true;
    for (int i = get_lane_id(); i < C_size; i += warpSize) {
      int cur_size_l = C_level_neighbors[i] & 0xffff;
      if (cur_size_l == L_size) {
        if (i < second_id) {
          is_maximal = false;
          break;
        }
        C_level_neighbors[i] = cur_size_l;
      }
    }
    is_maximal = __all_sync(FULL_MASK, is_maximal);
    if (!is_maximal) continue;
    local_mb_counter++;

    for (int i = get_lane_id(); i < L_size; i += warpSize) L_level[i] = 0;

    IterProcess(graph, L_vertices, L_level, L_size, C_vertices,
                C_level_neighbors, C_size, exe_stack, next_id,
                local_mb_counter);
  }
  if (get_lane_id() == 0 && local_mb_counter != 0) {
    atomicAdd(maximal_bicliques, local_mb_counter);
  }
}

__launch_bounds__(32, 1) __global__
    void IterFinderKernel_3(CSRBiGraph graph, int *global_buffer,
                            int *maximal_bicliques, int *processing_vertex) {
  int *C_vertices_sd = global_buffer + BUFFER_PER_BLOCK_2 * blockIdx.x;
  int *warp_buffer = C_vertices_sd + SD_BUFFER_PER_BLOCK;

  int *L_vertices, L_size;
  int *L_level = warp_buffer + MAX_DEGREE_BOUND;
  int *exe_stack = warp_buffer + 2 * MAX_DEGREE_BOUND;

  int *C_vertices = warp_buffer + 3 * MAX_DEGREE_BOUND;
  int *C_level_neighbors =
      warp_buffer + 3 * MAX_DEGREE_BOUND + MAX_2_H_DEGREE_BOUND;
  int C_size;

  int first_vertex, local_mb_counter = 0;

  while (true) {
    if (threadIdx.x == 0)
      first_vertex = graph.V_size - 1 - atomicAdd(processing_vertex, 1);
    first_vertex = get_value_from_lane_x(first_vertex);
    if (first_vertex < 0) break;

    // step 1: initialize L set
    L_size =
        graph.row_offset[first_vertex + 1] - graph.row_offset[first_vertex];
    L_vertices = graph.column_indices + graph.row_offset[first_vertex];
    if (L_size > MAX_DEGREE_BOUND && get_lane_id() == 0)
      printf("L_size is out of memory %d/%d\n", L_size, MAX_DEGREE_BOUND);

    if (L_size == 1) {
      if (get_lane_id() == 0) {
        int u0 = L_vertices[0];
        int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
        if (base_0[0] == first_vertex) local_mb_counter++;
      }
    } else {
      // step 2: initialize R set and maximality check
      int *R_vertices = exe_stack;
      int R_size = NeighborsIntersectL(graph, L_vertices, L_size, C_vertices,
                                       R_vertices);

      if (R_vertices[0] == first_vertex) {
        local_mb_counter++;

        // init C
        C_size = NeighborsUnionL(graph, L_vertices, L_size, C_level_neighbors,
                                 C_vertices_sd);
        C_size = seq_diff_warp(C_vertices_sd, C_size, R_vertices, R_size);

        if (C_size > MAX_2_H_DEGREE_BOUND && get_lane_id() == 0)
          printf("C_size is out of memory %d/%d\n", C_size,
                 MAX_2_H_DEGREE_BOUND);

        if (C_size > 0 && C_vertices_sd[C_size - 1] > first_vertex) {
          int next_id = C_size - 1;
          for (int i = get_lane_id(); i < C_size; i += warpSize) {
            if (C_vertices_sd[i] > first_vertex) {
              next_id = i;
              break;
            }
          }
          next_id = warp_min(next_id);
          next_id = get_value_from_lane_x(next_id);

          int real_cand = C_size - next_id;
          int warps = (real_cand - 1) / AVG_ITEMS_PER_WARP + 1;
          warps = warps > WARP_PER_BLOCK ? WARP_PER_BLOCK : warps;

          if (warps == 1) {
            C_vertices = C_vertices_sd;
            if (get_lane_id() == 0) exe_stack[0] = next_id;

            for (int i = get_lane_id(); i < L_size; i += warpSize) {
              L_level[i] = 0;
            }

            for (int i = get_lane_id(); i < C_size; i += warpSize) {
              C_level_neighbors[i] = 0x7fff0000;
            }
            for (int i = 0; i < L_size; i++) {
              int u0 = L_vertices[i];
              int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
              int size_0 =
                  graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

              seq_intersect_warp_for_iter_finder(C_vertices, C_level_neighbors,
                                                 C_size, base_0, size_0);
            }

            IterProcess(graph, L_vertices, L_level, L_size, C_vertices,
                        C_level_neighbors, C_size, exe_stack, next_id,
                        local_mb_counter);
          } else {
            if (get_lane_id() == 0) {
              ChildKernel<<<1, warps * warpSize>>>(
                  graph, global_buffer, maximal_bicliques, blockIdx.x,
                  first_vertex, next_id, C_size);
              cudaError_t cudaStatus = cudaGetLastError();
              if (cudaGetLastError() != cudaSuccess) {
                printf("ChildKernel launch failed: %s\n",
                       cudaGetErrorString(cudaStatus));
              }
              cudaStatus = cudaDeviceSynchronize();
              if (cudaGetLastError() != cudaSuccess) {
                printf("ChildKernel launch failed: %s\n",
                       cudaGetErrorString(cudaStatus));
              }
            }
          }
        }
      }
    }
  }

  if (get_lane_id() == 0 && local_mb_counter != 0)
    atomicAdd(maximal_bicliques, local_mb_counter);
}

IterFinderGpu3::IterFinderGpu3(CSRBiGraph *graph_in) : IterFinderGpu(graph_in) {
  graph_gpu_ = new CSRBiGraph();
  gpuErrchk(cudaSetDevice(0));
  graph_gpu_->CopyToGpu(*graph_in);
  gpuErrchk(cudaMalloc((void **)&dev_mb_, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&dev_processing_vertex_, sizeof(int)));
  size_t g_size = (size_t)MAX_BLOCKS * BUFFER_PER_BLOCK_2;
  gpuErrchk(cudaMalloc((void **)&dev_global_buffer_, g_size * sizeof(int)));
  maximal_nodes_ = 0;
  gpuErrchk(cudaMemset(dev_mb_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_processing_vertex_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_global_buffer_, 0, g_size * sizeof(int)));
}

IterFinderGpu3::~IterFinderGpu3() {
  graph_gpu_->Reset();
  delete graph_gpu_;
  gpuErrchk(cudaFree(dev_global_buffer_));
  gpuErrchk(cudaFree(dev_mb_));
  gpuErrchk(cudaFree(dev_processing_vertex_));
}

void IterFinderGpu3::Execute() {
  start_time_ = get_cur_time();
  IterFinderKernel_3<<<MAX_BLOCKS, 32>>>(*graph_gpu_, dev_global_buffer_,
                                         dev_mb_, dev_processing_vertex_);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(&maximal_nodes_, dev_mb_, sizeof(int),
                       cudaMemcpyDeviceToHost));
  exe_time_ = get_cur_time() - start_time_;
}

__device__ __forceinline__ int get_common_neighbors(int *row_offset, int *column_indices,  int const *vertices, int vertices_size, 
                                     int * const buffer, int * const result) {
  int now_vertex = vertices[vertices_size - 1];
  int size_temp = row_offset[now_vertex + 1] - row_offset[now_vertex];
  int *base = column_indices + row_offset[now_vertex];
  int buffer_size = 0;
  int result_size;
  if ((vertices_size - 1) & 1) {
    for (int i = get_lane_id(); i < size_temp; i += warpSize) {
      buffer[i] = base[i];
    }
    buffer_size = size_temp;
  } else {
    for (int i = get_lane_id(); i < size_temp; i += warpSize) {
      result[i] = base[i];
    }
    result_size = size_temp;
  }
  __syncwarp();
  for (int i = vertices_size - 2; i >= 0; i--) {
    now_vertex = vertices[i];
    size_temp = row_offset[now_vertex + 1] - row_offset[now_vertex];
    base = column_indices + row_offset[now_vertex];
    if (i & 1) {
      buffer_size = seq_intersect_warp(result, result_size, base, size_temp, buffer);
    }
    else {
      result_size = seq_intersect_warp(buffer, buffer_size, base, size_temp, result);
    }
  }
  return result_size;
}

__device__ __forceinline__ int get_all_neighbors(CSRBiGraph &graph, int const *vertices, const int &vertices_size, 
                                     int * const buffer, int * const result) {
  if (get_lane_id() == 0) {
    //printf("enter all\n");
  }
  int now_vertex = vertices[vertices_size - 1];
  int size_temp = graph.rev_row_offset[now_vertex + 1] - graph.rev_row_offset[now_vertex];
  int *base = graph.rev_column_indices + graph.rev_row_offset[now_vertex];
  int buffer_size;
  int result_size;

  if(vertices_size > UNION_THRESHOLD) {
    int cnt = 0, common_neighbors = 0;
    for (int i = 0; i < graph.V_size; i++) {
      size_temp = graph.row_offset[i + 1] - graph.row_offset[i];
      base = graph.column_indices + graph.row_offset[i];
      common_neighbors = seq_intersect_warp_cnt (vertices, vertices_size, base, size_temp);
      if(common_neighbors != 0) {
        if(get_lane_id() == 0)result[cnt] = i;
        cnt++;
      }
    }
    return cnt;
  }

  if ((vertices_size - 1) & 1) {
    for (int i = get_lane_id(); i < size_temp; i += warpSize) {
      buffer[i] = base[i]; 
    }
    buffer_size = size_temp;
  } else {
    for (int i = get_lane_id(); i < size_temp; i += warpSize) {
      result[i] = base[i];
    }
    result_size = size_temp;
  }
  for (int i = vertices_size - 2; i >= 0; i--) {
    now_vertex = vertices[i];
    size_temp = graph.rev_row_offset[now_vertex + 1] - graph.rev_row_offset[now_vertex];
    base = graph.rev_column_indices + graph.rev_row_offset[now_vertex];
    if (i & 1) {
      buffer_size = seq_union_warp_adv(result, result_size, base, size_temp, buffer);
      if (buffer_size >= MAX_2_H_DEGREE_BOUND && get_lane_id() == 0) {
        printf("error\n");
      }
    }
    else {
      result_size = seq_union_warp_adv(buffer, buffer_size, base, size_temp, result);
      if (result_size >= MAX_2_H_DEGREE_BOUND && get_lane_id() == 0) {
        printf("error\n");
      }
    }
  }
  if (get_lane_id() == 0) {
    //printf("exit all\n");
  }
  return result_size;
}

__device__ bool IterFinderWithMultipleVertex(CSRBiGraph graph, int *warp_buffer, WorkList<LargeTask> *global_large_worklist,
                                             const TinyTask &tt, int &local_mb_counter, unsigned long long *large_count, 
                                             unsigned long long *non_maximal = nullptr,
                                             int bound_height = 20, int bound_size = 1500) {
  int *L_vertices = warp_buffer;
  int *L_level = warp_buffer + MAX_DEGREE_BOUND;

  int *exe_stack = warp_buffer + 2 * MAX_DEGREE_BOUND;

  int *C_vertices = warp_buffer + 3 * MAX_DEGREE_BOUND;
  int *C_level_neighbors =
      warp_buffer + 3 * MAX_DEGREE_BOUND + MAX_2_H_DEGREE_BOUND;
  
  bool has_vertex = get_lane_id() < 4 && tt.vertices[get_lane_id()] != -1;
  int vertices_size = count_bit(__ballot_sync(0xffffffff, has_vertex));

  int L_size = get_common_neighbors(graph.row_offset, graph.column_indices, tt.vertices, vertices_size, exe_stack, L_vertices);
  __syncwarp();
  if (L_size > MAX_DEGREE_BOUND && get_lane_id() == 0)
    printf("L_size is out of memory %d/%d\n", L_size, MAX_DEGREE_BOUND);

  int *R_vertices = exe_stack;
  int R_size = get_common_neighbors(graph.rev_row_offset, graph.rev_column_indices, L_vertices, L_size, C_vertices, R_vertices);
  if(R_size > MAX_DEGREE_BOUND) {
    printf("error! R_size is larger than MAX_DEGREE_BOUND(%d/%d).\n", R_size, MAX_DEGREE_BOUND);
  }
 
  //maximality check
  bool is_maximal = true;
  if (vertices_size > 1) {
    int *L_vertices_last = L_level;
    int L_size_last = get_common_neighbors(graph.row_offset, graph.column_indices,
                                           tt.vertices, vertices_size - 1,
                                           C_vertices, L_vertices_last);
    int *R_vertices_last = C_vertices;
    int R_size_last = get_common_neighbors(graph.rev_row_offset, graph.rev_column_indices,
                                           L_vertices_last, L_size_last, C_level_neighbors,
                                           R_vertices_last);
    for(int i = get_lane_id(); i < R_size && R_vertices[i] < tt.vertices[vertices_size - 1]; i += warpSize) {
      if(i >= R_size_last || R_vertices[i] != R_vertices_last[i])is_maximal = false;
    }
    is_maximal = __all_sync(FULL_MASK, is_maximal);
  } else if (vertices_size == 1){
    is_maximal = R_vertices[0] == tt.vertices[0];
  }
  if(!is_maximal){
    if (get_lane_id() == 0 && non_maximal != nullptr) {
      atomicAdd(non_maximal, 1);
    }
    return false;
  }

  
  local_mb_counter++;
  

  if(L_size > 1){
    //generate C_vertices
    int size_temp = get_all_neighbors(graph, L_vertices, L_size, C_level_neighbors, C_vertices);
    int next_id;
    next_id = binary_search(tt.vertices[vertices_size - 1], C_vertices, 0, size_temp - 1) 
              - binary_search(tt.vertices[vertices_size - 1], R_vertices, 0, R_size - 1);
    int C_size = seq_diff_warp(C_vertices, size_temp, R_vertices, R_size);
    if( 
      ((C_size - next_id) > bound_height || L_size > bound_height) && (C_size - next_id) * L_size > bound_size 
      && !global_large_worklist->is_full())
    {
      if(vertices_size == 4) {
      } else {
        __syncwarp();
        
        while (true){
          if (!global_large_worklist->is_full()) {
            if (global_large_worklist->push_work(tt))
            {
              if (get_lane_id() == 0) {
                atomicAdd(large_count, 1);
              }
              return true;
            }
          } else 
          {
            break;//防止因为队列满导致无法结束
          }
        }
      }
    }
    if(C_size > MAX_2_H_DEGREE_BOUND)
      printf("error! C_size is larger than MAX_2_H_DEGREE_BOUND(%d/%d).\n", C_size, MAX_2_H_DEGREE_BOUND);

    for (int i = get_lane_id(); i < C_size; i += warpSize) {
      C_level_neighbors[i] = 0x7fff0000;
    }

    for (int i = 0; i < L_size; i++) {
      int u0 = L_vertices[i];
      int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
      int size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

      seq_intersect_warp_for_iter_finder(C_vertices, C_level_neighbors,
                                             C_size, base_0, size_0);
    }

    for(int i = get_lane_id(); i < C_size; i += warpSize) {
      int cur_size_l = C_level_neighbors[i] & 0xffff;  
      if(cur_size_l == L_size) {
        C_level_neighbors[i] = L_size;
      }
    }
    for (int i = get_lane_id(); i < L_size; i += warpSize) L_level[i] = 0;
    IterProcess(graph, L_vertices, L_level, L_size, C_vertices,
                    C_level_neighbors, C_size, exe_stack, next_id,
                    local_mb_counter, non_maximal);
  }
  return false;
}

__device__ int LargeToTiny(CSRBiGraph graph, int *warp_buffer, LargeTask task, TinyTask *result) {
  int *L = warp_buffer + MAX_DEGREE_BOUND;
  int *buffer = L + MAX_DEGREE_BOUND;
  int *C = buffer + MAX_2_H_DEGREE_BOUND;
  int *R = C + MAX_2_H_DEGREE_BOUND;
  int vertices_size;
  bool has_vertex = get_lane_id() < 4 && task.vertices[get_lane_id()] >= 0;
  vertices_size = count_bit(__ballot_sync(0xffffffff, has_vertex));
  int L_size = get_common_neighbors(graph.row_offset, graph.column_indices, task.vertices, vertices_size, buffer, L);
  int C_size = get_all_neighbors(graph, L, L_size, buffer, C);
  int R_size = get_common_neighbors(graph.rev_row_offset, graph.rev_column_indices, L, L_size, buffer, R);
  int next_id = binary_search(task.vertices[vertices_size - 1], C, 0, C_size - 1)
    - binary_search(task.vertices[vertices_size - 1], R, 0, R_size - 1);
  C_size = seq_diff_warp(C, C_size, R, R_size);
  for (int i = get_lane_id() + next_id; i < C_size; i += warpSize) {
    for (int j = 0; j < 4; j++) {
      result[i - next_id].vertices[j] = task.vertices[j];
    }
    result[i - next_id].vertices[vertices_size] = C[i];
  }
  return C_size - next_id;
}

#define DIVIDER_PER_BLOCK 1 
#define LARGE_WORKLIST_SIZE 0x48000

#define LOCAL_TINY_WORKLIST_SIZE 0x180 
#define LOCAL_TINY_WORKLIST_THRESHOLD LOCAL_TINY_WORKLIST_SIZE //* 0.6
#define CONSUMER_PER_BLOCK (WARP_PER_BLOCK - DIVIDER_PER_BLOCK)
__launch_bounds__(32 * WARP_PER_SM, 1) __global__
    void IterFinderKernel_6(CSRBiGraph graph, int *global_buffer, 
                            int *maximal_bicliques, int *processing_vertex,  
                            WorkList<LargeTask> *global_large_worklist, 
                            unsigned long long *global_count, 
                            unsigned long long *large_count, unsigned long long *tiny_count,
                            unsigned long long *non_maximal,
                            int bound_height, int bound_size, double clock_rate = 0
                            ) {
  unsigned long long sm_start = clock64(); 
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int *warp_buffer = global_buffer + (size_t)BUFFER_PER_WARP_2 * warp_id +
                     (size_t)SD_BUFFER_PER_BLOCK * (blockIdx.x + 1);
  int local_mb_counter = 0;

  __shared__ TinyTask local_tiny_worklist_ptr[LOCAL_TINY_WORKLIST_SIZE];
  for (int i = threadIdx.x; i < LOCAL_TINY_WORKLIST_SIZE; i += blockDim.x) {
    local_tiny_worklist_ptr[i].Init();
  }
  TinyTask *tiny_buffer = (TinyTask *)warp_buffer;
  volatile unsigned long long &gc = *global_count;
  volatile unsigned long long &lc = *large_count;
  volatile unsigned long long &tc = *tiny_count;
  volatile int &pv = *processing_vertex;

  unsigned long long llc = 0;
  unsigned long long ltc = 0;

  __shared__ WorkList<TinyTask> local_tiny_worklist;
  if (threadIdx.x == 0) {
    local_tiny_worklist.Init(local_tiny_worklist_ptr, LOCAL_TINY_WORKLIST_SIZE);
  }

  __threadfence_block();
  __syncthreads();
  
  if (threadIdx.x / warpSize < 1) {
    while (true) {
      __syncwarp();
      if (global_large_worklist->get_work_num() > 0) {
        LargeTask lt;
        size_t get_num = global_large_worklist->get(lt);
        if (get_num <= 0) continue;

        int twn_p = LargeToTiny(graph, warp_buffer, lt, tiny_buffer);
        if (get_lane_id() == 0) {
          atomicAdd(tiny_count, twn_p);
        }
        int pushed = 0;
        while (pushed < twn_p) {
          __syncwarp();
          if (local_tiny_worklist.get_work_num() < LOCAL_TINY_WORKLIST_THRESHOLD) {
            size_t pushing = local_tiny_worklist.push_works(tiny_buffer + pushed, 
                                                            twn_p - pushed);
            pushed += pushing;
          }
        }
        llc ++;
        continue;
      }
      if (pv >= graph.V_size) {
        if (get_lane_id() == 0) {
          if (llc > 0) {
            atomicAdd(large_count, -llc);
            llc = 0;
          }
        }
        if (gc == 0 && lc == 0 && tc == 0)break;
      }
    }
  } 
  else
  {
    int lgc = 0;
    while (true) {
      __syncwarp();
      if (!local_tiny_worklist.is_empty()) {
        TinyTask tt;
        if (local_tiny_worklist.get(tt)) {
          IterFinderWithMultipleVertex(graph, warp_buffer, global_large_worklist, 
                                     tt, local_mb_counter, large_count
                                     , non_maximal, bound_height, bound_size);
          ltc++;
          continue;
        }
      }
      __syncwarp();
      if (pv < graph.V_size) {
        int first_vertex;
        if (get_lane_id() == 0) {
          first_vertex = graph.V_size - 1 - atomicAdd(processing_vertex, 1);
          if (first_vertex >= 0) atomicAdd(global_count, 1);
        }
        first_vertex = get_value_from_lane_x(first_vertex);
        if (first_vertex >= 0) {
          LongTask lt(first_vertex, -1);
          IterFinderWithMultipleVertex(graph, warp_buffer, global_large_worklist, 
                                     lt, local_mb_counter, large_count,  
                                     non_maximal, bound_height, bound_size);
          lgc ++;
        }
      }
      else {
        if(get_lane_id() == 0) {
          if (ltc > 0) {
            atomicAdd(tiny_count, -ltc);
            ltc = 0;
          }
          if (lgc > 0) {
            atomicAdd(global_count, -lgc);
            lgc = 0;
          }
        }
        __syncwarp();
        if (gc == 0 && lc == 0 && tc == 0)
          break;
      }
    }
  }
  __syncthreads();
  unsigned long long sm_end = clock64();
  if (threadIdx.x == 0 && clock_rate != 0) {
    printf("SM exit: %lf\n", (sm_end - sm_start) / 1000.0 / clock_rate);
  }
  if (get_lane_id() == 0 && local_mb_counter != 0)  
  {
    /*printf("initialize clock: %lld\n", clock_initialize);
    printf("tiny generate clock: %lld\n", clock_tiny_generate);
    printf("iterate clock: %lld\n", clock_iterate);
    printf("queue clock: %lld\n", clock_queue);
    printf("other clock: %lld\n", clock_end - clock_start - clock_queue - clock_iterate - clock_tiny_generate - clock_initialize);*/

    atomicAdd(maximal_bicliques, local_mb_counter);
  }

}


IterFinderGpu6::IterFinderGpu6(CSRBiGraph *graph_in, int bound_height_ , int bound_size_) 
                : IterFinderGpu(graph_in), bound_height(bound_height_), bound_size(bound_size_) {
  graph_gpu_ = new CSRBiGraph();
  gpuErrchk(cudaSetDevice(0));
  graph_gpu_->CopyToGpu(*graph_in);
  gpuErrchk(cudaMalloc((void **)&dev_mb_, sizeof(int)));
  gpuErrchk(cudaMalloc((void **)&dev_processing_vertex_, sizeof(int)));
  LargeTask *large_worklist_ptr;
  gpuErrchk(cudaMalloc((void **)&large_worklist_ptr, sizeof(LargeTask) * LARGE_WORKLIST_SIZE));
  gpuErrchk(cudaMalloc((void **)&global_count, sizeof(unsigned long long)));
  gpuErrchk(cudaMalloc((void **)&large_count, sizeof(unsigned long long)));
  gpuErrchk(cudaMalloc((void **)&tiny_count, sizeof(unsigned long long)));
  gpuErrchk(cudaMalloc((void **)&global_large_worklist, sizeof(WorkList<LargeTask>)));
  size_t g_size = (size_t)MAX_BLOCKS * BUFFER_PER_BLOCK_2;
  gpuErrchk(cudaMalloc((void **)&dev_global_buffer_, g_size * sizeof(int)));

  LargeTask lts[LARGE_WORKLIST_SIZE];
  gpuErrchk(cudaMemcpy(large_worklist_ptr, lts, LARGE_WORKLIST_SIZE * sizeof(LargeTask), cudaMemcpyHostToDevice));

  WorkList<LargeTask> large_worklist;
  large_worklist.Init(large_worklist_ptr, LARGE_WORKLIST_SIZE, false);
  gpuErrchk(cudaMemcpy(global_large_worklist, &large_worklist, sizeof(WorkList<LargeTask>), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(global_count, 0, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(large_count, 0, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(tiny_count, 0, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(dev_mb_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_processing_vertex_, 0, sizeof(int)));
  gpuErrchk(cudaMemset(dev_global_buffer_, 0, g_size * sizeof(int)));
}

IterFinderGpu6::~IterFinderGpu6() {
  graph_gpu_->Reset();
  delete graph_gpu_;
  gpuErrchk(cudaFree(dev_global_buffer_));
  gpuErrchk(cudaFree(dev_mb_));
  gpuErrchk(cudaFree(dev_processing_vertex_));
}

void IterFinderGpu6::Execute() {
  start_time_ = get_cur_time();
  unsigned long long *non_maximal;
  cudaMalloc(&non_maximal, sizeof(unsigned long long));
  cudaMemset(non_maximal, 0, sizeof(unsigned long long));
  int *dev_test;
  cudaMalloc((void**)&dev_test, sizeof(int));
  cudaMemset(dev_test, 0, sizeof(int));
  printf("large worklist size: %d local tiny size: %d\n", LARGE_WORKLIST_SIZE, LOCAL_TINY_WORKLIST_SIZE);
  IterFinderKernel_6<<<MAX_BLOCKS, WARP_PER_BLOCK * 32>>>(
    *graph_gpu_, dev_global_buffer_, dev_mb_, dev_processing_vertex_,
       global_large_worklist, global_count, large_count, tiny_count,
     non_maximal, bound_height, bound_size, clock_rate);
  int host_test;
  unsigned long long host_non_maximal;
  cudaMemcpy(&host_test, dev_test, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_non_maximal, non_maximal, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  printf("%d\n", host_test);
  printf("non_maximal: %lld\n", host_non_maximal);
  gpuErrchk(cudaMemcpy(&maximal_nodes_, dev_mb_, sizeof(int),
                       cudaMemcpyDeviceToHost));
  exe_time_ = get_cur_time() - start_time_;
}
__launch_bounds__(32 * WARP_PER_SM, 1) __global__
    void IterFinderKernel_7(CSRBiGraph graph, int *global_buffer, 
                            int *maximal_bicliques, int *processing_vertex, int end_vertex, 
                            WorkList<LargeTask> *global_large_worklist, 
                            unsigned long long *global_count, 
                            unsigned long long *large_count, unsigned long long *tiny_count, unsigned long long *first_count, 
                            int *isProcessed, int ngpus) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int *warp_buffer = global_buffer + (size_t)BUFFER_PER_WARP_2 * warp_id +
                     (size_t)SD_BUFFER_PER_BLOCK * (blockIdx.x + 1);
  int local_mb_counter = 0;

  __shared__ TinyTask local_tiny_worklist_ptr[LOCAL_TINY_WORKLIST_SIZE];
  for (int i = threadIdx.x; i < LOCAL_TINY_WORKLIST_SIZE; i += blockDim.x) {
    local_tiny_worklist_ptr[i].Init();
  }
  TinyTask *tiny_buffer = (TinyTask *)warp_buffer;
  volatile unsigned long long &gc = *global_count;
  volatile unsigned long long &lc = *large_count;
  volatile unsigned long long &tc = *tiny_count;
  volatile unsigned long long &fc = *first_count;

  unsigned long long llc = 0;
  unsigned long long ltc = 0;

  __shared__ int pvLarge;
  __shared__ WorkList<TinyTask> local_tiny_worklist;
  __shared__ int allProcessed;
  __shared__ int processed;
  if (threadIdx.x == 0) {
    local_tiny_worklist.Init(local_tiny_worklist_ptr, LOCAL_TINY_WORKLIST_SIZE);
    pvLarge = 0;
    allProcessed = 0;
    processed = 0;
  }
  volatile int &ap = allProcessed;
  volatile int &volpvLarge = pvLarge;

  __threadfence_block();
  __syncthreads();
  
  if (threadIdx.x / warpSize < 1) {
    while (true) {
      __syncwarp();
      if (global_large_worklist->get_work_num() > 0) {
        LargeTask lt;
        size_t get_num = global_large_worklist->get(lt);
        if (get_num <= 0) continue;
        int twn_p = LargeToTiny(graph, warp_buffer, lt, tiny_buffer);
        if (get_lane_id() == 0) {
          atomicAdd(tiny_count, twn_p);
        }
        int pushed = 0;
        while (pushed < twn_p) {
          __syncwarp();
          if (local_tiny_worklist.get_work_num() < LOCAL_TINY_WORKLIST_THRESHOLD) {
            size_t pushing = local_tiny_worklist.push_works(tiny_buffer + pushed, 
                                                            twn_p - pushed);
            pushed += pushing;
          }
        }
        llc ++;
        if (get_lane_id() == 0 && lt.vertices[1] == -1) {
          atomicAdd(first_count , -1);
          //printf("fc: %lld\n", fc);
        }
        continue;
      }
      /*if (worklist_system->get_work_num() > 0) {
        LargeTask lt;
        size_t get_num = worklist_system->get(lt);
        if (get_num <= 0) continue;
        int twn_p = LargeToTiny(graph, warp_buffer, lt, tiny_buffer);
        if (get_lane_id() == 0) {
          atomicAdd(tiny_count, twn_p);
        }
        int pushed = 0;
        while (pushed < twn_p) {
          __syncwarp();
          if (local_tiny_worklist.get_work_num() < LOCAL_TINY_WORKLIST_THRESHOLD) {
            size_t pushing = local_tiny_worklist.push_works(tiny_buffer + pushed, 
                                                            twn_p - pushed);
            pushed += pushing;
          }
        }
        lsc ++;
        continue;
      }*/
      if (volpvLarge == 1 && ap == 1) {
        if (get_lane_id() == 0) {
          if (llc > 0) {
            atomicAdd(large_count, -llc);
            llc = 0;
          }
        }
	//if (get_lane_id() == 0)printf("%d %d %d %d\n", gc, lc, tc, test); 
        if (gc == 0 && lc == 0 && tc == 0)break;
      }
    }
  } 
  else
  {
    int first_vertex = end_vertex;
    int lgc = 0;
    while (true) {
      __syncwarp();
      if (!local_tiny_worklist.is_empty()) {
        TinyTask tt;
        if (local_tiny_worklist.get(tt)) {
          //if (global_large_worklist -> get_work_num() < LARGE_WORKLIST_THRESHOLD) {
          //  IterFinderWithMultipleVertex(graph, warp_buffer, worklist_system, 
          //                             tt, local_mb_counter, system_count);
          //} else {
          IterFinderWithMultipleVertex(graph, warp_buffer, global_large_worklist, 
                                       tt, local_mb_counter, large_count);
          //}
          ltc++;
          continue;
        }
      }
      __syncwarp();
      if (volpvLarge == 0 || ap == 0) {
        //if (global_large_worklist -> get_work_num() > 0) continue; 
        if (get_lane_id() == 0) {
          first_vertex = -1;
          int tfc;
          bool localIsEmpty;
          do {
            localIsEmpty = local_tiny_worklist.is_empty();
            tfc = fc; 
          } while (tfc < MAX_BLOCKS * 4 && localIsEmpty  && atomicCAS(first_count, tfc, tfc + 1) != tfc); 
          if (tfc < MAX_BLOCKS * 4 && localIsEmpty) {
            do {
          //int pos = atomicAdd(processing_vertex, 1);
              first_vertex = end_vertex - atomicAdd(processing_vertex, ngpus);
              //printf("fv: %d\n", first_vertex);
            } while (first_vertex >= 0 && atomicCAS_system(&isProcessed[first_vertex], 0, 1) == 1);
          /*if (pos < assignedVerticesSize) {
              //printf("%d\n", first_vertex);
	      atomicAdd(global_count, 1);
              //printf("%d/%d\n", pos, assignedVerticesSize);
              first_vertex = assignedVertices[pos];
              atomicAdd_block(&first_count, 1);
              //first_vertex = end_vertex - first_vertex;
            } else if (sf < sharedFirstSize){ 
              first_vertex = -1;
              atomicExch(&pvLarge, 1);
            }*/
            if (first_vertex >= 0) {
              atomicAdd(global_count, 1);
            } else {
              //printf("pvLarge\n");
              atomicExch(&pvLarge, 1);
              do {
                first_vertex = atomicAdd(&processed, 1);
              } while (first_vertex <= end_vertex && atomicCAS_system(&isProcessed[first_vertex], 0, 1) == 1);
              if (first_vertex <= end_vertex) {
                atomicAdd(global_count, 1);
                //printf("%d\n", first_vertex);
              } else {
                //printf("allprocessed\n");
                atomicExch(&allProcessed, 1);
              }
            }
            if (first_vertex < 0 || first_vertex > end_vertex) {
              atomicAdd(first_count, -1);
              //printf("fc: %d\n", fc);
            }
          }
        }
        first_vertex = get_value_from_lane_x(first_vertex);
        if (first_vertex >= 0 && first_vertex <= end_vertex) {
          //if(get_lane_id() == 0)printf("fv: %d\n", first_vertex);
          LongTask lt(first_vertex, -1);
          //if (global_large_worklist -> get_work_num() < LARGE_WORKLIST_THRESHOLD) {
          //  IterFinderWithMultipleVertex(graph, warp_buffer, worklist_system, 
          //                             lt, local_mb_counter, system_count);
          //} else {
          bool isLarge = IterFinderWithMultipleVertex(graph, warp_buffer, global_large_worklist, 
                                       lt, local_mb_counter, large_count);
          //}
          if (get_lane_id() == 0 && !isLarge) {
            atomicAdd(first_count, -1); 
            //printf("fc: %d\n", fc);
          }
          lgc ++;
        }
      } else {
        if(get_lane_id() == 0) {
          if (ltc > 0) {
            atomicAdd(tiny_count, -ltc);
            ltc = 0;
          }
          if (lgc > 0) {
            atomicAdd(global_count, -lgc);
            lgc = 0;
          }
        }
        __syncwarp();
        //printf("%d %d %d\n", gc, lc, tc);
        if (gc == 0 && lc == 0 && tc == 0)
          break;
      }
    }
  }
  __syncthreads();
  if (get_lane_id() == 0 && local_mb_counter != 0)  
  {
    atomicAdd(maximal_bicliques, local_mb_counter);
  }
}


IterFinderGpu7::IterFinderGpu7(CSRBiGraph *graph_in, int ngpus_) : IterFinderGpu(graph_in) {
  graph_gpu_ = graph_in;
  vsize = graph_in->V_size;
  int noGpus = 0;
  gpuErrchk(cudaGetDeviceCount(&noGpus));
  ngpus = std::min(noGpus, ngpus_);
  verticesEachGpu = (vsize + ngpus) / ngpus; 
}

IterFinderGpu7::~IterFinderGpu7() {
  graph_gpu_->Reset();
  delete graph_gpu_;
  gpuErrchk(cudaFree(dev_global_buffer_));
  gpuErrchk(cudaFree(dev_mb_));
  gpuErrchk(cudaFree(dev_processing_vertex_));
}

void IterFinderGpu7::Execute() {
  const int LargeSize = LARGE_WORKLIST_SIZE / ngpus;
  printf("large worklist size: %d local tiny size: %d\n", LARGE_WORKLIST_SIZE, LOCAL_TINY_WORKLIST_SIZE);
  cudaStream_t streams[ngpus];
  std::vector<cudaEvent_t>start_events(ngpus);
  std::vector<cudaEvent_t>preProcess_events(ngpus);
  std::vector<cudaEvent_t>end_events(ngpus);
  WorkList<LargeTask> *global_large_worklist[ngpus];
  WorkList<LargeTask> large_worklist[ngpus];
  unsigned long long *global_count[ngpus];
  unsigned long long *large_count[ngpus];
  unsigned long long *tiny_count[ngpus];
  unsigned long long *first_count[ngpus];

  int *all_mb;
  cudaMallocHost(&all_mb, ngpus * sizeof(int));
  int *local_processing_vertex[ngpus], *local_mb[ngpus], *local_global_buffer[ngpus], *host_processing_vertex, end_vertex[ngpus];
  cudaMallocHost(&host_processing_vertex, ngpus * sizeof(int));
  LargeTask *large_worklist_ptr[ngpus];
  LargeTask *lts[ngpus];
  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaMallocHost(&lts[gid], LARGE_WORKLIST_SIZE * sizeof(LargeTask)));
    for (int i = 0; i < LARGE_WORKLIST_SIZE; i++) {
      lts[gid][i] = LargeTask();
    }
  }
  
  size_t g_size = (size_t)MAX_BLOCKS * BUFFER_PER_BLOCK_2;
  
  int *isProcessed;
  cudaMallocManaged(&isProcessed, sizeof(int) * vsize);
  for (int i = 0; i < vsize; i++) {
    isProcessed[i] = 0;
  } 
 
//init shared_worklist 
  int *cost;
  cudaMallocManaged(&cost, sizeof(int) * vsize);

  CSRBiGraph *graph_gpu[ngpus]; 
 
  for (int i = 0; i < ngpus; i++) {
    cudaSetDevice(i);
    graph_gpu[i] = new CSRBiGraph();
    graph_gpu[i]->CopyToGpu(*graph_gpu_);
    host_processing_vertex[i] = i;
    //end_vertex[i] = std::min((i + 1) * verticesEachGpu, vsize) - 1;
    //std::cout<<i<<": "<<host_processing_vertex[i]<<" "<<end_vertex[i]<<std::endl;
    end_vertex[i] = vsize - 1;
    gpuErrchk(cudaMalloc((void **)&first_count[i], sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void **)&local_mb[i], sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&local_processing_vertex[i], sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&large_worklist_ptr[i], sizeof(LargeTask) * LargeSize));
    gpuErrchk(cudaMalloc((void **)&global_count[i], sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void **)&large_count[i], sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void **)&tiny_count[i], sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void **)&global_large_worklist[i], sizeof(WorkList<LargeTask>)));
    gpuErrchk(cudaMalloc((void **)&local_global_buffer[i], g_size * sizeof(int)));
    large_worklist[i].Init(large_worklist_ptr[i], LargeSize, false);
    
    gpuErrchk(cudaStreamCreate(&streams[i]));
    gpuErrchk(cudaEventCreate(&start_events[i]));
    gpuErrchk(cudaEventCreate(&preProcess_events[i]));
    gpuErrchk(cudaEventCreate(&end_events[i]));
  }
  
  start_time_ = get_cur_time();
  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaSetDevice(gid));
    gpuErrchk(cudaEventRecord(start_events[gid], streams[gid]))
    gpuErrchk(cudaMemcpyAsync(large_worklist_ptr[gid], lts[gid], LargeSize * sizeof(LargeTask), cudaMemcpyHostToDevice, streams[gid]));
    gpuErrchk(cudaMemcpyAsync(global_large_worklist[gid], &large_worklist[gid], sizeof(WorkList<LargeTask>), cudaMemcpyHostToDevice, streams[gid]));
    gpuErrchk(cudaMemsetAsync(global_count[gid], 0, sizeof(unsigned long long), streams[gid]));
    gpuErrchk(cudaMemsetAsync(large_count[gid], 0, sizeof(unsigned long long), streams[gid]));
    gpuErrchk(cudaMemsetAsync(tiny_count[gid], 0, sizeof(unsigned long long), streams[gid]));
    gpuErrchk(cudaMemsetAsync(local_mb[gid], 0, sizeof(int), streams[gid]));
    gpuErrchk(cudaMemsetAsync(first_count[gid], 0, sizeof(unsigned long long), streams[gid]));
    gpuErrchk(cudaMemcpyAsync(local_processing_vertex[gid], &host_processing_vertex[gid], sizeof(int), cudaMemcpyHostToDevice, streams[gid]));
    gpuErrchk(cudaMemsetAsync(local_global_buffer[gid], 0, g_size * sizeof(int), streams[gid]));
    IterFinderKernel_7<<<MAX_BLOCKS, WARP_PER_BLOCK * 32, 0, streams[gid]>>>(
      *graph_gpu[gid], local_global_buffer[gid], local_mb[gid], local_processing_vertex[gid], 
      end_vertex[gid], global_large_worklist[gid],   
      global_count[gid], large_count[gid], tiny_count[gid], first_count[gid], isProcessed, ngpus);
    gpuErrchk(cudaMemcpyAsync(&all_mb[gid], local_mb[gid], sizeof(int), cudaMemcpyDeviceToHost, streams[gid]))
    gpuErrchk(cudaEventRecord(end_events[gid], streams[gid]))

  }

  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaStreamSynchronize(streams[gid]));
    float time;
    cudaEventElapsedTime(&time, start_events[gid], end_events[gid]);
    std::cout<<"Processing time of gpu "<<gid<<": "<<time/1000<<"s"<<std::endl;
    gpuErrchk(cudaStreamDestroy(streams[gid]));
  }
  maximal_nodes_ = 0;
  for (int i = 0; i < ngpus; i++) {
    std::cout<<"gpu "<<i<<": "<<all_mb[i]<<std::endl;
    maximal_nodes_ += all_mb[i];
  }
  exe_time_ = get_cur_time() - start_time_;
}
