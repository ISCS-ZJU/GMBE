#ifndef MBEA_ITER_FINDER_GPU_H
#define MBEA_ITER_FINDER_GPU_H
#define LARGE_DEGREE 100

#include "WorkList.cuh"
#include "BicliqueFinder.h"
#include "GpuUtil.cuh"
#ifndef MAX_DEGREE_BOUND
#define MAX_DEGREE_BOUND 0x4000
#endif

#ifndef MAX_2_H_DEGREE_BOUND
#define MAX_2_H_DEGREE_BOUND 0x24000
#endif

#define UNION_THRESHOLD 10000

class IterFinderGpu : public BicliqueFinder {
 public:
  IterFinderGpu() = delete;
  IterFinderGpu(CSRBiGraph *graph_in);
  ~IterFinderGpu();
  virtual void Execute();

 protected:
  CSRBiGraph *graph_gpu_;
  int *dev_global_buffer_, *dev_mb_, *dev_processing_vertex_;
  double clock_rate;
  unsigned long long* total_clock_initialize, *total_clock_queue, *total_clock_generate_tiny, *total_clock_iterate, *dev_total_clock;
};

class IterFinderGpu2 : public IterFinderGpu {
 public:
  IterFinderGpu2() = delete;
  IterFinderGpu2(CSRBiGraph *graph_in);
  ~IterFinderGpu2();
  void Execute();
};

class IterFinderGpu3 : public IterFinderGpu {
 public:
  IterFinderGpu3() = delete;
  IterFinderGpu3(CSRBiGraph *graph_in);
  ~IterFinderGpu3();
  void Execute();
};

class IterFinderGpu4 : public IterFinderGpu {
 public:
  IterFinderGpu4() = delete;
  IterFinderGpu4(CSRBiGraph *graph_in);
  ~IterFinderGpu4();
  void Execute();
  private:
  ShortTask *global_worklist;
  int *worklist_num;
  ShortTask *local_worklist;
  unsigned *worklistG_lock;
  int *write_done;
};

class IterFinderGpu5 : public IterFinderGpu {
 public:
  IterFinderGpu5() = delete;
  IterFinderGpu5(CSRBiGraph *graph_in);
  ~IterFinderGpu5();
  void Execute();
  private:
  ShortTask *global_worklist;
  int *worklist_num;
  unsigned *worklist_lock;
  ShortTask *local_worklist;
  int *local_upper_bound;
};


class IterFinderGpu6 : public IterFinderGpu {
 public:
  IterFinderGpu6() = delete;
  IterFinderGpu6(CSRBiGraph *graph_in, int bound_height = 20, int bound_size = 1500);
  ~IterFinderGpu6();
  virtual void Execute();
  protected:
  WorkList<LargeTask> *global_large_worklist;
  unsigned long long *global_count;
  unsigned long long *large_count;
  unsigned long long *tiny_count;
  int bound_height;
  int bound_size;
};

class IterFinderGpu7: public IterFinderGpu {
public:
  IterFinderGpu7() = delete;
  IterFinderGpu7(CSRBiGraph *graph_in, int ngpus);
  ~IterFinderGpu7();
  void Execute();
private:
  int ngpus, verticesEachGpu, vsize;
};

__device__ __forceinline__ int NeighborsIntersectL(CSRBiGraph &graph,
                                                   int *L_vertices, int L_size,
                                                   int *buf, int *dst) {
  int *base_0, size_0, *base_1, size_1, u0, u1;
  u0 = L_vertices[0];
  base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
  size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

  for (int i = L_size - 2; i >= 0; i--) {
    u1 = L_vertices[L_size - 1 - i];
    base_1 = graph.rev_column_indices + graph.rev_row_offset[u1];
    size_1 = graph.rev_row_offset[u1 + 1] - graph.rev_row_offset[u1];
    if (i & 1) {
      size_0 = seq_intersect_warp(base_0, size_0, base_1, size_1, buf);
      base_0 = buf;
    } else {
      size_0 = seq_intersect_warp(base_0, size_0, base_1, size_1, dst);
      base_0 = dst;
    }
  }
  return size_0;
}

__device__ __forceinline__ int NeighborsIntersectR(CSRBiGraph &graph,
                                                   int *R_vertices, int R_size,
                                                   int *buf, int *dst) {
  int *base_0, size_0, *base_1, size_1, v0, v1;
  v0 = R_vertices[0];
  base_0 = graph.column_indices + graph.row_offset[v0];
  size_0 = graph.row_offset[v0 + 1] - graph.row_offset[v0];

  for (int i = R_size - 2; i >= 0; i--) {
    v1 = R_vertices[R_size - 1 - i];
    base_1 = graph.column_indices + graph.row_offset[v1];
    size_1 = graph.row_offset[v1 + 1] - graph.row_offset[v1];
    if (i & 1) {
      size_0 = seq_intersect_warp(base_0, size_0, base_1, size_1, buf);
      base_0 = buf;
    } else {
      size_0 = seq_intersect_warp(base_0, size_0, base_1, size_1, dst);
      base_0 = dst;
    }
  }
  return size_0;
}

__device__ __forceinline__ int NeighborsUnionL(CSRBiGraph &graph,
                                               int *L_vertices, int L_size,
                                               int *buf, int *dst) {
  int *base_0, size_0, *base_1, size_1, u0, u1;
  u0 = L_vertices[0];
  base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
  size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];

  if(L_size > UNION_THRESHOLD) {
    int cnt = 0, common_neighbors = 0;
    for (int i = 0; i < graph.V_size; i++) {
      size_1 = graph.row_offset[i + 1] - graph.row_offset[i];
      base_1 = graph.column_indices + graph.row_offset[i];
      common_neighbors = seq_intersect_warp_cnt (L_vertices, L_size, base_1, size_1);
      if(common_neighbors != 0) {
        if(get_lane_id() == 0)dst[cnt] = i;
        cnt++;
      }
    }
    return cnt;
  }


  for (int i = L_size - 2; i >= 0; i--) {
    u1 = L_vertices[L_size - 1 - i];
    base_1 = graph.rev_column_indices + graph.rev_row_offset[u1];
    size_1 = graph.rev_row_offset[u1 + 1] - graph.rev_row_offset[u1];
    if (i & 1) {
      size_0 = seq_union_warp_adv(base_0, size_0, base_1, size_1, buf);
      base_0 = buf;
    } else {
      size_0 = seq_union_warp_adv(base_0, size_0, base_1, size_1, dst);
      base_0 = dst;
    }
  }
  return size_0;
}

template <typename T = int>
__device__ __forceinline__ void seq_intersect_warp_for_iter_finder(
    T *src_0, T *level_neighbors_0, T size_0, T *src_1, T size_1,
    T incoming_val = 1, T level_bound = 0) {
  if (size_0 == 0 || size_1 == 0) return;

  for (T i = get_lane_id(); i < size_1; i += warpSize) {
    T pos = binary_search(src_1[i], src_0, 0, size_0 - 1);
    if (pos != -1) {
      if (incoming_val == 0) {
        int val_out = level_neighbors_0[pos] + 1;
        if ((val_out >> 16) >= level_bound) val_out |= 0x7fff0000;
        level_neighbors_0[pos] = val_out;
      } else {
        level_neighbors_0[pos] += incoming_val;
      }
    }
  }
}

__device__ __forceinline__ bool finder_push(
    CSRBiGraph &graph, int push_cid, int level, int *L_vertices, int *L_level,
    const int L_size, int *C_vertices, int *C_level_neighbors, const int C_size,
    bool init = false) {
  bool is_maximal = true;
  int cur_c = C_vertices[push_cid];
  int *base_cur_c = graph.column_indices + graph.row_offset[cur_c];
  int size_cur_c = graph.row_offset[cur_c + 1] - graph.row_offset[cur_c];

  // compute L
  int size_last_vs_cur = 0;
  for (int i = get_lane_id(); i < L_size; i += warpSize) {
    if (L_level[i] == level - 1) {
      size_last_vs_cur += 0x10000;
      if (binary_search(L_vertices[i], base_cur_c, 0, size_cur_c - 1) >= 0) {
        L_level[i] = level;
        size_last_vs_cur++;
      }
    }
  }
  size_last_vs_cur = warp_sum(size_last_vs_cur);
  size_last_vs_cur = get_value_from_lane_x(size_last_vs_cur);
  int size_cur_l = size_last_vs_cur & 0xffff;

  bool recompute = init || (size_last_vs_cur >> 16) > (size_cur_l << 1);
  int victim_level = recompute ? level : level - 1;

  if (recompute) {
    for (int i = get_lane_id(); i < C_size; i += warpSize) {
      C_level_neighbors[i] &= 0xffff0000;
    }
  }

  // compute C
  for (int i = 0; i < L_size; i++) {
    if (L_level[i] == victim_level) {
      int u0 = L_vertices[i];
      int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
      int size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];
      for (int j = get_lane_id(); j < size_0; j += warpSize) {
        int pos = binary_search(base_0[j], C_vertices, 0, C_size - 1);
        if (pos != -1) C_level_neighbors[pos] += recompute ? 1 : -1;
      }
    }
  }

  // maximality check
  for (int i = get_lane_id(); i < C_size && is_maximal; i += warpSize) {
    int c_level_neighbors = C_level_neighbors[i];
    if ((c_level_neighbors & 0xffff) == size_cur_l) {
      if (i < push_cid && (c_level_neighbors >> 16) >= level)
        is_maximal = false;
      else if ((c_level_neighbors >> 16) > level)
        C_level_neighbors[i] = (level << 16) | (c_level_neighbors & 0xffff);
    }
  }

  return __all_sync(FULL_MASK, is_maximal);
}

__device__ __forceinline__ void finder_pop(CSRBiGraph &graph, int pop_cid,
                                           int level, int *L_vertices,
                                           int *L_level, const int L_size,
                                           int *C_vertices,
                                           int *C_level_neighbors,
                                           const int C_size) {
  // recover L and C
  for (int i = 0; i < L_size; i++) {
    if (L_level[i] == level - 1) {
      int u0 = L_vertices[i];
      int *base_0 = graph.rev_column_indices + graph.rev_row_offset[u0];
      int size_0 = graph.rev_row_offset[u0 + 1] - graph.rev_row_offset[u0];
      for (int i = get_lane_id(); i < size_0; i += warpSize) {
        int pos = binary_search(base_0[i], C_vertices, 0, C_size);
        if (pos != -1) {
          int level_neighbors = C_level_neighbors[pos] + 1;
          if ((level_neighbors >> 16) >= level)
            level_neighbors = 0x3fff0000 | (level_neighbors & 0xffff);
          C_level_neighbors[pos] = level_neighbors;
        }
      }
    } else if (L_level[i] >= level && get_lane_id() == 0)
      L_level[i] = level - 1;
  }

#ifdef PRUNE_EN
  for (int i = get_lane_id(); i < C_size; i += warpSize) {
    int level_neighbors = C_level_neighbors[i];
    int c_level = level_neighbors >> 16;
    if (c_level >= level) {
      int prefix = (c_level == 0x3fff || i == pop_cid) ? 0x7fff : level - 1;
      C_level_neighbors[i] = (level_neighbors & 0xffff) | (prefix << 16);
    }
  }
#else
  if (get_lane_id() == 0) C_level_neighbors[pop_cid] |= 0x7fff0000;
#endif
}

#endif
