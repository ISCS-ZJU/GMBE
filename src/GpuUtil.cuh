#ifndef __GPU_UTIL_H
#define __GPU_UTIL_H

#include <cuda_runtime.h>
#include <stdio.h>
#define FULL_MASK 0xffffffff
#define POS_INFINITY 0x7fffffff

#ifndef MAX_SM
#define MAX_SM 108
#endif

#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 1
#endif

#define MAX_BLOCKS (MAX_SM * BLOCKS_PER_SM)

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK 16 
#endif

#define WARP_PER_SM (WARP_PER_BLOCK * BLOCKS_PER_SM)
#define AVG_ITEMS_PER_WARP 20

__device__ __forceinline__ unsigned get_lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ unsigned count_bit(unsigned bit_mask) {
  unsigned ret_val;
  asm volatile("popc.b32 %0, %1;" : "=r"(ret_val) : "r"(bit_mask));
  return ret_val;
}

__device__ __forceinline__ unsigned find_ms_bit(unsigned bit_mask) {
  unsigned ret_val;
  asm volatile("bfind.u32 %0, %1;" : "=r"(ret_val) : "r"(bit_mask));
  return ret_val;
}

__device__ __forceinline__ unsigned find_ls_bit(unsigned bit_mask) {
  return find_ms_bit(bit_mask & (-bit_mask));
}

template <typename T = int>
__device__ __forceinline__ T gpu_max(T num0, T num1) {
  return num0 > num1 ? num0 : num1;
}
template <typename T = int>
__device__ __forceinline__ T gpu_min(T num0, T num1) {
  return num0 < num1 ? num0 : num1;
}
template <typename T = int>
__device__ __forceinline__ void gpu_swap(T& num0, T& num1) {
  T tmp = num0;
  num0 = num1;
  num1 = tmp;
}

template <typename T = int>
__device__ __forceinline__ T linear_search(T key, T* list, T left, T right) {
  for (T i = left; i <= right; i++) {
    if (list[i] == key) return i;
  }
  return -1;
}

template <typename T = int>
__device__ __forceinline__ T binary_search(T key, const T* list, T left, T right) {
  while (left <= right) {
    T mid = (left + right) / 2;
    T value = list[mid];
    if (value == key) return mid;
    if (value < key)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

template <typename T = int>
__device__ __forceinline__ T binary_search_next_id(T key, T* list, T left,
                                                   T right) {
  while (left <= right) {
    T mid = (left + right) / 2;
    T value = list[mid];
    if (value == key) return -1;
    if (value < key)
      left = mid + 1;
    else
      right = mid - 1;
  }
  T value = list[left];
  if (value > key)
    return left;
  else if (value < key)
    return left + 1;
  else
    return -1;
}

template <typename T = int>
__device__ __forceinline__ T seq_intersect_warp_cnt(const T* src_0, const T size_0, const T* src_1,
                                                const T size_1) {
  if (size_0 == 0 || size_1 == 0) return 0;
  const T* lookup = src_0;
  T lookup_size = size_0;
  const T* search = src_1;
  T search_size = size_1;
  if (size_0 > size_1) {
    lookup = src_1;
    lookup_size = size_1;
    search = src_0;
    search_size = size_0;
  }
  T count = 0;
  for (T i = get_lane_id(); i < lookup_size; i += warpSize) {
    unsigned active_mask = __activemask();
    __syncwarp(active_mask);
    T pos = binary_search(lookup[i], search, 0, search_size - 1);
    unsigned match_mask = __ballot_sync(active_mask, pos != -1);
    T idx = count_bit(match_mask << (warpSize - get_lane_id() - 1));
    count += count_bit(match_mask);
  }
  
  count = __shfl_sync(FULL_MASK, count, 0);
  return count;
}


template <typename T = int>
__device__ __forceinline__ T seq_intersect_warp(T* src_0, T size_0, T* src_1,
                                                T size_1, T* dst) {
  if (size_0 == 0 || size_1 == 0) return 0;
  T* lookup = src_0;
  T lookup_size = size_0;
  T* search = src_1;
  T search_size = size_1;
  if (size_0 > size_1) {
    lookup = src_1;
    lookup_size = size_1;
    search = src_0;
    search_size = size_0;
  }
  T count = 0;
  for (T i = get_lane_id(); i < lookup_size; i += warpSize) {
    unsigned active_mask = __activemask();
    __syncwarp(active_mask);
    T pos = binary_search(lookup[i], search, 0, search_size - 1);
    unsigned match_mask = __ballot_sync(active_mask, pos != -1);
    T idx = count_bit(match_mask << (warpSize - get_lane_id() - 1));
    if (pos != -1) dst[count + idx - 1] = lookup[i];
    count += count_bit(match_mask);
  }
  
  count = __shfl_sync(FULL_MASK, count, 0);
  return count;
}

template <typename T = int>
__device__ __forceinline__ T seq_diff_warp(T* src_0, T size_0, T* src_1,
                                           T size_1) {
  if (size_0 == 0 || size_1 == 0) return size_0;
  T count = 0;
  for (T i = get_lane_id(); i < size_0; i += warpSize) {
    unsigned active_mask = __activemask();
    __syncwarp(active_mask);
    T pos = binary_search(src_0[i], src_1, 0, size_1 - 1);
    unsigned match_mask = __ballot_sync(active_mask, pos == -1);
    T idx = count_bit(match_mask << (warpSize - get_lane_id() - 1));
    if (pos == -1) src_0[count + idx - 1] = src_0[i];
    count += count_bit(match_mask);
  }
  count = __shfl_sync(FULL_MASK, count, 0);
  return count;
}

template <typename T = int>
#ifdef ADV_UNION
__device__ __forceinline__ T seq_union_warp_naive(T* src_0, T size_0, T* src_1,
                                                  T size_1, T* dst) {
#else
__device__ __forceinline__ T seq_union_warp_adv(T* src_0, T size_0, T* src_1,
                                                T size_1, T* dst) {
#endif
  int count = 0;
  if (get_lane_id() == 0) {
    for (int i = 0, j = 0; i < size_0 || j < size_1;) {
      if (i == size_0)
        dst[count++] = src_1[j++];
      else if (j == size_1)
        dst[count++] = src_0[i++];
      else if (src_0[i] < src_1[j])
        dst[count++] = src_0[i++];
      else if (src_0[i] > src_1[j])
        dst[count++] = src_1[j++];
      else {
        dst[count++] = src_0[i++];
        j++;
      }
    }
  }
  count = __shfl_sync(FULL_MASK, count, 0);
  return count;
}

template <typename T = int>
#ifdef ADV_UNION
__device__ __forceinline__ T seq_union_warp_adv(T* src_0, T size_0, T* src_1,
                                                T size_1, T* dst) {
#else
__device__ __forceinline__ T seq_union_warp_naive(T* src_0, T size_0, T* src_1,
                                                  T size_1, T* dst) {
#endif
  __shared__ T base_0_buf_sd[64 * WARP_PER_BLOCK];
  __shared__ T base_1_buf_sd[64 * WARP_PER_BLOCK];
  T warp_id = threadIdx.x / warpSize;
  T lane_id = get_lane_id();
  T* src_0_buf = base_0_buf_sd + 64 * warp_id;
  T* src_1_buf = base_1_buf_sd + 64 * warp_id;

  int window_A_begin = 0;
  int next_window_A_begin = 0;
  int res_cnt = 0;
  src_0_buf[lane_id] = (lane_id >= size_0) ? POS_INFINITY : src_0[lane_id];
  src_1_buf[lane_id] = (lane_id >= size_1) ? POS_INFINITY : src_1[lane_id];

  for (int i = lane_id; i < size_0 + size_1; i += warpSize) {
    int val = -1;
    int Ax = window_A_begin + lane_id;
    // binary search diagonal in local window for path
    int left = window_A_begin, right = window_A_begin + lane_id;

    while (Ax != left) {
      int count_0x = src_0_buf[Ax & 63];
      int count_0x_1 = src_0_buf[(Ax - 1) & 63];
      int val_Bx = src_1_buf[(i - Ax) & 63];
      int val_Bx_1 = src_1_buf[(i - Ax + 1) & 63];
      if (count_0x < val_Bx) {
        if (Ax == right) {
          val = count_0x;
          next_window_A_begin = Ax + 1;
          break;
        }
        left = Ax;
      } else if (count_0x == val_Bx || count_0x_1 == val_Bx_1) {
        next_window_A_begin = (count_0x == val_Bx) ? Ax : Ax - 1;
        break;
      } else if (count_0x_1 > val_Bx_1) {
        right = Ax;
      } else {
        if (count_0x_1 > val_Bx) {
          val = count_0x_1;
        } else if (count_0x_1 < val_Bx) {
          val = val_Bx;
        } else {
          val = count_0x_1;
        }
        next_window_A_begin = Ax;
        // Found it
        break;
      }
      Ax = (left + right) / 2;
    }

    if (Ax == window_A_begin) {
      next_window_A_begin = Ax;
      int count_0x = src_0_buf[Ax & 63];
      int val_Bx = src_1_buf[(i - Ax) & 63];
      if (count_0x > val_Bx) {
        val = val_Bx;
      } else if (count_0x < val_Bx) {
        val = count_0x;
      }
    }
    unsigned match_mask = __ballot_sync(__activemask(), val != -1);
    if (val != -1) {
      int index = res_cnt + count_bit(((1 << lane_id) - 1) & match_mask);
      dst[index] = val;
    }
    res_cnt += count_bit(match_mask);
    if ((i & 0xffffffe0) + 32 < size_0 + size_1) {
      next_window_A_begin = __shfl_sync(FULL_MASK, next_window_A_begin, 31);

      int cur_A_mask = (window_A_begin + 31) & 0xffffffe0;
      int cur_B_mask = ((i & 0xffffffe0) - window_A_begin + 31) & 0xffffffe0;

      window_A_begin = next_window_A_begin;
      int next_A_mask = (window_A_begin + 31) & 0xffffffe0;
      int next_B_mask = ((i & 0xffffffe0) - window_A_begin + 63) & 0xffffffe0;

      if (cur_A_mask != next_A_mask) {
        int xid = next_A_mask + lane_id;
        src_0_buf[xid & 63] = (xid >= size_0) ? POS_INFINITY : src_0[xid];
      }
      if (cur_B_mask != next_B_mask) {
        int xid = next_B_mask + lane_id;
        src_1_buf[xid & 63] = (xid >= size_1) ? POS_INFINITY : src_1[xid];
      }
    }
  }
  res_cnt = __shfl_sync(FULL_MASK, res_cnt, 0);

  return res_cnt;
}

template <typename T = int>
__device__ __forceinline__ T get_value_from_lane_x(T val, int srcLane = 0) {
  T ret_val = __shfl_sync(FULL_MASK, val, srcLane);
  return ret_val;
}

template <typename T = int>
__device__ __forceinline__ T warp_min(T val) {
  for (int lane_mask = warpSize >> 1; lane_mask > 0; lane_mask >>= 1) {
    int val_in = __shfl_xor_sync(FULL_MASK, val, lane_mask);
    val = (val < val_in) ? val : val_in;
  }
  return __shfl_sync(FULL_MASK, val, 0);
}

template <typename T = int>
__device__ __forceinline__ T warp_sum(T val) {
  for (int lane_mask = warpSize >> 1; lane_mask > 0; lane_mask >>= 1) {
    int val_in = __shfl_xor_sync(FULL_MASK, val, lane_mask);
    val = val + val_in;
  }
  return val;
}

template<class T>
__device__ void acquire_lock(T *lock) {
  if(get_lane_id() == 0) {
    T lock_value = 1;
    while(lock_value) {
      lock_value = atomicCAS(lock, 0, lock_value);
    }
  }
  __syncwarp();
}
template<class T>
__device__ void release_lock(T *lock) {
  __syncwarp();
  if(get_lane_id() == 0) {
    atomicExch(lock, 0);
  }
}

class ShortTask {
public:
  unsigned long long vertices;
  __device__ __host__ ShortTask() {
    vertices = (unsigned long long)-1;
  }
  __device__ ShortTask(int first, int second) {
    vertices = ((unsigned long long)first << 32) | second;
  }
  __device__ ShortTask(const ShortTask &task) {
    vertices = task.vertices;
  }
  __device__ ShortTask & operator = (const ShortTask &task) {
    if(this != &task) {
      vertices = task.vertices;
    }
    return *this;
  }
  __device__ void safeCopy(const ShortTask &task) {
    atomicExch(&vertices, task.vertices);
  }
  __device__ void Init() {
    vertices = (unsigned long long) -1;
  }
  __device__ bool isvalid () volatile {
    if (get_lane_id() == 0) {
      //printf("%lld\n", vertices);
    }
    return vertices != (unsigned long long)-1;
  }
  __device__ void invalidate () {
    atomicExch(&vertices, (unsigned long long)-1);
  }
  __device__ int firstVertex() const {
    return vertices >> 32;
  }
  __device__ int secondVertex() const {
    return vertices & 0xffffffff;
  }
};

class LongTask{
public:
  int vertices[4];
  __device__ LongTask(const LongTask &task) {
    for (int i = 0; i < 4; i++) {
      vertices[i] = task.vertices[i];
    }
  }
  __device__ __host__ LongTask() {
    for (int i = 0; i < 4; i++)
    {
        vertices[i] = -1;
    }
  }
  __device__ LongTask(const int &first, const int &second) {
    vertices[0] = first;
    vertices[1] = second;
    vertices[2] = vertices[3] = -1;
  }
  __device__ LongTask(const ShortTask &work) {
    vertices[0] = work.firstVertex();
    vertices[1] = work.secondVertex();
    vertices[2] = vertices[3] = -1;
  }
  __device__ void safeCopy(const LongTask &task, bool systemWide = false) {
    for (int i = 1; i < 4; i++) {
      vertices[i] = task.vertices[i];
    }
    if (systemWide) {
      __threadfence_system();
      atomicExch_system(&vertices[0], task.vertices[0]);
    } else if (__isGlobal(this))
    {
      __threadfence();
      atomicExch(&vertices[0], task.vertices[0]);
    }
    else if (__isShared(this)){
      __threadfence_block();
      atomicExch_block(&vertices[0], task.vertices[0]);
    }
  }
  __device__ void Init() {
    for (int i = 0; i < 4; i++) {
      vertices[i] = -1;
    }
  }
  __host__ __device__ LongTask & operator = (const LongTask &task) {
    if(this != &task) {
      for (int i = 0; i < 4; i++) {
        vertices[i] = task.vertices[i];
      }
    }
    return *this;
  }
  __device__ bool isvalid() volatile {
    //printf("%d\n", vertices[0]);
    return vertices[0] != -1;
  }
  
  __device__ void invalidate() {
    if (__isGlobal(this)) {
      atomicExch(&vertices[0], -1);
    } else if (__isShared(this)) {
      atomicExch_block(&vertices[0], -1);
    } else {
      atomicExch_system(&vertices[0], -1);
    }
  }
}; 
typedef LongTask LargeTask;
typedef LargeTask TinyTask;
#endif
