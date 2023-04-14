

#include "Utility.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;
#ifdef _MSC_VER
#include <windows.h>
#define fopen64 fopen
double get_cur_time() {
  LARGE_INTEGER nFreq;
  LARGE_INTEGER nTime;
  QueryPerformanceFrequency(&nFreq);
  QueryPerformanceCounter(&nTime);
  double time = (double)nTime.QuadPart / (double)nFreq.QuadPart;
  return time;
}

#else
#include <sys/time.h> /* gettimeofday */
double get_cur_time() {
  struct timeval tv;
  struct timezone tz;
  double cur_time;
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  return cur_time;
}
#endif

__device__ int binary_search(const int* seq, int num, int left, int right) {
  while (left <= right) {
    int mid = (left + right) / 2;
    if (seq[mid] == num)
      return mid;
    else if (seq[mid] < num)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

int seq_intersect_cnt(const int* base_0, int size_0, const int* base_1,
                      int size_1) {
  int cnt = 0;
  for (int i0 = 0, i1 = 0; i0 < size_0 && i1 < size_1;) {
    if (base_0[i0] == base_1[i1]) {
      cnt++;
      i0++;
      i1++;
    } else if (base_0[i0] > base_1[i1])
      i1++;
    else
      i0++;
  }
  return cnt;
}

std::vector<int> seq_intersect(const int* base_0, int size_0, const int* base_1,
                               int size_1) {
  std::vector<int> res;
  for (int i0 = 0, i1 = 0; i0 < size_0 && i1 < size_1;) {
    if (base_0[i0] == base_1[i1]) {
      res.emplace_back(base_0[i0]);
      i0++;
      i1++;
    } else if (base_0[i0] > base_1[i1])
      i1++;
    else
      i0++;
  }
  return res;
}
__global__ void generate_bitset(const int* base, const int size, const int vc, const int vc_id, const int* r_offset_ptr, 
                                const int* r_value_ptr, unsigned int* bitsets, const int size_r, int* bitsets_size, 
                                int* ismaximal, int *vvc)
{
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    unsigned int *all_bitset;
    if(threadIdx.x == 0)
    {
        //bitsets_size = 0;
        all_bitset = new unsigned int[size_r]();
    }
    all_bitset = (unsigned int *)__shfl_sync(0xffffffff, (unsigned long)all_bitset, 0);
    int l_id = threadIdx.x;
    if(l_id < size){
        int l = base[l_id];
        for (int r_id = r_offset_ptr[l]; r_id < r_offset_ptr[l + 1]; r_id++) 
        {
            int vertex = r_value_ptr[r_id];
            atomicOr(all_bitset + vertex, 1 << l_id);
        }
    }
    unsigned l_bs = 0xffffffff >> (32 - size);
    __shared__ int new_vc_id;
    g.sync();
    if(threadIdx.x == 0)
    {
        /*for(int i = 0; i < size_r; i++)
        {
            printf("%x ", all_bitset[i]);
        }printf("\n");*/
        *bitsets_size = 0;
        new_vc_id = 0;
        *ismaximal = 1;
        
    }
    
    g.sync();
    unsigned active_mask = __activemask();
    for(int i = threadIdx.x; i - threadIdx.x < size_r; i += 32)//Ensure no divergence
    {
        int flag = 1;
        unsigned active_mask = __ballot_sync(0xffffffff, i<size_r);
        if(i<size_r){
            if(all_bitset[i] == l_bs)
            {
                if(i < vc)
                {
                    atomicAdd(&new_vc_id, 1);
                }
                coalesced_group active = coalesced_threads();
                active.sync();
                if(i == vc && new_vc_id != vc_id)
                {
                    flag = 0;
                }
            }
            else if(all_bitset[i] != 0)
            {
                atomicAdd(bitsets_size, 1);
            }
            unsigned mask = __ballot_sync(active_mask, flag);
            if(mask != active_mask)//判断是否有线程flag为0
            {
                if(threadIdx.x == 0)
                {
                    *ismaximal = 0;
                }
                break;
            }
        }
        
    }
    g.sync();
    if(*ismaximal&&threadIdx.x == 0)
    {
        *vvc = -1;
        int count = 0;
        for(int i = 0; i < size_r; i++)
        {
            if(all_bitset[i] != 0 && all_bitset[i] != l_bs)
            {
                if(*vvc < 0 && i > vc)*vvc = count;
                bitsets[count++] = all_bitset[i];
            }
        }
    }
    if(threadIdx.x == 0)
    {
        delete [] all_bitset;
    }
}
__global__ void seq_intersect_cnt_gpu(const int*  base_0, int size_0,
                                      const int*  base_1, int size_1, int* res_sum,
                                      int total_threads = kMaxThread)
{
  __shared__ int block_0[kMaxThread];
  __shared__ int block_1[kMaxThread];

  int i_0 = 0, i_1 = 0, sum = 0;
  int bound_0 = total_threads, bound_1 = total_threads;

  while (i_0 < size_0 && i_1 < size_1) {
    bound_0 = min(size_0 - i_0, total_threads);
    bound_1 = min(size_1 - i_1, total_threads);

    if (i_0 + threadIdx.x < size_0)
      block_0[threadIdx.x] = base_0[i_0 + threadIdx.x];
    if (i_1 + threadIdx.x < size_1)
      block_1[threadIdx.x] = base_1[i_1 + threadIdx.x];
    //__syncthreads();
    __threadfence_block();
    for (int k = 0; k < bound_1; k++) {
      sum +=
          (threadIdx.x < size_0 - i_0) & (block_0[threadIdx.x] == block_1[k]);
    }
    if (block_0[bound_0 - 1] >= block_1[bound_1 - 1]) i_1 += bound_1;
    if (block_1[bound_1 - 1] >= block_0[bound_0 - 1]) i_0 += bound_0;
  }
  //__syncthreads();
  atomicAdd(res_sum, sum);
  //__syncthreads();
}

void seq_intersect_cnt_gpu_wrapper(const int* base_0, int size_0,
                                   const int* base_1, int size_1,
                                   int* res_sum) {
  int *dev_0, *dev_1, *dev_sum;
  gpuErrchk(cudaSetDevice(0));
  gpuErrchk(cudaMalloc((void**)&dev_0, size_0 * sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&dev_1, size_1 * sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&dev_sum, sizeof(int)));
  gpuErrchk(
      cudaMemcpy(dev_0, base_0, size_0 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(dev_1, base_1, size_1 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dev_sum, res_sum, sizeof(int), cudaMemcpyHostToDevice));
  seq_intersect_cnt_gpu<<<1, kMaxThread>>>(dev_0, size_0, dev_1, size_1,
                                           dev_sum);
  gpuErrchk(cudaGetLastError());
  // gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(res_sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(dev_0));
  gpuErrchk(cudaFree(dev_1));
  gpuErrchk(cudaFree(dev_sum));
}
