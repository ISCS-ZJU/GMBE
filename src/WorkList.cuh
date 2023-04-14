#ifndef __GPU_WORKLIST_H
#define __GPU_WORKLIST_H

#include "GpuUtil.cuh"
#include <stdio.h>
template<typename T>
class WorkList {
private:
public:
  T *list_ptr;
  volatile size_t front, rear;
  size_t size;
  bool local;
  __host__ __device__ WorkList();
  __host__ __device__ void Init(T *, size_t, bool local_ = true);
  __device__ bool is_empty();
  __device__ bool is_full();
  __device__ size_t push_works(T *, size_t);
  __device__ size_t push_works(int, int *, size_t);
  __device__ size_t push_work(const T&);
  __device__ size_t get(T&);
  __device__ size_t get_warp(T&);
  __device__ size_t get_work_num();
  __device__ size_t get_free_size();
  __device__ size_t get_works_from_global(WorkList<T> &global_worklist); 
  __device__ void require_read(const size_t &require_size, size_t &read_start, size_t &get_num); 
  __device__ void finish_read(const size_t &write_start, const size_t &get_num);
  __device__ void require_write(const size_t &require_size, size_t &write_start, size_t &push_num); 
  __device__ void finish_write(const size_t &write_start, const size_t &push_num);
};

template<typename T>
__host__ __device__ WorkList<T>::WorkList() {
}

template<typename T>
__host__ __device__ void WorkList<T>::Init(T *work_list_, size_t size_, bool local_) {
  list_ptr = work_list_;
  front = rear = 0;
  size = size_;  
  local = local_;
}

template<typename T>
__device__ bool WorkList<T>::is_empty() {
  return rear == front; 
}

template<typename T>
__device__ bool WorkList<T>::is_full() {
  return (rear + 1) % size == front;
}

template<typename T>
__device__ size_t WorkList<T>::get_work_num() {
  return (rear + size - front) % size;
}

template<typename T>
__device__ size_t WorkList<T>::get_free_size() {
  return size - (rear + size - front) % size - 1;
}

template<typename T>
__device__ void WorkList<T>::require_read(const size_t &require_size, size_t &read_start, size_t &get_num) {
  if (require_size <= 0) {
    read_start = (size_t)-1;
    get_num = 0;
    return;
  }
  if (get_lane_id() == 0) {
    size_t old;
    do {
      old = front;
      if(local)__threadfence_block();
      else __threadfence();//防止获取了旧rear，而获取新front，导致worknum计算错误
      size_t work_num = (rear + size - old) % size;
      get_num = work_num < require_size ? work_num : require_size;
      if (get_num <= 0) break;
      read_start = atomicCAS((unsigned long long*)&front, (unsigned long long)old, 
                             (unsigned long long)(old + get_num) % size);
    } while (old != read_start);
  }
  get_num = get_value_from_lane_x(get_num);
  if (get_num <= 0) {
    return;
  }
  read_start = get_value_from_lane_x(read_start);
}

template<typename T>
__device__ void WorkList<T>::require_write(const size_t &require_size, size_t &write_start, size_t &push_num) {
  if (require_size <= 0) {
    write_start = (size_t)-1;
    push_num = 0;
    return;
  }
  if (get_lane_id() == 0) {
    size_t old;
    do {
      old = rear;
      if(local)__threadfence_block();
      else __threadfence();
      size_t free_size = size - 1 - (old - front + size) %  size;
      push_num = free_size < require_size ? free_size : require_size;
      if (push_num <= 0)break;
      write_start = atomicCAS((unsigned long long*)&rear, (unsigned long long)old, 
                              (unsigned long long)(old + push_num) % size);
    } while (old != write_start);
  }
  push_num = get_value_from_lane_x(push_num);
  if (push_num <= 0) return;
  write_start = get_value_from_lane_x(write_start);
}


/*template<typename T>
__device__ size_t WorkList<T>::get_works_from_global(WorkList<T> &global_worklist) {
  size_t get_num;
  size_t read_start, write_start;
  if (get_lane_id() == 0) {
    size_t wt_inc = get_free_size();
    size_t wt = atomicAdd((unsigned long long*)&reading, (unsigned long long)wt_inc);
    size_t free_size = get_free_size();
    if (free_size < wt) {
      get_num = 0;
    }
    else {
      size_t require_size = free_size - wt;
      global_worklist.require_read(require_size, read_start, get_num);
    }
    if (get_num <= 0) {
      atomicAdd_block((unsigned long long*)&writing, (unsigned long long)-wt_inc);
    }
    else {
      size_t old;
      atomicAdd_block((unsigned long long*)&writing, (unsigned long long)-(wt_inc - get_num));
      do {
        old = rear;
        write_start = atomicCAS((unsigned long long*)&rear, (unsigned long long)old, (unsigned long long)(old + get_num) % size);
      } while (write_start != old);
      atomicAdd_block((unsigned long long*)&writing, (unsigned long long)-get_num);
    }
  }
  get_num = get_value_from_lane_x(get_num);
  if (get_num <= 0)
  {
    return 0;
  }
  write_start = get_value_from_lane_x(write_start);
  read_start = get_value_from_lane_x(read_start);
  for (int i = get_lane_id(); i < get_num; i += warpSize) {
    volatile T &rd_item = global_worklist.list_ptr[(read_start + i) % global_worklist.size];
    while (!rd_item.isvalid());
    volatile T &wt_item = list_ptr[(write_start + i) % size];
    while (wt_item.isvalid());
    list_ptr[(write_start + i) % size].safeCopy(global_worklist.list_ptr[(read_start + i) % global_worklist.size]);
    global_worklist.list_ptr[(read_start + i) % global_worklist.size].invalidate();
  }
  return get_num;
}*/

template<typename T>
__device__ size_t WorkList<T>::push_works(int first_vertex, int *second_vertices, size_t size_) {
  size_t push_num;
  size_t write_start;
  require_write(size_, write_start, push_num);
  if (push_num <= 0) return 0;
  for (size_t i = get_lane_id(); i < push_num; i += warpSize) {
    volatile T &item = list_ptr[(write_start + i) % size];
    while (item.isvalid());
    list_ptr[(write_start + i) % size].safeCopy(T(first_vertex, second_vertices[i]));
  }
  return push_num;
}

template<typename T>
__device__ size_t WorkList<T>::push_works(T* ptr, size_t size_) {
  size_t push_num;
  size_t write_start;
  require_write(size_, write_start, push_num);
  if (push_num <= 0) return 0;
  for (int i = get_lane_id(); i < push_num; i += warpSize) {
    volatile T &item = list_ptr[(write_start + i) % size];
    while (item.isvalid());
    list_ptr[(write_start + i) % size].safeCopy(ptr[i]);
  }
  return push_num;
}

template<typename T>
__device__ size_t WorkList<T>::push_work(const T &work) {
  size_t idx, push_num;
 
  require_write(1, idx, push_num);
  if (push_num <= 0)return 0;
  volatile T &item = list_ptr[idx]; 
  while(item.isvalid());
  if (get_lane_id() == 0) {
    list_ptr[idx].safeCopy(work);
  }
  return 1;
}

/*template<typename T>
__device__ size_t WorkList<T>::push_work(const T &work) {
  acquire_lock(&write_lock);
  if (is_full()) {
    release_lock(&write_lock);
    return 0;
  }
  if (get_lane_id() == 0) {
    list_ptr[write_done] = work;
    write_done = (write_done + 1) % size;
    rear = write_done;
  }
  if (local)__threadfence_block();
  else __threadfence();
  release_lock(&write_lock);
  return 1;
}*/

/*template<typename T>
__device__ size_t WorkList<T>::get(T &work) {
  size_t read_start, get_num;
  require_read(1, read_start, get_num);
  if (get_num != 0) {
    work = list_ptr[read_start];
    finish_read(read_start, get_num);
  }
  return get_num;
}*/

template<typename T>
__device__ size_t WorkList<T>::get(T &work) {
  size_t idx, get_num; 
  require_read(1, idx, get_num);

  if(get_num <= 0) {
    return 0;
  }
  volatile T &item = list_ptr[idx];
  while (!item.isvalid()); 
  work = list_ptr[idx];
  __syncwarp();
  if (get_lane_id() == 0) {
    list_ptr[idx].invalidate();
  }
  return 1;
}

/*template<typename T>
__device__ size_t WorkList<T>::get_idx() {
  atomicAdd( )
}*/

#endif
