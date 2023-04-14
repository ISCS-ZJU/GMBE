#include <map>
#include <unordered_map>

#include "BicliqueFinder.h"

BicliqueFinder::BicliqueFinder(CSRBiGraph* graph_in)
    : graph_(graph_in), maximal_nodes_(0), exe_time_(0), start_time_(0) {}

void BicliqueFinder::PrintResult(char* fn) {
  FILE* fp = (fn == nullptr || strlen(fn) == 0) ? stdout : fopen(fn, "a+");
  if (fn != nullptr) fseek(fp, 0, SEEK_END);
  fprintf(fp, "Total processing time: %lf seconds\n", exe_time_);
  fprintf(fp, "maximal nodes:%d\n", maximal_nodes_);
  fprintf(fp, "\n");
  if (fn != NULL) fclose(fp);
}

MbeaFinder::MbeaFinder(CSRBiGraph* graph_in) : BicliqueFinder(graph_in) {}

void MbeaFinder::Execute() {
  start_time_ = get_cur_time();
  std::vector<int> L, R, C;
  L.reserve(graph_->U_size);
  for (int i = 0; i < graph_->U_size; i++) L.emplace_back(i);
  for (int i = 0; i < graph_->V_size; i++) C.emplace_back(i);
  for (int i = 0; i < C.size(); i++) {
    biclique_find(L, R, C, C[i]);
  }
  exe_time_ = get_cur_time() - start_time_;
}

void MbeaFinder::biclique_find(const std::vector<int>& L,
                               const std::vector<int>& R,
                               const std::vector<int>& C, int vc) {
  std::vector<int> L_prime, R_prime, C_prime;
  const int* offset_ptr = graph_->row_offset;
  const int* value_ptr = graph_->column_indices;

  L_prime = seq_intersect(&L[0], L.size(), value_ptr + offset_ptr[vc],
                          offset_ptr[vc + 1] - offset_ptr[vc]);
  if (L_prime.size() == 0) return;
  R_prime = R;
  for (int c : C) {
    int nc = seq_intersect_cnt(&L_prime[0], L_prime.size(),
                               value_ptr + offset_ptr[c],
                               offset_ptr[c + 1] - offset_ptr[c]);
    if (nc == L_prime.size()) {
      if (c < vc) return;
      R_prime.emplace_back(c);
    } else if (nc != 0)
      C_prime.emplace_back(c);
  }
  maximal_nodes_++;
  for (int c : C_prime) {
    if (c > vc) biclique_find(L_prime, R_prime, C_prime, c);
  }
}

MbeaFinderBitset::MbeaFinderBitset(CSRBiGraph* graph_in)
    : BicliqueFinder(graph_in) {}

void MbeaFinderBitset::Execute() {
  start_time_ = get_cur_time();
  const int* offset_ptr = graph_->row_offset;
  const int* value_ptr = graph_->column_indices;
  for (int v = 0; v < graph_->V_size; v++) {
    std::vector<int> L;
    for (int i = offset_ptr[v]; i < offset_ptr[v + 1]; i++)
      L.emplace_back(value_ptr[i]);
    biclique_find(L, v, 0);
  }
  exe_time_ = get_cur_time() - start_time_;
}

void MbeaFinderBitset::biclique_find(const std::vector<int>& L, int vc,
                                     int vc_id) {
  const int* offset_ptr = graph_->row_offset;
  const int* value_ptr = graph_->column_indices;
  const int* r_offset_ptr = graph_->rev_row_offset;
  const int* r_value_ptr = graph_->rev_column_indices;

  std::vector<int> L_prime =
      seq_intersect(&L[0], L.size(), value_ptr + offset_ptr[vc],
                    offset_ptr[vc + 1] - offset_ptr[vc]);
  if (L_prime.size() > BITSET_BOUND) {
    std::vector<std::pair<int, int>> c_nc_array;
    for (int l : L_prime) {
      std::vector<std::pair<int, int>> c_nc_array_buf;
      auto c_nc_iter = c_nc_array.begin();
      for (int i = r_offset_ptr[l]; i < r_offset_ptr[l + 1]; i++) {
        int vertex = r_value_ptr[i];
        std::pair<int, int> p = std::make_pair(vertex, 1);
        while (c_nc_iter != c_nc_array.end() && c_nc_iter->first < vertex) {
          c_nc_array_buf.emplace_back(*c_nc_iter);
          c_nc_iter++;
        }
        if (c_nc_iter != c_nc_array.end() && c_nc_iter->first == vertex) {
          p.second = c_nc_iter->second + 1;
          c_nc_iter++;
        }
        c_nc_array_buf.emplace_back(p);
      }
      while (c_nc_iter != c_nc_array.end()) {
        c_nc_array_buf.emplace_back(*c_nc_iter);
        c_nc_iter++;
      }
      std::swap(c_nc_array, c_nc_array_buf);
      c_nc_array_buf.clear();
    }

    int R_prime_size = 0;
    for (auto& p : c_nc_array) {
      if (p.second == L_prime.size()) {
        if (p.first == vc && R_prime_size != vc_id) return;  // non-maximal
        R_prime_size++;
      } else if (p.first > vc) {
        biclique_find(L_prime, p.first, R_prime_size);
      }
    }
    maximal_nodes_++;
  } else {
    std::vector<std::pair<int, BITSET_T>> c_bs_array;
    for (int i = 0; i < L_prime.size(); i++) {
      std::vector<std::pair<int, BITSET_T>> c_bs_array_buf;
      int l = L_prime[i];
      auto c_bs_iter = c_bs_array.begin();
      for (int r_id = r_offset_ptr[l]; r_id < r_offset_ptr[l + 1]; r_id++) {
        int vertex = r_value_ptr[r_id];
        std::pair<int, int> p = std::make_pair(vertex, (1 << i));
        while (c_bs_iter != c_bs_array.end() && c_bs_iter->first < vertex) {
          c_bs_array_buf.emplace_back(*c_bs_iter);
          c_bs_iter++;
        }
        if (c_bs_iter != c_bs_array.end() && c_bs_iter->first == vertex) {
          p.second = c_bs_iter->second | (1 << i);
          c_bs_iter++;
        }
        c_bs_array_buf.emplace_back(p);
      }
      while (c_bs_iter != c_bs_array.end()) {
        c_bs_array_buf.emplace_back(*c_bs_iter);
        c_bs_iter++;
      }

      std::swap(c_bs_array, c_bs_array_buf);
      c_bs_array_buf.clear();
    }
    std::vector<BITSET_T> c_bs;
    int vvc = -1;
    BITSET_T l_bs = 0xffffffff >> (32 - L_prime.size());
    int R_prime_size = 0;

    for (auto& p : c_bs_array) {
      if (p.second == l_bs) {
        if (p.first == vc && R_prime_size != vc_id) return;  // non-maximal
        R_prime_size++;
      } else {
        if (vvc < 0 && p.first > vc) {
          vvc = c_bs.size();
        }
        bool insert_flag = true;
        for (int i = 0; i < c_bs.size() && insert_flag; i++) {
          int res_and = c_bs[i] & p.second;
          if (res_and == c_bs[i] && vvc < 0) {
            c_bs[i] = p.second;
            insert_flag = false;
          } else if ((c_bs[i] | p.second) == c_bs[i]) {
            if (vvc < 0 || i < vvc || c_bs[i] == p.second) insert_flag = false;
          }
        }
        if (insert_flag) c_bs.emplace_back(p.second);
      }
    }

    for (int i = vvc; i < c_bs.size(); i++) {
      biclique_find(~0, c_bs[i], i, c_bs);
    }

    maximal_nodes_++;
  }
}
void MbeaFinderBitset::biclique_find(BITSET_T parent_L, BITSET_T current_L,
                                     int vc_id, std::vector<BITSET_T>& c_bs) {
  for (int i = 0; i < vc_id; i++) {
    BITSET_T nc = current_L & c_bs[i];
    if (nc == current_L) {
      BITSET_T pnc = parent_L & c_bs[i];
      if (pnc != parent_L) return;
    }
  }

  maximal_nodes_++;
  for (int i = vc_id + 1; i < c_bs.size(); i++) {
    BITSET_T nc = current_L & c_bs[i];
    if (nc != current_L && nc != 0) biclique_find(current_L, nc, i, c_bs);
  }
}

// MbeaFinderGpu::MbeaFinderGpu(CSRBiGraph* graph_in) : BicliqueFinder(graph_in)
// {
//   int offset_size = (graph_->V_size + 1) * sizeof(int);
//   int edges_size = (graph_->EdgeSize()) * sizeof(int);
//   gpuErrchk(cudaSetDevice(0));
//   gpuErrchk(cudaMalloc((void**)&row_offset_ptr_dev_, offset_size));
//   gpuErrchk(cudaMalloc((void**)&colomn_indices_ptr_dev_, edges_size));
//   gpuErrchk(cudaMemcpy(row_offset_ptr_dev_, graph_->GetOffsetsPtr(),
//                        offset_size, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(colomn_indices_ptr_dev_, graph_->GetValuesPtr(),
//                        edges_size, cudaMemcpyHostToDevice));
// }

// MbeaFinderGpu::~MbeaFinderGpu() {
//   gpuErrchk(cudaFree(row_offset_ptr_dev_));
//   gpuErrchk(cudaFree(colomn_indices_ptr_dev_));
// }

// void MbeaFinderGpu::Execute() {
//   start_time_ = get_cur_time();
//   std::vector<int> L, R, C;
//   L.reserve(graph_->U_size);
//   for (int i = 0; i < graph_->U_size; i++) L.emplace_back(i);
//   for (int i = 0; i < graph_->V_size; i++) C.emplace_back(i);
//   for (int i = 0; i < C.size(); i++) {
//     biclique_find(L, R, C, C[i]);
//   }
//   exe_time_ = get_cur_time() - start_time_;
// }

// void MbeaFinderGpu::biclique_find(const std::vector<int>& L,
//                                   const std::vector<int>& R,
//                                   const std::vector<int>& C, int vc) {
//   std::vector<int> L_prime, R_prime, C_prime;
//   const int* offset_ptr = graph_->GetOffsetsPtr();
//   const int* value_ptr = graph_->GetValuesPtr();

//   L_prime = seq_intersect(&L[0], L.size(), value_ptr + offset_ptr[vc],
//                           offset_ptr[vc + 1] - offset_ptr[vc]);
//   if (L_prime.size() == 0) return;
//   R_prime = R;

//   int *dev_l, *dev_c, *host_nc_cnt;

//   host_nc_cnt = new int[C.size()];

//   for (int i = 0; i < C.size(); i++) {
//     int* base_c = graph_->GetValuesPtr() + offset_ptr[C[i]];
//     int size_c = offset_ptr[C[i] + 1] - offset_ptr[C[i]];
//     int sum = 0;
//     seq_intersect_cnt_gpu_wrapper(&L_prime[0], L_prime.size(), base_c,
//     size_c,
//                                   &sum);
//     host_nc_cnt[i] = sum;
//   }
//   for (int i = 0; i < C.size(); i++) {
//     int nc = host_nc_cnt[i];
//     int c = C[i];
//     if (nc == L_prime.size()) {
//       if (c < vc) {
//         free(host_nc_cnt);
//         return;
//       }
//       R_prime.emplace_back(c);
//     } else if (nc != 0)
//       C_prime.emplace_back(c);
//   }
//   free(host_nc_cnt);
//   maximal_nodes_++;

//   for (int c : C_prime) {
//     if (c > vc) biclique_find(L_prime, R_prime, C_prime, c);
//   }
// }