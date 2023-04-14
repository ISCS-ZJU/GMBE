#include <algorithm>
#include <set>

#include "BiGraph.h"

CSRBiGraph::CSRBiGraph() {
  U_size = V_size = edge_size = 0;
  U_degree = V_degree = U_2degree = V_2degree = 0;
  row_offset = column_indices = rev_row_offset = rev_column_indices = nullptr;
  device_graph = false;
}

void CSRBiGraph::Read(const char* filename, Option opt) {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    std::cerr << "Can't open file " << filename << std::endl;
    return;
  }
  std::vector<std::vector<int>> v_adj_lists;
  std::vector<std::vector<int>> u_adj_lists;
  U_size = V_size = edge_size = 0;
  U_degree = V_degree = U_2degree = V_2degree = 0;
  std::string line;
  int item;
  std::set<int> two_hop_neighbors;

  while (std::getline(infile, line)) {
    v_adj_lists.emplace_back(std::vector<int>());
    std::stringstream ss(line);
    while (ss >> item) {
      U_size = std::max(U_size, item + 1);
      v_adj_lists.back().emplace_back(item);
      edge_size++;
    }
    V_degree = std::max(V_degree, (int)v_adj_lists.back().size());
  }
  V_size = v_adj_lists.size();

  if (!opt.fast_mode) {
    u_adj_lists.resize(U_size, std::vector<int>());

    for (int vid = 0; vid < V_size; vid++) {
      for (int uid : v_adj_lists[vid]) {
        u_adj_lists[uid].emplace_back(vid);
      }
    }
    for (int uid = 0; uid < U_size; uid++) {
      U_degree = std::max(U_degree, (int)u_adj_lists[uid].size());
      two_hop_neighbors.clear();
      for (int vid : u_adj_lists[uid]) {
        for (int u : v_adj_lists[vid]) {
          two_hop_neighbors.insert(u);
        }
      }
      U_2degree = std::max(U_2degree, (int)two_hop_neighbors.size() - 1);
    }
    for (int vid = 0; vid < V_size; vid++) {
      two_hop_neighbors.clear();
      for (int uid : v_adj_lists[vid]) {
        for (int v : u_adj_lists[uid]) {
          two_hop_neighbors.insert(v);
        }
      }
      V_2degree = std::max(V_2degree, (int)two_hop_neighbors.size() - 1);
    }

    if ((opt.uvtrans == 1 && U_size < V_size) ||
        (opt.uvtrans == 2 && U_size > V_size)) {
      std::swap(U_size, V_size);
      std::swap(U_degree, V_degree);
      std::swap(U_2degree, V_2degree);
      std::swap(u_adj_lists, v_adj_lists);
    }

    if (opt.order == 1 || opt.order == 2) {
      std::sort(v_adj_lists.begin(), v_adj_lists.end(),
                [&](std::vector<int> vec0, std::vector<int> vec1) -> bool {
                  if (vec0.size() == vec1.size()) {
                    for (int i = 0; i < vec0.size(); i++)
                      if (vec0[i] != vec1[i]) return vec0[i] < vec1[i];
                    return false;
                  } else
                    return opt.order == 1 ? vec0.size() < vec1.size()
                                          : vec0.size() > vec1.size();
                });
      for (auto& vec : u_adj_lists) vec.clear();
      u_adj_lists.resize(U_size, std::vector<int>());
      for (int vid = 0; vid < V_size; vid++) {
        for (int uid : v_adj_lists[vid]) {
          u_adj_lists[uid].emplace_back(vid);
        }
      }
    } else if (opt.order == 3) {
        int size = V_size;
        std::vector<std::vector<int> >N2(size);
        std::vector<int>NON2(size);
        auto start =get_cur_time();
        /*for(int i = 0; i < size; i++)
        {
          std::vector<int> two_hop_neighbors;
          for(auto neighbor: l_adj_lists_[i])
          {
            two_hop_neighbors = seq_union(two_hop_neighbors, r_adj_lists_[neighbor]);
          }
          N2[i] = two_hop_neighbors;
          NON2[i] = two_hop_neighbors.size();
        }*/
        auto computeExa2HopN = [&]() {
          std::vector<int> bfs (size, -1);
          for(int i=0;i<size;i++){
            int n2 = 0;
            for(auto &v:v_adj_lists[i]){ //v is in R
              for(auto &u: u_adj_lists[v]){
                if(u!=i){
                  if(bfs[u]!=i){
                    bfs[u] = i;
                    n2++;
                  }
                }
              }

            }
            NON2[i]=n2;
          }
        };
        computeExa2HopN();

        int maxDeg2 = 0;
        for(auto &deg2: NON2){
          if(deg2>maxDeg2){
            maxDeg2 = deg2;
          }
        }

        int *buck = new int[maxDeg2+1]();
        for(auto &deg2: NON2){
          buck[deg2]++;
        }


        int pos =0;
        for(int i=0;i<maxDeg2+1;i++){
          int size = buck[i];
          buck[i] = pos;
          pos = size+pos;
        }
        //buck store the starting position


        //first sort by 2-hop neighbours
        int *sortedV = new int [size];
        int *vLoc = new int[size]; // location of a vertex in a the sorted array.

        for(int i = 0; i < size; i++){
          sortedV[buck[NON2[i]]] = i;
          vLoc[i] = buck[NON2[i]];
          buck[NON2[i]]++;
        }
        // after the above loop, given a degree buck[degree] is the starting position for vertices with degree+1
        for (int i=maxDeg2;i>0;i--){
          buck[i]=buck[i-1];
        }
        buck[0] = 0;

        std::vector<int> reorder_map(size);
        std::vector<int> bfs (size, -1);
        for(int i=0;i<size;i++){
          int u = sortedV[i];
          reorder_map[i] = u;
          for(auto &v:v_adj_lists[u]){ // v is in R
            for(auto &u_prime:u_adj_lists[v]){
              if(bfs[u_prime]!=u){
                bfs[u_prime]=u;
                if(NON2[u_prime]>NON2[u]){
                  int deg_u_prime=NON2[u_prime];
                  int p_u_prime=vLoc[u_prime];
                  int positionw = buck[deg_u_prime];
                  int w = sortedV[positionw];
                  if(u_prime!=w){
                    vLoc[u_prime] = positionw;
                    sortedV[p_u_prime] = w;
                    vLoc[w] = p_u_prime;  //the previous u here almost kill me.
                    sortedV[positionw] = u_prime;
                  }
                  buck[deg_u_prime]++;
                  NON2[u_prime]--;
                }
              }

            }
          }
        }
        std::vector<std::vector<int>> n_l_adj_list(V_size);

        for (int i = 0; i < V_size; i++)
          n_l_adj_list[i] = std::move(v_adj_lists[reorder_map[i]]);
        std::swap(v_adj_lists, n_l_adj_list);
        n_l_adj_list.clear();

        for (int i = 0; i < U_size; i++) u_adj_lists[i].clear();
        for (int i = 0; i < V_size; i++) {
          for (int r_id : v_adj_lists[i]) {
            u_adj_lists[r_id].emplace_back(i);
          }
        }

        delete [] vLoc;
        delete [] sortedV;
        delete [] buck;
    }

    printf("Graph:%s ", filename);
    printf("|U|:%8d |V|:%8d |E|:%8d ", U_size, V_size, edge_size);
    printf("d(U):%8d d2(U):%8d ", U_degree, U_2degree);
    printf("d(V):%8d d2(V):%8d\n", V_degree, V_2degree);
  } else {
    if ((opt.uvtrans == 1 && U_size < V_size) ||
        (opt.uvtrans == 2 && U_size > V_size)) {
      u_adj_lists.resize(U_size, std::vector<int>());
      for (int vid = 0; vid < V_size; vid++) {
        for (int uid : v_adj_lists[vid]) {
          u_adj_lists[uid].emplace_back(vid);
        }
      }
      std::swap(U_size, V_size);
      std::swap(u_adj_lists, v_adj_lists);
    }
    if (opt.order != 0) {
      std::sort(v_adj_lists.begin(), v_adj_lists.end(),
                [&](std::vector<int> vec0, std::vector<int> vec1) -> bool {
                  if (vec0.size() == vec1.size()) {
                    for (int i = 0; i < vec0.size(); i++)
                      if (vec0[i] != vec1[i]) return vec0[i] < vec1[i];
                    return false;
                  } else
                    return opt.order == 1 ? vec0.size() < vec1.size()
                                          : vec0.size() > vec1.size();
                });
    }
    for (auto& vec : u_adj_lists) vec.clear();
    u_adj_lists.resize(U_size, std::vector<int>());
    for (int vid = 0; vid < V_size; vid++) {
      for (int uid : v_adj_lists[vid]) {
        u_adj_lists[uid].emplace_back(vid);
      }
    }
    printf("Graph:%s ", filename);
    printf("|U|:%8d |V|:%8d |E|:%8d\n", U_size, V_size, edge_size);
  }

  row_offset = new int[V_size + 1];
  column_indices = new int[edge_size];
  rev_row_offset = new int[U_size + 1];
  rev_column_indices = new int[edge_size];
  int edge_cnt = 0;
  for (int i = 0; i < V_size; i++) {
    row_offset[i] = edge_cnt;
    for (int j = 0; j < v_adj_lists[i].size(); j++) {
      column_indices[edge_cnt + j] = v_adj_lists[i][j];
    }
    edge_cnt += v_adj_lists[i].size();
    v_adj_lists[i].clear();
  }

  row_offset[V_size] = edge_cnt;
  v_adj_lists.clear();
  edge_cnt = 0;
  for (int i = 0; i < U_size; i++) {
    rev_row_offset[i] = edge_cnt;
    for (int j = 0; j < u_adj_lists[i].size(); j++) {
      rev_column_indices[edge_cnt + j] = u_adj_lists[i][j];
    }
    edge_cnt += u_adj_lists[i].size();
    u_adj_lists[i].clear();
  }
  rev_row_offset[U_size] = edge_cnt;
  u_adj_lists.clear();
}

// void CSRBiGraph::Read(const char* filename) {
//   std::vector<int> row_offset_buf;
//   std::vector<int> column_indices_buf;
//   // get graph G
//   U_size = 0;
//   std::ifstream infile(filename);
//   if (!infile.is_open()) {
//     std::cerr << "Can't open file " << filename << std::endl;
//     return;
//   }
//   std::string line;
//   int item;
//   while (std::getline(infile, line)) {
//     row_offset_buf.emplace_back(column_indices_buf.size());
//     std::stringstream ss(line);
//     while (ss >> item) {
//       U_size = std::max(U_size, item + 1);
//       column_indices_buf.emplace_back(item);
//     }
//   }
//   row_offset_buf.emplace_back(column_indices_buf.size());
//   V_size = row_offset_buf.size() - 1;
//   edge_size = column_indices_buf.size();
//   infile.clear();
//   row_offset = new int[row_offset_buf.size()];
//   column_indices = new int[column_indices_buf.size()];
//   memcpy(row_offset, &row_offset_buf[0], row_offset_buf.size() *
//   sizeof(int)); memcpy(column_indices, &column_indices_buf[0],
//          column_indices_buf.size() * sizeof(int));

//   // get graph G_rev
//   std::vector<std::vector<int>> adj_lists(U_size, std::vector<int>());
//   for (int i = 0; i < V_size; i++)
//     for (int j = row_offset[i]; j < row_offset[i + 1]; j++)
//       adj_lists[column_indices[j]].emplace_back(i);
//   rev_row_offset = new int[U_size + 1];
//   rev_column_indices = new int[edge_size];
//   rev_row_offset[0] = 0;
//   int cur_offset = 0;
//   for (int i = 0; i < U_size; i++) {
//     memcpy(rev_column_indices + cur_offset, &adj_lists[i][0],
//            adj_lists[i].size() * sizeof(int));
//     cur_offset += adj_lists[i].size();
//     rev_row_offset[i + 1] = cur_offset;
//   }
// }

void CSRBiGraph::ReadAndReorder(const char* filename) {
  std::vector<std::vector<int>> adj_lists;
  // get graph G
  U_size = 0;
  V_size = 0;
  edge_size = 0;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    std::cerr << "Can't open file " << filename << std::endl;
    return;
  }


  std::string line;
  int item;
  while (std::getline(infile, line)) {
    adj_lists.emplace_back(std::vector<int>());
    std::stringstream ss(line);
    while (ss >> item) {
      U_size = std::max(U_size, item + 1);
      adj_lists.back().emplace_back(item);
    }
  }

  std::sort(adj_lists.begin(), adj_lists.end(),
            [&](std::vector<int> vec0, std::vector<int> vec1) -> bool {
              if (vec0.size() == vec1.size()) {
                for (int i = 0; i < vec0.size(); i++)
                  if (vec0[i] != vec1[i]) return vec0[i] < vec1[i];
                return false;
              } else
                return vec0.size() < vec1.size();
            });
  std::vector<std::vector<int>> rev_adj_lists(U_size, std::vector<int>());

  for (int i = 0; i < adj_lists.size(); i++) {
    if (i == 0 || adj_lists[i - 1] != adj_lists[i]) {
      for (int l : adj_lists[i]) rev_adj_lists[l].emplace_back(V_size);
      edge_size += adj_lists[i].size();
      V_size++;
    }
    adj_lists[i].clear();
  }

  std::sort(rev_adj_lists.begin(), rev_adj_lists.end(),
            [&](std::vector<int> vec0, std::vector<int> vec1) -> bool {
              return vec0.size() < vec1.size();
            });
  for (int i = 0; i < rev_adj_lists.size(); i++) {
    for (int r : rev_adj_lists[i]) {
      adj_lists[r].emplace_back(i);
    }
  }
  
  //std::swap(adj_lists, rev_adj_lists);
  //std::swap(U_size, V_size);

  row_offset = new int[V_size + 1];
  column_indices = new int[edge_size];
  rev_row_offset = new int[U_size + 1];
  rev_column_indices = new int[edge_size];
  int edge_cnt = 0;
  for (int i = 0; i < V_size; i++) {
    row_offset[i] = edge_cnt;
    for (int j = 0; j < adj_lists[i].size(); j++) {
      column_indices[edge_cnt + j] = adj_lists[i][j];
    }
    edge_cnt += adj_lists[i].size();
    adj_lists[i].clear();
  }

  row_offset[V_size] = edge_cnt;
  adj_lists.clear();
  edge_cnt = 0;
  for (int i = 0; i < U_size; i++) {
    rev_row_offset[i] = edge_cnt;
    for (int j = 0; j < rev_adj_lists[i].size(); j++) {
      rev_column_indices[edge_cnt + j] = rev_adj_lists[i][j];
    }
    edge_cnt += rev_adj_lists[i].size();
    rev_adj_lists[i].clear();
  }
  rev_row_offset[U_size] = edge_cnt;
  rev_adj_lists.clear();
}

void CSRBiGraph::CopyToGpu(CSRBiGraph& copygraph) {
  U_size = copygraph.U_size;
  V_size = copygraph.V_size;
  edge_size = copygraph.edge_size;
  gpuErrchk(cudaMalloc((void**)&row_offset, (V_size + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&column_indices, edge_size * sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&rev_row_offset, (U_size + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&rev_column_indices, edge_size * sizeof(int)));

  gpuErrchk(cudaMemcpy(row_offset, copygraph.row_offset,
                       (V_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(column_indices, copygraph.column_indices,
                       edge_size * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(rev_row_offset, copygraph.rev_row_offset,
                       (U_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(rev_column_indices, copygraph.rev_column_indices,
                       edge_size * sizeof(int), cudaMemcpyHostToDevice));
  device_graph = true;
}

void CSRBiGraph::Reset() {
  if (device_graph) {
    gpuErrchk(cudaFree(row_offset));
    gpuErrchk(cudaFree(column_indices));
    gpuErrchk(cudaFree(rev_row_offset));
    gpuErrchk(cudaFree(rev_column_indices));
    row_offset = column_indices = rev_row_offset = rev_column_indices = nullptr;
    device_graph = false;
  } else {
    auto check_free_ptr = [&](int* ptr) {
      if (ptr != nullptr) {
        delete[] ptr;
        ptr = nullptr;
      }
    };
    U_size = V_size = edge_size = 0;
    check_free_ptr(row_offset);
    check_free_ptr(column_indices);
    check_free_ptr(rev_row_offset);
    check_free_ptr(rev_column_indices);
  }
}
