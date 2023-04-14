#ifndef MBEA_BIGRAPH_H
#define MBEA_BIGRAPH_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Utility.h"

struct Option {
  int uvtrans;
  int order;
  int fast_mode;
  Option() {
    uvtrans = 0;
    order = 1;
    fast_mode = 0;
  }
};
struct CSRBiGraph {
  CSRBiGraph();
  void Read(const char *filename, Option opt);
  void ReadAndReorder(const char *filename);
  void CopyToGpu(CSRBiGraph &copygraph);
  void Reset();

  int U_size, V_size, edge_size, U_degree, V_degree, U_2degree, V_2degree;

  // CSR_format
  int *row_offset;          // equal to R_size + 1
  int *column_indices;      // equal to edge size
  int *rev_row_offset;      // equal to L_size + 1
  int *rev_column_indices;  // equal to edge size
  bool device_graph;
};

#endif
