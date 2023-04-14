#ifndef MBEA_BICLIQUE_FINDER_H
#define MBEA_BICLIQUE_FINDER_H

#include <iostream>
#include <vector>

#include "BiGraph.h"
#include "Utility.h"

#define BITSET_BOUND 32

class Biclique {
 public:
  int Size() { return L.size() * R.size(); }
  std::vector<int> L, R;
};

class BicliqueFinder {
 public:
  BicliqueFinder() = delete;
  BicliqueFinder(CSRBiGraph* graph_in);
  virtual void Execute() = 0;
  void PrintResult(char* fn = nullptr);

 protected:
  CSRBiGraph* graph_;
  Biclique maximum_biclique_;
  int maximal_nodes_;
  double exe_time_, start_time_;
};

class MbeaFinder : public BicliqueFinder {
 public:
  MbeaFinder() = delete;
  MbeaFinder(CSRBiGraph* graph_in);
  void Execute();

 private:
  void biclique_find(const std::vector<int>& L, const std::vector<int>& R,
                     const std::vector<int>& C, int vc);
};

typedef unsigned int BITSET_T;

class MbeaFinderBitset : public BicliqueFinder {
 public:
  MbeaFinderBitset() = delete;
  MbeaFinderBitset(CSRBiGraph* graph_in);
  virtual void Execute();

 private:
  void biclique_find(const std::vector<int>& L, int vc, int vc_id);
  void biclique_find(BITSET_T parent_L, BITSET_T current_L, int vc_id,
                     std::vector<BITSET_T>& c_bs);
};

#endif