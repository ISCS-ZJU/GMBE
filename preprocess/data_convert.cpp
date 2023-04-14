#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>


void Convert(const std::string& infile,
                         const std::string& outfile) {
  std::ifstream fin(infile);
  std::ofstream fout(outfile);
  if (!fin.is_open() || !fout.is_open()) {
    std::cout << "Open file error!" << std::endl;
    exit(1);
  }

  std::map<int, std::set<int>> item_map;
  std::map<int, int> tag;
  int now = 0;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty() || line[0] == '%' || line[0] == '#') continue;
    std::stringstream ss(line);
    int src, dst;
    ss >> src >> dst;
    if(tag.find(dst) == tag.end()) {
      tag[dst] = now++;
    }
    item_map[src].insert(tag[dst]); // adjust the start id from 1 to 0
  }
  fin.close();

  for (auto& p : item_map) {
    for (auto it = p.second.begin(); it != p.second.end(); it++) {
      if (it != p.second.begin()) fout << " ";
      fout << *it;
    }
    fout << std::endl;
  }
  fout.close();
}

int main(int argc, char *argv[]) {
  int opt;
  std::string infile, outfile;
  infile = argv[1];
  outfile = argv[2];
  if(infile != ""){
    int pos = infile.rfind(".txt");
      int start_pos = infile.rfind("/");
      std::cout << infile << " >> " << outfile << std::endl;
      Convert(infile, outfile);
  }
  return 0;
}
