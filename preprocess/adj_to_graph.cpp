#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;
int main(int argc, char *argv[])
{
    string infilename = argv[1];
    string outfilename = argv[2];
    cout<<infilename<<" "<<outfilename<<endl;
    ifstream infile(infilename);
    ofstream outfile(outfilename);
    string line;
    vector<vector<int>> graph; 
    int rmax = -1;
    while(getline(infile, line))
    {
        vector<int>neighbors;
        stringstream ss(line);
        string v;
        while(getline(ss, v, ' '))
        {
            int r = stoi(v);
            rmax = r > rmax ? r : rmax;
            neighbors.emplace_back(r);
        }
        graph.emplace_back(neighbors);
    }
    int nowL = 0;
    outfile<<graph.size()<<" "<<rmax + 1<<endl;
    for(auto neighbors: graph)
    {
        for(auto R:neighbors)
        {
            outfile<<nowL<<" "<<R<<endl;
        }
        nowL++;
    }
    infile.close();
    outfile.close();
    return 0;
}

