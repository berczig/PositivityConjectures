#include "GlobalDataPreparer.h"
#include "GlobalUIODataPreparer.h"

using namespace std;

GlobalDataPreparer::GlobalDataPreparer(vector<int> uio_sizes, vector<vector<int>> partitions) : uio_sizes(uio_sizes), partitions(partitions){
    for (int i = 0; i < uio_sizes.size(); i++){
        uiopreparers.emplace_back(uio_sizes[i], partitions[i]);
        GlobalUIODataPreparer G = uiopreparers.back();
        G.calculateCoefficients();
        cout << "lol" << endl;
    }
}

void GlobalDataPreparer::generateData(){}

void GlobalDataPreparer::saveData(string filepath){}