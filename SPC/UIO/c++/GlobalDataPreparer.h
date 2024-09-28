#ifndef GLOBALDATAPREPARER_H
#define GLOBALDATAPREPARER_H

#include <string>
#include <iostream>
#include <array>
#include <vector>
#include "GlobalUIODataPreparer.h"

using namespace std;


class GlobalDataPreparer {
public:
    vector<int> uio_sizes;
    vector<vector<int>> partitions;
    vector<GlobalUIODataPreparer> uiopreparers;

    GlobalDataPreparer(vector<int> uio_sizes, vector<vector<int>> partitions);
    void generateData();
    void saveData(string filepath);
};

#endif  // GLOBALDATAPREPARER_H
