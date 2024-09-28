#ifndef GLOBALUIODATAPREPARER_H
#define GLOBALUIODATAPREPARER_H

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include "UIODataExtractor.h"

using namespace std;


class GlobalUIODataPreparer {
public:
    int n_uios;
    int uio_size;
    vector<vector<int>> encodings;
    vector<int> partition;
    vector<UIODataExtractor> extractors;
    vector<int> coeffs;

    GlobalUIODataPreparer(int uio_size, vector<int> partition);
    GlobalUIODataPreparer(const GlobalUIODataPreparer& other);
    vector<int> calculateCoefficients();
};

#endif  // UIO_H
