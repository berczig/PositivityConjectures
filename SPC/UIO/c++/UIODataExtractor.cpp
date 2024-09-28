#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <tuple>
#include "UIODataExtractor.h"
#include "UIO.h"
#include "combinatorics.h"
#include "SequenceGenerator.h"
#include "misc.h"
#include <memory>

using namespace std;
    // Assuming core_generator_class type, but no usage in the class was shown in the Python code



//CacheDecorator<vector<string>,int, int> cd(getKPermutationsOfN);
CacheDecorator<vector<string>,int, int, vector<int>> cd(getCyclicKPermutationsOfN);


// Constructor
UIODataExtractor::UIODataExtractor(shared_ptr<UIO> uio) : uio(uio) {}

// Count Eschers
int UIODataExtractor::countEschers(const vector<int>& partition)  {
    int count = 0;
    vector<string> P = calculateCyclicKPermutationsOfN(uio->N, sum(partition), partition);
    int escher_cycle_types = calculateNumberOfCycleTypes(partition);
    //cout << "part: " << endl;
    //printvec(partition);
    //cout << "escher_cycle_types: " << escher_cycle_types << endl << endl;


    //vector<string> interPermuteStrings(vector<string>& words){

    if (partition.size() == 1) {
        for (const auto& seq : P) {
            if (uio->isescher(seq)) {
                count += 1;
            }
        }
    }

    else if (partition.size() == 2) {
        int a = partition[0];
        for (const auto& seq : P) {
            if (uio->isescher(string(seq.begin(), seq.begin() + a)) &&
                uio->isescher(string(seq.begin() + a, seq.end()))) {
                count += 1;
            }
        }
    }

    else if (partition.size() == 3) {
        int a = partition[0], b = partition[1];
        for (const auto& seq : P) {
            if (uio->isescher(string(seq.begin(), seq.begin() + a)) &&
                uio->isescher(string(seq.begin() + a, seq.begin() + a + b)) &&
                uio->isescher(string(seq.begin() + a + b, seq.end()))) {
                count += 1;
            }
        }
    }

    else if (partition.size() == 4) {
        int a = partition[0], b = partition[1], c = partition[2];
        for (const auto& seq : P) {
            if (uio->isescher(string(seq.begin(), seq.begin() + a)) &&
                uio->isescher(string(seq.begin() + a, seq.begin() + a + b)) &&
                uio->isescher(string(seq.begin() + a + b, seq.begin() + a + b + c)) &&
                uio->isescher(string(seq.begin() + a + b + c, seq.end()))) {
                count += 1;
            }
        }
    }
    return escher_cycle_types*count;
}

// Get Coefficient
int UIODataExtractor::getCoefficient(const vector<int>& partition)  {
    if (partition.size() == 1) {
        return countEschers(partition);
    }
    else if (partition.size() == 2) {
        int dc = countEschers(partition);
        int sc = countEschers({partition[0]+partition[1]});
        return dc - sc;
    }
    else if (partition.size() == 3) {
        int n = partition[0], k = partition[1], l = partition[2];
        return 2 * countEschers({n + k + l}) +
                countEschers(partition) -
                countEschers({n + l, k}) -
                countEschers({n + k, l}) -
                countEschers({l + k, n});
    }
    else if (partition.size() == 4) {
        int a = partition[0], b = partition[1], c = partition[2], d = partition[3];
        return countEschers({a, b, c, d}) -
                countEschers({a + b, c, d}) -
                countEschers({a + c, b, d}) -
                countEschers({a + d, b, c}) -
                countEschers({b + c, a, d}) -
                countEschers({b + d, a, c}) -
                countEschers({c + d, a, b}) +
                countEschers({a + b, c + d}) +
                countEschers({a + c, b + d}) +
                countEschers({a + d, b + c}) +
                2 * countEschers({a + b + c, d}) +
                2 * countEschers({a + b + d, c}) +
                2 * countEschers({a + c + d, b}) +
                2 * countEschers({b + c + d, a}) -
                6 * countEschers({a + b + c + d});
    }
    return 0;  // Default case
}

// Overload the output operator to print the UIODataExtractor object
ostream& operator<<(ostream& os, const UIODataExtractor& extractor) {
    os << "EXTRACTOR OF [" << UIODataExtractor::vectorToString(extractor.uio->encoding) << "]";
    return os;
}


// Helper function to convert a vector to a string
string UIODataExtractor::vectorToString(const vector<int>& vec) {
    ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}
