#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "UIO.h"
using namespace std;

// Constants representing the comparison relationships
const char UIO::INCOMPARABLE = 100;
const char UIO::LESS = 101;
const char UIO::GREATER = 102;
const char UIO::EQUAL = 103;

// Constructor
UIO::UIO(const vector<int>& uio_encoding, int id) : N(uio_encoding.size()), ID(id){
    encoding = uio_encoding;
    repr = vectorToString(encoding);

    // Initialize the comparison matrix
    comparison_matrix.resize(N, vector<char>(N, UIO::EQUAL));

    // Fill the comparison matrix based on the encoding
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (uio_encoding[j] <= i) {
                comparison_matrix[i][j] = INCOMPARABLE;
                comparison_matrix[j][i] = INCOMPARABLE;
            } else {
                comparison_matrix[i][j] = LESS;
                comparison_matrix[j][i] = GREATER;
            }
        }
    }


}

// Utility function to convert a vector to a string
string UIO::vectorToString(const vector<int>& vec) const {
    stringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

// isescher method - checks if the sequence is an Escher sequence
bool UIO::isescher(const string& seq) const {
    for (size_t i = 0; i < seq.size() - 1; ++i) {
        if (!isarrow(seq, i, i + 1)) {
            return false;
        }
    }
    return isarrow(seq, seq.size() - 1, 0);
}

// isarrow method - checks if there is an arrow between elements i and j
bool UIO::isarrow(const string& escher, int i, int j, bool verbose) const {
    if (verbose) {
        std::cout << "arrow " << escher << " " << i << " " << j 
                    << " " << (comparison_matrix[escher[i]][escher[j]] != GREATER) << std::endl;
    }
    return comparison_matrix[escher[i]][escher[j]] != GREATER;  // EQUAL also intersects
}

// isarrow method - checks if there is an arrow between elements i and j
bool UIO::isarrow(const string& escher1, const string& escher2, int i, int j) const {
    return comparison_matrix[escher1[i]][escher2[j]] != GREATER;  // EQUAL also intersects
}

// Overload the output operator to print the UIO object
ostream& operator<<(std::ostream& os, const UIO& uio) {
    os << uio.repr;
    return os;
}