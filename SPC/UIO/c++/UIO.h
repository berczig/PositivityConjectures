#ifndef UIO_H
#define UIO_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

class UIO {
public:
    // Constants representing the comparison relationships
    static const int INCOMPARABLE;
    static const int LESS;
    static const int GREATER;
    static const int EQUAL;
    
    

    // Class members
    const int N;  // Size of the encoding
    vector<int> encoding;  // The UIO encoding
    vector<vector<int>> comparison_matrix;  // Matrix of comparisons
    string repr;  // String representation of the UIO encoding

    // Constructor
    UIO(const vector<int>& uio_encoding);

    // Utility function to convert a vector to a string
    string vectorToString(const vector<int>& vec) const;

    // isescher method - checks if the sequence is an Escher sequence
    bool isescher(const string& seq) const;

    // isarrow method - checks if there is an arrow between elements i and j
    bool isarrow(const string& escher, int i, int j, bool verbose = false) const;

    // Overload the output operator to print the UIO object
    friend ostream& operator<<(ostream& os, const UIO& uio);
};

#endif  // UIO_H
