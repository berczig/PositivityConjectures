#ifndef UIODATAEXTRACTOR_H
#define UIODATAEXTRACTOR_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include "UIO.h"  // Assuming UIO class is defined in a separate file
#include "CacheDecorator.h"
#include "SequenceGenerator.h"
#include <memory>

using namespace std;

class UIODataExtractor {
public:
    // Constructor
    shared_ptr<UIO> uio;
    UIODataExtractor(shared_ptr<UIO> uio);

    // Count Eschers
    int countEschers(const vector<int>& partition) ;

    // Get Coefficient
    int getCoefficient(const vector<int>& partition) ;

    // Overload the output operator to print the UIODataExtractor object
    friend ostream& operator<<(ostream& os, const UIODataExtractor& extractor);

private:
    // Helper function to convert a vector to a string
    static string vectorToString(const vector<int>& vec);
};

#endif // UIODATAEXTRACTOR_H
