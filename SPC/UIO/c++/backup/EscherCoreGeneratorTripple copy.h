#ifndef ESCHERCOREGENERATORTRIPPLE_H
#define ESCHERCOREGENERATORTRIPPLE_H

#include "EscherCoreGeneratorAbstract.h"
#include <memory>

class EscherCoreGeneratorTripple : public EscherCoreGeneratorAbstract {
public:
    EscherCoreGeneratorTripple(shared_ptr<UIO>& uio, const vector<int>& partition);

    char compareTwoCoreElements(int a, int b) override;

    vector<string> getCoreLabels(const vector<int>& partition) override;
    map<string, vector<string>> getCoreComparisions(const vector<int>& partition) override;

    vector<char> generateCore(const string& escher) override;

};

#endif  // ESCHERCOREGENERATORTRIPPLE_H
