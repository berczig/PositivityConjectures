#ifndef ESCHERCOREGENERATORBASIC_H
#define ESCHERCOREGENERATORBASIC_H

#include "EscherCoreGeneratorAbstract.h"
#include <memory>

class EscherCoreGeneratorBasic : public EscherCoreGeneratorAbstract {
public:
    EscherCoreGeneratorBasic(shared_ptr<UIO>& uio, const vector<int>& partition);

    char compareTwoCoreElements(int a, int b) override;

    vector<string> getCoreLabels(const vector<int>& partition) override;
    map<string, vector<string>> getCoreComparisions(const vector<int>& partition) override;

    vector<int> generateCore(const string& escher) override;

};

#endif  // ESCHERCOREGENERATORBASIC_H
