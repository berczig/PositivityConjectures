#ifndef ESCHERCOREGENERATORABSTRACT_H
#define ESCHERCOREGENERATORABSTRACT_H

#include "CoreGenerator.h"

class EscherCoreGeneratorAbstract : public CoreGenerator {
public:
    EscherCoreGeneratorAbstract(shared_ptr<UIO>& uio, const vector<int>& partition);

    vector<int> getpseudosubescherstartingpoints(const string& escher, int k);
    vector<int> getInsertionPoints(const string& u, const string& v, int lcm_);
    tuple<vector<int>, vector<int>> getInsertionsSubeshers(const string& u, const string& v);
    vector<int> get_shortb_insertion_and_subescher_of_2_eschers(const string& u, const string& v);
    void add_one_to_last_element(vector<int>& list);
    string concat(const string& first_escher, const string& second_escher, int insertionpoint);
};

#endif  // ESCHERCOREGENERATORABSTRACT_H
