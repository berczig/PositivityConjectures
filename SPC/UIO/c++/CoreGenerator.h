#ifndef COREGENERATOR_H
#define COREGENERATOR_H

#include "UIO.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std;


struct core {
    vector<int> values;
    core(vector<int> values) : values(values){}
    string toString(){
        string s; 
        for (char c : values){
            s += to_string(c)+",";
        }
        return s;
    }
};

struct coreRepresentation{
    vector<unsigned char> comparisons;
    coreRepresentation(vector<unsigned char> comparisons) : comparisons(comparisons){}
    bool operator<(const coreRepresentation& other) const {
        return comparisons < other.comparisons;
    }

    string toString() const{
        string s; 
        for (unsigned char c : comparisons){
            s += to_string(c)+",";
        }
        return s;
    }
};

class CoreGenerator {
public:
    shared_ptr<UIO> uio;
    const vector<int> partition;
    vector<tuple<int, int>> comp_indices;

    CoreGenerator(shared_ptr<UIO>& uio, const vector<int>& partition);
    virtual ~CoreGenerator() = default;

    /**
     * escher - full escher of length sum(partition), not a tuple
     */
    virtual vector<int> generateCore(const string& escher) = 0;
    virtual char compareTwoCoreElements(int a, int b) = 0;

    virtual map<string, vector<string>> getCoreComparisions(const vector<int>& partition) = 0;
    virtual vector<string> getCoreLabels(const vector<int>& partition) = 0;

    void calculate_comp_indices(const vector<int>& partition);
    vector<unsigned char> getCoreRepresentation(const core& core);


    int getCoreRepresentationLength(const vector<int>& partition);
    int getCoreLength(const vector<int>& partition);
    vector<tuple<string, string>> getOrderedCoreComparisions(const vector<int>& partition);
};

#endif  // COREGENERATOR_H
