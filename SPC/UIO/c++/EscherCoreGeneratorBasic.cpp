#include "EscherCoreGeneratorBasic.h"
#include "misc.h"
#include <tuple>
using namespace std;

EscherCoreGeneratorBasic::EscherCoreGeneratorBasic(shared_ptr<UIO>& uio, const vector<int>& partition)
    : EscherCoreGeneratorAbstract(uio, partition) {
        calculate_comp_indices(partition);
        }

char EscherCoreGeneratorBasic::compareTwoCoreElements(int a, int b) {
    return a < b ? UIO::LESS : UIO::GREATER;
}

vector<string> EscherCoreGeneratorBasic::getCoreLabels(const vector<int>& partition) {
    return {"0", "subescher start interval", "subescher end interval", "1.insert"};
} 

map<string, vector<string>> EscherCoreGeneratorBasic::getCoreComparisions(const vector<int>& partition) {
    return {{"0", {"subescher start interval", "1.insert"}},
            {"subescher end interval", {"1.insert"}}};
}

vector<int> EscherCoreGeneratorBasic::generateCore(const string& escher) {
    const string u = string(escher.begin(), escher.begin() + partition[0]);
    const string v = string(escher.begin()+ partition[0], escher.begin() + partition[0] + partition[1]);
    vector<int> core = get_shortb_insertion_and_subescher_of_2_eschers(u,v);
    add_one_to_last_element(core);
    core.insert(core.begin(), 0);
    return core;
}

