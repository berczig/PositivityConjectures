#include "SequenceGenerator.h"
#include "combinatorics.h"
#include <string>
#include <vector>
#include <tuple>

using namespace std;


vector<string> calculateCyclicKPermutationsOfN(int N, int K, vector<int> partition){
    static map<tuple<int, int, vector<int>>, vector<string>> storage;
    tuple<int, int, vector<int>> key = {N, K, partition};
    if (storage.find(key) != storage.end()){
        return storage[key];
    }

    try {
        vector<string> sequences = getCyclicKPermutationsOfN(N,K,partition);
        storage[key] = sequences;
        return sequences;
    } catch (const invalid_argument& e) {
    }

}
