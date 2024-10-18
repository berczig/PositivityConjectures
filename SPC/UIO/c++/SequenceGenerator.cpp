#include "SequenceGenerator.h"
#include "combinatorics.h"
#include <string>
#include <vector>
#include <tuple>

using namespace std;


vector<string> calculateCyclicKPermutationsOfN(int N, int K, vector<int> partition){
    // cout << "N: " << N << " K: " << K << " part: " << partition[0] << partition[1] << partition[2] << partition[3] << endl;
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
        cerr << e.what();
    }

}

vector<string> calculateKPermutationsOfN(int N, int K){
    static map<tuple<int, int>, vector<string>> storage2;
    tuple<int, int> key = {N, K};
    if (storage2.find(key) != storage2.end()){
        return storage2[key];
    } 

    try {
        vector<string> sequences = getKPermutationsOfN(N,K);
        storage2[key] = sequences;

        return sequences;
    } catch (const invalid_argument& e) {
        cerr << e.what();
    }

}
