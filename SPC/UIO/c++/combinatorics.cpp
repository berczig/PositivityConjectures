#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "combinatorics.h"
#include <chrono>
#include "stringy.h"
#include <unordered_set>
#include <unordered_map>
#include <math.h> 
#include <stdexcept>
#include "misc.h"
 
using namespace std;

void printvec(vector<int> v){
    for (int i: v)
        std::cout << i << ' ';
    cout << endl;
}
void print2dvec(vector<vector<int>> v){
    cout << "print2dvec" << endl;
    for (vector<int> vec: v)
        printvec(vec);
}

/* returns all k subsets out of n elements (without order)
*/
vector<string> comb(int N, int K)
{
    vector<string> results = vector<string>();

    string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
 
    // print integers and permute bitmask
    do {
        string comb = "";
        for (int i = 0; i < N; ++i){   
            if (bitmask[i]){
                comb.push_back(i);
            } 
        }
        results.push_back(comb);
    } while (prev_permutation(bitmask.begin(), bitmask.end()));
    return results;
}

/* return all possible permutations of all vectors in the combinations vector
*/
vector<string> perm(vector<string> combinations){
    vector<string> results = vector<string>();
    for (string comb : combinations){
        sort(comb.begin(), comb.end());
        do {
            string perm(comb);
            results.push_back(perm);
        } while (next_permutation(comb.begin(), comb.end()));
    }
    return results;
}

/**
 * Get all possible K-length permutations of 0, ..., N-1
 */
vector<string> getKPermutationsOfN(int N, int K){
    vector<string> D = comb(N, K);
    return perm(D);
}

/**
 * Permute string acording to partition
 */
void sortByPartition(string& s, vector<int> partition){
    int startpos = 0;
    vector<string> chunks;
    for (int i = 0; i < partition.size(); i++){
        int size = partition[i];
        string part_s = s.substr(startpos, size);

        // sort each chunk internally by cyclic-shifting the first entry to be the smallest
        int smallest = 255;
        for (int i=0; i<part_s.size(); i++)    
            if (part_s[i] < smallest)
                smallest = part_s[i];    
        while (part_s[0] != smallest){
            rotate(part_s.begin(), part_s.begin() + 1, part_s.end());
        }
        
        chunks.push_back(part_s);
        startpos += size;
    }

    // sort chunks of same size against each
    int lastindex = 0;
    for (auto it = partition.begin(); it < partition.end();){
        int p = *it;
        int counted = count(partition.begin(), partition.end(), p);

        // sort on slice of same length chunks
        sort(chunks.begin()+lastindex, chunks.begin()+lastindex+counted);

        lastindex += counted;
        it += counted;
    }

    // concat back to string
    string out;
    for (string cs : chunks)
        out += cs;
    s = out;
}

long factorial(const int n){
    long f = 1;
    for (int i=1; i<=n; ++i)
        f *= i;
    return f;
}

int calculateNumberOfCycleTypes(vector<int> partition){
    int product = 1;
    int last_cycle_length = -1;
    for (int cycle_length : partition){
        if (cycle_length != last_cycle_length){
            last_cycle_length = cycle_length;
            int occurances = count(partition.begin(), partition.end(), cycle_length);
            product *= pow(cycle_length, occurances) * factorial(occurances);
        }
    }
    return product;
}

/**
 * Get all possible K-length permutations of 0, ..., N-1 modulo escher-cyclic
 */
vector<string> getCyclicKPermutationsOfN(int N, int K, vector<int> partition){
    if (N < K)
        throw invalid_argument("N < K");
    if (sum(partition) != K)
        throw invalid_argument("sum(partition) != K");
    vector<string> combinations = comb(N, K);
    vector<string> results;
    unordered_set<string> resultsSet;
    for (string comb : combinations){
        sort(comb.begin(), comb.end());

        do {
            string perm(comb);
            sortByPartition(perm, partition);

            auto it = resultsSet.find(perm);
            if(it == resultsSet.end()) {
                resultsSet.emplace(perm);
            }
        } while (next_permutation(comb.begin(), comb.end()));
    }

    // to vector
    for (string s : resultsSet){
        results.push_back(s);
    }

    return results;
}























/* returns all possible partition of N by recursion
*/
vector<vector<int>> getPartitionUpToN_(int N){
    vector<vector<int>> results = vector<vector<int>>();

    if ( N > 1){
        for (int i = 1; i < N; i++){
            vector<vector<int>> subsums = getPartitionUpToN_(N-i);
            for (vector<int> subsum : subsums){
                subsum.push_back(i);
                results.push_back(subsum);
            }
        }
    }

    vector<int> res = {N};
    results.push_back(res);

    return results;
}

/* returns all possible partition of N
*/
vector<vector<int>> getPartitionUpToN(int N){
    vector<vector<int>> sums = getPartitionUpToN_(N);
    for (vector<int>& sum : sums){
        sort(sum.begin(), sum.end(), greater<int>());
    }    
    return sums;
}

vector<vector<int>> generate_all_uio_encodings(int n){
    vector<vector<int>> encodings = {};
    generate_all_uio_encodings_(encodings, vector<int>{0}, n, 1);
    cout << "generated " << encodings.size() << " UIOS! Encodings" << endl;
    return encodings;
}

void generate_all_uio_encodings_(vector<vector<int>>& encodings, vector<int> current_uio, int n, int index){
    if (index == n){
        encodings.push_back(current_uio);
    }else{
        for (int j = current_uio[index-1]; j < index+1; j++){
            vector<int> current_uio_copy(current_uio);
            current_uio_copy.push_back(j);
            generate_all_uio_encodings_(encodings, current_uio_copy, n, index+1);
        }
    }
}

void print1dstring(vector<string> v){
    for (string& s : v){
        for (int i : s){
            cout << i+1 << " ";
        }
        cout << endl;
    }
}

void printstring(string s){
    for (int i : s){
        cout << i << " ";
    }
    cout << endl;
}

 
// int main()
// {
//     //getKPermutationsOfN(3,3);
//     //getKLengthPartitionsOfN(6, 2);
//     vector<vector<int>> D;
//     //D = getPartitionUpToN(12);
//     //vector<string> E = getKPermutationsOfN(10,6);

//     auto start = std::chrono::system_clock::now();
//     vector<string> E = getCyclicKPermutationsOfN(4, 4, {4});

//     auto end = std::chrono::system_clock::now();
//     std::chrono::duration<double> elapsed_seconds = end-start;
//     cout << elapsed_seconds.count() << endl;

//     cout << calculateNumberOfCycleTypes({4}) << endl;

//     //D = generate_all_uio_encodings(5);
//     print1dstring(E);
//     //print2dvec(D);
//     //print2dvec(getKPermutationsOfN(5,2));
// }