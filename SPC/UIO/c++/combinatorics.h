#include <vector>
using namespace std;

#ifndef COMBINATORICS
#define COMBINATORICS

void printvec(vector<int> v);
void print2dvec(vector<vector<int>> v);
vector<string> comb(int N, int K);
vector<string> perm(vector<vector<int>> combinations);
vector<string> getKPermutationsOfN(int N, int K);
vector<string> getCyclicKPermutationsOfN(int N, int K, vector<int> partition);
int calculateNumberOfCycleTypes(vector<int> partition);
vector<vector<int>> getPartitionUpToN(int N);
vector<vector<int>> getPartitionUpToN_(int N);
vector<vector<int>> generate_all_uio_encodings(int n);
void generate_all_uio_encodings_(vector<vector<int>>& encodings, vector<int> current_uio, int n, int index);

void printstring(string s);

#endif