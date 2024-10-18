#ifndef SEQUENCEGENERATOR_H
#define SEQUENCEGENERATOR_H

#include <string>
#include <iostream>
#include "CacheDecorator.h"
#include <vector>
#include <map>

using namespace std;

vector<string> calculateCyclicKPermutationsOfN(int N, int K, vector<int> partition);
vector<string> calculateKPermutationsOfN(int N, int K);

#endif  // UIO_H
