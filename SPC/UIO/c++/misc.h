#ifndef MISC_H
#define MISC_H

#include <string>
#include <vector>

using namespace std;

void printLoadingBar(int progress, int total, string symbol);
void write_file(string filepath, vector<vector<string>> data);
int sum(const vector<int>& vec);

#endif