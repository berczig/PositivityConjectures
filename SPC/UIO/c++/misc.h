#ifndef MISC_H
#define MISC_H

#include <string>
#include <vector>

using namespace std;

void printLoadingBar(int progress, int total, string symbol);
void write_file(string filepath, vector<vector<string>> data);
int sum(const vector<int>& vec);
string partitionToString(vector<int> p);
int mod(int a, int b);
// Recursive function to return gcd of a and b 
long long gcd(long long int a, long long int b);
// Function to return LCM of two numbers 
long long lcm(int a, int b);
string cyclicslice(const string& s, int start, int end);
string repeat(const string& input, unsigned num);
string getBasePath();
double timeStop();
#endif