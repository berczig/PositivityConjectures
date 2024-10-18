#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <limits.h>  // For PATH_MAX
#endif

using namespace std;

void printLoadingBar(int progress, int total, string symbol) {
    int barWidth = 50; // Width of the loading bar
    float progressRatio = static_cast<float>(progress) / total;
    int pos = static_cast<int>(barWidth * progressRatio); // Calculate the position for the progress

    cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            cout << symbol; // Filled part of the bar
        } else {
            cout << " "; // Empty part of the bar
        }
    }
    cout << "] " << int(progressRatio * 100) << "%"; // Show percentage
    if (progress == total)
         cout << "\n";
    cout.flush(); // Flush to make sure it prints instantly
}

void write_file(string filepath, vector<vector<string>> data){
    ofstream myfile;
    myfile.open(filepath);

    for (auto row = data.begin(); row != data.end(); row++){
        for (auto it = row->begin(); it != row->end(); it++){
            myfile << *it;
            if (it != row->end()-1)
                myfile << ",";
            else
                myfile << "\n";
        }
        
    }

    myfile.close();
}

string partitionToString(vector<int> p){ 
    stringstream ss;
    ss << "(";
    for (auto it = p.begin(); it != p.end(); it++){
        int v = *it;
        ss << v;
        if (it != p.end()-1)
            ss << ",";
    }
    ss << ")";
    return ss.str();
}

// Helper function to sum the elements of a vector
int sum(const vector<int>& vec) {
    int total = 0;
    for (int v : vec) {
        total += v;
    }
    return total;
}


int mod(int a, int b){
    return ((a % b) + b) % b;
}

// Recursive function to return gcd of a and b 
long long gcd(long long int a, long long int b)
{
  if (b == 0)
    return a;
  return gcd(b, a % b);
}

// Function to return LCM of two numbers 
long long lcm(int a, int b)
{
    return (a / gcd(a, b)) * b;
}

string cyclicslice(const string& s, int start, int end){
    int n = s.size();
    start = mod(start, n);
    end = mod(end, n);

    if (start < end)
        return s.substr(start, end-start);
    else if (start == end)
        return s;
    return s.substr(start, n-start) + s.substr(0, end);
}

string repeat(const string& input, unsigned num){
    string ret;
    ret.reserve(input.size() * num);
    while (num--)
        ret += input;
    return ret;
}



// Function to check if a file exists in a directory
bool fileExists(const string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Function to get the current executable path
string getExecutablePath() {
    char buffer[1024];
    string exePath;

#ifdef _WIN32
    GetModuleFileName(NULL, buffer, 1024);
    exePath = string(buffer);
#else
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        exePath = string(buffer);
    } else {
        throw runtime_error("Could not determine executable path.");
    }
#endif

    // Remove the executable name to get the directory
    size_t lastSlashPos = exePath.find_last_of("/\\");
    if (lastSlashPos != string::npos) {
        return exePath.substr(0, lastSlashPos);
    }
    return exePath;
}

/**
 * Get path of the folder containing the github readme
 */
string getBasePath() {
    string currentDir = getExecutablePath();

    // Loop to move up directories
    while (!currentDir.empty()) {
        // Check if "README.md" exists in the current directory
        string readmePath = currentDir + "/README.md";
        if (fileExists(readmePath)) {
            return currentDir;
        }

        // Find the parent directory by removing the last segment
        size_t lastSlashPos = currentDir.find_last_of("/\\");
        if (lastSlashPos != string::npos) {
            currentDir = currentDir.substr(0, lastSlashPos);
        } else {
            currentDir.clear();  // No more parents, we're at the root
        }
    }

    throw std::runtime_error("README.md not found in any parent directories.");
}

double timeStop(){
    static bool activated = false;
    static chrono::_V2::system_clock::time_point start;
    chrono::_V2::system_clock::time_point end;
    chrono::duration<double> elapsed_seconds;
    switch (activated)
    {
    case true:
        end = chrono::system_clock::now();
        elapsed_seconds = end-start;
        activated = false;
        return elapsed_seconds.count();
    case false:
        start = chrono::system_clock::now();
        activated = true;
        break;
    }

    return 0.0;
}