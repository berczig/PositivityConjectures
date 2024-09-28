#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void printLoadingBar(int progress, int total, string symbol) {
    int barWidth = 50; // Width of the loading bar
    float progressRatio = static_cast<float>(progress) / total;
    int pos = static_cast<int>(barWidth * progressRatio); // Calculate the position for the progress

    cout.flush();
    cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            cout << symbol; // Filled part of the bar
        } else {
            cout << " "; // Empty part of the bar
        }
    }
    cout << "] " << int(progressRatio * 100); // Show percentage
    if (progress != total)
        cout << " %\r";
    cout.flush(); // Flush to make sure it prints instantly
}

void write_file(string filepath, vector<vector<string>> data){
    ofstream myfile;
    myfile.open (filepath);

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

// Helper function to sum the elements of a vector
int sum(const vector<int>& vec) {
    int total = 0;
    for (int v : vec) {
        total += v;
    }
    return total;
}