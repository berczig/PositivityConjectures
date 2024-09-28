#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "combinatorics.h"
#include <functional>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "stringy.h"
#include "UIO.h"
#include "UIODataExtractor.h"
#include "GlobalUIODataPreparer.h"
#include "misc.h"
#include "SequenceGenerator.h"
#include "GlobalDataPreparer.h"

using namespace std;




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

void generateAndSaveCoeffs(int uio_size, vector<vector<int>> partitions){
    vector<vector<int>> encodings = generate_all_uio_encodings(uio_size);

    unordered_map<string, vector<int>> data;
    vector<string> encodings_str;

    for (vector<int>& partition : partitions){
        int coefsum = 0;
        vector<int> coeffs;
        auto start = std::chrono::system_clock::now();    

        for (vector<int> uio_encoding: encodings){
            
            shared_ptr<UIO> uio = make_shared<UIO>(uio_encoding);
            encodings_str.emplace_back(uio->repr);

            UIODataExtractor extractor = UIODataExtractor(uio);  // Replace nullptr with core generator object if required
            
            int coef = extractor.getCoefficient(partition);
            coeffs.push_back(coef);
            coefsum += coef;
            //std::cout << extractor.uio << " coef: " << coef << std::endl;

        }
        data[partitionToString(partition)] = coeffs;
        cout << "coef sum: " << coefsum << endl;

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << elapsed_seconds.count() << endl;
    }

    vector<vector<string>> textdata;

    // header
    vector<string> header;
    header.emplace_back("UIO_Encoding");
    for (auto it = data.begin(); it != data.end(); it++){
        header.emplace_back("\""+it->first+"\"");
    }
    textdata.push_back(header);

    // coeffs
    for (int rowID = 0; rowID < encodings.size(); rowID++){
        vector<string> row;
        row.emplace_back("\""+encodings_str[rowID]+"\"");
        for (auto it = data.begin(); it != data.end(); it++){
            vector<int> coefs = it->second;
            row.emplace_back(to_string(coefs[rowID]));
        }
        textdata.push_back(row);
    }

    write_file("c_uio_data_n="+to_string(uio_size)+".csv", textdata);

}

// Example usage in main
int main() {

    int uio_size = 9;
     // (4, 5)","(3, 6)","(2, 7)","(1, 8)
    vector<vector<int>> partitions = {{8, 1}, {7, 2}, {6,3},{5,4}};
    //vector<vector<int>> partitions = {{3,1}};
    vector<int> part = {6,3};
    
    GlobalDataPreparer G = GlobalDataPreparer({uio_size}, {part});
    G.generateData();


    //generateAndSaveCoeffs(uio_size, partitions);


    return 0;
}