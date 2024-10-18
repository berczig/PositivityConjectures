#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <chrono>
#include <fstream>
#include <cmath>
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
#include "EscherCoreGeneratorBasic.h"
#include <windows.h>

using namespace std;






void generateAndSaveCoeffs(int uio_size, vector<vector<int>> partitions){
    vector<vector<int>> encodings = generate_all_uio_encodings(uio_size);

    unordered_map<string, vector<int>> data;
    vector<string> encodings_str;

    for (vector<int>& partition : partitions){
        int coefsum = 0;
        vector<int> coeffs;
        auto start = std::chrono::system_clock::now();    

        for (int UIOID = 0; UIOID < encodings.size(); UIOID++){
            vector<int> uio_encoding = encodings[UIOID];
            shared_ptr<UIO> uio = make_shared<UIO>(uio_encoding, UIOID);
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

void generateSingleData(int uio_size, vector<int> partition){
    vector<int> uio_sizes = {uio_size};
    vector<vector<int>> partitions = {partition};
    GlobalDataPreparer G = GlobalDataPreparer(uio_sizes,partitions);
    G.generateData();

    // Save
    //G.saveJSONData(getBasePath()+"/SPC/Saves,Tests/C_Trainingdata/cpp_data.json");
}

void generateAllData(){
    const int MAX_UIO_SIZE = 9;
    const int MAX_PART_SUM = 9;
    const int MAX_PART_LENGTH = 4;

    for (int part_sum = 4; part_sum <= MAX_PART_SUM; part_sum++){
        cout << "part_sum: " << part_sum << endl;
        
        // Generate all partition of n=part_sum
        vector<vector<int>> partitions = getPartitionUpToN(part_sum, 2, MAX_PART_LENGTH);

        // For each partition match it with multiple uio sizes
        vector<vector<int>> partitions_matched;
        partitions_matched.reserve((MAX_UIO_SIZE - part_sum + 1) * partitions.size());
        vector<int> uio_sizes;
        uio_sizes.reserve(partitions_matched.size());

        for (int uio_size = part_sum; uio_size <= MAX_UIO_SIZE; uio_size++){
            for (vector<int>& partition : partitions){
                partitions_matched.push_back(partition);
                uio_sizes.push_back(uio_size);
            }
        }

        // Print
        cout << "uio sizes: " << partitionToString(uio_sizes) << endl;
        cout << "partitions: " << endl;
        for (vector<int>& part : partitions_matched){
            cout << " - " << partitionToString(part) << endl;
        }

        // Generate Data 
        GlobalDataPreparer G = GlobalDataPreparer(uio_sizes,partitions_matched);
        G.generateData();

        // Save
        G.saveJSONData(getBasePath()+"/SPC/Saves,Tests/C_Trainingdata/cpp_data_all_partitions_of_"+to_string(part_sum)+".json");

        cout << endl;
    }
}

// Example usage in main
int main() {
    auto start = std::chrono::system_clock::now();

    //generateAllData();
    generateSingleData(6, {2,2,2});

    // vector<int> uio_sizes = {9};
    // vector<vector<int>> partitions = {{4,3,2}};

    // vector<int> partition = {4,2};
    // vector<int> encoding = {0,0,0,0,0,0};
    // shared_ptr<UIO> uio = make_shared<UIO>(encoding);
    // EscherCoreGeneratorBasic x = EscherCoreGeneratorBasic(uio, partition);


    // Check UIO
    // vector<int> encod = {0,0,0,0,0,0,0,0,0,0};
    // vector<int> part = {5,5};
    // shared_ptr<UIO> uio = make_shared<UIO>(encod);
    // UIODataExtractor ext = UIODataExtractor(uio);
    // cout << "coef: " << ext.getCoefficient(part) << endl;



    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    cout << "Elapsed Time: " << elapsed_seconds.count() << " seconds" << endl;
    return 0;
}