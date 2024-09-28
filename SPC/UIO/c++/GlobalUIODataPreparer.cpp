#include <string>
#include <iostream>
#include <vector>
#include "GlobalUIODataPreparer.h"
#include "UIODataExtractor.h"
#include "UIO.h"
#include "combinatorics.h"
#include "misc.h"
#include <memory>

using namespace std;

GlobalUIODataPreparer::GlobalUIODataPreparer(int uio_size, vector<int> partition) : partition(partition), uio_size(uio_size){
    // Generate UIOs and UIOExtractors
    encodings = generate_all_uio_encodings(uio_size);
    n_uios = encodings.size();

    for (int UIOID = 0; UIOID < n_uios; UIOID++){
        vector<int> uio_encoding = encodings[UIOID];
        shared_ptr<UIO> uio = make_shared<UIO>(uio_encoding);
        UIODataExtractor extr = UIODataExtractor(uio);
        //encodings_str.emplace_back(uio.repr);
        extractors.push_back(extr);
    }
}

// Copy Constructor (only definition here)
GlobalUIODataPreparer::GlobalUIODataPreparer(const GlobalUIODataPreparer& other)
    : partition(other.partition), uio_size(other.uio_size), encodings(other.encodings),
      extractors(other.extractors), coeffs(other.coeffs), n_uios(other.n_uios) {
    cout << "copy" << endl;
}

vector<int> GlobalUIODataPreparer::calculateCoefficients(){
    cout << "generating coefficients for " << n_uios << " uios" << endl;
    for (int UIOID = 0; UIOID < n_uios; UIOID++){
        UIODataExtractor& extractor = extractors[UIOID];
        coeffs.push_back(extractor.getCoefficient(partition));
        if (UIOID % 10 == 0)
            printLoadingBar(UIOID+1, n_uios, "#");
    }
}


    // unordered_map<string, vector<int>> data;
    // vector<string> encodings_str;

    // for (vector<int>& partition : partitions){
    //     int coefsum = 0;
    //     vector<int> coeffs;
    //     auto start = std::chrono::system_clock::now();    
    //     SequenceGenerator sg = SequenceGenerator();

    //     for (vector<int> uio_encoding: encodings){
            
    //         UIO uio(uio_encoding);
    //         encodings_str.emplace_back(uio.repr);

    //         UIODataExtractor extractor(uio, sg);  // Replace nullptr with core generator object if required
            
    //         int coef = extractor.getCoefficient(partition);
    //         coeffs.push_back(coef);
    //         coefsum += coef;
    //         //std::cout << extractor.uio << " coef: " << coef << std::endl;

    //     }
    //     data[partitionToString(partition)] = coeffs;
    //     cout << "coef sum: " << coefsum << endl;

    //     auto end = std::chrono::system_clock::now();
    //     std::chrono::duration<double> elapsed_seconds = end-start;
    //     cout << elapsed_seconds.count() << endl;
    // }