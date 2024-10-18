#include <string>
#include <iostream>
#include <vector>
#include "GlobalUIODataPreparer.h"
#include "UIODataExtractor.h"
#include "UIO.h"
#include "combinatorics.h"
#include "misc.h"
#include <memory>
#include <map>

using namespace std;

GlobalUIODataPreparer::GlobalUIODataPreparer(int uio_size, vector<int> partition) : partition(partition), uio_size(uio_size){
    // Generate UIOs and UIOExtractors
    encodings = generate_all_uio_encodings(uio_size);
    n_uios = encodings.size();
    cout << "create GlobalUIODataPreparer with uio_size = " << uio_size << " part = " << partitionToString(partition) << endl;
    for (int UIOID = 0; UIOID < n_uios; UIOID++){
        vector<int>& uio_encoding = encodings[UIOID];
        shared_ptr<UIO> uio = make_shared<UIO>(uio_encoding, UIOID);
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

void GlobalUIODataPreparer::calculateCoefficients(){
    cout << "generating coefficients for " << n_uios << " uios" << endl;
    for (int UIOID = 0; UIOID < n_uios; UIOID++){
        UIODataExtractor& extractor = extractors[UIOID];
        coeffs.push_back(extractor.getCoefficient(partition));
        if (UIOID % 10 == 0)
            printLoadingBar(UIOID, n_uios, "#");
    }
    printLoadingBar(n_uios, n_uios, "#");
}

void GlobalUIODataPreparer::countCoreRepresentations(){
    cout << "generating and counting CoreRepresentations for " << n_uios << " uios" << endl;
    int total_corereps = 0;
    for (int UIOID = 0; UIOID < n_uios; UIOID++){
        if (UIOID % 10 == 0)
            printLoadingBar(UIOID, n_uios, "#");

        UIODataExtractor& extractor = extractors[UIOID];
        //timeStop();
        vector<coreRepresentation> corereps = extractor.getCoreRepresentations(partition);
        //cout << "T:" << timeStop() << endl;
        total_corereps += corereps.size();

        for (coreRepresentation corerep : corereps){
            // count this observed category
            if (counter.find(corerep) == counter.end()){
                map<int, int> specific_corerep_global_uio_counter;
                specific_corerep_global_uio_counter[UIOID] = 1;
                counter[corerep] = specific_corerep_global_uio_counter;
            }else{
                map<int, int>& s_c_g_u_c = counter[corerep];
                if (s_c_g_u_c.find(UIOID) == s_c_g_u_c.end()){
                    s_c_g_u_c[UIOID] = 1;
                }else{
                    s_c_g_u_c[UIOID]++;
                }
            }
        }
    }
    printLoadingBar(n_uios, n_uios, "#");

    cout << "Found " << total_corereps << " total corereps" << endl;
    cout << "Found " << counter.size() << " distinct corereps" << endl;
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