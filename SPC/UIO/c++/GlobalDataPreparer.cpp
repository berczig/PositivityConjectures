#include "GlobalDataPreparer.h"
#include "GlobalUIODataPreparer.h"
#include "UIODataExtractor.h"
#include "misc.h"
#include <fstream>
#include "json.hpp"

// for convenience
using json = nlohmann::json;
using namespace std;

GlobalDataPreparer::GlobalDataPreparer(vector<int> uio_sizes, vector<vector<int>> partitions) : uio_sizes(uio_sizes), partitions(partitions){
    uiopreparers.reserve(uio_sizes.size());
    for (int i = 0; i < uio_sizes.size(); i++){
        cout << "IN size:" << uio_sizes[i] << endl;
        cout << "IN partition: " << partitionToString(partitions[i]) << endl;
        uiopreparers.emplace_back(uio_sizes[i], partitions[i]);
    }
}

void GlobalDataPreparer::generateData(){
    // generate coeffs
    for (GlobalUIODataPreparer& prep : uiopreparers){
        prep.calculateCoefficients(); 
        prep.countCoreRepresentations();
        for (const auto& item : prep.counter){
            coreRepresentation corerep = item.first;
            map<int, int> counts = item.second;
            //cout << "c: " << +corerep.comparisons[0] << " uioID: " << counts.begin()->first << " count: " << counts.begin()->second << endl;
            //cout << "counts: " << counts.size() << endl;
        }
    }
}

void GlobalDataPreparer::saveCSVData(string filepath){

    vector<string> header;
    vector<string> encodings_str;
    vector<vector<string>> textdata;

    // header
    header.emplace_back("UIO_Encoding");
    for (GlobalUIODataPreparer& preparer : uiopreparers){
        header.emplace_back("\""+ partitionToString(preparer.partition) +"\"");
    }
    textdata.push_back(header);

    // UIO encodings
    GlobalUIODataPreparer& preparer = uiopreparers[0];
    for (UIODataExtractor& ext : preparer.extractors){
        encodings_str.emplace_back(ext.uio->repr);
    }

    // rows with coeffs
    for (int rowID = 0; rowID < encodings_str.size(); rowID++){
        vector<string> row;
        row.emplace_back("\""+encodings_str[rowID]+"\"");
        for (GlobalUIODataPreparer& preparer : uiopreparers){
            row.emplace_back(to_string(preparer.coeffs[rowID]));
        }
        textdata.push_back(row);
    }

    write_file(filepath, textdata);
}

void GlobalDataPreparer::saveJSONData(string filepath){
    json jsonout;
    vector<json> GlobalUioDatas;
    cout << "save JSON file: " << filepath << endl;
    for (GlobalUIODataPreparer& prep : uiopreparers){
        json temp;
        temp["partition"] = prep.partition;
        temp["uio_size"] = prep.uio_size;
        temp["coefficients"] = prep.coeffs;

        cout << "part: " << partitionToString(prep.partition) << " uio_size: " << prep.uio_size << " coeffs: " << prep.coeffs.size() << " distinct corereps: " << prep.counter.size() << endl;

        //vector<json> corerep_json;
        map<vector<unsigned char>, map<int, int>> convert;
        for (const auto& item : prep.counter){
            convert[item.first.comparisons] = item.second;
        }

        temp["core_representations"] = convert;
        GlobalUioDatas.push_back(temp);
    }
    jsonout["data"] = GlobalUioDatas;
    
    // write prettified JSON to another file
    std::ofstream o(filepath);
    // o  << std::setw(4) << jsonout << std::endl;
    o  << jsonout << std::endl;
}








