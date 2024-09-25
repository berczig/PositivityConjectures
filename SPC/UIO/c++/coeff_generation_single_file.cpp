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
#include <fstream>


using namespace std;


template <typename R, typename... A>
class CacheDecorator {
  public:
    CacheDecorator(std::function<R(A...)> f) : f_(f) {}

    R operator()(A... a) {
        std::tuple<A...> key(a...);
        auto search = map_.find(key);
        if (search != map_.end()) {
            return search->second;
        }

        auto result = f_(a...);
        map_[key] = result;
        return result;
    }

  private:
    std::function<R(A...)> f_;
    std::map<std::tuple<A...>, R> map_;
};


class UIO {
public:
    // Constants representing the comparison relationships
    static const int INCOMPARABLE;
    static const int LESS;
    static const int GREATER;
    static const int EQUAL;

    // Class members
    int N;  // Size of the encoding
    std::vector<int> encoding;  // The UIO encoding
    std::vector<std::vector<int>> comparison_matrix;  // Matrix of comparisons
    std::string repr;  // String representation of the UIO encoding

    // Constructor
    UIO(const std::vector<int>& uio_encoding) {
        N = uio_encoding.size();
        encoding = uio_encoding;
        repr = vectorToString(encoding);

        // Initialize the comparison matrix
        comparison_matrix.resize(N, std::vector<int>(N, EQUAL));

        // Fill the comparison matrix based on the encoding
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (uio_encoding[j] <= i) {
                    comparison_matrix[i][j] = INCOMPARABLE;
                    comparison_matrix[j][i] = INCOMPARABLE;
                } else {
                    comparison_matrix[i][j] = LESS;
                    comparison_matrix[j][i] = GREATER;
                }
            }
        }
    }

    // Utility function to convert a vector to a string
    std::string vectorToString(const std::vector<int>& vec) const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) oss << ", ";
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    }

    // isescher method - checks if the sequence is an Escher sequence
    bool isescher(const string& seq) const {
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            if (!isarrow(seq, i, i + 1)) {
                return false;
            }
        }
        return isarrow(seq, seq.size() - 1, 0);
    }

    // isarrow method - checks if there is an arrow between elements i and j
    bool isarrow(const string& escher, int i, int j, bool verbose = false) const {
        if (verbose) {
            std::cout << "arrow " << escher << " " << i << " " << j 
                      << " " << (comparison_matrix[escher[i]][escher[j]] != GREATER) << std::endl;
        }
        return comparison_matrix[escher[i]][escher[j]] != GREATER;  // EQUAL also intersects
    }

    // Overload the output operator to print the UIO object
    friend std::ostream& operator<<(std::ostream& os, const UIO& uio) {
        os << uio.repr;
        return os;
    }
};


const int UIO::INCOMPARABLE = 100;
const int UIO::LESS = 101;
const int UIO::GREATER = 102;
const int UIO::EQUAL = 103;


//CacheDecorator<vector<string>,int, int> cd(getKPermutationsOfN);
CacheDecorator<vector<string>,int, int, vector<int>> cd(getCyclicKPermutationsOfN);


class UIODataExtractor {

    // Assuming core_generator_class type, but no usage in the class was shown in the Python code

public:
    // Constructor
    const UIO& uio;
    UIODataExtractor(const UIO& uio) : uio(uio) {}

    // Count Eschers
    int countEschers(const std::vector<int>& partition) const {
        int count = 0;
        vector<string> P = cd(uio.N, sum(partition), partition);
        int escher_cycle_types = calculateNumberOfCycleTypes(partition);
        //cout << "part: " << endl;
        //printvec(partition);
        //cout << "escher_cycle_types: " << escher_cycle_types << endl << endl;


        //vector<string> interPermuteStrings(vector<string>& words){

        if (partition.size() == 1) {
            for (const auto& seq : P) {
                if (uio.isescher(seq)) {
                    count += 1;
                }
            }
        }

        else if (partition.size() == 2) {
            int a = partition[0];
            for (const auto& seq : P) {
                if (uio.isescher(string(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(string(seq.begin() + a, seq.end()))) {
                    count += 1;
                }
            }
        }

        else if (partition.size() == 3) {
            int a = partition[0], b = partition[1];
            for (const auto& seq : P) {
                if (uio.isescher(string(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(string(seq.begin() + a, seq.begin() + a + b)) &&
                    uio.isescher(string(seq.begin() + a + b, seq.end()))) {
                    count += 1;
                }
            }
        }

        else if (partition.size() == 4) {
            int a = partition[0], b = partition[1], c = partition[2];
            for (const auto& seq : P) {
                if (uio.isescher(string(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(string(seq.begin() + a, seq.begin() + a + b)) &&
                    uio.isescher(string(seq.begin() + a + b, seq.begin() + a + b + c)) &&
                    uio.isescher(string(seq.begin() + a + b + c, seq.end()))) {
                    count += 1;
                }
            }
        }
        return escher_cycle_types*count;
    }

    // Get Coefficient
    int getCoefficient(const std::vector<int>& partition) const {
        if (partition.size() == 1) {
            return countEschers(partition);
        }
        else if (partition.size() == 2) {
            int dc = countEschers(partition);
            int sc = countEschers({partition[0]+partition[1]});
            //cout << uio << ": " << dc << " " << sc << endl;
            return dc - sc;
        }
        else if (partition.size() == 3) {
            int n = partition[0], k = partition[1], l = partition[2];
            return 2 * countEschers({n + k + l}) +
                   countEschers(partition) -
                   countEschers({n + l, k}) -
                   countEschers({n + k, l}) -
                   countEschers({l + k, n});
        }
        else if (partition.size() == 4) {
            int a = partition[0], b = partition[1], c = partition[2], d = partition[3];
            return countEschers({a, b, c, d}) -
                   countEschers({a + b, c, d}) -
                   countEschers({a + c, b, d}) -
                   countEschers({a + d, b, c}) -
                   countEschers({b + c, a, d}) -
                   countEschers({b + d, a, c}) -
                   countEschers({c + d, a, b}) +
                   countEschers({a + b, c + d}) +
                   countEschers({a + c, b + d}) +
                   countEschers({a + d, b + c}) +
                   2 * countEschers({a + b + c, d}) +
                   2 * countEschers({a + b + d, c}) +
                   2 * countEschers({a + c + d, b}) +
                   2 * countEschers({b + c + d, a}) -
                   6 * countEschers({a + b + c + d});
        }
        return 0;  // Default case
    }

    // Overload the output operator to print the UIODataExtractor object
    friend std::ostream& operator<<(std::ostream& os, const UIODataExtractor& extractor) {
        os << "EXTRACTOR OF [" << vectorToString(extractor.uio.encoding) << "]";
        return os;
    }

private:
    // Helper function to sum the elements of a vector
    int sum(const std::vector<int>& vec) const {
        int total = 0;
        for (int v : vec) {
            total += v;
        }
        return total;
    }

    // Helper function to convert a vector to a string
    static std::string vectorToString(const std::vector<int>& vec) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) oss << ", ";
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    }
};

void write_file(string filepath, vector<vector<string>> data){
    std::ofstream myfile;
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

    // for (auto it = data.begin(); it != data.end(); it++){
    //     myfile << it->first;
    //     if (it != data.end())
    //         myfile << ",";
    //     else
    //         myfile << "\n";
    // }

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

void generateAndSaveCoeffs(int uio_size, vector<vector<int>> partitions){
    vector<vector<int>> encodings = generate_all_uio_encodings(uio_size);

    unordered_map<string, vector<int>> data;
    vector<string> encodings_str;

    for (vector<int>& partition : partitions){
        int coefsum = 0;
        vector<int> coeffs;
        auto start = std::chrono::system_clock::now();    

        for (vector<int> uio_encoding: encodings){
            
            UIO uio(uio_encoding);
            encodings_str.emplace_back(uio.repr);

            UIODataExtractor extractor(uio);  // Replace nullptr with core generator object if required
            
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
    generateAndSaveCoeffs(uio_size, partitions);


    return 0;
}