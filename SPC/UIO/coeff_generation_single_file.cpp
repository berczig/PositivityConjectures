#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>


using namespace std;



void generateUniquePermutations(int N, int k, std::vector<int>& current, int start, std::vector<std::vector<int>>& result) {
    // Base case: if the current permutation has k elements, add it to the result
    if (current.size() == k) {
        result.push_back(current);
        return;
    }

    // Try every number from 'start' to N-1
    for (int i = start; i < N; ++i) {
        current.push_back(i);  // Add the number to the current permutation
        generateUniquePermutations(N, k, current, i + 1, result);  // Recur with next number starting from i+1
        current.pop_back();  // Backtrack: remove the last number
    }
}

std::vector<std::vector<int>> getKPermutationsOfN(int N, int k) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;

    cout << "perm\n";

    // Start the backtracking recursion with the start index of 0
    generateUniquePermutations(N, k, current, 0, result);

    
    for (int i = 0; i < result.size(); i++){
        cout << result[i][0] << " " << result[i][1] << "\n";
    }
    
    return result;
}




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
                    cout << "here\n";
                    comparison_matrix[i][j] = INCOMPARABLE;
                    comparison_matrix[j][i] = INCOMPARABLE;
                } else {
                    cout << "actually here\n";
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
    bool isescher(const std::vector<int>& seq) const {
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            if (!isarrow(seq, i, i + 1)) {
                return false;
            }
        }
        return isarrow(seq, seq.size() - 1, 0);
    }

    // isarrow method - checks if there is an arrow between elements i and j
    bool isarrow(const std::vector<int>& escher, int i, int j, bool verbose = false) const {
        if (verbose) {
            std::cout << "arrow " << vectorToString(escher) << " " << i << " " << j 
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




class UIODataExtractor {
private:
    const UIO& uio;
    // Assuming core_generator_class type, but no usage in the class was shown in the Python code

public:
    // Constructor
    UIODataExtractor(const UIO& uio) : uio(uio) {}

    // Count Eschers
    int countEschers(const std::vector<int>& partition) const {
        int count = 0;
        std::vector<std::vector<int>> P = getKPermutationsOfN(uio.N, sum(partition));

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
                if (uio.isescher(std::vector<int>(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a, seq.end()))) {
                    count += 1;
                }
            }
        }
        else if (partition.size() == 3) {
            int a = partition[0], b = partition[1];
            for (const auto& seq : P) {
                if (uio.isescher(std::vector<int>(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a, seq.begin() + a + b)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a + b, seq.end()))) {
                    count += 1;
                }
            }
        }
        else if (partition.size() == 4) {
            int a = partition[0], b = partition[1], c = partition[2];
            for (const auto& seq : P) {
                if (uio.isescher(std::vector<int>(seq.begin(), seq.begin() + a)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a, seq.begin() + a + b)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a + b, seq.begin() + a + b + c)) &&
                    uio.isescher(std::vector<int>(seq.begin() + a + b + c, seq.end()))) {
                    count += 1;
                }
            }
        }
        return count;
    }

    // Get Coefficient
    int getCoefficient(const std::vector<int>& partition) const {
        if (partition.size() == 1) {
            return countEschers(partition);
        }
        else if (partition.size() == 2) {
            return countEschers(partition) - countEschers({uio.N});
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

// Example usage in main
int main() {
    std::vector<int> encoding = {0, 0, 2,2};  // Sample encoding
    UIO uio(encoding);

    UIODataExtractor extractor(uio);  // Replace nullptr with core generator object if required

    std::vector<int> partition = {2,2};
    std::cout << "Count of Eschers: " << extractor.countEschers(partition) << std::endl;

    std::cout << extractor << std::endl;

    return 0;
}