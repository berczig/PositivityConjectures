#include "CoreGenerator.h"
#include <algorithm>
#include <stdexcept>

CoreGenerator::CoreGenerator(shared_ptr<UIO>& uio, const vector<int>& partition)
    : uio(uio), partition(partition) {}

void CoreGenerator::calculate_comp_indices(const vector<int>& partition) {
    if (comp_indices.empty()) {
        comp_indices.clear();
        vector<string> labels = getCoreLabels(partition);
        map<string, vector<string>> comp = getCoreComparisions(partition);
        for (int first_index = 0; first_index < labels.size(); ++first_index) {
            const string& first_label = labels[first_index];
            if (comp.find(first_label) != comp.end()) {
                for (const string& second_label : comp[first_label]) {
                    int second_index = distance(labels.begin(), find(labels.begin(), labels.end(), second_label));
                    comp_indices.emplace_back(first_index, second_index);
                }
            }
        }
        // cout << "Reduced size of core representations.\n";
    }
}

vector<unsigned char> CoreGenerator::getCoreRepresentation(const core& core) {
    vector<unsigned char> rep;
    for (auto const& tup : comp_indices) {
        int f_index, s_index;
        tie(f_index,s_index) = tup;
        rep.push_back(compareTwoCoreElements(core.values[f_index], core.values[s_index]));
    }
    return rep;
}

int CoreGenerator::getCoreRepresentationLength(const vector<int>& partition) {
    vector<string> labels = getCoreLabels(partition);
    map<string, vector<string>> comp = getCoreComparisions(partition);
    int length = 0;
    for (const string& label : labels) {
        if (comp.find(label) != comp.end()) {
            length += comp[label].size();
        }
    }
    return length;
}

int CoreGenerator::getCoreLength(const vector<int>& partition) {
    return getCoreLabels(partition).size();
}

vector<tuple<string, string>> CoreGenerator::getOrderedCoreComparisions(const vector<int>& partition) {
    vector<tuple<string, string>> order;
    map<string, vector<string>> comps = getCoreComparisions(partition);
    for (const string& first_label : getCoreLabels(partition)) {
        if (comps.find(first_label) != comps.end()) {
            for (const string& second_label : comps[first_label]) {
                order.emplace_back(first_label, second_label);
            }
        }
    }
    return order;
}


// Implement other static functions...
