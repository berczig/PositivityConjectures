#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;

void permuteStrings_(vector<string>& newwords, string current, vector<string>& words, int index){
    if (index == words.size()){
        newwords.push_back(current);
    }else{
        string word = words[index];
        permuteStrings_(newwords, current+word, words, index+1);
        for (int i = 1; i < word.size(); i++){
            rotate(word.begin(), word.begin() + 1, word.end());
            permuteStrings_(newwords, current+word, words, index+1);
        }
    }
}

/**
 * generates all possible permutations with base-words in words
 */
vector<string> permuteStrings(vector<string>& words){
    vector<string> newwords = {};
    permuteStrings_(newwords, "", words, 0);
    return newwords;
}

/**
 * generates all possible permutations with base-words in words and switch the order of the words. "hello" -> "lohel"
 */
vector<string> interPermuteStrings(vector<string>& words){
    sort(words.begin(), words.end());
    vector<string> totalnewwords = vector<string>();
    do{
        vector<string> newwords = permuteStrings(words);
        totalnewwords.insert(totalnewwords.end(), newwords.begin(), newwords.end());

    } while (next_permutation(words.begin(), words.end()));
    //sort(totalnewwords.begin(), totalnewwords.end());
    return totalnewwords;
}


// int main() {
//     vector<string> IN = {"0123", "456", "78"};
//     vector<string> result = interPermuteStrings(IN);
   
//     // Output the result
//     // for (const string& s : result) {
//     //     cout << s << endl;
//     // }

//     string s = "";
//     vector<string> a;
//     s.push_back(2);
//     vector<int> ints = {10,20,30};
//     cout << "hello" << ints[s[0]];

//     return 0;
// }
