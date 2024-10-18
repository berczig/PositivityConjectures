#include "EscherCoreGeneratorAbstract.h"
#include "misc.h"

EscherCoreGeneratorAbstract::EscherCoreGeneratorAbstract(shared_ptr<UIO>& uio, const vector<int>& partition)
    : CoreGenerator(uio, partition) {}

void EscherCoreGeneratorAbstract::add_one_to_last_element(vector<int>& list){
    for (int i = 0; i < list.size(); i++){list[i] = 2*list[i];}
    list[list.size()-1] = list.back()+1;
}

vector<int> EscherCoreGeneratorAbstract::get_shortb_insertion_and_subescher_of_2_eschers(const string& u, const string& v){
    int n = u.size();
    int k = v.size();
    vector<int> insertions;
    vector<int> escherstartpoints;
    tie(insertions, escherstartpoints) = getInsertionsSubeshers(u,v);

    if (insertions.size() == 0)
        return {-1, -1, -1};

    else{
        vector<int> points;
        points.reserve(3);
        bool has_bigger_than_0_startpoint = false;

        for (int escherstartpoint : escherstartpoints){
            if (escherstartpoint > 0){
                points.emplace_back(escherstartpoint);
                points.emplace_back(escherstartpoint+k-1);
                has_bigger_than_0_startpoint = true;
                break;
            }
        }

        if (has_bigger_than_0_startpoint == false){
            points.emplace_back(-1);
            points.emplace_back(insertions[0]+1);
        }

        points.emplace_back(insertions[0]);
        
        return points;
    }
}


tuple<vector<int>, vector<int>> EscherCoreGeneratorAbstract::getInsertionsSubeshers(const string& u, const string& v){
    int n = u.size();
    int k = v.size();
    int lcm_ = lcm(n,k);
    string uu = repeat(u, lcm_/n);

    vector<int> insertions = getInsertionPoints(u,v,lcm_);
    vector<int> subeschers;

    if (insertions.size() > 0){
        int insertion = insertions[0];
        const string extrav = cyclicslice(v, insertion+1, insertion+2);
        const string sub = uu.substr(0, insertion+1) + extrav;
        subeschers = getpseudosubescherstartingpoints(sub, k);
    }

    return make_tuple(insertions, subeschers);
}

vector<int> EscherCoreGeneratorAbstract::getInsertionPoints(const string& u, const string& v, int lcm_ = 0){
    vector<int> points;
    int n = u.size();
    int k = v.size();
    if (lcm_ == 0)
        lcm_ = lcm(n,k);
    for (int i = 0; i < lcm_; i++){
        if (uio->isarrow(u,v,mod(i,n), mod(i+1, k)) && uio->isarrow(v,u,mod(i,k), mod(i+1, n)))
            points.emplace_back(i);
    }
    return points;
}

vector<int> EscherCoreGeneratorAbstract::getpseudosubescherstartingpoints(const string& escher, int k){
    vector<int> subeschersstartingpoint;
    int h = escher.size();

    for (int m = -1; m < h-1; m++){
        bool cond1 = uio->isarrow(escher,   mod(m+k,h),    mod(m+1,h));
        bool cond2 = uio->isarrow(escher,   mod(m,h),        mod(m+k+1,h));
        if (cond1 && cond2)
            subeschersstartingpoint.push_back(m+1);
    }
    
    return subeschersstartingpoint;
}

string EscherCoreGeneratorAbstract::concat(const string& first_escher, const string& second_escher, int insertionpoint){
    int f = first_escher.size();
    int k = second_escher.size();
    return string(first_escher.begin(), first_escher.begin() + mod(insertionpoint, f) + 1)
    + cyclicslice(second_escher, insertionpoint+1, insertionpoint+k+1)
    + string(first_escher.begin() + mod(insertionpoint, f) + 1, first_escher.end());
}

