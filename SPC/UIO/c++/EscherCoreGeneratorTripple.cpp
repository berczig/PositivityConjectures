#include "EscherCoreGeneratorTripple.h"
#include "misc.h"
#include <tuple>
using namespace std;

EscherCoreGeneratorTripple::EscherCoreGeneratorTripple(shared_ptr<UIO>& uio, const vector<int>& partition)
    : EscherCoreGeneratorAbstract(uio, partition) {
        calculate_comp_indices(partition);
        }

char EscherCoreGeneratorTripple::compareTwoCoreElements(int a, int b) {
    return a < b ? UIO::LESS : UIO::GREATER;
}

vector<string> EscherCoreGeneratorTripple::getCoreLabels(const vector<int>& partition) {
    return {"0", "subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
                "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uv_w start", 
                "subescher uv_w end", "uv_w 1. insert", "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
                "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert"};
} 

map<string, vector<string>> EscherCoreGeneratorTripple::getCoreComparisions(const vector<int>& partition) {
    return {
        {"0", {"subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start"}},
        {"subescher uv end",  {"uv 1. insert","uw 1. insert","vw 1. insert"}},
        {"subescher uw end",  {"uv 1. insert","uw 1. insert","vw 1. insert"}},
        {"subescher vw end",  {"uv 1. insert","uw 1. insert","vw 1. insert"}}
        };
}

vector<int> EscherCoreGeneratorTripple::generateCore(const string& escher) {
    const string u = string(escher.begin(), escher.begin() + partition[0]);
    const string v = string(escher.begin()+ partition[0], escher.begin() + partition[0] + partition[1]);
    const string w = string(escher.begin() + partition[0] + partition[1], escher.begin() + partition[0] + partition[1] + partition[2]);
    // const string u = escher.substr(0, partition[0]);
    // const string v = escher.substr(partition[0], partition[1]);
    // const string w = escher.substr(partition[0]+partition[1], partition[2]);

    vector<int> core_u_v = get_shortb_insertion_and_subescher_of_2_eschers(u,v);
    vector<int> core_v_w = get_shortb_insertion_and_subescher_of_2_eschers(v,w);
    vector<int> core_u_w = get_shortb_insertion_and_subescher_of_2_eschers(u,w);
    vector<int> core_uw_v;
    vector<int> core_vw_u;
    vector<int> core_uv_w;

    if (core_u_v.back() == -1){
        core_uv_w = {-1,-1,-1};
    }else{
        const string uv = concat(u,v, core_u_v.back());
        core_uv_w = get_shortb_insertion_and_subescher_of_2_eschers(uv, w);
    }

    if (core_u_w.back() == -1){
        core_uw_v = {-1,-1,-1};
    }else{
        const string uw = concat(u,w, core_u_w.back());
        core_uw_v = get_shortb_insertion_and_subescher_of_2_eschers(uw, v);
    }

    if (core_v_w.back() == -1){
        core_vw_u = {-1,-1,-1};
    }else{
        const string vw = concat(v,w, core_v_w.back());
        core_vw_u = get_shortb_insertion_and_subescher_of_2_eschers(vw, u);
    }

    add_one_to_last_element(core_u_v);  
    add_one_to_last_element(core_v_w); 
    add_one_to_last_element(core_u_w); 
    add_one_to_last_element(core_uv_w);
    add_one_to_last_element(core_uw_v);
    add_one_to_last_element(core_vw_u);

    // merge
    //string core = core_u_v + core_v_w + core_u_w + core_uv_w + core_uw_v + core_vw_u;
    vector<int> core;
    core.reserve( core_u_v.size() + core_v_w.size()
        + core_u_w.size() + core_uv_w.size() 
        + core_uw_v.size() + core_vw_u.size());
    core.insert(core.end(), core_u_v.begin(), core_u_v.end());
    core.insert(core.end(), core_v_w.begin(), core_v_w.end());
    core.insert(core.end(), core_u_w.begin(), core_u_w.end());
    core.insert(core.end(), core_uv_w.begin(), core_uv_w.end());
    core.insert(core.end(), core_uw_v.begin(), core_uw_v.end());
    core.insert(core.end(), core_vw_u.begin(), core_vw_u.end());

    core.insert(core.begin(), 0);
    //cout << "core: " << partitionToString(core) << endl;
    return core;
}

