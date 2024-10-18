#include "EscherCoreGeneratorQuadruple.h"
#include "misc.h"
#include <tuple>
using namespace std;

EscherCoreGeneratorQuadruple::EscherCoreGeneratorQuadruple(shared_ptr<UIO>& uio, const vector<int>& partition)
    : EscherCoreGeneratorAbstract(uio, partition) {
        calculate_comp_indices(partition);
        }

char EscherCoreGeneratorQuadruple::compareTwoCoreElements(int a, int b) {
    return a < b ? UIO::LESS : UIO::GREATER;
}

vector<string> EscherCoreGeneratorQuadruple::getCoreLabels(const vector<int>& partition) {
    return {"0", "subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
                "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uz start", "subescher uz end", "uz 1. insert",
                "subescher vz start", "subescher vz end", "vz 1. insert", "subescher wz start", "subescher wz end", "wz 1. insert",
                "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert", 
                "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
                "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert", 
                "subescher uv_z start", "subescher uv_z end", "uv_z 1. insert",
                "subescher uz_v start", "subescher uz_v end", "uz_v 1. insert",
                "subescher vz_u start", "subescher vz_u end", "vz_u 1. insert",
                "subescher uw_z start", "subescher uw_z end", "uw_z 1. insert", 
                "subescher uz_w start", "subescher uz_w end", "uz_w 1. insert",
                "subescher wz_u start", "subescher wz_u end", "wz_u 1. insert",
                "subescher vw_z start", "subescher vw_z end", "vw_z 1. insert",
                "subescher vz_w start", "subescher vz_w end", "vz_w 1. insert",
                "subescher wz_v start", "subescher wz_v end", "wz_v 1. insert",
                "subescher uvw_z start", "subescher uvw_z end", "uvw_z 1. insert", 
                "subescher uvz_w start", "subescher uvz_w end", "uvz_w 1. insert",
                "subescher uwz_v start", "subescher uwz_v end", "uwz_v 1. insert", 
                "subescher vwz_u start", "subescher vwz_u end", "vwz_u 1. insert",
                "subescher uv_wz start", "subescher uv_wz end", "uv_wz 1. insert", 
                "subescher uw_vz start", "subescher uw_vz end", "uw_vz 1. insert",
                "subescher vw_uz start", "subescher vw_uz end", "vw_uz 1. insert"};
} 

map<string, vector<string>> EscherCoreGeneratorQuadruple::getCoreComparisions(const vector<int>& partition) {

    return {
            {"0", {"subescher uv start", "subescher uw start", "subescher vw start", "subescher uv_w start", "subescher uw_v start", "subescher vw_u start", 
                   "subescher uz start", "subescher vz start", "subescher wz start", 
                   "subescher uv_w start", "subescher uw_v start", "subescher vw_u start",
                   "subescher uv_z start", "subescher uz_v start", "subescher vz_u start", 
                    "subescher uw_z start", "subescher uz_w start", "subescher wz_u start", 
                    "subescher vw_z start", "subescher vz_w start", "subescher wz_v start",
                    "subescher uvw_z start", "subescher uvz_w start", "subescher uwz_v start", "subescher vwz_u start",
                    "subescher uv_wz start", "subescher uw_vz start", "subescher vw_uz start"}},
            {"subescher uv end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}},
            {"subescher uw end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}},
            {"subescher vw end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}},
            {"subescher uz end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}},
            {"subescher vz end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}},
            {"subescher wz end",  {"uv 1. insert","uw 1. insert","vw 1. insert", "uz 1. insert", "vz 1. insert", "wz 1. insert"}}
        };
}

vector<int> EscherCoreGeneratorQuadruple::generateCore(const string& escher) {
    const string u = string(escher.begin(), escher.begin() + partition[0]);
    const string v = string(escher.begin()+ partition[0], escher.begin() + partition[0] + partition[1]);
    const string w = string(escher.begin() + partition[0] + partition[1], escher.begin() + partition[0] + partition[1] + partition[2]);
    const string z = string(escher.begin() + partition[0] + partition[1] + partition[2], escher.begin() + partition[0] + partition[1] + partition[2] + partition[3]);

    // 6 - get data of double pairs
    vector<int> core_u_v = get_shortb_insertion_and_subescher_of_2_eschers(u,v);
    vector<int> core_v_w = get_shortb_insertion_and_subescher_of_2_eschers(v,w);
    vector<int> core_u_w = get_shortb_insertion_and_subescher_of_2_eschers(u,w);
    vector<int> core_u_z = get_shortb_insertion_and_subescher_of_2_eschers(u,z);
    vector<int> core_v_z = get_shortb_insertion_and_subescher_of_2_eschers(v,z);
    vector<int> core_w_z = get_shortb_insertion_and_subescher_of_2_eschers(w,z);

    // 9
    vector<int> core_uv_w;
    vector<int> core_uv_z;
    vector<int> core_uv_wz;

    vector<int> core_uw_v;
    vector<int> core_uw_z;
    vector<int> core_uw_vz;

    vector<int> core_vw_u;
    vector<int> core_vw_z;
    vector<int> core_vw_uz;

    // 6
    vector<int> core_uz_w;
    vector<int> core_uz_v;

    vector<int> core_vz_w;
    vector<int> core_vz_u;

    vector<int> core_wz_u;
    vector<int> core_wz_v;

    // 4
    vector<int> core_uvw_z;
    vector<int> core_uvz_w;
    vector<int> core_uwz_v;
    vector<int> core_vwz_u;

    string uv; 
    string uw; 
    string uz;
    string vw;
    string vz; 
    string wz; 



    // Calculate the core of ab_cd

    if (core_u_v.back() == -1){
        core_uv_w = {-1,-1,-1};
        core_uv_z = {-1,-1,-1};
        core_uv_wz = {-1,-1,-1};
    }else{
        uv = concat(u,v, core_u_v.back());
        core_uv_w = get_shortb_insertion_and_subescher_of_2_eschers(uv, w);
        core_uv_z = get_shortb_insertion_and_subescher_of_2_eschers(uv, z);
        if (core_w_z.back() != -1){
            wz = concat(w,z, core_w_z.back());
            core_uv_wz = get_shortb_insertion_and_subescher_of_2_eschers(uv, wz);
        }else{
            core_uv_wz = {-1,-1,-1};
        }
    }

    if (core_u_w.back() == -1){
        core_uw_v = {-1,-1,-1};
        core_uw_z = {-1,-1,-1};
        core_uw_vz = {-1,-1,-1};
    }else{
        uw = concat(u,w, core_u_w.back());
        core_uw_v = get_shortb_insertion_and_subescher_of_2_eschers(uw, v);
        core_uw_z = get_shortb_insertion_and_subescher_of_2_eschers(uw, z);
        if (core_v_z.back() != -1){
            vz = concat(v,z, core_v_z.back());
            core_uw_vz = get_shortb_insertion_and_subescher_of_2_eschers(uw, vz);
        }else{
            core_uw_vz = {-1,-1,-1};
        }
    }



    if (core_v_w.back() == -1){
        core_vw_u = {-1,-1,-1};
        core_vw_z = {-1,-1,-1};
        core_vw_uz = {-1,-1,-1};
    }else{
        vw = concat(v,w, core_v_w.back());
        core_vw_u = get_shortb_insertion_and_subescher_of_2_eschers(vw, u);
        core_vw_z = get_shortb_insertion_and_subescher_of_2_eschers(vw, z);
        if (core_u_z.back() != -1){
            uz = concat(u,z, core_u_z.back());
            core_vw_uz = get_shortb_insertion_and_subescher_of_2_eschers(vw, uz);
        }else{
            core_vw_uz = {-1,-1,-1};
        }
    }

    // Calculate the core of a triple pair

    if (core_u_z.back() == -1){
        core_uz_w = {-1,-1,-1};
        core_uz_v = {-1,-1,-1};
    }else{
        uz = concat(u,z, core_u_z.back());
        core_uz_w = get_shortb_insertion_and_subescher_of_2_eschers(uz, w);
        core_uz_v = get_shortb_insertion_and_subescher_of_2_eschers(uz, v);
    }


    if (core_v_z.back() == -1){
        core_vz_w = {-1,-1,-1};
        core_vz_u = {-1,-1,-1};
    }else{
        vz = concat(v,z, core_v_z.back());
        core_vz_w = get_shortb_insertion_and_subescher_of_2_eschers(vz, w);
        core_vz_u = get_shortb_insertion_and_subescher_of_2_eschers(vz, u);
    }


    if (core_w_z.back() == -1){
        core_wz_u = {-1,-1,-1};
        core_wz_v = {-1,-1,-1};
    }else{
        wz = concat(w,z, core_w_z.back());
        core_wz_u = get_shortb_insertion_and_subescher_of_2_eschers(wz, u);
        core_wz_v = get_shortb_insertion_and_subescher_of_2_eschers(wz, v);
    }


    // Calculate the core of a quadruple pair of type abc_d

    if (core_uv_w.back() == -1 && core_uw_v.back() == -1 && core_vw_u.back() == -1 ){
        core_uvw_z = {-1,-1,-1};
    }else{
        if (core_uv_w.back() != -1){
            const string uvw = concat(uv, w, core_uv_w.back());
            core_uvw_z = get_shortb_insertion_and_subescher_of_2_eschers(uvw, z);
        }else if (core_uw_v.back() != -1){
            const string uwv = concat(uw, v, core_uw_v.back());
            core_uvw_z = get_shortb_insertion_and_subescher_of_2_eschers(uwv, z);
        }else if (core_vw_u.back() != -1){
            const string vwu = concat(vw, u, core_vw_u.back());
            core_uvw_z = get_shortb_insertion_and_subescher_of_2_eschers(vwu, z);
        }
    }

    if (core_uv_z.back() == -1 && core_uz_v.back() == -1 && core_vz_u.back() == -1 ){
        core_uvz_w = {-1,-1,-1};
    }else{
        if (core_uv_z.back() != -1){
            const string uvz = concat(uv, z, core_uv_z.back());
            core_uvz_w = get_shortb_insertion_and_subescher_of_2_eschers(uvz, w);
        }else if (core_uz_v.back() != -1){
            const string uzv = concat(uz, v, core_uz_v.back());
            core_uvz_w = get_shortb_insertion_and_subescher_of_2_eschers(uzv, w);
        }else if (core_vz_u.back() != -1){
            const string vzu = concat(vz, u, core_vz_u.back());
            core_uvz_w = get_shortb_insertion_and_subescher_of_2_eschers(vzu, w);
        }
    }

    if (core_uw_z.back() == -1 && core_uz_w.back() == -1 && core_wz_u.back() == -1 ){
        core_uwz_v = {-1,-1,-1};
    }else{
        if (core_uw_z.back() != -1){
            const string uwz = concat(uw, z, core_uw_z.back());
            core_uwz_v = get_shortb_insertion_and_subescher_of_2_eschers(uwz, v);
        }else if (core_uz_w.back() != -1){
            const string uzw = concat(uz, w, core_uz_w.back());
            core_uwz_v = get_shortb_insertion_and_subescher_of_2_eschers(uzw, v);
        }else if (core_wz_u.back() != -1){
            const string wzu = concat(wz, u, core_wz_u.back());
            core_uwz_v = get_shortb_insertion_and_subescher_of_2_eschers(wzu, v);
        }
    }

    if (core_vw_z.back() == -1 && core_vz_w.back() == -1 && core_wz_v.back() == -1 ){
        core_vwz_u = {-1,-1,-1};
    }else{
        if (core_vw_z.back() != -1){
            const string vwz = concat(vw, z, core_vw_z.back());
            core_vwz_u = get_shortb_insertion_and_subescher_of_2_eschers(vwz, u);
        }else if (core_vz_w.back() != -1){
            const string vzw = concat(vz, w, core_vz_w.back());
            core_vwz_u = get_shortb_insertion_and_subescher_of_2_eschers(vzw, u);
        }else if (core_wz_v.back() != -1){
            const string wzv = concat(wz, v, core_wz_v.back());
            core_vwz_u = get_shortb_insertion_and_subescher_of_2_eschers(wzv, u);
        }
    }


    add_one_to_last_element(core_u_v); 
    add_one_to_last_element(core_v_w); 
    add_one_to_last_element(core_u_w); 
    add_one_to_last_element(core_u_z); 
    add_one_to_last_element(core_v_z); 
    add_one_to_last_element(core_w_z); 
    add_one_to_last_element(core_uv_w); 
    add_one_to_last_element(core_uv_z); 
    add_one_to_last_element(core_uv_wz); 
    add_one_to_last_element(core_uw_v); 
    add_one_to_last_element(core_uw_z); 
    add_one_to_last_element(core_uw_vz); 
    add_one_to_last_element(core_vw_u); 
    add_one_to_last_element(core_vw_z); 
    add_one_to_last_element(core_vw_uz); 
    add_one_to_last_element(core_uz_w); 
    add_one_to_last_element(core_uz_v); 
    add_one_to_last_element(core_vz_w); 
    add_one_to_last_element(core_vz_u); 
    add_one_to_last_element(core_wz_u); 
    add_one_to_last_element(core_wz_v); 
    add_one_to_last_element(core_uvw_z); 
    add_one_to_last_element(core_uvz_w); 
    add_one_to_last_element(core_uwz_v); 
    add_one_to_last_element(core_vwz_u);


    // merge
    vector<int> core;
    core.reserve( core_u_v.size() + core_v_w.size() + core_u_w.size() + core_u_z.size() + 
    core_v_z.size() + core_w_z.size() + core_uv_w.size() + core_uv_z.size() + 
    core_uv_wz.size() + core_uw_v.size() + core_uw_z.size() + core_uw_vz.size() + 
    core_vw_u.size() + core_vw_z.size() + core_vw_uz.size() + core_uz_w.size() + 
    core_uz_v.size() + core_vz_w.size() + core_vz_u.size() + core_wz_u.size() + 
    core_wz_v.size() + core_uvw_z.size() + core_uvz_w.size() + core_uwz_v.size() + core_vwz_u.size());

    core.insert(core.end(), core_u_v.begin(), core_u_v.end()); 
    core.insert(core.end(), core_v_w.begin(), core_v_w.end()); 
    core.insert(core.end(), core_u_w.begin(), core_u_w.end()); 
    core.insert(core.end(), core_u_z.begin(), core_u_z.end()); 
    core.insert(core.end(), core_v_z.begin(), core_v_z.end()); 
    core.insert(core.end(), core_w_z.begin(), core_w_z.end()); 

    core.insert(core.end(), core_uv_w.begin(), core_uv_w.end()); 
    core.insert(core.end(), core_uw_v.begin(), core_uw_v.end());  
    core.insert(core.end(), core_vw_u.begin(), core_vw_u.end());

    core.insert(core.end(), core_vw_z.begin(), core_vw_z.end());
    core.insert(core.end(), core_uv_z.begin(), core_uv_z.end());
    core.insert(core.end(), core_uw_z.begin(), core_uw_z.end()); 

    core.insert(core.end(), core_uz_w.begin(), core_uz_w.end()); 
    core.insert(core.end(), core_uz_v.begin(), core_uz_v.end()); 
    core.insert(core.end(), core_vz_w.begin(), core_vz_w.end()); 

    core.insert(core.end(), core_vz_u.begin(), core_vz_u.end()); 
    core.insert(core.end(), core_wz_u.begin(), core_wz_u.end()); 

    core.insert(core.end(), core_wz_v.begin(), core_wz_v.end()); 
    core.insert(core.end(), core_uv_wz.begin(), core_uv_wz.end()); 

    core.insert(core.end(), core_uw_vz.begin(), core_uw_vz.end());   
    core.insert(core.end(), core_vw_uz.begin(), core_vw_uz.end()); 

    core.insert(core.end(), core_uvw_z.begin(), core_uvw_z.end()); 
    core.insert(core.end(), core_uvz_w.begin(), core_uvz_w.end()); 
    core.insert(core.end(), core_uwz_v.begin(), core_uwz_v.end()); 
    core.insert(core.end(), core_vwz_u.begin(), core_vwz_u.end()); 

    core.insert(core.begin(), 0);

    return core;
}

