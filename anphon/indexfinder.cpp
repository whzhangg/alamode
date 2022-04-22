
#pragma once

#include <map>
#include <vector>
#include "kpoint.h"
#include "LBTE_equation.h"

namespace PHON_NS {

class KwithIndex{
public:
    int k2, k3, index;
    KwithIndex(int k2_in, int k3_in, int index_in):
        k2(k2_in), k3(k3_in), index(index_in) {};
};

// provide an interface
class IndexFinder{
// it will return zero if the pair does not exist
// return the given index if it does
public:
    IndexFinder(int nkin): nkfull(nkin) {
        absorb_counter = 0;
        emitt_counter = 0;
        mapper_absorb.clear();
        mapper_emitt.clear();
    };

    ~IndexFinder();

    KwithIndex find_index_absorb(int, int);
    KwithIndex find_index_emitt(int, int);

    void add_triplets_absorb(int, std::vector<KsListGroup> &);
    void add_triplets_emitt(int, std::vector<KsListGroup> &);

private:
    std::vector<std::map<int, KwithIndex>> mapper_absorb;
    std::vector<std::map<int, KwithIndex>> mapper_emitt;
    int nkfull;
    int absorb_counter;
    int emitt_counter;
};

KwithIndex IndexFinder::find_index_absorb(int ik_in, int k2)
{
    
}

void IndexFinder::add_triplets_absorb(int ik_in, std::vector<KsListGroup> &triplet)
{
    std::map<int, KwithIndex> map_for_ik;
    map_for_ik.clear();
    int k2, k3;
    for (auto i = 0; i < triplet.size(); ++i){

        auto pair = triplet[i];
        for (auto j = 0; j < pair.group.size(); ++j){
            k2 = pair.group[j].ks[0];
            k3 = pair.group[j].ks[1];
            map_for_ik[k2] = KwithIndex(k2, k3, absorb_counter);
        }
        absorb_counter += 1;
    }
    mapper_absorb.push_back(map_for_ik);
}

void IndexFinder::add_triplets_emitt(int ik_in, std::vector<KsListGroup> &triplet)
{
    std::map<int, KwithIndex> map_for_ik;
    map_for_ik.clear();
    int k2, k3;
    for (auto i = 0; i < triplet.size(); ++i){

        auto pair = triplet[i];
        for (auto j = 0; j < pair.group.size(); ++j){
            k2 = pair.group[j].ks[0];
            k3 = pair.group[j].ks[1];
            map_for_ik[k2] = KwithIndex(k2, k3, emitt_counter);
        }
        emitt_counter += 1;
    }
    mapper_emitt.push_back(map_for_ik);
}

} // name space