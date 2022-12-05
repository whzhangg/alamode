
#pragma once

#include <map>
#include <vector>
#include "kpoint.h"
#include "LBTE_equation.h"

namespace PHON_NS {

// provide an interface
class IndexFinder{
// it will return zero if the pair does not exist
// return the given index if it does
public:
    IndexFinder() {
        absorb_counter = 0;
        emitt_counter = 0;
        mapper_ikq2_q3.clear();
        mapper_ikq3_q2.clear();
        mapper_ik_q2q3.clear();
    };

    ~IndexFinder();

    int find_index_ikk2_k3(const int, const int);
    int find_index_ikk3_k2(const int, const int);
    int find_index_ik_k2k3(const int, const int);

    void add_triplets_absorb(int, std::vector<KsListGroup> &);
    void add_triplets_emitt(int, std::vector<KsListGroup> &);

private:
    std::vector<std::map<int, int>> mapper_ikq2_q3;
    std::vector<std::map<int, int>> mapper_ikq3_q2;
    std::vector<std::map<int, int>> mapper_ik_q2q3;
    int absorb_counter;
    int emitt_counter;
};

int IndexFinder::find_index_ikk2_k3(const int ik, const int k2)
{
    auto mapper = mapper_ikq2_q3[ik];
    if ( mapper.count(k2) == 0 ) return -1;
    return mapper.at(k2);
}

int IndexFinder::find_index_ikk3_k2(const int ik, const int k2)
{
    auto mapper = mapper_ikq3_q2[ik];
    if ( mapper.count(k2) == 0 ) return -1;
    return mapper.at(k2);
}

int IndexFinder::find_index_ik_k2k3(const int ik, const int k2)
{
    auto mapper = mapper_ik_q2q3[ik];
    if ( mapper.count(k2) == 0 ) return -1;
    return mapper.at(k2);
}

void IndexFinder::add_triplets_absorb(int ik_in, std::vector<KsListGroup> &triplet)
{
    std::map<int, int> ikq2_q3, ikq3_q2;
    ikq2_q3.clear();
    ikq3_q2.clear();

    int k2, k3;
    for (auto i = 0; i < triplet.size(); ++i){

        auto pair = triplet[i];
        for (auto j = 0; j < pair.group.size(); ++j){
            k2 = pair.group[j].ks[0];
            k3 = pair.group[j].ks[1];
            ikq2_q3[k2] = absorb_counter;
            ikq3_q2[k3] = absorb_counter;
        }
        absorb_counter += 1;
    }
    mapper_ikq2_q3.push_back(ikq2_q3);
    mapper_ikq3_q2.push_back(ikq3_q2);
}

void IndexFinder::add_triplets_emitt(int ik_in, std::vector<KsListGroup> &triplet)
{
    std::map<int, int> ik_q2q3;
    ik_q2q3.clear();
    int k2, k3;
    for (auto i = 0; i < triplet.size(); ++i){

        auto pair = triplet[i];
        for (auto j = 0; j < pair.group.size(); ++j){
            k2 = pair.group[j].ks[0];
            k3 = pair.group[j].ks[1];
            ik_q2q3[k2] = emitt_counter;
        }
        emitt_counter += 1;
    }
    mapper_ik_q2q3.push_back(ik_q2q3);
}

} // name space