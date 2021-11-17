/*
 iterativebte.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include "kpoint.h"
#include <vector>
#include <set>
#include <complex>

namespace PHON_NS {
class Iterativebte : protected Pointers {
 public:
    Iterativebte(class PHON *);
    ~Iterativebte();

    void setup_iterative();
    void do_iterativebte();

    bool do_iterative;
    bool direct_solution;
    double *Temperature;
    unsigned int ntemp;
    int max_cycle;
    int min_cycle;
    double mixing_factor;

    double convergence_criteria;  // dF(i+1) - dF(i) < cc
    double ***kappa;

    std::fstream fs_result;

 private:

    void set_default_variables();
    void deallocate_variables();

    int kplength_emitt;
    int kplength_absorb;
    int nk_3ph, nklocal, ns, ns2;
    bool use_triplet_symmetry;
    bool sym_permutation;

    double ***L_absorb; // L q0 + q1 -> q2
    double ***L_emitt;  // L q0 -> q1 + q2
    double ***vel;
    double ***dFold;
    double ***dFnew;
    double ***damping4;
    double **isotope_damping_loc;
    double **boundary_damping_loc;

    std::vector<std::vector<KsListGroup>> localnk_triplets_emitt;
    std::vector<std::vector<KsListGroup>> localnk_triplets_absorb;
    std::vector<int> nk_l, nk_job;

    void iterative_solver(); 
    void direct_solver();       // not implemented

    void get_triplets();
    void setup_L_smear();
    void setup_L_tetra();

    void calc_damping4();
    void calc_Q_from_L(double **&, double **&);
    void calc_boson(int, double **&, double **&);
    void calc_kappa(int, double ***&, double **&); 

    void average_vector_degenerate_at_k(int, double **&);
    void average_scalar_degenerate_at_k(int, double *&);

    bool check_convergence_kappa(double **&, double **&); // check if convergence cirteria is meet
    bool check_convergence_dF(const std::vector<double> &);

    void write_result();
    void write_Q_dF(int, double **&, double ***&);
    void write_kappa_iterative();
};
}
