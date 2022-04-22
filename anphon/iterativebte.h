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
 friend class MatrixA;
 public:
    Iterativebte(class PHON *);
    ~Iterativebte();

    void do_iterativebte();

    bool do_iterative;
    double *Temperature;
    unsigned int ntemp;

    int max_cycle;
    int min_cycle;
    double mixing_factor;

    int solution_method;  
    // 0: default, iterative solve
    // 1: iterative solver, symmetry version
    // 2: direct solver (cg)

    double convergence_criteria;  // dF(i+1) - dF(i) < cc
    double ***kappa;

    std::fstream fs_result;

 private:

    int kplength_emitt;
    int kplength_absorb;
    int nk_3ph, nklocal, ns, ns2;
    bool use_triplet_symmetry;
    bool sym_permutation;
    bool has_rta_damping;
    bool has_4ph_damping;

    double ***vel;
    double ***damping4;
    double **rta_damping_loc;  // temperature independent rta damping i.e. boundary, isotope

    //double **isotope_damping_loc;
    //double **boundary_damping_loc;

    std::vector<int> nk_l;

    //helper functions
    void calc_n0(int, double **&);
    void calc_dndT(int, double **&);
    void calc_kappa(int, double ***&, double **&); 
    void average_vector_degenerate_at_k(int, double **&);
    void average_scalar_degenerate_at_k(int, double *&);
    void write_kappa_iterative();
    
    void setup_iterative();
    void setup_control_variables();
    void distribute_q();

    void prepare_data();    // prepare volecity, L, tau_RTA
    void prepare_group_vel();
    void prepare_fixed_tau();
    void calc_damping4();

    void LBTE_wrapper();        // provide a wrapper around solver for LBTE
    void calc_righthandside(const int, double ***&);
    void calc_righthandside_full(const int, double ***&);
    void calc_n1overtau(const int, double **&);

    void naive_iteration(); // solve LBTE with current iteration, without symmetry, should serve as a stable reference method
    void symmetry_iteration();       // not implemented
    
    double local_residual_sum_squared(double ***&);
    double residual_norm(double ***&);
    // naive iterative solver
};
}
