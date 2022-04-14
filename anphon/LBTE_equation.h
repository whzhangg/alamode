/*
This class is responsible to formulate the matrix part of the linear equation, including the 
preconditioner and the remaining parts
*/

#pragma once

#include "pointers.h"
#include "kpoint.h"
#include <vector>
#include <set>
#include <complex>

namespace PHON_NS {
class MatrixA : protected Pointers {
public:
   MatrixA(class PHON *);
   ~MatrixA();

   void set_temperature(const int);
   void row_product(const double ***&, double ***&);

private:
   void setup_matrix(const std::vector<int> &, const int);
   void set_temperature();
   void get_triplets();

   void calc_L();
   void calc_L_smear();
   void calc_L_tetra();
   void calc_A(const int, double ***&, double ***&);
   void calc_rta_diag(double **&);

   std::vector<std::vector<KsListGroup>> localnk_triplets_emitt;
   std::vector<std::vector<KsListGroup>> localnk_triplets_absorb;
   int kplength_emitt;
   int kplength_absorb;

   double *Temperature;
   unsigned int ntemp;

   double ***L_absorb; // L q0 + q1 -> q2
   double ***L_emitt;  // L q0 -> q1 + q2

   double ***A_absorb;
   double ***A_emitt;
   double **diag;

   std::vector<int> &nk_l;
   int nk_3ph, nklocal, ns, ns2;
   bool additional_scattering;


   int nk_3ph, nklocal, ns, ns2;
};
}
