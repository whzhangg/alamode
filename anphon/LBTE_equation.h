/*
This class is responsible to formulate the matrix part of the linear equation, including the 
preconditioner and the remaining parts
*/

#pragma once

#include "pointers.h"
#include "kpoint.h"
#include <vector>
#include <map>
#include <set>
#include <complex>

namespace PHON_NS {

// provide an interface
class IndexFinder{
// it will return zero if the pair does not exist
// return the given index if it does
public:
   IndexFinder(int nkin): nkfull(nkin) {
      mapper_absorb.clear();
      mapper_emitt.clear();
   };

   ~IndexFinder();

   int find_index_absorb(int, int, int);
   int find_index_emitt(int, int, int);

   void add_triplets_absorb(std::vector<KsListGroup> &);
   void add_triplets_emitt(std::vector<KsListGroup> &);

private:
   std::map<int, std::map<int, int>> mapper_absorb;
   std::map<int, std::map<int, int>> mapper_emitt;
   int nkfull;

};

class MatrixA{
// matrixA is a friend of iterativebte, it would access global information through 
public:
   MatrixA(Iterativebte *&);
   ~MatrixA();

   void set_temperature(const int);
   void setup_matrix(const std::vector<int> &, const int);
   double matrix_elements(const int, const int);
   double M(const int, const int);
   double P(const int);
   void row_product(const double ***&, double ***&);

private:
   Iterativebte *lbte;
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

   std::vector<int> nk_l;
   int nk_3ph, nklocal, ns, ns2;
   bool additional_scattering;


   int nk_3ph, nklocal, ns, ns2;
};
}
