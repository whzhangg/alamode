/*
 iterativebte.cpp
*/

#include "mpi_common.h"
#include "conductivity.h"
#include "iterativebte.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "interpolation.h"
#include "parsephon.h"
#include "isotope.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "phonon_dos.h"
#include "thermodynamics.h"
#include "phonon_velocity.h"
#include "anharmonic_core.h"
#include "system.h"
#include "write_phonons.h"
#include "symmetry_core.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iterator>
#include "LBTE_equation.h"

using namespace PHON_NS;

MatrixA::MatrixA(PHON *phon) : Pointers(phon)
{
    Temperature = nullptr;
    L_absorb = nullptr;
    L_emitt = nullptr;
    A_absorb = nullptr;
    A_emitt = nullptr;
    diag = nullptr;
}

MatrixA::~MatrixA()
{
    if (Temperature) deallocate(Temperature);
    if (L_absorb) deallocate(L_absorb);
    if (L_emitt) deallocate(L_emitt);
    if (A_absorb) deallocate(A_absorb);
    if (A_emitt) deallocate(A_emitt);
    if (diag) deallocate(diag);
}

void MatrixA::setup_matrix(const std::vector<int> &nkl_in, const int nklocal_in) 
{
    nklocal = nklocal_in;
    nk_l.clear();
    for (auto ik = 0; ik < nklocal; ++ik) {
        nk_l.push_back(nkl_in[ik]);
    }

    nk_3ph = dos->kmesh_dos->nk;
    ns = dynamical->neval;
    ns2 = ns * ns;

    set_temperature();
    get_triplets();
    cal_L();
}

void MatrixA::set_temperature()
{
    allocate(Temperature, ntemp);
    ntemp = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    for (auto i = 0; i < ntemp; ++i) {
        Temperature[i] = system->Tmin + static_cast<double>(i) * system->dT;
    }
}

void MatrixA::get_triplets()
{
    sym_permutation = false;
    use_triplet_symmetry = true;
    localnk_triplets_emitt.clear();  // pairs k3 = k1 - k2 ( -k1 + k2 + k3 = G )
    localnk_triplets_absorb.clear(); // pairs k3 = k1 + k2 (  k1 + k2 + k3 = G )

    int counter = 0;
    int counter2 = 0;

    for (unsigned int i = 0; i < nklocal; ++i) {

        auto ik = nk_l[i];
        std::vector<KsListGroup> triplet;
        std::vector<KsListGroup> triplet2;

        // k3 = k1 - k2
        dos->kmesh_dos->get_unique_triplet_k(ik, symmetry->SymmList,
                                             use_triplet_symmetry,
                                             sym_permutation, triplet);

        // k3 = - (k1 + k2)
        dos->kmesh_dos->get_unique_triplet_k(ik, symmetry->SymmList,
                                             use_triplet_symmetry,
                                             sym_permutation, triplet2, 1);

        counter += triplet.size();
        counter2 += triplet2.size();

        localnk_triplets_emitt.push_back(triplet);
        localnk_triplets_absorb.push_back(triplet2);

    }
    kplength_emitt = counter;   // remember the number of unique pairs
    kplength_absorb = counter2;
}

void MatrixA::calc_L()
{
    if (mympi->my_rank == 0) {
        std::cout << " Calculate once for the transition probability L(absorb) and L(emitt)" << std::endl;
        std::cout << " Size of L (MB) (approx.) = " << memsize_in_MB(sizeof(double), kplength_absorb + kplength_emitt, ns, ns2)
                  << " ... ";
    }

    allocate(L_absorb, kplength_absorb, ns, ns2);
    allocate(L_emitt, kplength_emitt, ns, ns2);
    if (integration->ismear >= 0) {
        calc_L_smear();
    } else if (integration->ismear == -1) {
        calc_L_tetra();
    }

    if (mympi->my_rank == 0) {
        std::cout << "     DONE !" << std::endl;
    }
}

void MatrixA::calc_L_smear()
{
    // we calculate V for all pairs L+(local_nk*eachpair,ns,ns2) and L-

    unsigned int arr[3];
    int k1, k2, k3;
    int s1, s2, s3;
    int ib;
    double omega1, omega2, omega3;

    double v3_tmp;

    unsigned int counter;
    double delta = 0;

    auto epsilon = integration->epsilon;

    const auto omega_tmp = dos->dymat_dos->get_eigenvalues();
    const auto evec_tmp = dos->dymat_dos->get_eigenvectors();

    double epsilon2[2];

    // emitt
    counter = 0;
    unsigned int ik, j;

    for (ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;    // k index in full grid

        // emitt k1 -> k2 + k3 
        // V(-q1, q2, q3) delta(w1 - w2 - w3)
        for (j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

            auto pair = localnk_triplets_emitt[ik][j];

            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                arr[0] = dos->kmesh_dos->kindex_minus_xk[k1] * ns + s1;
                omega1 = omega_tmp[k1][s1];

                for (ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;

                    arr[1] = k2 * ns + s2;
                    arr[2] = k3 * ns + s3;
                    omega2 = omega_tmp[k2][s2];
                    omega3 = omega_tmp[k3][s3];

                    if (integration->ismear == 0) {
                        delta = delta_lorentz(omega1 - omega2 - omega3, epsilon);
                    } else if (integration->ismear == 1) {
                        delta = delta_gauss(omega1 - omega2 - omega3, epsilon);
                    } else if (integration->ismear == 2) {
                        integration->adaptive_sigma->get_sigma(k2, s2, k3, s3, epsilon2);
                        delta = delta_gauss(omega1 - omega2 - omega3, epsilon2[0]);
                    }

                    v3_tmp = std::norm(anharmonic_core->V3(arr,
                                                           dos->kmesh_dos->xk,
                                                           omega_tmp,
                                                           evec_tmp));

                    L_emitt[counter][s1][ib] = (pi / 4.0) * v3_tmp * delta / static_cast<double>(nk_3ph);
                }
            }
            counter += 1;
        }
    }
    if (counter != kplength_emitt) {
        exit("setup_L", "Emitt: pair length not equal!");
    }

    counter = 0;
    for (ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;    // k index in full grid

        // absorption k1 + k2 -> -k3
        // V(q1, q2, q3) since k3 = - (k1 + k2)
        for (j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

            auto pair = localnk_triplets_absorb[ik][j];

            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                arr[0] = k1 * ns + s1;
                omega1 = omega_tmp[k1][s1];

                for (ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;

                    arr[1] = k2 * ns + s2;
                    arr[2] = k3 * ns + s3;
                    omega2 = omega_tmp[k2][s2];
                    omega3 = omega_tmp[k3][s3];

                    if (integration->ismear == 0) {
                        delta = delta_lorentz(omega1 + omega2 - omega3, epsilon);
                    } else if (integration->ismear == 1) {
                        delta = delta_gauss(omega1 + omega2 - omega3, epsilon);
                    } else if (integration->ismear == 2) {
                        integration->adaptive_sigma->get_sigma(k2, s2, k3, s3, epsilon2);
                        delta = delta_gauss(omega1 + omega2 - omega3, epsilon2[0]);  
                        // we use epsilon2[0] for both absorption and emission, as in shengBTE
                        // this is different from the adaptive in SERTA case
                    }

                    v3_tmp = std::norm(anharmonic_core->V3(arr,
                                                           dos->kmesh_dos->xk,
                                                           omega_tmp,
                                                           evec_tmp));

                    L_absorb[counter][s1][ib] = (pi / 4.0) * v3_tmp * delta / static_cast<double>(nk_3ph);
                }
            }
            counter += 1;
        }
    }

    if (counter != kplength_absorb) {
        exit("setup_L", "absorb: pair length not equal!");
    }

}

void MatrixA::calc_L_tetra()
{
    // generate index for, emitt
    std::vector<std::vector<int>> ikp_emitt;
    ikp_emitt.clear();
    int cnt = 0;
    for (auto ik = 0; ik < nklocal; ++ik) {
        std::vector<int> counterk;
        counterk.clear();
        for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {
            counterk.push_back(cnt);
            cnt += 1;
        }
        ikp_emitt.push_back(counterk);
    }
    // absorb
    std::vector<std::vector<int>> ikp_absorb;
    ikp_absorb.clear();
    cnt = 0;
    for (auto ik = 0; ik < nklocal; ++ik) {
        std::vector<int> counterk;
        counterk.clear();
        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {
            counterk.push_back(cnt);
            cnt += 1;
        }
        ikp_absorb.push_back(counterk);
    }

    unsigned int arr[3];
    int k1, k2, k3;
    int s1, s2, s3;
    int ib;
    double omega1, omega2, omega3;

    double v3_tmp;
    double xk_tmp[3];
    double delta = 0;

    unsigned int *kmap_identity;
    allocate(kmap_identity, nk_3ph);
    for (auto i = 0; i < nk_3ph; ++i) kmap_identity[i] = i;

    double *energy_tmp;
    double *weight_tetra;
    allocate(energy_tmp, nk_3ph);
    allocate(weight_tetra, nk_3ph);

    const auto omega_tmp = dos->dymat_dos->get_eigenvalues();
    const auto evec_tmp = dos->dymat_dos->get_eigenvectors();

    for (auto ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (s1 = 0; s1 < ns; ++s1) {

            omega1 = omega_tmp[k1][s1];

            for (ib = 0; ib < ns2; ++ib) {
                s2 = ib / ns;
                s3 = ib % ns;

                // emitt k1 -> k2 + k3 : V(-q1, q2, q3) delta(w1 - w2 - w3)
                for (k2 = 0; k2 < nk_3ph; k2++) {
                    
                    for (auto i = 0; i < 3; ++i) {
                        xk_tmp[i] = dos->kmesh_dos->xk[k1][i] - dos->kmesh_dos->xk[k2][i];
                    }

                    k3 = dos->kmesh_dos->get_knum(xk_tmp);

                    omega2 = omega_tmp[k2][s2];
                    omega3 = omega_tmp[k3][s3];

                    energy_tmp[k2] = omega2 + omega3;
                }
                integration->calc_weight_tetrahedron(nk_3ph,
                                                     kmap_identity,
                                                     energy_tmp,
                                                     omega1,
                                                     dos->tetra_nodes_dos->get_ntetra(),
                                                     dos->tetra_nodes_dos->get_tetras(),
                                                     weight_tetra);

                for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

                    auto pair = localnk_triplets_emitt[ik][j];
                    auto counter = ikp_emitt[ik][j];

                    k2 = pair.group[0].ks[0];
                    k3 = pair.group[0].ks[1];

                    arr[0] = dos->kmesh_dos->kindex_minus_xk[k1] * ns + s1;
                    arr[1] = k2 * ns + s2;
                    arr[2] = k3 * ns + s3;
                    delta = weight_tetra[k2];
                    v3_tmp = std::norm(anharmonic_core->V3(arr,
                                                           dos->kmesh_dos->xk,
                                                           omega_tmp,
                                                           evec_tmp));

                    L_emitt[counter][s1][ib] = (pi / 4.0) * v3_tmp * delta;
                }

                // absorption k1 + k2 -> -k3 : V(q1, q2, q3) delta(w1 + w2 - w3)
                for (k2 = 0; k2 < nk_3ph; k2++) {

                    for (auto i = 0; i < 3; ++i) {
                        xk_tmp[i] = dos->kmesh_dos->xk[k1][i] + dos->kmesh_dos->xk[k2][i];
                    }
                    k3 = dos->kmesh_dos->get_knum(xk_tmp);

                    omega2 = omega_tmp[k2][s2];
                    omega3 = omega_tmp[k3][s3];

                    energy_tmp[k2] = -omega2 + omega3;
                }
                integration->calc_weight_tetrahedron(nk_3ph,
                                                     kmap_identity,
                                                     energy_tmp,
                                                     omega1,
                                                     dos->tetra_nodes_dos->get_ntetra(),
                                                     dos->tetra_nodes_dos->get_tetras(),
                                                     weight_tetra);

                for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

                    auto pair = localnk_triplets_absorb[ik][j];
                    auto counter = ikp_absorb[ik][j];

                    k2 = pair.group[0].ks[0];
                    k3 = pair.group[0].ks[1];

                    arr[0] = k1 * ns + s1;
                    arr[1] = k2 * ns + s2;
                    arr[2] = k3 * ns + s3;
                    delta = weight_tetra[k2];
                    v3_tmp = std::norm(anharmonic_core->V3(arr,
                                                           dos->kmesh_dos->xk,
                                                           omega_tmp,
                                                           evec_tmp));

                    L_absorb[counter][s1][ib] = (pi / 4.0) * v3_tmp * delta;
                }
            } // ib
        } // s1    
    } // ik

    deallocate(kmap_identity);
    deallocate(energy_tmp);
    deallocate(weight_tetra);
}

void MatrixA::set_temperature(const int itemp)
{
    allocate(A_absorb, kplength_absorb, ns, ns2);
    allocate(A_emitt, kplength_emitt, ns, ns2);
    calc_A(itemp, A_absorb, A_emitt);

    allocate(diag, nklocal, ns);
    calc_rta_diag(diag);
}

void MatrixA::calc_A(const int itemp, double ***&A_absorb, double ***&A_emitt)
{
    double **n0; 
    allocate(n0, nk_3ph, ns);
    iterativebte->calc_n0(itemp, n0);

    unsigned k1, k2, k3, s1, s2, s3;
    double nq1, nq2, nq3;

    unsigned counter = 0;
    for (auto ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;    // k index in full grid

        // emitt k1 -> k2 + k3 
        // V(-q1, q2, q3) delta(w1 - w2 - w3)
        for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

            auto pair = localnk_triplets_emitt[ik][j];

            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                nq1 = n0[k1][s1];

                for (auto ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;

                    nq2 = n0[k2][s2];
                    nq3 = n0[k3][s3];

                    A_emitt[counter][s1][ib] = L_emitt[counter][s1][ib] * nq1 * (nq2 + 1.0) * (nq3 + 1.0);
                }
            }
            counter += 1;
        }
    }
    if (counter != kplength_emitt) {
        exit("setup_L", "Emitt: pair length not equal!");
    }
    counter = 0;
    for (auto ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;    // k index in full grid

        // absorption k1 + k2 -> -k3
        // V(q1, q2, q3) since k3 = - (k1 + k2)
        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

            auto pair = localnk_triplets_absorb[ik][j];

            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                nq1 = n0[k1][s1];

                for (auto ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;

                    nq2 = n0[k2][s2];
                    nq3 = n0[k3][s3];

                    A_absorb[counter][s1][ib] = L_absorb[counter][s1][ib] * nq1 * nq2 * (nq3 + 1.0);
                }
            }
            counter += 1;
        }
    }
    if (counter != kplength_absorb) {
        exit("setup_L", "absorb: pair length not equal!");
    }

    deallocate(n0);
}

void MatrixA::calc_rta_diag(double **&q1)
{
    unsigned int ik, tmpk;
    unsigned int k1, k2, k3;
    unsigned int s1, s2, s3;
    double n1, n2, n3;

    double **Qemit;
    double **Qabsorb;
    allocate(Qemit, nklocal, ns);
    allocate(Qabsorb, nklocal, ns);

    for (auto ik = 0; ik < nklocal; ++ik) {
        for (s1 = 0; s1 < ns; ++s1) {
            Qemit[ik][s1] = 0.0;
            Qabsorb[ik][s1] = 0.0;
        }
    }

    unsigned int counter = 0;
    // emit
    for (ik = 0; ik < nklocal; ++ik) {

        tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

            auto pair = localnk_triplets_emitt[ik][j];
            auto multi = static_cast<double>(pair.group.size());
            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    Qemit[ik][s1] += 0.5 * A_emitt[counter][s1][ib] * multi;
                }
            }
            counter += 1;
        }

    }
    if (counter != kplength_emitt) {
        exit("setup_L", "Emitt: pair length not equal!");
    }

    // absorb k1 + k2 -> -k3
    counter = 0;
    for (ik = 0; ik < nklocal; ++ik) {

        tmpk = nk_l[ik];
        k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

            auto pair = localnk_triplets_absorb[ik][j];
            auto multi = static_cast<double>(pair.group.size());
            k2 = pair.group[0].ks[0];
            k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    Qabsorb[ik][s1] += A_absorb[counter][s1][ib] * multi;
                }
            }
            counter += 1;
        }
    }

    if (counter != kplength_absorb) {
        exit("setup_L", "absorb: pair length not equal!");
    }

    for (ik = 0; ik < nklocal; ++ik) {
        for (s1 = 0; s1 < ns; ++s1) {
            q1[ik][s1] = Qemit[ik][s1] + Qabsorb[ik][s1];
        }
    }
    deallocate(Qemit);
    deallocate(Qabsorb);
}

void MatrixA::row_product(const double ***&fq_in, double ***product)
{
    // we perform a row product with a full Fq, both fq_in and product will be 
    // in the full BZ
    for (auto ik = 0; ik < nklocal; ik ++) {

        auto tmpk = nk_l[ik];
        auto num_equivalent = dos->kmesh_dos->kpoint_irred_all[tmpk].size();
        auto kref = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto is = 0; is < ns; is ++)  {
            for (auto x = 0; x < 3; x ++) {
                product[ik][is][x] = 0.0;
            }
        }
    }

    
}
