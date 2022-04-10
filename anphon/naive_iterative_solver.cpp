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

using namespace PHON_NS;

/* 
Functions in this file is responsible for solving iteration equation and return the solution f

*/

void Iterativebte::naive_iteration(double ***&A_absorb, double ***&A_emitt, double ***&bq, double **&n1overtau, double ***&solution)
{
    // we solve the equation
    // bq = [ \sum_{q2,q3} (A_{q1,q2}^{q3} + 1/2 A_{q1}^{q2,q3} ) + n1overtau(q1) ] f_{q1} + \sum_{q2,q3} [ (f_{q2} - f_{q3})A_{q1,q2}^{q3} + 1/2 (f_{q2} + f_{q3}) A_{q1}^{q2,q3} ]
    
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

    int nsym = symmetry->SymmList.size();

    double ***fq_old;
    double ***fq_new;
    allocate(fq_old, nk_3ph, ns, 3);
    allocate(fq_new, nk_3ph, ns, 3);

    // calculate RTA
    

    for (auto itr = 0; itr < max_cycle; ++itr) {
        if (mympi->my_rank == 0) std::cout << "   -> iter " << std::setw(3) << itr << ": ";

    }

    deallocate(fq_old);
    deallocate(fq_new);
}


void Iterativebte::iterative_solver()
{
    double ***dFold; 
    double ***dFnew; 

    allocate(dFold, nk_3ph, ns, 3);
    allocate(dFnew, nk_3ph, ns, 3);
    allocate(kappa, ntemp, 3, 3);
    
    std::vector<double> convergence_history;   // store | f_n - f_{n-1} | L2 norm
    convergence_history.empty();

    double **Q;
    double **kappa_new;
    double **kappa_old;
    double **fb;
    double **dndt;
    allocate(kappa_new, 3, 3);
    allocate(kappa_old, 3, 3);
    allocate(Q, nklocal, ns);
    allocate(dndt, nklocal, ns);
    allocate(fb, nk_3ph, ns);


    if (conductivity->fph_rta > 0) {
        calc_damping4();
    }


    if (mympi->my_rank == 0) {
        std::cout << std::endl << " Iteration starts ..." << std::endl << std::endl;
    }

    // we solve iteratively for each temperature
    int ik, is, ix, iy;
    double n1, n2, n3;

    // generate index for, emitt
    std::vector<std::vector<int>> ikp_emitt;
    ikp_emitt.clear();
    int cnt = 0;
    for (ik = 0; ik < nklocal; ++ik) {
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
    for (ik = 0; ik < nklocal; ++ik) {
        std::vector<int> counterk;
        counterk.clear();
        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {
            counterk.push_back(cnt);
            cnt += 1;
        }
        ikp_absorb.push_back(counterk);
    }

    int nsym = symmetry->SymmList.size();
    // start iteration
    double **Wks;
    allocate(Wks, ns, 3);

    double norm;
    double local_squared_norm;
    int s1, s2, s3;
    int k1, k2, k3, k3_minus;

    for (auto itemp = 0; itemp < ntemp; ++itemp) {

        if (mympi->my_rank == 0) {
            std::cout << " Temperature step ..." << std::setw(10) << std::right
                      << std::fixed << std::setprecision(2) << Temperature[itemp] << " K" <<
                      "    -----------------------------" << std::endl;
            std::cout << "      Kappa [W/mK]        xx          xy          xz" <<
                      "          yx          yy          yz" <<
                      "          zx          zy          zz    |df' - df|" << std::endl;
        }

        double beta = 1.0 / (thermodynamics->T_to_Ryd * Temperature[itemp]);

        calc_boson(itemp, fb, dndt);
        calc_Q_from_L(fb, Q);

        for (ik = 0; ik < nklocal; ik ++) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;  // k index in full grid
            average_scalar_degenerate_at_k(k1, Q[ik]);
        }

        for (ik = 0; ik < nk_3ph; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (ix = 0; ix < 3; ++ix) {
                    dFold[ik][is][ix] = 0.0;
                }
            }
        }

        int generating_sym;
        bool time_reverse = false;   // keep track if we further apply time reversal symmetry
        int isym;

        for (auto itr = 0; itr < max_cycle; ++itr) {
            if (mympi->my_rank == 0) {
                std::cout << "   -> iter " << std::setw(3) << itr << ": ";
            }

            local_squared_norm = 0.0;

            for (ik = 0; ik < nk_3ph; ++ik) {
                for (is = 0; is < ns; ++is) {
                    for (ix = 0; ix < 3; ++ix) {
                        dFnew[ik][is][ix] = 0.0;
                    }
                }
            }

            for (ik = 0; ik < nklocal; ++ik) {

                auto tmpk = nk_l[ik];
                auto num_equivalent = dos->kmesh_dos->kpoint_irred_all[tmpk].size();
                auto kref = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

                for (auto ieq = 0; ieq < num_equivalent; ++ieq) {

                    k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][ieq].knum;      // k1 will go through all points

                    generating_sym = -1;
                    for (isym = 0; isym < nsym; ++isym) {
                        auto krot = dos->kmesh_dos->knum_sym(kref, symmetry->SymmList[isym].rot);
                        auto minuskrot = dos->kmesh_dos->kindex_minus_xk[krot];
                        if (k1 == krot) {
                            generating_sym = isym;
                            time_reverse = false;
                        } else if ( symmetry->time_reversal_sym && k1 == minuskrot) {
                            generating_sym = isym;
                            time_reverse = true;
                        }
                    }
                    if (generating_sym == -1) {
                        exit("iterative solution", "cannot find all equivalent k");
                    }

                    // calculate W here
                    for (s1 = 0; s1 < ns; ++s1) {

                        for (ix = 0; ix < 3; ++ix) {
                            Wks[s1][ix] = 0.0;
                        }

                        // emitt k1 -> k2 + k3
                        for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

                            auto pair = localnk_triplets_emitt[ik][j];
                            int kp_index = ikp_emitt[ik][j];

                            for (auto ig = 0; ig < pair.group.size(); ig++) {

                                k2 = dos->kmesh_dos->knum_sym(pair.group[ig].ks[0],
                                                              symmetry->SymmList[generating_sym].rot);
                                k3 = dos->kmesh_dos->knum_sym(pair.group[ig].ks[1],
                                                              symmetry->SymmList[generating_sym].rot);
                                if (time_reverse) {
                                    k2 = dos->kmesh_dos->kindex_minus_xk[k2];
                                    k3 = dos->kmesh_dos->kindex_minus_xk[k3];
                                }

                                for (int ib = 0; ib < ns2; ++ib) {
                                    s2 = ib / ns;
                                    s3 = ib % ns;

                                    n1 = fb[k1][s1];
                                    n2 = fb[k2][s2];
                                    n3 = fb[k3][s3];
                                    for (ix = 0; ix < 3; ++ix) {
                                        Wks[s1][ix] -= 0.5 * (dFold[k2][s2][ix] + dFold[k3][s3][ix]) * n1 * (n2 + 1.0)
                                              * (n3 + 1.0) * L_emitt[kp_index][s1][ib];
                                    }
                                }
                            }
                        }

                        // absorb k1 + k2 -> -k3
                        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

                            auto pair = localnk_triplets_absorb[ik][j];
                            int kp_index = ikp_absorb[ik][j];

                            for (auto ig = 0; ig < pair.group.size(); ig++) {

                                k2 = dos->kmesh_dos->knum_sym(pair.group[ig].ks[0],
                                                              symmetry->SymmList[generating_sym].rot);
                                k3 = dos->kmesh_dos->knum_sym(pair.group[ig].ks[1],
                                                              symmetry->SymmList[generating_sym].rot);
                                if (time_reverse) {
                                    k2 = dos->kmesh_dos->kindex_minus_xk[k2];
                                    k3 = dos->kmesh_dos->kindex_minus_xk[k3];
                                }
                                
                                k3_minus = dos->kmesh_dos->kindex_minus_xk[k3];

                                for (int ib = 0; ib < ns2; ++ib) {
                                    s2 = ib / ns;
                                    s3 = ib % ns;

                                    n1 = fb[k1][s1];
                                    n2 = fb[k2][s2];
                                    n3 = fb[k3][s3];
                                    for (ix = 0; ix < 3; ++ix) {
                                        Wks[s1][ix] +=
                                              (dFold[k2][s2][ix] - dFold[k3_minus][s3][ix]) * n1 * n2 * (n3 + 1.0)
                                                    * L_absorb[kp_index][s1][ib];
                                    }
                                }
                            }
                        }

                    } // s1

                    average_vector_degenerate_at_k(k1, Wks);

                    for (s1 = 0; s1 < ns; ++s1) {

                        double Q_final = Q[ik][s1];
                        if (has_rta_damping) {
                            Q_final += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * rta_damping_loc[ik][s1];
                        }
                        
                        if (conductivity->fph_rta > 0) {
                            Q_final += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * damping4[itemp][ik][s1];
                        }

                        if (Q_final < 1.0e-50 || dos->dymat_dos->get_eigenvalues()[k1][s1] < eps8) {
                            for (ix = 0; ix < 3; ix++) dFnew[k1][s1][ix] = 0.0;
                        } else {
                            for (ix = 0; ix < 3; ix++) {
                                dFnew[k1][s1][ix] = (-vel[k1][s1][ix] * dndt[ik][s1] / beta - Wks[s1][ix]) / Q_final;
                            }
                        }
                        if (itr > 0) {
                            for (ix = 0; ix < 3; ix++) {
                                dFnew[k1][s1][ix] = dFnew[k1][s1][ix] * mixing_factor 
                                                  + dFold[k1][s1][ix] * (1.0 - mixing_factor);
                                local_squared_norm += std::pow( dFnew[k1][s1][ix] - dFold[k1][s1][ix], 2.0);
                            }
                        }

                    }

                } // ieq
            } // ik

            // check convergence, if converged, stop, if not, update dF and print kappa
            norm = 0.0;
            MPI_Allreduce(&local_squared_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            convergence_history.push_back(norm);

            auto converged1 = false;
            auto converged2 = false;
            if (itr >= min_cycle) converged1 = check_convergence_dF(convergence_history);

            if (converged1) {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa_new[ix][iy] = kappa_old[ix][iy];
                    }
                }
            } else {
                MPI_Allreduce(&dFnew[0][0][0],
                            &dFold[0][0][0],
                            nk_3ph * ns * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                calc_kappa(itemp, dFold, kappa_new);
                
                if (mympi->my_rank == 0) {
                    for (ix = 0; ix < 3; ++ix) {
                        for (iy = 0; iy < 3; ++iy) {
                            std::cout << std::setw(12) << std::scientific
                                    << std::setprecision(2) << kappa_new[ix][iy];
                        }
                    }
                    norm = std::pow(norm, 0.5);
                    std::cout << std::setw(14) << std::scientific << std::setprecision(2) << norm << std::endl;
                }

                if (itr >= min_cycle) converged2 = check_convergence_kappa(kappa_old, kappa_new);
            }

            if (converged1 || converged2) {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_old[ix][iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   -> Converged is achieved";
                    if (converged1) std::cout << " (dF converged)                                             ";
                               else std::cout << " (kappa converged)                                          ";
                    std::cout << "                                     "
                              << std::setw(14) << std::scientific << std::setprecision(2) << norm << std::endl;
                }
                break;

            } else {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa_old[ix][iy] = kappa_new[ix][iy];
                    }
                }
            }

            if (itr == (max_cycle - 1)) {
                // update kappa even if not converged
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_new[ix][iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   -> iter     Warning !! max cycle reached but kappa not converged " << std::endl;
                }
            }

        } // iter
        write_Q_dF(itemp, Q, dFold);

    } // itemp

    deallocate(Q);
    deallocate(dndt);
    deallocate(kappa_new);
    deallocate(kappa_old);
    deallocate(fb);
    deallocate(Wks);

    if (mympi->my_rank == 0) {
        fs_result.close();
    }
}


bool Iterativebte::check_convergence_kappa(double **&k_old, double **&k_new)
{
    // check diagonal components only, since they are the most important
    double max_diff = -100;
    double diff;
    for (auto ix = 0; ix < 3; ++ix) {
        diff = std::abs(k_new[ix][ix] - k_old[ix][ix]) / std::abs(k_old[ix][ix]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff < convergence_criteria;
}


bool Iterativebte::check_convergence_dF(const std::vector<double> &history)
{
    auto size = history.size();
    double last = history[size - 1];
    double lastlast = history[size - 2];
    if (last > lastlast) {
        return true;
    } else return false;
}


void Iterativebte::calc_Q_from_L(double **&n, double **&q1)
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
                n1 = n[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    n2 = n[k2][s2];
                    n3 = n[k3][s3];
                    Qemit[ik][s1] += 0.5 * (n1 * (n2 + 1.0) * (n3 + 1.0)) * L_emitt[counter][s1][ib] * multi;
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
                n1 = n[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    n2 = n[k2][s2];
                    n3 = n[k3][s3];
                    Qabsorb[ik][s1] += (n1 * n2 * (n3 + 1.0)) * L_absorb[counter][s1][ib] * multi;
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
