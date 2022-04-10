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

Iterativebte::Iterativebte(PHON *phon) : Pointers(phon)
{
    // public
    do_iterative = true;
    Temperature = nullptr;
    ntemp = 0;
    min_cycle = 5;
    max_cycle = 20;
    mixing_factor = 0.9;
    convergence_criteria = 0.02;
    solution_method = 0; 
    kappa = nullptr;
    // private
    use_triplet_symmetry = true;
    sym_permutation = true;
    has_rta_damping = false;
    has_4ph_damping = false;
    // pointers
    L_absorb = nullptr;
    L_emitt = nullptr;
    vel = nullptr;
    damping4 = nullptr;
    rta_damping_loc = nullptr;
}

Iterativebte::~Iterativebte()
{
    if (Temperature) deallocate(Temperature);
    if (kappa) deallocate(kappa);
    if (vel) deallocate(vel);
    if (L_absorb) deallocate(L_absorb);
    if (L_emitt) deallocate(L_emitt);
    if (damping4) deallocate(damping4);
    if (rta_damping_loc) deallocate(rta_damping_loc);
}

void Iterativebte::do_iterativebte()
{
    setup_iterative();
    prepare_data();
    LBTE_wrapper();
    write_kappa_iterative();
}


void Iterativebte::setup_iterative()
{
    nk_3ph = dos->kmesh_dos->nk;
    ns = dynamical->neval;
    ns2 = ns * ns;

    setup_control_variables();
    distribute_q();

    allocate(kappa, ntemp, 3, 3);
    for (auto itemp = 0; itemp < ntemp; itemp ++) {
        for (auto i = 0; i < 3; i++) {
            for (auto j = 0; i < 3; i++) {
                kappa[itemp][i][j] = 0.0;
            }
        }
    }

    allocate(Temperature, ntemp);
    ntemp = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    for (auto i = 0; i < ntemp; ++i) {
        Temperature[i] = system->Tmin + static_cast<double>(i) * system->dT;
    }

    if (mympi->my_rank == 0) {
        std::cout << std::endl;
        std::cout << " Iterative solution" << std::endl;
        std::cout << " ==================" << std::endl;
        std::cout << " MIN_CYCLE = " << min_cycle << ", MAX_CYCLE = " << max_cycle << std::endl;
        std::cout << " ITER_THRESHOLD = " << std::setw(10) << std::right << 
                                             std::setprecision(4) << convergence_criteria << std::endl;
        if (conductivity->len_boundary > eps) std::cout << " LEN_BOUNDARY = " << std::setw(10) << std::right << 
                                             std::setprecision(4) << conductivity->len_boundary << std::endl;
        if (conductivity->fph_rta) std::cout << " 4ph = 1";
        std::cout << std::endl;
        std::cout << " Distribute q point ... ";
        std::cout << " Number of q point pre process: " << std::setw(5) << nklocal << std::endl;
        std::cout << std::endl;
    }

    get_triplets();
    // write_results();
}

void Iterativebte::setup_control_variables()
{
    MPI_Bcast(&max_cycle, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_cycle, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mixing_factor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&convergence_criteria, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    sym_permutation = false;
    use_triplet_symmetry = true;
    if (anharmonic_core->quartic_mode > 0) conductivity->fph_rta = 1;
}

void Iterativebte::distribute_q()
{
    // setup nk_l
    auto nk_ir = dos->kmesh_dos->nk_irred;
    nk_l.clear();
    for (auto i = 0; i < nk_ir; ++i) {
        if (i % mympi->nprocs == mympi->my_rank) nk_l.push_back(i);
    }
    nklocal = nk_l.size();
}

void Iterativebte::get_triplets()
{
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


void Iterativebte::prepare_data()
{
    if (mympi->my_rank == 0) {
        std::cout << std::endl << " Preparing input data for LBTE" << std::endl;
    }
    
    prepare_group_vel();
    prepare_L();
    prepare_fixed_tau();
}

void Iterativebte::prepare_group_vel()
{
    allocate(vel, nk_3ph, ns, 3);
    phonon_velocity->get_phonon_group_velocity_mesh_mpi(*dos->kmesh_dos,
                                                        system->lavec_p,
                                                        fcs_phonon->fc2_ext,
                                                        vel); //this will gather to rank0 process
    MPI_Bcast(&vel[0][0][0], nk_3ph * ns * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Iterativebte::prepare_L()
{
    if (mympi->my_rank == 0) {
        std::cout << " Calculate once for the transition probability L(absorb) and L(emitt)" << std::endl;
        std::cout << " Size of L (MB) (approx.) = " << memsize_in_MB(sizeof(double), kplength_absorb + kplength_emitt, ns, ns2)
                  << " ... ";
    }

    allocate(L_absorb, kplength_absorb, ns, ns2);
    allocate(L_emitt, kplength_emitt, ns, ns2);
    if (integration->ismear >= 0) {
        setup_L_smear();
    } else if (integration->ismear == -1) {
        setup_L_tetra();
    }

    if (mympi->my_rank == 0) {
        std::cout << "     DONE !" << std::endl;
    }
}

void Iterativebte::prepare_fixed_tau()
{
    
    if (isotope->include_isotope || conductivity->len_boundary > eps) has_rta_damping = true;

    if (has_rta_damping) {
        allocate(rta_damping_loc, nklocal, ns);
        for (auto ik = 0; ik < nklocal; ik++) {
            for (auto is = 0; is < ns; is++) { rta_damping_loc[ik][is] = 0.0; }
        }
    }
    
    if (isotope->include_isotope) {
        double **isotope_damping;
        allocate(isotope_damping, dos->kmesh_dos->nk_irred, ns);
        if (mympi->my_rank == 0) {
            for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ik++) {
                for (auto is = 0; is < ns; is++) {
                    isotope_damping[ik][is] = isotope->gamma_isotope[ik][is];
                }
            }
        }

        MPI_Bcast(&isotope_damping[0][0], dos->kmesh_dos->nk_irred * ns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (auto ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            for (auto is = 0; is < ns; is++) {
                rta_damping_loc[ik][is] += isotope_damping[tmpk][is];
            }
        }
        deallocate(isotope_damping);
    }

    if (conductivity->len_boundary > eps) {
        double vel_norm;
        for (auto ik = 0; ik < nklocal; ++ik) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
            vel_norm = 0.0;
            for (auto is = 0; is < ns; is++) { 
                for (auto j = 0; j < 3; ++j) {
                    vel_norm += vel[k1][is][j] * vel[k1][is][j];
                }
                
                vel_norm = std::sqrt(vel_norm);
                rta_damping_loc[ik][is] += (vel_norm / conductivity->len_boundary) * time_ry ;
            }
        }
    }

    if (conductivity->fph_rta > 0) {
        has_4ph_damping = true;
        calc_damping4();  // value to  damping4 (ntemp, nklocal, ns);
    }
}

void Iterativebte::setup_L_smear()
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

void Iterativebte::setup_L_tetra()
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

void Iterativebte::calc_damping4() 
{
    // call conductivity to do the 4ph part
    conductivity->fph_rta = 1;

    conductivity->setup_kappa_4ph();
    conductivity->calc_anharmonic_imagself4();

    double ***damping4_ir = nullptr;
    allocate(damping4_ir, ntemp, dos->kmesh_dos->nk_irred ,ns);

    if (mympi->my_rank == 0) {

        double **damping4_dense = nullptr;
        allocate(damping4_dense, dos->kmesh_dos->nk_irred * ns, ntemp);

        conductivity->interpolate_data( conductivity->kmesh_4ph, dos->kmesh_dos, 
                                        conductivity->damping4, damping4_dense);

        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ++ik) {
                for (auto is = 0; is < ns; ++is) {
                    damping4_ir[itemp][ik][is] = damping4_dense[ik * ns + is][itemp];
                }
            }
        }

        deallocate(damping4_dense);
    }

    MPI_Bcast(&damping4_ir[0][0][0], ntemp * dos->kmesh_dos->nk_irred * ns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    allocate(damping4, ntemp, nklocal, ns);
    for (auto ik = 0; ik < nklocal; ++ik) {
        auto tmpk = nk_l[ik];
        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            for (auto is = 0; is < ns; ++is) {
                damping4[itemp][ik][is] = damping4_ir[itemp][tmpk][is];
            }
        }
    }

    deallocate(damping4_ir);
}


void Iterativebte::LBTE_wrapper()
{
    // don't keep redundent variables around, like beta, which is trival to calculate
    for (auto itemp = 0; itemp < ntemp; ++itemp) {
        if (mympi->my_rank == 0) {
            std::cout << " Temperature step ..." << std::setw(10) << std::right << std::fixed << std::setprecision(2) << Temperature[itemp] << " K"
                      << "    -----------------------------" << std::endl;
        }
        // set up the b vector:
        double ***bq;  // right hand side of the equation
        double ***fq;  // solution
        double ***A_absorb;
        double ***A_emitt;
        allocate(A_absorb, kplength_absorb, ns, ns2);
        allocate(A_emitt, kplength_emitt, ns, ns2);
        allocate(bq, nklocal, ns, 3);
        allocate(fq, nklocal, ns, 3);

        calc_righthandside(itemp, bq);
        calc_A(itemp, A_absorb, A_emitt);

        double **n1overtau = nullptr;
        if (has_4ph_damping || has_rta_damping) {
            allocate(n1overtau, nklocal, ns); 
            for (auto ik = 0; ik < nklocal; ik++) {
                for (auto is = 0; is < ns; is++) {
                    n1overtau[ik][is] = 0.0;
                }
            }
            calc_n1overtau(itemp, n1overtau);
        }

        if (solution_method == 0) {
            naive_iteration(A_absorb, A_emitt, bq, n1overtau, fq);
        } else if (solution_method == 1) {
            symmetry_iteration(A_absorb, A_emitt, bq, n1overtau, fq);
        } else if (solution_method == 2) {
            direct_solver(A_absorb, A_emitt, bq, n1overtau, fq);
        }

        calc_kappa(itemp, fq, kappa[itemp]);

        deallocate(A_absorb);
        deallocate(A_emitt);
        deallocate(bq);
        deallocate(fq);

        if (has_4ph_damping || has_rta_damping) {
            deallocate(n1overtau);
        }

    } // itemp
}

void Iterativebte::calc_righthandside(const int itemp, double ***&result) 
{
    // equation 2.19, actually right hand side
    // b = - \beta^{-1} v_q (dn/dT)
    double beta = 1.0 / (thermodynamics->T_to_Ryd * Temperature[itemp]);

    double **dndt;
    allocate(dndt, nklocal, ns);
    calc_dndT(itemp, dndt);

    for (auto ik = 0; ik < nklocal; ++ik) {
        auto tmpk = nk_l[ik];
        auto k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto is = 0; is < ns; ++is) {
            double part = (-1.0 / beta) * dndt[ik][is];

            for (auto alpha = 0; alpha < 3; ++alpha) {
                result[ik][is][alpha] = part * vel[k1][is][alpha];
            }

        }
    }

    deallocate(dndt);
}

void Iterativebte::calc_A(const int itemp, double ***&A_absorb, double ***&A_emitt)
{
    double **n0; 
    allocate(n0, nk_3ph, ns);
    calc_n0(itemp, n0);

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

void Iterativebte::calc_n1overtau(const int itemp, double **&n1overtau)
{
    double **tau; 
    allocate(tau, nklocal, ns);
    for (auto ik = 0; ik < nklocal; ik++) {
        for (auto is = 0; is < ns; is++) {
            tau[ik][is] = 0.0;
        }
    }

    if (has_4ph_damping) {
        for (auto ik = 0; ik < nklocal; ik++) {
            for (auto is = 0; is < ns; is++) {
                tau[ik][is] += damping4[itemp][ik][is];
            }
        }
    }
    if (has_rta_damping) {
        for (auto ik = 0; ik < nklocal; ik++) {
            for (auto is = 0; is < ns; is++) {
                tau[ik][is] += rta_damping_loc[ik][is];
            }
        }
    }

    double **n0; 
    allocate(n0, nk_3ph, ns);
    calc_n0(itemp, n0);
    for (auto ik = 0; ik < nklocal; ik++) {
        auto tmpk = nk_l[ik];
        auto k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;    // k index in full grid
        for (auto is = 0; is < ns; is++) {
            n1overtau[ik][is] += n0[k1][is] * (n0[k1][is] + 1.0) * 2 * tau[ik][is];
        }
    }
    deallocate(tau);
}


void Iterativebte::calc_n0(int itemp, double **&b_out)
{
    auto etemp = Temperature[itemp];
    double omega;
    for (auto ik = 0; ik < nk_3ph; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            omega = dos->dymat_dos->get_eigenvalues()[ik][is];
            b_out[ik][is] = thermodynamics->fB(omega, etemp);
        }
    }
}

void Iterativebte::calc_dndT(int itemp, double **&dndt_out)
{
    auto etemp = Temperature[itemp];
    double omega;
    const double t_to_ryd = thermodynamics->T_to_Ryd;

    for (auto ik = 0; ik < nklocal; ++ik) {
        auto ikr = nk_l[ik];
        auto k1 = dos->kmesh_dos->kpoint_irred_all[ikr][0].knum;
        for (auto is = 0; is < ns; ++is) {
            omega = dos->dymat_dos->get_eigenvalues()[k1][is];
            auto x = omega / (t_to_ryd * etemp);
            dndt_out[ik][is] = std::pow(1.0 / (2.0 * sinh(0.5 * x)), 2) * x / etemp;
        }
    }
}

void Iterativebte::average_scalar_degenerate_at_k(int k1, double *&val)
{
    double *tmp_q;
    double *tmp_omega;
    allocate(tmp_q, ns);
    allocate(tmp_omega, ns);

    const auto tol_omega = 1.0e-7; // Approximately equal to 0.01 cm^{-1}
    int s1, s2;

    for (s1 = 0; s1 < ns; ++s1) tmp_omega[s1] = dos->dymat_dos->get_eigenvalues()[k1][s1];

    for (s1 = 0; s1 < ns; ++s1) {
        double sum_scalar = 0.0;
        int n_deg = 0;
        for (s2 = 0; s2 < ns; ++s2) {
            if (std::abs(tmp_omega[s2] - tmp_omega[s1]) < tol_omega) {
                sum_scalar += val[s2];
                n_deg += 1;
            }
        }
        tmp_q[s1] = sum_scalar / static_cast<double>(n_deg);
    }

    
    for (s1 = 0; s1 < ns; ++s1) val[s1] = tmp_q[s1];

    deallocate(tmp_q);
    deallocate(tmp_omega);
}

void Iterativebte::average_vector_degenerate_at_k(int k1, double **&val)
{
    double *tmp_omega;
    allocate(tmp_omega, ns);
    double **tmp_W;
    allocate(tmp_W, ns, 3);

    const auto tol_omega = 1.0e-7; // Approximately equal to 0.01 cm^{-1}
    int s1, s2;

    for (s1 = 0; s1 < ns; ++s1) tmp_omega[s1] = dos->dymat_dos->get_eigenvalues()[k1][s1];

    for (s1 = 0; s1 < ns; ++s1) {
        double sum_vector[3];
        for (auto i = 0; i < 3; ++i) sum_vector[i] = 0.0;
        int n_deg = 0;
        for (s2 = 0; s2 < ns; ++s2) {
            if (std::abs(tmp_omega[s2] - tmp_omega[s1]) < tol_omega) {
                for (auto i = 0; i < 3; ++i) {
                    sum_vector[i] += val[s2][i];
                }
                n_deg += 1;
            }
        }
        for (auto i = 0; i < 3; ++i) {
            tmp_W[s1][i] = sum_vector[i] / static_cast<double>(n_deg);
        }
    }

    for (s1 = 0; s1 < ns; ++s1) {
        for (auto i = 0; i < 3; ++i) {
            val[s1][i] = tmp_W[s1][i];
        }
    }

    deallocate(tmp_W);
    deallocate(tmp_omega);
}

void Iterativebte::calc_kappa(int itemp, double ***&df, double **&kappa_out)
{
    auto etemp = Temperature[itemp];
    double omega;
    double beta = 1.0 / (thermodynamics->T_to_Ryd * etemp);
    double **tmpkappa;
    allocate(tmpkappa, 3, 3);

    for (auto ix = 0; ix < 3; ++ix) {
        for (auto iy = 0; iy < 3; ++iy) {
            tmpkappa[ix][iy] = 0.0;
        }
    }

    const double conversionfactor = Ryd / (time_ry * Bohr_in_Angstrom * 1.0e-10 * nk_3ph * system->volume_p);

    for (auto k1 = 0; k1 < nk_3ph; ++k1) {
        for (auto s1 = 0; s1 < ns; ++s1) {

            omega = dos->dymat_dos->get_eigenvalues()[k1][s1];  
            double n1 = thermodynamics->fB(omega, etemp);
            double factor = beta * omega * n1 * (n1 + 1.0);

            for (auto ix = 0; ix < 3; ++ix) {
                for (auto iy = 0; iy < 3; ++iy) {
                    tmpkappa[ix][iy] += - factor * vel[k1][s1][ix] * df[k1][s1][iy];
                    // df in unit bohr/K
                }
            }
        }
    }

    for (auto ix = 0; ix < 3; ++ix) {
        for (auto iy = 0; iy < 3; ++iy) {
            kappa_out[ix][iy] = tmpkappa[ix][iy] * conversionfactor;
        }
    }

    deallocate(tmpkappa);
}
