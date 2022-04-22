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
    vel = nullptr;
    damping4 = nullptr;
    rta_damping_loc = nullptr;
}

Iterativebte::~Iterativebte()
{
    if (Temperature) deallocate(Temperature);
    if (kappa) deallocate(kappa);
    if (vel) deallocate(vel);
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

void Iterativebte::prepare_data()
{
    if (mympi->my_rank == 0) {
        std::cout << std::endl << " Preparing input data for LBTE" << std::endl;
    }
    prepare_group_vel();
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
        allocate(bq, nklocal, ns, 3);
        allocate(fq, nklocal, ns, 3);

        calc_righthandside(itemp, bq);

        double **n1overtau = nullptr;
        if (has_4ph_damping || has_rta_damping) {
            allocate(n1overtau, nklocal, ns); 
            calc_n1overtau(itemp, n1overtau);
        }
        
        solve_bte(itemp, bq, n1overtau, fq);
        calc_kappa(itemp, fq, kappa[itemp]);

        deallocate(bq);
        deallocate(fq);
        if (has_4ph_damping || has_rta_damping) {
            deallocate(n1overtau);
        }

    } // itemp
}

void Iterativebte::solve_bte(int itemp, double ***&bq, double **&n1overtau, double ***&fq)
{
    if (solution_method == 0) {
        naive_iteration(itemp, bq, n1overtau, fq);
    } else if (solution_method == 1) {
        symmetry_iteration(itemp, bq, n1overtau, fq);
    }
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
            n1overtau[ik][is] = n0[k1][is] * (n0[k1][is] + 1.0) * 2 * tau[ik][is];
        }
    }

    deallocate(n0);
    deallocate(tau);
}

// helper functions

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

void Iterativebte::write_kappa_iterative()
{
    if (mympi->my_rank == 0) {

        auto file_kappa = input->job_title + ".kl_iter";

        std::ofstream ofs_kl;

        ofs_kl.open(file_kappa.c_str(), std::ios::out);
        if (!ofs_kl) exit("write_kappa_iterative", "Could not open file_kappa");

        ofs_kl << "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]" << std::endl;
        ofs_kl << "# Iterative result." << std::endl;

        if (isotope->include_isotope) ofs_kl << "# Isotope effects are included." << std::endl;
        if (conductivity->fph_rta > 0) ofs_kl << "# 4ph is included non-iteratively." << std::endl;
        if (conductivity->len_boundary > eps) {
                ofs_kl << "# Size of boundary " << std::scientific << std::setprecision(2) 
                                    << conductivity->len_boundary * 1e9 << " [nm]" << std::endl;
        }

        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                   << Temperature[itemp];
            for (auto ix = 0; ix < 3; ++ix) {
                for (auto iy = 0; iy < 3; ++iy) {
                    ofs_kl << std::setw(15) << std::scientific
                           << std::setprecision(4) << kappa[itemp][ix][iy];
                }
            }
            ofs_kl << std::endl;
        }
        ofs_kl.close();
        std::cout << std::endl;
        std::cout << " -----------------------------------------------------------------" << std::endl << std::endl;
        std::cout << " Lattice thermal conductivity is stored in the file " << file_kappa << std::endl;
    }
}

void Iterativebte::naive_iteration(int itemp, double ***&bq, double **&n1overtau, double ***&fq)
{
    double ***residual;
    double ***fq_old;
    allocate(residual, nklocal, ns, 3);
    allocate(fq_old, nklocal, ns, 3);

    MatrixA A(this);
    // Maybe I need to use friend, instead of inherit from pointer.
    A.setup_matrix(nk_l, nklocal); // this need to move to outside temperature loop
    A.set_temperature(itemp);
    for (auto ik = 0; ik < nklocal; ik++) {
        for (auto is = 0; is < ns; is++) {
            for (auto ix = 0; ix < 3; ix++) {
                fq_old[ik][is][ix] = 0.0;
                residual[ik][is][ix] = 0.0;
            }
        }
    }

    int step = 0;
    double norm_b = residual_norm(bq);
    double norm_r = residual_norm(residual);
    bool converged = (norm_r / norm_b) < convergence_criteria;

    // TODO should fq has length nk_3ph?
    while (!converged && step < max_cycle){
        for (auto k1 = 0; k1 < nklocal; k1++) {
            for (auto s1 = 0; s1 < ns; s1++) {
                auto one_over_p = n1overtau[k1][s1] + A.P(k1 * ns + s1);
                for (auto ix = 0; ix < 3; ix ++) {
                    fq[k1][s1][ix] = fq_old[k1][s1][ix] + one_over_p * residual[k1][s1][ix];
                }
            }
        }

    }


    deallocate(residual);
    deallocate(fq_old);
}

double Iterativebte::residual_norm(double ***&r)
{
    double sum = 0.0;
    auto each_squared = local_residual_sum_squared(r);
    MPI_Allreduce(&each_squared, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::pow(sum, 0.5);
}

double Iterativebte::local_residual_sum_squared(double ***&r)
{
    double norm = 0.0;
    for (auto ik = 0; ik < nklocal; ik++) {
        for (auto is = 0; is < ns; is++) {
            for (auto ix = 0; ix < 3; ix++) {
                norm += std::pow( r[ik][is][ix], 2.0);
            }
        }
    } 
    return norm;
}