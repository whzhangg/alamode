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

void Iterativebte::write_result()
{
    // write Q and W for all phonon, only phonon in irreducible BZ is written
    // restart of iterative calculation is not clear
    int i;
    int nk_ir = dos->kmesh_dos->nk_irred;
    double Ry_to_kayser = Hz_to_kayser / time_ry;

    if (mympi->my_rank == 0) {
        std::cout << " Prepare result file ..." << std::endl;

        fs_result.open(conductivity->get_filename_results(3).c_str(), std::ios::out);

        if (!fs_result) {
            exit("setup_result_io",
                 "Could not open file_result3");
        }

        fs_result << "## General information" << std::endl;
        fs_result << "#SYSTEM" << std::endl;
        fs_result << system->natmin << " " << system->nkd << std::endl;
        fs_result << system->volume_p << std::endl;
        fs_result << "#END SYSTEM" << std::endl;

        fs_result << "#KPOINT" << std::endl;
        fs_result << dos->kmesh_dos->nk_i[0] << " " << dos->kmesh_dos->nk_i[1] << " " << dos->kmesh_dos->nk_i[2]
                          << std::endl;
        fs_result << dos->kmesh_dos->nk_irred << std::endl;

        for (int i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
            fs_result << std::setw(6) << i + 1 << ":";
            for (int j = 0; j < 3; ++j) {
                fs_result << std::setw(15)
                                  << std::scientific << dos->kmesh_dos->kpoint_irred_all[i][0].kval[j];
            }
            fs_result << std::setw(12)
                              << std::fixed << dos->kmesh_dos->weight_k[i] << std::endl;
        }
        fs_result.unsetf(std::ios::fixed);

        fs_result << "#END KPOINT" << std::endl;

        fs_result << "#CLASSICAL" << std::endl;
        fs_result << thermodynamics->classical << std::endl;
        fs_result << "#END CLASSICAL" << std::endl;

        fs_result << "#FCSXML" << std::endl;
        fs_result << fcs_phonon->file_fcs << std::endl;
        fs_result << "#END  FCSXML" << std::endl;

        fs_result << "#SMEARING" << std::endl;
        fs_result << integration->ismear << std::endl;
        fs_result << integration->epsilon * Ry_to_kayser << std::endl;
        fs_result << "#END SMEARING" << std::endl;

        fs_result << "#TEMPERATURE" << std::endl;
        fs_result << system->Tmin << " " << system->Tmax << " " << system->dT << std::endl;
        fs_result << "#END TEMPERATURE" << std::endl;

        fs_result << "##END General information" << std::endl;

        fs_result << "##Phonon Frequency" << std::endl;
        fs_result << "#K-point (irreducible), Branch, Omega (cm^-1), Group velocity (m/s)" << std::endl;

        double factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;
        for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
            const int ik = dos->kmesh_dos->kpoint_irred_all[i][0].knum;
            for (auto is = 0; is < dynamical->neval; ++is) {
                fs_result << std::setw(6) << i + 1 << std::setw(6) << is + 1;
                fs_result << std::setw(15) << writes->in_kayser(dos->dymat_dos->get_eigenvalues()[ik][is]);
                fs_result << std::setw(15) << vel[ik][is][0] * factor
                                  << std::setw(15) << vel[ik][is][1] * factor
                                  << std::setw(15) << vel[ik][is][2] * factor << std::endl;
            }
        }

        fs_result << "##END Phonon Frequency" << std::endl << std::endl;
        fs_result << "##Q and W at each temperature" << std::endl;
    }
}

void Iterativebte::write_Q_dF(int itemp, double **&q, double ***&df)
{
    auto etemp = Temperature[itemp];

    auto nk_ir = dos->kmesh_dos->nk_irred;
    double **Q_tmp;
    double **Q_all;
    allocate(Q_all, nk_ir, ns);
    allocate(Q_tmp, nk_ir, ns);
    for (auto ik = 0; ik < nk_ir; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            Q_all[ik][is] = 0.0;
            Q_tmp[ik][is] = 0.0;
        }
    }
    for (auto ik = 0; ik < nklocal; ++ik) {
        auto tmpk = nk_l[ik];
        for (auto is = 0; is < ns; ++is) {
            Q_tmp[tmpk][is] = q[ik][is];
        }
    }
    MPI_Allreduce(&Q_tmp[0][0], &Q_all[0][0],
                  nk_ir * ns, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    deallocate(Q_tmp);

    // now we have Q
    if (mympi->my_rank == 0) {
        fs_result << std::setw(10) << etemp << std::endl;

        for (auto ik = 0; ik < nk_ir; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                auto k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
                fs_result << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << std::endl;
                fs_result
                      << std::setw(15) << std::scientific << std::setprecision(5) << Q_all[ik][is]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][0]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][1]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][2] << std::endl;
            }
        }
        fs_result << std::endl;
    }
    deallocate(Q_all);
}
