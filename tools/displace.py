#!/usr/bin/env python
#
# displace.py
#
# Simple script to generate input files of given displacement patterns.
# Currently, VASP, Quantum-ESPRESSO, LAMMPS, OpenMX, and xTAPP are supported.
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

"""
Input file generator for displaced configurations.
"""

from __future__ import print_function
import argparse
import numpy as np
import interface.VASP as vasp
import interface.QE as qe
import interface.xTAPP as xtapp
import interface.OpenMX as openmx
import interface.LAMMPS as lammps

parser = argparse.ArgumentParser()

parser.add_argument('--mag',
                    type=float, default=0.02,
                    help="Magnitude of displacement in units of \
                        Angstrom (default: 0.02)")

parser.add_argument('--prefix',
                    type=str, default="disp",
                    help="Prefix of the files to be created. (default: disp)")

parser.add_argument('--QE',
                    metavar='supercell.pw.in',
                    help="Quantum-ESPRESSO input file with equilibrium atomic positions (default: None)")

parser.add_argument('--VASP',
                    metavar='SPOSCAR',
                    help="VASP POSCAR file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('--xTAPP',
                    metavar='supercell.cg',
                    help="xTAPP CG file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('--LAMMPS',
                    metavar='supercell.lammps',
                    help="LAMMPS structure file with equilibrium atomic positions (default: None)")

parser.add_argument('--OpenMX',
                    metavar='supercell.dat',
                    help="dat file with equilibrium atomic \
                        positions (default: None)")

parser.add_argument('pattern_file', metavar='prefix.pattern_*', type=str, nargs='+',
                    help="ALM pattern file(s) generated with MODE = suggest")

parser.add_argument('--random', action="store_true", dest="random", default=False,
                    help="Generate randomly-displaced structures.")

parser.add_argument('--temp', type=float,
                    help="Target temperature of the random distribution of \
                        Q (default: 300). Used if --MD is not given.")

parser.add_argument('--start', type=int, default=1,
                    help="Specify where to start using the data. Used if --MD is given.\
                         (default: 1)")

parser.add_argument('--end', type=int, default=None,
                    help="Specify where to finish using the data. Used if --MD is given. "
                         " (default: None)")

parser.add_argument('-e', '--every', type=int, default=50,
                    help="Specify the interval of data sampling. Used if --MD is given. "
                         "(default: 50)")

parser.add_argument('-md', '--load_mddata', type=str,
                    help="Specify the file containing displacements of MD trajectories.")

parser.add_argument('--prim', type=str,
                    help="Specify the file containing structure data of the primitive lattice.")

parser.add_argument('--evec', type=str,
                    help="Specify the file containing harmonic eigenvalues and eigenvectors.")

parser.add_argument('-nd', '--num_disp', type=int,
                    help="Specify the number of displacement patterns.")

parser.add_argument('-cl', '--classical', action="store_true", dest="classical", default=False,
                     help="Use classical expectation value for <Q^2>.")

parser.add_argument('--print', action="store_true", dest="print_disp", default=False,
                    help="Print displacements to stdout")

parser.add_argument('--pes', help="Specify the target mode to compute PES.")
parser.add_argument('--Qrange', help='Range of normal coordinate Q in units of eV/Ang.')


def parse_displacement_patterns(files_in):

    pattern = []

    for file in files_in:
        pattern_tmp = []

        f = open(file, 'r')
        tmp, basis = f.readline().rstrip().split(':')
        if basis == 'F':
            print("Warning: DBASIS must be 'C'")
            exit(1)

        while True:
            line = f.readline()

            if not line:
                break

            line_split_by_colon = line.rstrip().split(':')
            is_entry = len(line_split_by_colon) == 2

            if is_entry:
                pattern_set = []
                natom_move = int(line_split_by_colon[1])
                for i in range(natom_move):
                    disp = []
                    line = f.readline()
                    line_split = line.rstrip().split()
                    disp.append(int(line_split[0]))
                    for j in range(3):
                        disp.append(float(line_split[j + 1]))

                    pattern_set.append(disp)
                pattern_tmp.append(pattern_set)

        print("File %s containts %i displacement patterns"
              % (file, len(pattern_tmp)))

        for entry in pattern_tmp:
            if entry not in pattern:
                pattern.append(entry)

        f.close()

    print("Number of unique displacement patterns = %d" % len(pattern))

    return pattern


def char_xyz(entry):

    if entry % 3 == 0:
        return 'x'
    if entry % 3 == 1:
        return 'y'
    if entry % 3 == 2:
        return 'z'


def gen_displacement(counter_in, pattern, disp_mag, nat, invlavec):

    poscar_header = "Disp. Num. %i" % counter_in
    poscar_header += " ( %f Angstrom" % disp_mag

    disp = np.zeros((nat, 3))

    for displace in pattern:
        atom = displace[0] - 1

        poscar_header += ", %i : " % displace[0]

        str_direction = ""

        for i in range(3):
            if abs(displace[i + 1]) > 1.0e-10:
                if displace[i + 1] > 0.0:
                    str_direction += "+" + char_xyz(i)
                else:
                    str_direction += "-" + char_xyz(i)

            disp[atom][i] += displace[i + 1] * disp_mag

        poscar_header += str_direction

    poscar_header += ")"

    if invlavec is not None:
        for i in range(nat):
            disp[i] = np.dot(disp[i], invlavec.T)

    return poscar_header, disp


def get_number_of_zerofill(npattern):

    nzero = 1

    while True:
        npattern //= 10

        if npattern == 0:
            break

        nzero += 1

    return nzero


def check_options(args):

    conditions = [args.VASP is None,
                  args.QE is None,
                  args.xTAPP is None,
                  args.LAMMPS is None,
                  args.OpenMX is None]

    if conditions.count(True) == len(conditions):
        print(
            "Error : Either --VASP, --QE, --xTAPP, --LAMMPS, --OpenMX option must be given.")
        exit(1)

    elif len(conditions) - conditions.count(True) > 1:
        print("Error : --VASP, --QE, --xTAPP, --LAMMPS, and --OpenMX cannot be given simultaneously.")
        exit(1)

    elif args.VASP:
        code = "VASP"
        struct_format = "VASP POSCAR"
        str_outfiles = "%s{counter}.POSCAR" % args.prefix
        file_original = args.VASP
        suffix = None

    elif args.QE:
        code = "QE"
        struct_format = "Quantum-ESPRESSO pw.in format"
        str_outfiles = "%s{counter}.pw.in" % args.prefix
        file_original = args.QE
        suffix = "pw.in"

    elif args.xTAPP:
        code = "xTAPP"
        struct_format = "xTAPP cg format"
        str_outfiles = "%s{counter}.cg" % args.prefix
        file_original = args.xTAPP
        suffix = None

    elif args.LAMMPS:
        code = "LAMMPS"
        struct_format = "LAMMPS structure format"
        str_outfiles = "%s{counter}.lammps" % args.prefix
        file_original = args.LAMMPS
        suffix = None

    elif args.OpenMX:
        code = "OpenMX"
        struct_format = "OpenMX dat format"
        str_outfiles = "%s{counter}.dat" % args.prefix
        file_original = args.OpenMX
        suffix = None

    return code, file_original, struct_format, str_outfiles, suffix


def get_original_parameter_set(code, file_original):
    # Read the original file

    param_dic = {}

    if code == "VASP":
        aa, aa_inv, elems, nats, x_frac = vasp.read_POSCAR(file_original)
        nat = np.sum(nats)

        param_dic['nat'] = nat
        param_dic['lavec'] = aa
        param_dic['invlavec'] = aa_inv
        param_dic['elems'] = elems
        param_dic['nats'] = nats
        param_dic['x_frac'] = x_frac

    elif code == "QE":
        list_namelist, list_ATOMIC_SPECIES, \
            list_K_POINTS, list_CELL_PARAMETERS, list_OCCUPATIONS, \
            nat, lavec, kd_symbol, x_frac, aa_inv = qe.read_original_QE(
                file_original)

        param_dic['nat'] = nat
        param_dic['lavec'] = lavec
        param_dic['invlavec'] = aa_inv
        param_dic['kd_symbol'] = kd_symbol
        param_dic['x_frac'] = x_frac
        param_dic['namelist'] = list_namelist
        param_dic['ATOMIC_SPECIES'] = list_ATOMIC_SPECIES
        param_dic['K_POINTS'] = list_K_POINTS
        param_dic['CELL_PARAMETERS'] = list_CELL_PARAMETERS
        param_dic['OCCUPATIONS'] = list_OCCUPATIONS

    elif code == "xTAPP":
        str_header, nat, nkd, aa, aa_inv, x_frac, kd \
            = xtapp.read_CG_file(file_original)

        param_dic['nat'] = nat
        param_dic['lavec'] = aa
        param_dic['invlavec'] = aa_inv
        param_dic['x_frac'] = x_frac
        param_dic['kd'] = kd
        param_dic['nkd'] = nkd
        param_dic['str_header'] = str_header

    elif code == "LAMMPS":
        common_settings, nat, x_cart, kd, charge \
            = lammps.read_lammps_structure(file_original)
#        aa_inv = None

        param_dic['nat'] = nat
        param_dic['kd'] = kd
        param_dic['charge'] = charge
        param_dic['common_settings'] = common_settings
        param_dic['x_cart'] = x_cart
        param_dic['lavec'] = None
        param_dic['invlavec'] = None

    elif code == "OpenMX":
        aa, aa_inv, nat, x_frac = openmx.read_OpenMX_input(file_original)

        param_dic['lavec'] = aa
        param_dic['invlavec'] = aa_inv
        param_dic['nat'] = nat
        param_dic['x_frac'] = x_frac

    return param_dic


def generate_finite_displacements(file_pattern, nat, aa_inv):

    header_list = []
    disp_list = []
    disp_pattern = parse_displacement_patterns(file_pattern)
    counter = 1

    for pattern in disp_pattern:
        header, disp = gen_displacement(counter, pattern,
                                        disp_length,
                                        nat, aa_inv)
        counter += 1
        header_list.append(header)
        disp_list.append(disp)

    return header_list, disp_list


def write_displaced_structures(params_orig, header_list, disp_list):

    nzerofills = get_number_of_zerofill(len(disp_list))

    counter = 1
    for header, disp in zip(header_list, disp_list):
        if code == "VASP":
            vasp.generate_input(prefix, counter, header,
                                disp, nzerofills, params_orig)

        elif code == "QE":
            qe.generate_input(prefix, counter,
                              disp, nzerofills, params_orig)

        elif code == "xTAPP":
            xtapp.generate_input(prefix, counter,
                                 disp, nzerofills, params_orig)

        elif code == "LAMMPS":
            lammps.generate_input(prefix, counter, header,
                                  disp, nzerofills, params_orig)

        elif code == "OpenMX":
            openmx.generate_input(prefix, counter,
                                  disp, nzerofills, params_orig, file_original)

        counter += 1


if __name__ == '__main__':

    print("*****************************************************************")
    print("    displace.py --  Generator of displaced configurations        ")
    print("*****************************************************************")
    print("")

    args = parser.parse_args()
    file_pattern = args.pattern_file

    code, file_original, struct_format, str_outfiles, suffix = check_options(args)
    disp_length = args.mag
    prefix = args.prefix
    #params = get_original_parameter_set(code, file_original)
    #structure = vasp.VaspParser()
    #structure = xtapp.xTappParser()
    #structure = qe.QEParser()
    #structure = lammps.LammpsParser()
    structure = openmx.OpenmxParser()
    structure.load_initial_structure(file_original)

    print(" Output format                  : %s" % struct_format)
    print(" Structure before displacements : %s" % file_original)
    print(" Output file names              : %s" % str_outfiles)
    print(" Magnitude of displacements     : %s Angstrom" % disp_length)
    #print(" Number of atoms                : %i" % params['nat'])
    print("")
    print("-----------------------------------------------------------------")

#    header_list, disp_list = generate_finite_displacements(file_pattern, params['nat'], params['invlavec'])
    header_list, disp_list = generate_finite_displacements(file_pattern,
                                                           structure.nat,
                                                           structure.inverse_lattice_vector)

    #nzerofills = get_number_of_zerofill(len(disp_list))

    #write_displaced_structures(params, header_list, disp_list)

    structure.generate_structures(prefix, header_list, disp_list)

    print("")
    print("All input files are created.")
