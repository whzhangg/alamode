#!/usr/bin/env python
#
# genPOSCAR.py
#
# Simple script to generate POSCAR files for given displacement patterns
#
# Copyright (c) 2014 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory 
# or http://opensource.org/licenses/mit-license.php for information.
#
"""
This python script generates POSCAR files compatible with 
the VASP code for the given displacement patterns.
"""
import sys, os
import optparse
import numpy as np

usage = "usage: %prog [options] file.pattern_HARMONIC file.pattern_ANHARM3 ... \n \
      file.pattern_* can be generated by 'alm' with MODE = suggest."
parser = optparse.OptionParser(usage=usage)
parser.add_option('-m', '--magnitude', help="Magnitude of displacement in units of Angstrom (default: 0.02)")
parser.add_option('-p', '--prefix', help="Prefix of the files to be created (default: None)")
parser.add_option('-i', '--input', help="Original POSCAR file with equilibrium atomic positions (default: POSCAR)")


def read_POSCAR(file_in):
    file_pos = open(file_in, 'r')
    
    str_tmp = file_pos.readline()
    a = float(file_pos.readline().rstrip())
    lavec = np.zeros((3,3))

    for i in range(3):
        arr = file_pos.readline().rstrip().split()
        if len(arr) != 3:
            print "Could not read POSCAR properly"
            exit(1)

        for j in range(3):
            lavec[i,j] = a * float(arr[j])

    lavec = np.matrix(lavec).transpose()
    invlavec = lavec.I

    elements = file_pos.readline().rstrip().split()
    nat_elem = [int(tmp) for tmp in file_pos.readline().rstrip().split()]

    nat = np.sum(nat_elem)
    basis = file_pos.readline().rstrip()
    x = np.zeros((nat, 3))

    for i in range(nat):
        arr = file_pos.readline().rstrip().split()
        for j in range(3):
            x[i,j] = float(arr[j])

    if basis == "Direct" or basis == "direct" or basis == "D" or basis == "d":
        xf = np.matrix(x)
    else:
        xf = np.matrix(x)
        for i in range(nat):
            xf[i,:] = xf[i,:] * invlavec.transpose()

    file_pos.close()
    return lavec, invlavec, elements, nat_elem, xf


def parse_displacement_patterns(files_in):

	pattern = []

	for file in files_in:
		pattern_tmp = []

		f = open(file, 'r')

		tmp, basis = f.readline().rstrip().split(':')

		if basis == 'F':
			print "Warning: DBASIS must be 'C'"
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


		print "File %s containts %i displacement patterns" % (file, len(pattern_tmp))


		for entry in pattern_tmp:
			if not entry in pattern:
				pattern.append(entry)

		f.close()

	print "Number of unique displacement patterns = ", len(pattern)

	return pattern

def char_xyz(entry):
	if entry % 3 == 0:
		return 'x'
	elif entry % 3 == 1:
		return 'y'
	elif entry % 3 == 2:
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

	disp = np.matrix(disp)

	for i in range(nat):
		disp[i,:] = disp[i,:] * invlavec.transpose()

	return poscar_header, disp


def gen_POSCAR(prefix, counter, header, nzerofills, lavec, elems, nat, disp, coord):
    filename = prefix + "POSCAR." + str(counter).zfill(nzerofills)
    f = open(filename, 'w')
    f.write("%s\n" % header)
    f.write("%s\n" % "1.0")
    for i in range(3):
        f.write("%20.15f %20.15f %20.15f\n" % (lavec[0,i], lavec[1,i], lavec[2,i]))
    for i in range(len(elems)):
        f.write("%s " % elems[i])
    f.write("\n")
    for i in range(len(nat)):
        f.write("%d " % nat[i])
    f.write("\n")
    f.write("Direct\n")

    for i in range(len(disp)):
        for j in range(3):
            f.write("%20.15f" % (coord[i,j] + disp[i,j])) 
        f.write("\n")
    f.close()

def get_number_of_zerofill(npattern):

	nzero = 1

	while True:
		npattern /= 10

		if npattern == 0:
			break

		nzero += 1

	return nzero


if __name__ == '__main__':
	options, args = parser.parse_args()
	file_pattern = args[0:]

	if len(file_pattern) == 0:
		print "Usage: genPOSCAR.py [options] file1.pattern_HARMONIC file2.pattern_ANHARM3 ..."
		print "file.pattern_* can be generated by 'alm' with MODE = suggest."
		print 
		print "For details of available options, please type\n$ python genPOSCAR.py -h"
		exit(1)

	if options.prefix == None:
		print "--prefix option not given"
		print "Output format of POSCAR.{number} will be used."
		prefix = ""

	else:
		print "Prefix of the output files : ", options.prefix
		print "Output format of %s.POSCAR.{number} will be used." % options.prefix
		prefix = options.prefix + "."
       
	if options.magnitude == None:
		disp_length = 0.02
		print "--magnitude option not given."

	else:
		disp_length = float(options.magnitude)
	
	print "Atomic displacement of %f angstrom will be used." % disp_length

	if options.input == None:
		fname_poscar_orig = "POSCAR"
		print "--input option not given"
	else:
		fname_poscar_orig = options.input

	print "Original atomic positions will be read from the file %s in the working directory." % fname_poscar_orig
	print

	aa, aa_inv, elems, nats, x_frac = read_POSCAR(fname_poscar_orig)

	print "Equilibrium atomic positions were successfully read from the file %s" % fname_poscar_orig
	print "Number of atoms : ", np.sum(nats)

	disp_pattern = parse_displacement_patterns(args[:])

	nzerofills = get_number_of_zerofill(len(disp_pattern))

	counter = 0
	for pattern in disp_pattern:
		counter += 1
		header, disp = gen_displacement(counter, pattern, disp_length, np.sum(nats), aa_inv)
		gen_POSCAR(prefix, counter, header, nzerofills, aa, elems, nats, disp, x_frac)

	print "POSCAR files are created."

		

	





