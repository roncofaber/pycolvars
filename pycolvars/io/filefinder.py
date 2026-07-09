#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:17:38 2025

@author: roncofaber
"""

def count_lammps_keywords(file_path):
    lammps_keywords = {
        "units", "atom_style", "pair_style", "read_data", "read_restart",
        "pair_coeff", "mass", "velocity", "fix", "run", "minimize",
        "timestep", "thermo", "dump", "log", "boundary", "region",
        "create_box", "create_atoms", "lattice", "neighbor", "neigh_modify",
        "comm_modify", "group", "compute", "variable", "thermo_style",
        "thermo_modify", "write_restart", "write_data"
    }
    
    count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()
                count += sum(1 for word in words if word in lammps_keywords)
        return count
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def count_colvars_keywords(file_path):
    colvars_keywords = {
        "colvarsTrajFrequency", "colvarsRestartFrequency", "colvars",
        "colvar", "distance", "angle", "dihedral", "rmsd", "hBond",
        "coordNum", "orientation", "orientationAngle", "spinAngle",
        "spin", "torsion", "path", "cylinder", "group1", "group2",
        "centers", "lowerBoundary", "upperBoundary", "width", "colvarsConfig",
        "harmonicWalls", "distanceZ", "main", "ref", "forceNoPBC", "axis",
        "hillWeight", "newHillFrequency", "hillWidth", "useGrids", "rebinGrids",
        "multipleReplicas", "replicasRegistry", "replicaUpdateFrequency",
        "replicaID", "writeFreeEnergyFile", "writePartialFreeEnergyFile"
    }
    
    count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                words = line.split()
                count += sum(1 for word in words if word in colvars_keywords)
        return count
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def find_most_likely_colvars_file(file_paths):
    max_colvars_count = 0
    most_likely_file = None
    
    for file_path in file_paths:
        colvars_count = count_colvars_keywords(file_path)
        if colvars_count > max_colvars_count:
            max_colvars_count = colvars_count
            most_likely_file = file_path
    
    if most_likely_file:
        return most_likely_file
    else:
        raise "None of the files are likely to be COLVARS input files."
