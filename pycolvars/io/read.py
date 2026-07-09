#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:22:01 2025

@author: roncofaber
"""

def read_colvars_input_file(input_file):
    with open(input_file, "r") as fin:
    
        # we are not reading anything
        read_metadyn = False
        read_walls   = False
    
        other_args = []
    
        # iterate through file
        for line in fin:
            cline = line.strip()
    
            # check what we are reading
            if cline.startswith("metadynamics"):
                read_metadyn = True
            if cline.startswith("harmonicWalls"):
                read_walls   = True
                new_other_args = dict()
    
            # this one we need!
            if read_metadyn:
                if cline.strip().startswith("name"):
                    metadyn_name = cline.split()[1]
                if cline.startswith("colvars"):
                    colvars_name = cline.split()[1:]
                if cline.startswith("}"):
                    read_metadyn = False
    
            # all those optional
            if read_walls:
                if cline.strip().startswith("name"):
                    new_other_args["wall_name"]  = cline.split()[1]
                if cline.startswith("colvars"):
                    new_other_args["wall_colvar"] = cline.split()[1:]
                if cline.startswith("upperWalls"):
                    new_other_args["upper_wall"] = cline.split()[1:]
                if cline.startswith("lowerWalls"):
                    new_other_args["lower_wall"] = cline.split()[1:]
                if cline.startswith("upperWallConstant"):
                    new_other_args["wall_constant"] = cline.split()[1:]
                if cline.startswith("}"):
                    other_args.append(new_other_args)
                    read_walls = False
    
    return metadyn_name, colvars_name, other_args
