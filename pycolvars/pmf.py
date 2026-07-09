#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:02:47 2021

Set of functions to simplify the life of a colvar scientist.

@author: asanzmatias, roncofaber
"""

# functions to import

# data manipulation
import numpy as np
from scipy.signal import savgol_filter

# os and stuff
import os
import glob
import dill
import warnings
import itertools

# MPS
import pycolvars.min_energy_path as mep
import pycolvars._utilities as aux
from .plotting.plot_functions import plot_colvars_data
from .io.filefinder import find_most_likely_colvars_file
from .io.read import read_colvars_input_file

# plot
import matplotlib.pyplot as plt

from ase import units

kcal_2_kt = (units.kcal/units.mol)/(298*units.kB)

#%% MAIN OBJECT
class PMF_obj():

    def __init__(self):

        return

    # def __getattr__(self, name):
    #     ''' will only get called for undefined attributes '''
    #     warnings.warn('No member "%s" contained in settings config.' % name)
    #     return ''

    # save object as pkl file
    def save_object(self, oname=None):

        if oname is None:
            oname = self.savename

        if oname.endswith(".pkl"):
            oname =  self.pmf_path + "/" + oname
        else:
            oname = self.pmf_path + "/" + oname.split(".")[-1] + ".pkl"

        with open(oname, 'wb') as fout:
            dill.dump(self, fout)

        print("Saved everything as {}".format(oname))
        return

    # restart object from pkl file previously saved
    def restart_pmf_from_pickle(self, filein, pkl_type):

        # open previously generated gpw file
        with open(filein, "rb") as fin:
            restart = dill.load(fin)

        self.__dict__ = restart.__dict__.copy()

        assert hasattr(self, "colvars_pmf_check"), "Cannot load non compatible object!"

        assert self.colvars_pmf_check == pkl_type, "Wrong PMF? Check single"\
            "or multiple replica!"

        return

    # read a N ndimal pmf file
    def read_colvars_data(self, pmf_file, simple=False, assign=False):
        
        # read raw PMF data
        raw_data = np.loadtxt(pmf_file)
        
        raw_data[:,-1] = kcal_2_kt*raw_data[:,-1] #convert in kT

        if simple:
            return raw_data

        # read pmf file in detail
        with open(pmf_file, "r") as fin:
            line = fin.readline()

            ndim = int(line.strip().split()[1])

            ax_size   = []
            ax_bounds = []
            ax_width  = []
            for cc in range(ndim):
                line = fin.readline()

                sline = line.strip().split()

                cmin   = float(sline[1])
                dcoord = float(sline[2])
                size   = int(sline[3])
                cmax   = cmin + size*dcoord

                ax_size.append(size)
                ax_bounds.append([cmin, cmax])
                ax_width.append(dcoord)

        # generate axes
        axes = []
        for dim in range(ndim):
            axes.append(np.unique(raw_data[:,dim]))

        # generate main pmf
        main_pmf = np.resize(raw_data[:,-1], ax_size)

        if assign:
            self._main_pmf = main_pmf
            self.axes      = axes
            self.ndim      = ndim
            self.ax_size   = ax_size
            self.ax_bounds = ax_bounds
            self.ax_width  = ax_width
            self.raw_data  = raw_data
            self.grid      = raw_data[:,:ndim]

        else:
            return main_pmf, raw_data, ndim, axes, ax_size, ax_bounds, ax_width

    # read input from a .lmp/.inp file!
    def read_colvars_input(self, input_file, assign=False):

        metadyn_name, colvars_name, other_args = read_colvars_input_file(input_file)

        if assign:
            self.metadyn_name = metadyn_name
            self.colvars_name = colvars_name
            self.other_args   = other_args
        else:
            return metadyn_name, colvars_name, other_args

    # find colvars input file given the proper path
    def find_colvars_input_file(self, pmf_path):

        # get all relevant files
        types = ('/*.lmp', '/*.inp') # the tuple of file types
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(pmf_path + files))

        colvars_input = find_most_likely_colvars_file(files_grabbed)

        self.input_file = colvars_input

        return

    def average_pmf_files(self, pmf_path, nframes=20, every=1,
                          make_nan=True):

        files_list =  glob.glob(pmf_path + "/*.pmf")
        files_list.sort(key=lambda x: os.path.getctime(x))

        # check that no partial pmf are in the list
        new_list = []
        for file in files_list:
            if "partial" not in file:
                new_list.append(file)
        files_list = new_list

        # check that there are pmfs, if any...
        if len(files_list) <= 1:
            if len(files_list) == 0:
                print("No pmf file found - abort ")
            else:
                print("Single pmf file found - ignore averaging ")

            return [], self.main_pmf, self.raw_data[:,-1]

        pmf_list = []
        raw_datas = []
        for file in files_list[-nframes*every::every]:
            
            raw_data = np.loadtxt(file)
            raw_datas.append(kcal_2_kt*raw_data[:,-1]) #convert in kT
            
            pmf_data = np.resize(kcal_2_kt*raw_data[:,-1], self.ax_size)
            
            if make_nan:
                pmf_data[pmf_data >= pmf_data.max()] = np.NaN

            pmf_list.append(pmf_data)

        return pmf_list, np.mean(pmf_list, axis=0), np.mean(raw_datas, axis=0)

    def initialize_MPS_analysis(self, force=False):

        if self.mps_initialized and not force:
            print("MPS already initialized - skipped")
            return

        #load a surface file to a map_handler instance
        map_handler = mep.surface_handler(self.raw_data)

        #Check if the file has been sucessfully read
        if map_handler.good:
            #map_handler method to compute -- takes lot of time
            map_handler.get_global_network()

        self.map_handler     = map_handler
        self.mps_initialized = True

        return

    def get_MPS_path(self,
                     origin,
                     target,
                     npaths = 0,
                     ):

        # check MPS was already done
        assert self.mps_initialized, "MPS not initialized - perform initialization first!"

        #Select initial and final points from the pmf (minima closer to the points that you give):
        ori_idx = self.map_handler.select_origin_by_range(origin, True)
        tar_idx = self.map_handler.select_target_by_range(target, True)

        assert isinstance(ori_idx, int), "No minima found in the origin range!"
        assert isinstance(tar_idx, int), "No minima found in the target range!"

        # get paths until "npaths"
        self.map_handler.get_npaths(npaths)

        # get path - copyed from their function but does not save shit
        trace  = (self.map_handler.propagation.gnw_paths[npaths])[7]
        points = np.argwhere(trace != 0).flatten()
        order  = np.argsort(trace[points]).flatten()
        points = points[order]

        path = np.column_stack((points,
                                self.map_handler.coords[points],
                                self.map_handler.energy[points]))

        return path, ori_idx, tar_idx

    def get_bias(self, divide=1, grad=False):

        V_bias = (np.max(self.main_pmf) - self.main_pmf)/(kcal_2_kt*divide)

        if not grad:
            return V_bias

        else:
            V_grad = []
            for cc, width in enumerate(self.ax_width):
                V_grad.append(savgol_filter(V_bias, 7, 1, deriv=1, delta=width,
                              axis=cc))
            V_grad = [np.reshape(ii, self.ax_size) for ii in V_grad]
            # V_grad = np.gradient(V_bias, *self.ax_width)

            return V_bias, V_grad

    def get_free_energy(self, coord=None):
        
        if coord is None:
            return self.main_pmf

        idxs = []
        for cc, axis in enumerate(self.axes):
            idx, __ = aux.find_nearest(axis, coord[cc])
            idxs.append(idx)

        return self.main_pmf[tuple(idxs)]


    def write_state_file(self, fname="restart.python.state",
                         step=0, repID=None, divide=1):

        lower = np.array(self.ax_bounds)[:,0]
        upper = np.array(self.ax_bounds)[:,1]

        V_bias, V_grad = self.get_bias(grad=True, divide=divide)

        V_bias = V_bias.flatten()

        V_grad = [ii.flatten() for ii in V_grad]
        V_grad = list(itertools.chain.from_iterable(zip(*V_grad)))

        header0 = [
            "configuration {\n",
            "  step    {}\n".format(step),
            "  #dt 1.000000e+00\n",
            "  #version 2020-01-27\n"
            "}\n\n",
            ]

        header1 = [
            "metadynamics {\n",
            "  configuration {\n",
            "    step {:d}\n".format(step),
            "    name {}\n".format(self.metadyn_name),
            "    replicaID {}\n".format(repID) if repID is not None else "",
            "  }\n",
            "  hills_energy\n",
            "grid_parameters {\n",
            "  n_colvars {}\n".format(self.ndim),
            "  lower_boundaries {}".format(aux.list2string(lower)),
            "  upper_boundaries {}".format(aux.list2string(upper)),
            "  widths {}".format(aux.list2string(self.ax_width)),
            "  sizes {}".format(aux.list2string(self.ax_size, "int")),
            "}\n",
            ]

        header2 = [
            "  hills_energy_gradients\n",
            "grid_parameters {\n",
            "  n_colvars {}\n".format(self.ndim),
            "  lower_boundaries {}".format(aux.list2string(lower)),
            "  upper_boundaries {}".format(aux.list2string(upper)),
            "  widths {}".format(aux.list2string(self.ax_width)),
            "  sizes {}".format(aux.list2string(self.ax_size, "int")),
            "}\n",
            ]

        with open(self.pmf_path + "/" + fname, "w") as fout:
            for line in header0:
                fout.write(line)
            for cname in self.colvars_name:
                fout.write("colvar {\n")
                fout.write("  name {}\n".format(cname))
                fout.write("  #x  SOMETHING\n")
                fout.write("}\n\n")
            for other_arg in self.other_args:
                if "wall_name" in other_arg.keys():
                    fout.write("restraint {\n")
                    fout.write("  configuration {\n")
                    fout.write("  step {}\n".format(step))
                    fout.write("  name {}\n".format(other_arg["wall_name"]))
                    fout.write("  }\n")
                    fout.write("}\n\n")
            for line in header1:
                fout.write(line)
            for argument in aux.grouper(V_bias, 3):
                fout.write(aux.list2string(argument))
            for line in header2:
                fout.write(line)
            for argument in aux.grouper(V_grad, 3):
                fout.write(aux.list2string(argument))
            fout.write("}\n\n")
            # fout.write("")

        return

    def get_1D_slice(self, locs, axis, pmf_data=None):
        
        if pmf_data is None:
            pmf_data = self.main_pmf
         
        # make it a list
        locs = [locs] if type(locs) is not list else locs

        indexes = []
        values  = []
        for loc in locs:
            idx, val = aux.find_nearest(self.axes[axis], loc)

            indexes.append(idx)
            values.append(val)

        slices = []
        for idx in indexes:
            slices.append(pmf_data.take(idx, axis=axis))


        return np.squeeze(slices), values

    def plot_colvars_data(self,
                          pmf_data   = None,
                          axes       = None,
                          fixed_axis = None,
                          slider     = False,

                          xlabel    = None,
                          ylabel    = None,
                          title     = None,
                          yshift    = 0,
                          savename  = None,
                          vmax      = None,
                          contours  = True,
                          con_steps = 5,
                          path      = None, # add path from mps
                          
                          squared   = False,
                          
                          cmap = "viridis",
                          hillshade = False,
                          ):


        plot_colvars_data(self,
                        pmf_data   = pmf_data,
                        axes       = axes,
                        fixed_axis = fixed_axis,
                        slider     = slider,

                        xlabel     = xlabel,
                        ylabel     = ylabel,
                        title      = title,
                        yshift     = yshift,
                        savename   = savename,
                        vmax       = vmax,
                        contours   = contours,
                        con_steps  = con_steps,
                        path       = path,
                        
                        squared    = squared,
                        
                        cmap       = cmap,
                        hillshade = hillshade
                        )

        return
    
    def get_harmonic_potential(self, axis, eps0, k=1, width=1):
        
        def harmonic(eps, eps0, k=1, width=1):
            return kcal_2_kt*0.5*k*((eps-eps0)/width)**2
        
        return harmonic(self.axes[axis], eps0, k=k, width=width)

    @property
    def main_pmf(self):
        return self._main_pmf


# end of object

#%%
class SingleReplica(PMF_obj):

    def __init__(self,
                 pmf_path,   # path to pmf_file
                 pmf_name    = None, # name of pmf file - if want to average just put one of them
                 average_pmf = False, # average multiple pmf files, if found
                 nframes     = 20,    # take only last nframes files to average
                 do_mps      = False, # perform MPS analysis initalization
                 force_mps   = False, # force to redo MPS even if already done
                 save        = False,
                 savename    = "pmf_object.pkl"
                 ):

        self.savename = savename

        # get name of pmf file
        self.pmf_path = pmf_path
        
        if pmf_name is None:
            pmf_candidates = glob.glob(pmf_path + "/*.pmf")
            
            pmf_candidates = [ii for ii in pmf_candidates if "partial" not in ii]
            
            pmf_name = max(pmf_candidates, key=os.path.getctime).split("/")[-1]
            
        self.pmf_name = pmf_name
        self.pmf_file = pmf_path + "/" + pmf_name

        # check if it's a new wfs file or a already processed pkl file
        if self.pmf_name.endswith(".pmf"):
            # read COLVARS data
            self.initialize_single_replica(average_pmf, nframes)

        # restart from pickle file
        elif self.pmf_name.endswith(".pkl"):
            self.restart_pmf_from_pickle(self.pmf_file, "single")


        # do mps analysis to get barrier heights
        if do_mps:
            self.initialize_MPS_analysis(force=force_mps)


        # save self as picke file
        if save:
            self.save_object(savename)

        return

    # initialize PMF object from scratch
    def initialize_single_replica(self, average_pmf, nframes):

        # initialize variables
        self.map_handler     = None
        self.mps_initialized = False

        self.colvars_pmf_check = "single" # dyummy attribute to make sure same class

        # read pmf data and assign variables
        self.read_colvars_data(self.pmf_file, assign=True)

        # find and read input file
        self.find_colvars_input_file(self.pmf_path)
        self.read_colvars_input(self.input_file, assign=True)


        # do pmf averaging - update main_pmf and raw_data
        if average_pmf:
            pmf_list, av_pmf, av_raw_data = self.average_pmf_files(self.pmf_path, nframes)

            self.pmf_list       = pmf_list
            self.main_pmf       = av_pmf
            self.raw_data[:,-1] = av_raw_data

        return



class MultipleReplica(PMF_obj):

    def __init__(self,
                 pmf_path,   # path to pmf_file
                 pmf_name    = "*.pmf", # name of pmf file - if want to average just put one of them
                 do_mps      = False, # perform MPS analysis initalization
                 force_mps   = False, # force to redo MPS even if already done
                 save        = False,
                 savename    = "pmf_mr_object.pkl",
                 average_pmf = True
                 ):

        self.savename  = savename
        self.pmf_path  = pmf_path
        self.pmf_name  = pmf_name

        # check if it's a new wfs file or a already processed pkl file
        # restart from pickle file
        if pmf_name.endswith(".pkl"):
            self.restart_pmf_from_pickle(self.pmf_file, "multiple")
        elif pmf_name.endswith(".pmf"):
            # read COLVARS data
            self.initialize_multiple_replica(pmf_path, pmf_name, average_pmf)

        # do mps analysis to get barrier heights
        if do_mps:
            self.initialize_MPS_analysis(force=force_mps)


        # save self as picke file
        if save:
            self.save_object(savename)

        return

    # initialize the MR object by finding relevant stuff
    def initialize_multiple_replica(self, pmf_path, pmf_name, average_pmf):

        # initialize variables
        self.map_handler     = None
        self.mps_initialized = False

        self.colvars_pmf_check = "multiple" # dyummy attribute to make sure same class
        
        # find best pmf candidate        
        # get name of pmf file
        if pmf_name is None:
            pmf_name = "/*.pmf"
        
        # get possible pmf files
        pmf_candidates = glob.glob(pmf_path + "*/*.pmf")
        pmf_candidates = [ii for ii in pmf_candidates if "partial" not in ii]
        # Sort by modification time (oldest first)
        pmf_candidates.sort(key=os.path.getmtime)  

        # check long enough
        assert len(pmf_candidates) > 0, "Are you in the right directory?"

        # assign stuff
        self.mr_dirs = np.unique([sdir.split("/")[-2] for sdir in pmf_candidates])

        # get latest pmf and use that one to read
        self.pmf_file = pmf_candidates[-1]
        self.pmf_name = self.pmf_file.split("/")[-1]
        self.main_dir = self.pmf_file.split("/")[-2]

        # read pmf data and assign variables
        self.read_colvars_data(self.pmf_file, assign=True)

        # find and read input file
        self.find_colvars_input_file(self.pmf_path + self.mr_dirs[0])
        self.read_colvars_input(self.input_file, assign=True)
        
        self._all_pmfs = None
        if average_pmf:
            assert len(pmf_candidates) >= len(self.mr_dirs), "FABO check this"
            
            all_pmfs = []
            for pmf_candidate in pmf_candidates:
                
                if self.main_dir in pmf_candidate:
                
                    raw_data  = self.read_colvars_data(pmf_candidate, simple=True)
                    all_pmfs.append(np.resize(raw_data[:,-1], self.ax_size))
                
            self._all_pmfs = np.array(all_pmfs)
            
        part_pmf = []
        for sdir in self.mr_dirs:

            partial_file = glob.glob(pmf_path + sdir + "/*.partial*.pmf")

            assert isinstance(partial_file, list), "Something went wrong "\
                "looking for the partial pmf files..."
            
            # get newest file
            partial_file = max(partial_file, key=os.path.getctime)

            raw_data  = self.read_colvars_data(partial_file, simple=True)
            part_pmf.append(np.resize(raw_data[:,-1], self.ax_size))

        self.part_pmf  = np.array(part_pmf)
        self.n_replica = len(self.part_pmf)

        return
    
    def plot_partial_pmf(self,
                         xlabel    = None,
                         ylabel    = None,
                         title     = None,
                         yshift    = 0,
                         savename  = None,
                         vmax      = None,
                         contours  = True,
                         con_steps = None,
                         ):

        assert self.colvars_pmf_check == "multiple", "This is not a MR calculation"


        # adapt values for
        if vmax is None:
            vmax = 40/self.n_replica
        if con_steps is None:
            con_steps = np.floor(5/np.sqrt(self.n_replica))


        # get good grid, maybe...
        n = int(np.floor(np.sqrt(self.n_replica)))
        m = int(np.ceil(self.n_replica/n))

        assert n*m >= self.n_replica, "Fabo you did a mistake here..."

        # generate figure
        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(11,9),
                                constrained_layout=True)

        for cc, part_pmf in enumerate(self.part_pmf):

            # get current axis
            cax = axs.flat[cc]

            # data to plot
            pivotted = part_pmf.T
            xrange   = self.ax_bounds[0]
            yrange   = self.ax_bounds[1]

            fs = 11

            # plot energy surface
            pos = cax.imshow(pivotted,
                             aspect="auto",
                             origin="lower",
                             extent=(xrange[0], xrange[-1], yrange[0]+yshift, yrange[-1]+yshift),
                             vmin=0,
                             vmax=vmax,
                             interpolation="none",
                             cmap="viridis")

            # plot contours
            if contours:

                levels = np.arange(0, vmax+1, con_steps)

                cax.contour(
                    pivotted,
                    levels = levels,
                    extent=(xrange[0], xrange[-1], yrange[0]+yshift, yrange[-1]+yshift),
                    colors='black',
                    linewidths=0.25
                    )

            # make nice stuff
            if cc%(m) == 0:
                cax.set_ylabel(ylabel, fontsize=fs, fontweight="bold")
            else:
                cax.set_yticklabels([])
            if cc >= m*(n-1):
                cax.set_xlabel(xlabel, fontsize=fs, fontweight="bold")
            else:
                cax.set_xticklabels([])
            if cc > 0 and (cc+1)%m == 0:
                # colorbar
                cbar = plt.colorbar(pos, ax=cax)
                cbar.set_label(r"$\Delta$G [kT]", fontsize=fs+1, fontweight="bold")

            # add grid
            cax.grid(linestyle="--", color="#888888")


        if title is not None:
            fig.suptitle(title, fontsize=fs+3, fontweight="bold")

        # fig.tight_layout()
        fig.show()

        return
    
    def get_pmfs(self, last=1, every=1, atol=1e-8, align=False):
        
        all_pmfs = self._all_pmfs[-last::every,:].copy()
        
        first_vals = all_pmfs[:, 0][:, np.newaxis]
        mask = np.isclose(all_pmfs, first_vals, atol=atol)
        
        
        all_pmfs[mask] = np.nan
        
        # mean_free_energy = all_pmfs.mean(axis=0)
        # sem_free_energy = all_pmfs.std(axis=0, ddof=1) #/ np.sqrt(all_pmfs.shape[0])
        # free_energy = mean_free_energy
        
        if align:
            means = np.nanmean(all_pmfs[:,125:180], axis=1, keepdims=True)
            all_pmfs = all_pmfs - means
        
        return all_pmfs
