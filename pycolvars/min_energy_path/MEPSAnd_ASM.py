import numpy as np
import pandas as pd
import multiprocessing as mp
import io, time, math, sys
import matplotlib as mpl
from matplotlib import cm
import _pickle as pickle
import tkinter as tki
from tkinter import filedialog, simpledialog, TRUE, FALSE, messagebox, N, S, W, E, END
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


network_dependencies = True
try:
    import igraph #conda install -c conda-forge python-igraph
    from fa2 import ForceAtlas2 #pip install fa2
    import cairo # conda install -c anaconda pycairo
except:
    network_dependencies = False

# #########################################################################
#     Copyright (C) 2019 Íñigo Marcos Alcalde                             #
# #########################################################################
#                               LICENSE                                  #
# #########################################################################
#     MEPSAnd is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
# #########################################################################
# #########################################################################
# Contact: pagomez@cbm.csic.es, eduardo.lopez@ufv.es and imarcos@cbm.csic.es
# #########################################################################
# Citation: Marcos-Alcalde, I., Lopez-Viñas, E. & Gómez-Puertas, P. (2020). MEPSAnd: Minimum Energy Path Surface Analysis over n-dimensional surfaces. Bioinformatics 36, 956–958.
# #########################################################################

class progress_reporter:
    def count(self,i):
        if (self.progress_reference[i] >= self.next_progress or i == 10):
            etas = int((time.time() - self.t0)/(i+1) * (self.niter - (i+1))+0.5)
            etam = int(etas/60)
            etah = int(etam/60)
            etad = int(etah/24)
            etas -= (etam*60)
            etam -= (etah*60)
            etah -= (etad*24)
            if (etad > 0):
                time_str = str(etad) + " d " + str(etah) + " h"
            elif (etah > 0):
                time_str = str(etah) + " h " + str(etam) + " m"
            elif (etam > 0):
                time_str = str(etam) + " m " + str(etas) + " s"
            else:
                time_str = str(etas) + " s"
            if (self.fp > 0):
                new_percent = np.rint(100*(10**self.fp)*(i+1)/self.niter)/(10**self.fp)
            else:
                new_percent = int(np.rint(100*(i+1)/self.niter))
            print ("\r" + self.indent + str(new_percent) +
          " % | E.T.A. " + time_str + 12*" ", end="")
            sys.stdout.flush()
            self.next_progress = self.progress_reference[i] + self.progress_step
    def __init__(self, niter,progress_step = 1,fp=0,indent="\t"):
        self.indent = indent
        self.fp = fp
        self.niter = niter
        self.progress_reference = 100 * (np.arange(self.niter) / self.niter)
        self.progress_step = progress_step
        self.next_progress = progress_step
        self.t0 = time.time()



def reduce_dtype(inp_array):
    if (inp_array.shape[0] > 0):
        a = np.min(inp_array)
        b = np.max(inp_array)
        at = str(np.min_scalar_type(a))
        bt = str(np.min_scalar_type(b))
        ua = at[0] == "u"
        ub = bt[0] == "u"
        fa = "float" in at
        fb = "float" in bt
        ia = 0
        if (ua):
            ia += 1
        if (fa):
            ia += 5
        else:
            ia += 3
        na = int(at[ia:])
        ib = 0
        if (ub):
            ib += 1
        if (fb):
            ib += 5
        else:
            ib += 3
        nb = int(bt[ib:])
        u = ua and ub
        f = fa or fb
        n = max(na,nb)
        if (ua != ub):
            ct = str(np.min_scalar_type(-b))
            uc = ct[0] == "u"
            fc = "float" in ct
            ic = 0
            if (uc): #this could be assumed as true
                ic += 1
            if (fc):
                ic += 5
            else:
                ic += 3
            nc = int(ct[ic:])
            n = max(n,nc)
        dtype = ""
        if (u and not f):
            dtype += "u"
        if (f):
            dtype += "float"
        else:
            dtype += "int"
        tmp_dtype = dtype + str(n)
        while np.any((inp_array.astype(tmp_dtype)-inp_array)!=0):
            n *= 2
            tmp_dtype = dtype + str(n)
        return inp_array.astype(tmp_dtype)
    else:
        return inp_array


def get_periodic_point_neighbors(point,coords,con_cut_up, con_cut_down,state_cut_up, state_cut_down,a,b,c,d,indices,sum_reference, con_cut_down_1, con_cut_down_2, con_cut_up_1, con_cut_up_2, state_cut_down_1, state_cut_down_2, state_cut_up_1, state_cut_up_2, diagonals = False):
    point_coords = coords[point,:]
    np.logical_and(point_coords[0] <= con_cut_up[:,0], point_coords[0] >= con_cut_down[:,0],out=a)
    np.logical_and(point_coords[0] <= con_cut_up_1[:,0], point_coords[0] >= con_cut_down_1[:,0],out=b)
    np.logical_and(point_coords[0] <= con_cut_up_2[:,0], point_coords[0] >= con_cut_down_2[:,0],out=c)
    np.logical_or(a,b,out=d)
    np.logical_or(c,d,out=a)
    a[point] = False
    d = indices[a]
    for x in range(1,point_coords.shape[0]):
        e1 = np.logical_and(point_coords[x] <= con_cut_up[d,x], point_coords[x] >= con_cut_down[d,x])
        e2 = np.logical_and(point_coords[x] <= con_cut_up_1[d,x], point_coords[x] >= con_cut_down_1[d,x])
        e3 = np.logical_and(point_coords[x] <= con_cut_up_2[d,x], point_coords[x] >= con_cut_down_2[d,x])
        d = d[np.logical_or(np.logical_or(e1,e2),e3)]

        f1 = np.logical_and(point_coords<=state_cut_up[d], point_coords>=state_cut_down[d])
        f2 = np.logical_and(point_coords<=state_cut_up_1[d], point_coords>=state_cut_down_1[d])
        f3 = np.logical_and(point_coords<=state_cut_up_2[d], point_coords>=state_cut_down_2[d])
        f4 = np.logical_or(f1,f2)
        f5 = np.logical_or(f3,f4)
        f6 = np.sum(f5,axis=1)
    if (diagonals):
        return reduce_dtype(d), reduce_dtype(d[f6 >= sum_reference])
    else:
        return reduce_dtype(d)

def get_point_neighbors(point,coords,con_cut_up, con_cut_down,state_cut_up, state_cut_down,a,indices,sum_reference, diagonals = False):
    point_coords = coords[point,:]
    np.logical_and(point_coords[0] <= con_cut_up[:,0], point_coords[0] >= con_cut_down[:,0],out=a)
    a[point] = False
    d = indices[a]
    for x in range(1,point_coords.shape[0]):
        d = d[np.logical_and(point_coords[x] <= con_cut_up[d,x], point_coords[x] >= con_cut_down[d,x])]

        f = np.sum(np.logical_and(point_coords<=state_cut_up[d], point_coords>=state_cut_down[d]),axis=1)
    if (diagonals):
        return reduce_dtype(d), reduce_dtype(d[f >= sum_reference])
    else:
        return reduce_dtype(d[f >= sum_reference])

class connectivity_handler:
    def clean_neighbors(self):
        self.neighbors = []
        self.connectivity_done=False

######CUTOFF HANDLING
    def save_cutoffs(self, file_path):
        out = np.column_stack((self.state_cutoffs,self.edge_cutoffs))
        return np.savetxt(file_path, out, delimiter = "\t", fmt='% g')

    def load_periodicity(self, file_path):
        print ("Loading custom periodicity")
        df = pd.read_csv(file_path, sep='\s+', header = None)
        in_periodicity = (df.values).flatten()
#         print (in_periodicity)
        if (np.shape(in_periodicity)[0] == self.coords.shape[1] and len(np.shape(in_periodicity)) == 1):
            self.periodicity = reduce_dtype(np.absolute(in_periodicity))
            print ("Done")
        else:
            print ("Custom periodicity array has the wrong shape an will not be loaded")
            return False
        self.clean_neighbors()

    def save_periodicity(self, file_path):
        out = np.expand_dims(self.periodicity, axis=1)
        return np.savetxt(file_path, out, delimiter = "\t", fmt='% g')

    def get_def_edge_cutoffs(self):
        self.edge_cutoffs = np.zeros(np.shape(self.coords)[1])
        print ("Calculating default cutoffs.")
        for c in range(np.shape(self.coords)[1]):
            a = np.unique(self.coords[:,c])
            if a.shape[0] > 0:
                b = a[1:] - a[:-1]
                if (b.shape[0]>0):
                    self.edge_cutoffs[c] = (np.min(b)) * 1.5
                else:
                    self.edge_cutoffs[c] = 0
            else:
                self.edge_cutoffs[c] = 0
        #print("Default edge cutoffs: " + str(self.edge_cutoffs))
        self.con_cut_up = reduce_dtype(np.add(self.edge_cutoffs, self.coords).astype(np.float64))
        self.con_cut_down = reduce_dtype(np.add(-self.edge_cutoffs, self.coords).astype(np.float64))

    def load_cutoffs(self, cut_path):
        df=pd.read_csv(cut_path, sep='\s+',header=None)
        in_edge_cutoffs = df.values
        edge_good = self.load_custom_edge_cutoffs(np.array(in_edge_cutoffs))
        if (edge_good == False):
            print( "ERROR", "Cutoffs could not be read. Default will be used instead")


    def load_custom_edge_cutoffs(self, in_cutoffs):
        print ("Loading custom cutoffs")
        if (np.shape(in_cutoffs)[0] == self.coords.shape[1] and len(np.shape(in_cutoffs)) == 2) and np.shape(in_cutoffs)[1] == 2:
            self.state_cutoffs = reduce_dtype(in_cutoffs[:,0])
            self.edge_cutoffs = reduce_dtype(in_cutoffs[:,1])
        else:
            print ("Custom cutoffs array has the wrong shape an will not be loaded")
            self.get_def_edge_cutoffs()
            return False
        #print("Current edge cutoffs: " + str(self.edge_cutoffs))
        self.con_cut_up = reduce_dtype(np.add(self.edge_cutoffs, self.coords).astype(np.float64))
        self.con_cut_down = reduce_dtype(np.add(-self.edge_cutoffs, self.coords).astype(np.float64))
        self.state_cut_up = reduce_dtype(np.add(self.state_cutoffs, self.coords).astype(np.float64))
        self.state_cut_down = reduce_dtype(np.add(-self.state_cutoffs, self.coords).astype(np.float64))
        self.clean_neighbors()
        #self.get_def_state_cutoffs()
        return True

    def get_def_state_cutoffs(self):
        #print ("Calculating default node_cutoffs")
        self.state_cutoffs = (self.edge_cutoffs / 1.5) * 0.5
        #print("Default node_cutoffs: " + str(self.state_cutoffs))
        self.state_cut_up = reduce_dtype(np.add(self.state_cutoffs, self.coords).astype(np.float64))
        self.state_cut_down = reduce_dtype(np.add(-self.state_cutoffs, self.coords).astype(np.float64))

    def get_connectivity(self,points):
        neighbors = [None] * points.shape[0]
        if (self.diagonals):
            diag_neighbors = [None] * points.shape[0]
        indices = reduce_dtype(np.arange(self.coords.shape[0]))
        a = np.zeros(self.coords.shape[0], dtype=np.bool_)
        sum_reference=self.coords.shape[1] - 1

        progress = progress_reporter(points.shape[0],1,indent="\t\tComputing connectivity: ")
        for x in range(points.shape[0]):
            if (self.diagonals):
                diag_neighbors[x], neighbors[x] = get_point_neighbors(points[x],self.coords,self.con_cut_up, self.con_cut_down,self.state_cut_up, self.state_cut_down,a,indices,sum_reference,diagonals=self.diagonals)
            else:
                neighbors[x] = get_point_neighbors(points[x],self.coords,self.con_cut_up, self.con_cut_down,self.state_cut_up, self.state_cut_down,a,indices,sum_reference,diagonals=self.diagonals)
            progress.count(x)
        print ("Done")
        if (self.diagonals):
            return tuple(diag_neighbors), tuple(neighbors)
        else:
            return(tuple(neighbors))


    def get_periodic_connectivity(self,points):
        coords = np.copy(self.coords)
        min_coords = np.min(coords,axis=0)
        for i in np.argwhere(self.periodicity > 0).flatten():
            coords[:,i]-=min_coords[i]
            points_to_fix = np.argwhere(coords[:,i] > self.periodicity[i]).flatten()
            for point in points_to_fix:
                correction = int(coords[point,i] / self.periodicity[i])
                coords[point,i] = coords[point,i] - (correction * self.periodicity[i])

        periodicity_stack = np.repeat(np.expand_dims(self.periodicity,axis=0), self.coords.shape[0], axis=0)

        con_cut_up_1 = self.con_cut_up - periodicity_stack
        con_cut_up_2 = self.con_cut_up + periodicity_stack
        con_cut_down_1 = self.con_cut_down - periodicity_stack
        con_cut_down_2 = self.con_cut_down + periodicity_stack

        state_cut_up_1 = self.state_cut_up - periodicity_stack
        state_cut_up_2 = self.state_cut_up + periodicity_stack
        state_cut_down_1 = self.state_cut_down - periodicity_stack
        state_cut_down_2 = self.state_cut_down + periodicity_stack

        neighbors = [np.empty(0,dtype='int8')] * points.shape[0]
        if (self.diagonals):
            diag_neighbors = [np.empty(0,dtype='int8')] * points.shape[0]
        indices = np.arange(self.coords.shape[0])
        a = np.zeros(self.coords.shape[0], dtype=np.bool_)
        b = np.zeros(self.coords.shape[0], dtype=np.bool_)
        c = np.zeros(self.coords.shape[0], dtype=np.bool_)
        d = np.zeros(self.coords.shape[0], dtype=np.bool_)
        sum_reference=self.coords.shape[1] - 1

        progress = progress_reporter(points.shape[0],1,indent="\t\tComputing periodic connectivity: ")
        for x in range(points.shape[0]):
            if (self.diagonals):
                diag_neighbors[x], neighbors[x] = get_periodic_point_neighbors(points[x],self.coords,self.con_cut_up, self.con_cut_down,self.state_cut_up, self.state_cut_down,a,b,c,d,indices,sum_reference, con_cut_down_1, con_cut_down_2, con_cut_up_1, con_cut_up_2, state_cut_down_1, state_cut_down_2, state_cut_up_1, state_cut_up_2,diagonals=self.diagonals)
            else:
                neighbors[x] = get_periodic_point_neighbors(points[x],self.coords,self.con_cut_up, self.con_cut_down,self.state_cut_up, self.state_cut_down,a,b,c,d,indices,sum_reference, con_cut_down_1, con_cut_down_2, con_cut_up_1, con_cut_up_2, state_cut_down_1, state_cut_down_2, state_cut_up_1, state_cut_up_2,diagonals=self.diagonals)
            progress.count(x)
        print ("Done")
        if (self.diagonals):
            return tuple(diag_neighbors), tuple(neighbors)
        else:
            return(tuple(neighbors))


######GET CONNECTIVITY
    def compute_neighbors_from_grid_midpoint(self,points,grid_indices_raw,grid,midpoint_coords,diag=False,do_periodic=False):
        if (diag):
            mask = np.ones([3]*self.coords.shape[1],dtype=np.bool_)
            mask[midpoint_coords] = 0
        else:
            mask = np.zeros([3]*self.coords.shape[1],dtype=np.bool_)
            for i in range(self.coords.shape[1]):
                midpoint_coords[i][0] = 0
                mask[midpoint_coords] = 1
                midpoint_coords[i][0] = 2
                mask[midpoint_coords] = 1
                midpoint_coords[i][0] = 1
        neigh_indices_list = np.array((np.where(mask)))-1
        neigh_indices_list_ref = np.copy(neigh_indices_list)

        mask = np.ones(neigh_indices_list[0].shape[0],dtype=np.bool_)
        neighbors = [np.empty(0,dtype='int8')] * points.shape[0]
        for point in range(grid_indices_raw.shape[0]):
            indices = tuple(grid_indices_raw[point].tolist())
            mask.fill(True)
            for i in range(self.coords.shape[1]):
                neigh_indices_list[i] = neigh_indices_list_ref[i] + indices[i]
                if (do_periodic):
                    neigh_indices_list[neigh_indices_list == grid.shape[i]] = 0
                else:
                    mask[neigh_indices_list[i] == -1] = False
                    mask[neigh_indices_list[i] >= grid.shape[i]] = False
            if (np.any(mask)):
                neighbors[point] = grid[tuple(neigh_indices_list[:,mask].tolist())]
                neighbors[point] = neighbors[point][neighbors[point]!=-1]
        for i in range(len(neighbors)):
            if neighbors[i] is not None:
                neighbors[i] = reduce_dtype(neighbors[i])
        return tuple(neighbors)

    def get_grid_neighbors(self,diag=False,do_periodic=False):
        points = np.arange(self.coords.shape[0])
        mins = np.min(self.coords,axis=0)
        steps = np.zeros(self.coords.shape[1],dtype=np.float64)
        for i in range(self.coords.shape[1]):
            uniq_values = np.unique(self.coords[:,i])
            if(uniq_values.shape[0] > 1):
                steps[i] = np.min(uniq_values[1:]-uniq_values[:-1])
        grid_indices_raw = np.rint(((self.coords-mins)/steps)).astype(np.int_)
        grid_dims = np.max(grid_indices_raw, axis = 0)+1
        grid = np.empty(grid_dims,dtype = np.int_)
        grid.fill(-1)
        for i in range(self.coords.shape[0]):
            indices = tuple(grid_indices_raw[i].tolist())
            grid[indices] = i
        midpoint_coords = []
        for i in range(self.coords.shape[1]):
            midpoint_coords.append(np.ones(1,dtype=np.int_))
        midpoint_coords = tuple(midpoint_coords)
        if (diag):
            return self.compute_neighbors_from_grid_midpoint(points,grid_indices_raw,grid,midpoint_coords,diag=True,do_periodic=do_periodic),self.compute_neighbors_from_grid_midpoint(points,grid_indices_raw,grid,midpoint_coords,diag=False,do_periodic=do_periodic)
        else:
            return self.compute_neighbors_from_grid_midpoint(points,grid_indices_raw,grid,midpoint_coords,diag=False,do_periodic=do_periodic)


    def get_global_connectivity(self):
        if (self.connectivity_done==False):
            print("Computing neighbors")
            do_periodic = np.any(self.periodicity.flatten()!=0)
            if (self.assume_grid):
                if (self.diagonals):
                    self.diag_neighbors, self.neighbors = self.get_grid_neighbors(diag=True,do_periodic=do_periodic)
                else:
                    self.neighbors = self.get_grid_neighbors(diag=False,do_periodic=do_periodic)
                self.connectivity_done=True
            else:
                points = np.arange(self.coords.shape[0])
                if (do_periodic):
                    self.neighbors=self.get_periodic_connectivity(points)
                else:
                    if (self.diagonals):
                        self.diag_neighbors, self.neighbors=self.get_connectivity(points)
                    else:
                        self.neighbors=self.get_connectivity(points)
                for point in points:
                    if(self.neighbors[point].shape[0] == 0):
                        print("No neighbors found por point: " + str(point))
                self.connectivity_done=True
            if (self.connectivity_done):
                counts = np.zeros(len(self.neighbors), dtype = 'int64')
                for i in range(len(self.neighbors)):
                    c = 0
                    if (self.diagonals and self.diag_neighbors[i] is not None):
                        c += self.diag_neighbors[i].shape[0]
                    elif (self.neighbors[i] is not None):
                        c += self.neighbors[i].shape[0]
                    counts[c] += 1
                indices = np.argwhere(counts!=0).flatten()
                for i in indices:
                    print("\t" + str(counts[i]) + " points have " + str(i) + " neighbors")

######SAVE/LOAD CONNECTIVITY
    def save_connectivity(self, filepath):
        if (self.connectivity_done==True):
            sio = io.StringIO()
            for neighbors in self.neighbors:
                np.savetxt(sio,neighbors.reshape([1,neighbors.shape[0]]), fmt='%u')
        f = open(filepath,"w+")
        f.write(sio.getvalue())
        f.close()
    def load_connectivity(self, filepath):
        self.neighbors = []
        for line in open(filepath, 'r'):
            self.neighbors.append(reduce_dtype(np.array(line.strip().split(), dtype=np.int_)))
        if (len(self.neighbors) == self.coords.shape[0]):
            self.connectivity_done=True
        else:
            self.neighbors = []
            self.connectivity_done=False
            print("Connectivity file could not be read")
    def load_diag_connectivity(self, filepath):
        self.diag_neighbors = []
        for line in open(filepath, 'r'):
            self.diag_neighbors.append(reduce_dtype(np.array(line.strip().split(), dtype=np.int_)))
        if (len(self.diag_neighbors) == self.coords.shape[0]):
            self.diagonals=True
        else:
            self.diag_neighbors = []
            self.diagonals=False
            print("Diagonal connectivity file could not be read")
######INIT

    def __init__(self, coords, edge_cutoffs = np.empty(0),state_cutoffs = np.empty(0),periodicity = np.empty(0),diagonals=False,assume_grid=True):
        self.diagonals = diagonals
        self.assume_grid = assume_grid
        self.coords = coords
        self.clean_neighbors()
        self.periodicity=np.zeros(coords.shape[1])
        if(coords.shape[1] == periodicity.shape[0]):
            self.periodicity=periodicity
        if (np.shape(edge_cutoffs)[0] == 0):
            self.get_def_edge_cutoffs()
        else:
            self.load_custom_edge_cutoffs(edge_cutoffs)
        if (np.shape(state_cutoffs)[0] == 0):
            self.get_def_state_cutoffs()
        else:
            self.load_custom_state_cutoffs(state_cutoffs)

class graph_plot_handler:
    def __init__(self,energy, neighbors, clusters, clusters_energy, minima_indices, barrier_indices, minimum_clusters_location, barrier_clusters_location, barriers_as_edges = True):
        self.energy = energy
        self.neighbors = neighbors
        self.barriers_as_edges = barriers_as_edges
        self.clusters = clusters
        self.clusters_energy = clusters_energy
        self.minima_indices = minima_indices
        self.barrier_indices = barrier_indices
        self.minimum_clusters_location = minimum_clusters_location
        self.barrier_clusters_location = barrier_clusters_location
        self.layout_done = False
        self.create_graph()

    #def get_weights(self):
    #    tmp = np.array(self.G.es['energy'])
    #    tmp = tmp - np.nanmin(tmp)
    #    self.weights = 100 * (0.1 + tmp /np.nanmax(tmp)) / 1.1

    def create_graph(self):
        print("Creating graph")
        self.G = igraph.Graph()
        self.G.to_undirected()
        self.vertices_location = np.copy(self.minimum_clusters_location)
        if(self.barriers_as_edges):
            total_vertices = self.minima_indices.shape[0]
            self.G.add_vertices(total_vertices)
        else:
            total_vertices = self.minima_indices.shape[0]+self.barrier_indices.shape[0]

            self.G.add_vertices(total_vertices)
            barriers_to_add = np.argwhere(self.barrier_clusters_location!=-1).flatten()
            self.vertices_location[barriers_to_add] = self.barrier_clusters_location[barriers_to_add]

        edges_energy = []
        edges_to_add = []
        self.edge_minimum_clusters_1 = []
        self.edge_minimum_clusters_2 = []
        self.edge_barrier_clusters = []
        for barrier_index in self.barrier_indices:
            neighbors = self.neighbors[barrier_index]
            for i, mi in enumerate(neighbors[:-1]):
                for mj in neighbors[i+1:]:
                    if(self.barriers_as_edges):
                        self.edge_minimum_clusters_1.append(mi)
                        self.edge_minimum_clusters_2.append(mj)
                        self.edge_barrier_clusters.append(barrier_index)
                        edges_to_add.append((mi,mj))
                        edges_energy.append(self.clusters_energy[barrier_index])
                    else:
                        self.edge_minimum_clusters_1.append(mi)
                        self.edge_minimum_clusters_2.append(barrier_index)
                        self.edge_barrier_clusters.append(barrier_index)
                        edges_to_add.append((mi,barrier_index))
                        edges_energy.append(self.clusters_energy[barrier_index])

                        self.edge_minimum_clusters_1.append(mj)
                        self.edge_minimum_clusters_2.append(barrier_index)
                        self.edge_barrier_clusters.append(barrier_index)
                        edges_to_add.append((mj,barrier_index))
                        edges_energy.append(self.clusters_energy[barrier_index])


        self.edge_minimum_clusters_1 = np.array(self.edge_minimum_clusters_1,dtype=np.int_)
        self.edge_minimum_clusters_2 = np.array(self.edge_minimum_clusters_2,dtype=np.int_)
        self.edge_barrier_clusters = np.array(self.edge_barrier_clusters,dtype=np.int_)
        self.G.add_edges(edges_to_add)
        self.edges_energy = np.array(edges_energy)


        #ATTRIBUTES
        if (self.barriers_as_edges):
            node_points = self.minima_indices
        else:
            node_points = np.arange(self.clusters_energy.shape[0])

        self.G.vs['name'] = node_points
        self.G.vs['size'] = 20
        self.G.vs['color'] = 'red'
        self.G.vs['frame_color'] = 'black'
        self.G.vs['frame_width'] = 0
        self.G.vs['shape'] = 'circle'
        self.G.vs['label'] = node_points
        self.G.vs['label_dist'] = 0
        self.G.vs['label_color'] = 'black'
        self.G.vs['label_size'] = 0
        self.G.vs['label_angle'] = 0
        self.G.vs['order'] = node_points
        self.G.vs['alpha'] = 1
        self.G.vs['frame_alpha'] = 1
        self.G.vs['energy'] = self.clusters_energy[node_points]

        self.G.es['color'] = 'red'
        self.G.es['width'] = 10
        self.G.es['order'] = np.arange(self.edges_energy.shape[0])
        self.G.es['alpha'] = 1
        self.G.es['energy'] = self.edges_energy
#        self.get_weights()
#        self.G.es['weight'] = self.weights
        print("Done")
        print("Total vertices: " + str(self.G.vcount()))
        print("Total edges: " + str(self.G.ecount()))


    def compute_layout(self,edgeWeightInfluence=0,outboundAttractionDistribution=True,scalingRatio=1000,gravity=1,verbose=True,steps = 4000,resume=True):
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=outboundAttractionDistribution,
            edgeWeightInfluence=edgeWeightInfluence,
            scalingRatio=scalingRatio,
            gravity=gravity,
            verbose=verbose)
        if (self.layout_done == False or resume == False):
            self.layout = forceatlas2.forceatlas2_igraph_layout(self.G, pos=None, iterations=steps)
            self.layout_done = True
        else:
            self.layout = forceatlas2.forceatlas2_igraph_layout(self.G, pos=self.layout.coords, iterations=steps)

    def get_energy_ranges(self, energies, npoints):
        energy_range = np.linspace(start = np.nanmin(energies), stop = np.nanmax(energies), num = npoints+1, endpoint = True)
        up_energy_range = energy_range[1:]
        down_energy_range = energy_range[:-1]
        return up_energy_range, down_energy_range

    def assign_cmap_colors(self, points = None, minima = True, barriers = True, cmap_name = 'rainbow', nodes = True, edges = True, independent = False): # COMPROBAR QUE TODAS LAS COMBINACIONES FUNCIONAN BIEN
        print("Assigning color map")
        cmap = cm.get_cmap(cmap_name)
        ncolors = cmap.N
        visited_nodes = None
        if(nodes == True):
            visited_nodes = self.get_visited_nodes(points, minima = minima, barriers = barriers)
            if (visited_nodes.shape[0] > 0):
                nodes_energy = self.clusters_energy[visited_nodes]
                nodes_energy_minmax = np.array((np.min(nodes_energy),np.max(nodes_energy)))
            else:
                nodes = False
        if(edges == True):
            edge_ids = self.get_visited_edges(points)
            if (edge_ids.shape[0] > 0):
                barriers_energy = self.edges_energy[edge_ids]
                barriers_energy = barriers_energy[~np.isnan(barriers_energy)]
                barriers_energy_minmax = np.array((np.min(barriers_energy),np.max(barriers_energy)))
            else:
                edges = False
        if (nodes == True and edges == True and independent == False):
            mixed_energies = np.concatenate((nodes_energy_minmax,barriers_energy_minmax))
            up_energy_range_nodes, down_energy_range_nodes = self.get_energy_ranges(mixed_energies, ncolors)
            up_energy_range_edges = up_energy_range_nodes
            down_energy_range_edges = down_energy_range_nodes
        else:
            if(nodes == True):
                up_energy_range_nodes, down_energy_range_nodes = self.get_energy_ranges(nodes_energy_minmax, ncolors)
            elif(edges == True):
                up_energy_range_edges, down_energy_range_edges = self.get_energy_ranges(barriers_energy_minmax, ncolors)
        vcolors = self.G.vs['color']
        if (nodes == True):
            node_alphas = self.G.vs['alpha']
            for x in visited_nodes:
                node_energy = self.clusters_energy[x]
                color_index = ((np.argwhere(np.logical_and(node_energy >= down_energy_range_nodes, node_energy <= up_energy_range_nodes)))[-1])[0]
                rgba_color = list(cmap(color_index))
                rgba_color[3] = node_alphas[x]
                color = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                vcolors[x] = color
            self.G.vs['color'] = vcolors
        ecolors = self.G.es['color']
        if (edges == True):
            edge_alphas = self.G.es['alpha']
            for edge_id in edge_ids:
                barriers_energy = self.edges_energy[edge_id]
                color_index = ((np.argwhere(np.logical_and(barriers_energy >= down_energy_range_edges,barriers_energy <= up_energy_range_edges)))[-1])[0]
                rgba_color = list(cmap(color_index))
                rgba_color[3] = edge_alphas[edge_id]
                color = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                ecolors[edge_id] = color
            self.G.es['color'] = ecolors
        print("Done")

    def get_visited_nodes(self,points, minima = True, barriers = True):
        if (points is None):
            points = np.arange(self.energy.shape[0])
        visited_vertices = pd.unique(self.vertices_location[points])
        visited_vertices = visited_vertices[visited_vertices!=-1]

        if (self.barriers_as_edges == False and (minima == False or barriers == False)):
            visited_vertices_mask = np.zeros(visited_vertices.shape,dtype=np.bool_)
            if (minima == True):
                visited_vertices_mask[visited_vertices<self.minima_indices.shape[0]] = True
            if (barriers == True):
                visited_vertices_mask[visited_vertices>=self.minima_indices.shape[0]] = True
            visited_vertices = visited_vertices[visited_vertices_mask]
        return visited_vertices

    def get_visited_edges(self,points=None):
        if (points is None):
            points = np.arange(self.energy.shape[0])
        if (points.shape[0] == self.energy.shape[0]):
            return np.arange(self.G.ecount())

        visited_minima = pd.unique(self.vertices_location[points])
        visited_minima = visited_minima[visited_minima!=-1]
        visited_barriers = pd.unique(self.barrier_clusters_location[points])
        visited_barriers = visited_barriers[visited_barriers!=-1]

        visited_clusters_mask = np.zeros(self.clusters_energy.shape,dtype=np.bool_)

        visited_clusters_mask[visited_minima] = True
        visited_clusters_mask[visited_barriers] = True

        edge_minima_mask = np.logical_and(visited_clusters_mask[self.edge_minimum_clusters_1],visited_clusters_mask[self.edge_minimum_clusters_2])
        edge_barriers_mask = visited_clusters_mask[self.edge_barrier_clusters]

        return np.argwhere(np.logical_and(edge_minima_mask,edge_barriers_mask)).flatten()

    def set_node_indices_as_label(self, points, minima = False, barriers = False):
        visited_nodes = self.get_visited_nodes(points, minima = minima, barriers = barriers)
        if (visited_nodes.shape[0] > 0):
            vertices = self.G.vs.select(name_in=visited_nodes)
            vertices['label'] = visited_nodes

    def set_node_energy_as_label(self, points, minima = False, barriers = False):
        visited_nodes = self.get_visited_nodes(points, minima = minima, barriers = barriers)
        if (visited_nodes.shape[0] > 0):
            vertices = self.G.vs.select(name_in=visited_nodes)
            vertices['label'] = vertices['energy']

    def set_node_attributes(self, points = None, minima = False, barriers = False, color = None, size = None, alpha = None, frame_color = None, frame_width = None, frame_alpha = None, shape = None, label = None, label_dist = None, label_angle = None, label_color = None, label_size = None, bring_to = None, add_tags = None, remove_tags = None):
        visited_nodes = self.get_visited_nodes(points, minima = minima, barriers = barriers)
        if (visited_nodes.shape[0] > 0):
            vertices = self.G.vs.select(name_in=visited_nodes)
            if size is not None:
                vertices['size'] = size
            if alpha is not None:
                vertices['alpha'] = alpha
                if color is None:
                    colors = vertices['color']
                    alphas = vertices['alpha']
                    for i in range(len(colors)):
                        rgba_color = list(mpl.colors.to_rgba(colors[i]))
                        rgba_color[3] = alpha
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                    vertices['color'] = colors
            if color is not None:
                alphas = vertices['alpha']
                colors = vertices['color']
                rgba_color = list(mpl.colors.to_rgba(color))
                if alpha is None:
                    for i in range(len(alphas)):
                        rgba_color[3] = alphas[i]
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                else:
                    rgba_color[3] = alpha
                    colors = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                vertices['color'] = colors
            if add_tags is not None:
                for tag in add_tags:
                    if tag in self.G.vs:
                        vertices[tag] = True
                    else:
                        self.G.vs[tag] = np.zeros(self.node_points.shape,dtype = np.bool_).tolist()
                        vertices[tag] = True
            if remove_tags is not None:
                for tag in remove_tags:
                    if tag in self.G.vs:
                        vertices[tag] = False

            if frame_alpha is not None:
                vertices['frame_alpha'] = frame_alpha
                if frame_color is None:
                    colors = vertices['frame_color']
                    alphas = vertices['frame_alpha']
                    for i in range(len(colors)):
                        rgba_color = list(mpl.colors.to_rgba(colors[i]))
                        rgba_color[3] = frame_alpha
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                    vertices['frame_color'] = colors
            if frame_color is not None:
                alphas = vertices['frame_alpha']
                colors = vertices['frame_color']
                rgba_color = list(mpl.colors.to_rgba(frame_color))
                if frame_alpha is None:
                    for i in range(len(alphas)):
                        rgba_color[3] = alphas[i]
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                else:
                    rgba_color[3] = frame_alpha
                    colors = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                vertices['frame_color'] = colors

            if frame_width is not None:
                vertices['frame_width'] = frame_width

            if shape is not None:
                vertices['shape'] = shape

            if label is not None:
                vertices['label'] = label

            if label_dist is not None:
                vertices['label_dist'] = label_dist

            if label_color is not None:
                vertices['label_color'] = label_color

            if label_size is not None:
                vertices['label_size'] = label_size

            if label_angle is not None:
                vertices['label_angle'] = label_angle

            if bring_to is not None:
                if bring_to == "front":
                    tmp_order = np.argsort(self.G.vs['order'])
                    tmp_order[visited_nodes] = np.max(tmp_order)+1
                    self.G.vs['order'] = np.argsort(tmp_order)
                if bring_to == "back":
                    tmp_order = np.argsort(self.G.vs['order'])
                    tmp_order[visited_nodes] = np.min(tmp_order)-1
                    self.G.vs['order'] = np.argsort(tmp_order)

    def set_edge_attributes(self, points = None, color = None, width = None, alpha = None, bring_to = None, add_tags = None, remove_tags = None):
        edge_ids = self.get_visited_edges(points).tolist()
        if (len(edge_ids) > 0):
            if (width is not None):
                self.G.es[edge_ids]['width'] = width
            if alpha is not None:
                self.G.es[edge_ids]['alpha'] = alpha
                if color is None:
                    colors = self.G.es[edge_ids]['color']
                    alphas = self.G.es[edge_ids]['alpha']
                    for i in range(len(colors)):
                        rgba_color = list(mpl.colors.to_rgba(colors[i]))
                        rgba_color[3] = alpha
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                    self.G.es[edge_ids]['color'] = colors
            if color is not None:
                alphas = self.G.es[edge_ids]['alpha']
                colors = self.G.es[edge_ids]['color']
                rgba_color = list(mpl.colors.to_rgba(color))
                if alpha is None:
                    for i in range(len(alphas)):
                        rgba_color[3] = alphas[i]
                        colors[i] = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                else:
                    rgba_color[3] = alpha
                    colors = mpl.colors.to_hex(rgba_color,keep_alpha = True)
                self.G.es[edge_ids]['color'] = colors
            if add_tags is not None:
                for tag in add_tags:
                    if tag in self.G.vs:
                        self.G.es[edge_ids][tag] = True
                    else:
                        self.G.es[tag] = np.zeros(self.edge_reference.shape[0],dtype = np.bool_).tolist()
                        self.G.es[edge_ids][tag] = True
            if remove_tags is not None:
                for tag in remove_tags:
                    if tag in self.G.vs:
                        self.G.es[edge_ids][tag] = False
            if bring_to is not None:
                if bring_to == "front":
                    tmp_order = np.argsort(self.G.es['order'])
                    tmp_order[edge_ids] = np.max(tmp_order)+1
                    self.G.es['order'] = np.argsort(tmp_order)
                if bring_to == "back":
                    tmp_order = np.argsort(self.G.es['order'])
                    tmp_order[edge_ids] = np.min(tmp_order)-1
                    self.G.es['order'] = np.argsort(tmp_order)

    def get_r_g_b(self,colors):
        out_colors=np.zeros((len(colors),3))
        for i,color in enumerate(colors):
            rgb_color = (np.array(mpl.colors.to_rgb(color)) * 255).astype(np.int_)
            out_colors[i,0] = rgb_color[0]
            out_colors[i,1] = rgb_color[1]
            out_colors[i,2] = rgb_color[2]
        return out_colors

    def save_graph_to_graphml(self,file_name = None):
        if (file_name is not None):
            vcolor = self.G.vs['color'].copy()
            frame_color = self.G.vs['frame_color'].copy()
            label_color = self.G.vs['label_color'].copy()
            ecolor = self.G.vs['color'].copy()

            rgbs = self.get_r_g_b(self.G.vs['color'])
            self.G.vs['r'] = rgbs[:,0]
            self.G.vs['g'] = rgbs[:,1]
            self.G.vs['b'] = rgbs[:,2]
            rgbs = self.get_r_g_b(self.G.es['color'])
            self.G.es['r'] = rgbs[:,0]
            self.G.es['g'] = rgbs[:,1]
            self.G.es['b'] = rgbs[:,2]

            del self.G.vs['color']
            del self.G.vs['frame_color']
            del self.G.vs['label_color']
            del self.G.es['color']

            self.G.write_graphml(file_name)

            del self.G.vs['r']
            del self.G.vs['g']
            del self.G.vs['b']
            del self.G.es['r']
            del self.G.es['g']
            del self.G.es['b']

            self.G.vs['color'] = vcolor
            self.G.vs['frame_color'] = frame_color
            self.G.vs['label_color'] = label_color
            self.G.es['color'] = ecolor

    def save_plot_to_file(self,file_name = None, width = 4000, height = 4000, background = 'white'):
        if (file_name is not None):
            igraph.plot(self.G,bbox=(width,height), background = background,layout=self.layout,target = file_name,
                vertex_size=self.G.vs['size'],
                vertex_color=self.G.vs['color'],
                vertex_frame_color=self.G.vs['frame_color'],
                vertex_frame_width=self.G.vs['frame_width'],
                vertex_shape=self.G.vs['shape'],
                vertex_label=self.G.vs['label'],
                vertex_label_dist=self.G.vs['label_dist'],
                vertex_label_color=self.G.vs['label_color'],
                vertex_label_size=self.G.vs['label_size'],
                vertex_label_angle=self.G.vs['label_angle'],
                vertex_order=self.G.vs['order'],
                edge_color=self.G.es['color'],
                edge_width=self.G.es['width'],
                edge_order=self.G.es['order'],
                )

    def view_plot(self, width = 4000, height = 4000, background = 'white'):
        igraph.plot(self.G,bbox=(width,height), background = background,layout=self.layout,
            vertex_size=self.G.vs['size'],
            vertex_color=self.G.vs['color'],
            vertex_frame_color=self.G.vs['frame_color'],
            vertex_frame_width=self.G.vs['frame_width'],
            vertex_shape=self.G.vs['shape'],
            vertex_label=self.G.vs['label'],
            vertex_label_dist=self.G.vs['label_dist'],
            vertex_label_color=self.G.vs['label_color'],
            vertex_label_size=self.G.vs['label_size'],
            vertex_label_angle=self.G.vs['label_angle'],
            vertex_order=self.G.vs['order'],
            edge_color=self.G.es['color'],
            edge_width=self.G.es['width'],
            edge_order=self.G.es['order'],
            )


class surface_handler:
    def __init__(self, inputfile, edge_cutoffs = np.empty(0), diagonals = False, assume_grid = True, barriers_as_edges = True):
        print(inputfile)
        try:
#ASM
#            df=pd.read_csv(inputfile, sep='\s+',header=None, comment='#' )

#             df=pd.read_csv(inputfile, sep='\s+',comment='#' )
            data = inputfile #df.values
            self.coords = reduce_dtype(data[:,0:-1].astype(np.float64))
            uniq_coords, uniq_counts = np.unique(self.coords,axis = 0,return_counts=True)
            if (np.all(uniq_counts==1)):
                self.energy = reduce_dtype(data[:,-1])
                print ("Surface loaded:\n\tPoints: " + str(self.energy.shape[0]) + "\n\tDimensions: " + str(self.coords.shape[1]+1))
                for i in range(self.coords.shape[1]):
                    print("\tCoordinate " + str(i+1) + " (min,max): " + str((np.min(self.coords[:,i]),np.max(self.coords[:,i]))))
                print("\tEnergy (min,max): " + str((np.min(self.energy),np.max(self.energy))))
                self.diagonals = diagonals
                self.assume_grid = assume_grid
                self.barriers_as_edges = barriers_as_edges
                self.connectivity = connectivity_handler(self.coords, edge_cutoffs, diagonals = self.diagonals, assume_grid=self.assume_grid)
                self.propagation = propagation_handler(self.coords, self.energy, self.connectivity)
                self.good = True
                self.gnw_graph_done = False
            else:
                print("Surface could be parsed but has coordinate redudancy and will be rejected.")
                print("The offeding point coordinates are:")
                for i in np.argwhere(uniq_counts>1):
                    print(uniq_coords[i])
                self.good = False
        except:
            print("Surface could not be loaded")
            self.good = False

    def load_cutoffs(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.connectivity.load_cutoffs(filepath)
                self.connectivity.connectivity_done = False
                self.propagation_handler.gnw_done = False
    def save_cutoffs(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.connectivity.save_cutoffs(filepath)

    def load_peridiocity(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.connectivity.load_peridiocity(filepath)
                self.connectivity.connectivity_done = False
                self.propagation_handler.gnw_done = False

    def save_peridiocity(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.connectivity.save_peridiocity(filepath)

    def load_connectivity(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.connectivity.load_connectivity(filepath)
                self.connectivity.connectivity_done = False
                self.propagation_handler.gnw_done = False

    def save_connectivity(self,filepath=None):
        if(self.good):
            if (self.connectivity.connectivity_done):
                if (type(filepath) == str):
                    self.connectivity.save_connectivity(filepath)

    def get_global_network(self):
        if(self.good):
            self.propagation.get_gnw()

    def set_OT(self,OT=[None,None]):
        if(self.good):
            if (type(OT) == list):
                if (len(OT) == 2):
                    tmp_OT=[None,None]
                    if (type(OT[0]) == int):
                        tmp_OT[0] = np.array([OT[0]])
                    if (type(OT[1]) == int):
                        tmp_OT[1] = np.array([OT[1]])
                    self.propagation.set_OT(OT)

    def select_origin_by_range(self,coord_range, minimum = True):
        assert self.good
        #ASM
        if (isinstance(coord_range, np.ndarray) and isinstance(minimum,bool)):
            point = self.propagation.select_origin_by_range(coord_range, minimum = True)
            return point

    def select_target_by_range(self,coord_range, minimum = True):
        assert self.good
        #FR
        if (isinstance(coord_range, np.ndarray) and isinstance(minimum,bool)):
        #if (type(coord_range) == 'numpy.ndarray' and type(minimum) == bool):
            point = self.propagation.select_target_by_range(coord_range, minimum = True)
            return point

    def select_origin_by_minimum_id(self,index):
        if(self.good):
            if (type(index) == int):
                self.propagation.select_origin_by_minimum_id(index)

    def select_target_by_minimum_id(self,index):
        if(self.good):
            if (type(index) == int):
                self.propagation.select_target_by_minimum_id(index)

    def get_npaths(self,n=0):
        if(self.good):
            if (type(n) == int):
                self.propagation.get_npaths(n)

    def get_well_sampling(self):
        if(self.good):
            self.propagation.get_well_sampling()

    def save_minbar_clusters_to_txt(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.propagation.save_minbar_clusters_to_txt(filepath)

    def save_simplified_path_to_txt(self,filepath=None,n=0):
        if(self.good):
            if (type(filepath) == str and type(n) == int):
                out = self.propagation.save_simplified_path_to_txt(filepath,n)
                return out

    def save_fragmentwise_path_to_txt(self,filepath=None,n=0):
        if(self.good):
            if (type(filepath) == str and type(n) == int):
                self.propagation.save_fragmentwise_path_to_txt(filepath,n)

    def save_well_sampling_to_txt(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                self.propagation.save_well_sampling_to_txt(filepath)

    def save_session(self,filepath=None):
        if(self.good):
            if (type(filepath) == str):
                with open(filepath, 'wb') as output:
                    return pickle.dump(self, output, -1)

    def create_gnw_graph(self):
        if(self.good):
            self.propagation.get_gnw()
            self.gnw_graph = graph_plot_handler(self.energy, self.propagation.gnw_neighbors, self.propagation.gnw_clusters, self.propagation.gnw_energy, self.propagation.gnw_minimum_clusters_indices, self.propagation.gnw_barrier_clusters_indices, self.propagation.gnw_minimum_clusters_location, self.propagation.gnw_barrier_clusters_location)
            self.gnw_graph_done = True


def node_detection(energy, connectivity, flat_is_node = False, allowed_mask = None, indices=None):
    if (indices is None):
        if (allowed_mask is None):
            indices = reduce_dtype(np.arange(energy.shape[0]))
        else:
            indices = reduce_dtype(np.argwhere(allowed_mask == True).flatten())
    node_mask = np.zeros(np.shape(energy)[0], dtype = np.bool_)

    for p in indices:
        node_mask[p] = node_eval(p,energy, connectivity, flat_is_node, allowed_mask)
    return reduce_dtype(np.argwhere(node_mask == True).flatten())

def node_detection_mp_helper(args):
    return node_eval(args[0],args[1],args[2],args[3],args[4])


def node_eval(point, energy, connectivity, flat_is_node = False, allowed_mask = None):
    available_points = connectivity[point]
    if (allowed_mask is not None):
        available_points = available_points[allowed_mask[available_points]]
    if (available_points.shape[0] > 0):
        neighs_energies = energy[available_points]
        min_neigh = np.min(neighs_energies)
        if (energy[point] <= min_neigh):
            if(flat_is_node == True):
                return True
            elif(energy[point] != np.nanmax(neighs_energies)):
                return True
        return False
    else:
        return True

def get_candidates(coord_range, coords):
    mask = np.ones(np.shape(coords)[0])
    for c in range(np.shape(coords)[1]):
        mask_down = coords[:,c] >= coord_range[c,0]
        mask_up = coords[:,c] <= coord_range[c,1]
        mask = np.logical_and(mask,np.logical_and(mask_down,mask_up))
    return reduce_dtype(np.argwhere(mask).flatten())

def minima_in_range (point_range, coords, energy, connectivity, has_to_be_node = True, flat_is_node = False):
    candidates = get_candidates(point_range, coords)
    if (np.shape(candidates)[0]>0):
        if (has_to_be_node == True):
            points_bool = np.zeros(np.shape(energy), dtype = bool)
            candidates = candidates[np.argsort(energy[candidates])]
            candidates_energy = energy[candidates]
            sorted_energy = np.unique(candidates_energy)
            for energy_value in sorted_energy:
                print ("ENERGY: " + str(energy_value))
                selected_candidates = candidates[candidates_energy == energy_value]
                for candidate in selected_candidates:
                    if (node_eval(candidate, energy, connectivity, flat_is_node) == True):
                        points_bool[candidate] = True
                if (np.sum(points_bool) != 0):
                    return reduce_dtype(np.argwhere(points_bool == True).flatten())
            return np.empty(0, dtype = 'int8')
        else:
            points = (candidates[energy[candidates] == np.min(energy[candidates])]).astype(np.int_)
            return reduce_dtype(points)
    else:
        return np.empty(0, dtype = 'int8')

class propagation_handler:
    def __init__(self, coords, energy, connectivity):
        self.connectivity = connectivity
        self.energy = energy
        self.coords = coords
        self.steepest = True
        self.gnw_done = False
        self.lv0trace = False
        self.lv1trace = False
        self.lv2trace = False
        self.lv3trace = False
        self.lv0path = False
        self.lv1path = False
        self.lv2path = False
        self.lv3path = False
        self.fpsr_done = False
        self.gnw_paths = []
        self.OT_points = [np.array([-1]),np.array([-1])]
        self.OT_cluster_ids = [np.array([-1]),np.array([-1])]
        self.OT_not_node_fix = [None,None]

    def down_propagate(self,point,allowed_mask,steepest=True):
        indices = reduce_dtype(np.arange(self.energy.shape[0]))
        propagation = np.zeros(self.energy.shape[0], dtype = np.bool_)
        not_indexed = np.ones(self.energy.shape[0], dtype = np.bool_)
        available = np.zeros(self.energy.shape[0], dtype = np.bool_)
        down_propagation(np.array([point,]), self.connectivity.neighbors, self.energy, allowed_mask, propagation, not_indexed, available, indices, steepest=steepest)
        return (propagation)

    def down_propagate_OT(self,point,candidate_nodes):
        allowed_mask = np.zeros(self.energy.shape[0], dtype = np.bool_)
        for candidate_node in candidate_nodes:
            allowed_mask[self.gnw_node_propagation_clusters[candidate_node]] = True
        propagation = self.down_propagate(point,allowed_mask,steepest=self.steepest)
        visited_nodes = np.unique(self.gnw_minimum_clusters_location[propagation!=0])
        visited_nodes = visited_nodes[visited_nodes!=-1]
        if(visited_nodes.shape[0] > 0):
            min_e = np.min(self.gnw_energy[visited_nodes])
            visited_nodes = visited_nodes[self.gnw_energy[visited_nodes] == min_e]
            if(visited_nodes.shape[0] > 1):
                print("Warning: The provided point can be indistinctively assigned to  " + str(visited_nodes.shape[0]) + " node. Selecting one arbitrarily.")
                print("\tThe index of the node selected is " + str(visited_nodes[0]) + "\n\t")
            if (point not in self.gnw_clusters[visited_nodes[0]]):
                allowed_mask.fill(False)
                allowed_mask[self.gnw_node_propagation_clusters[visited_nodes[0]]] = True
                M_points = self.gnw_clusters[visited_nodes[0]]
                point_array = np.array([point,])
                up_trace = OT_propagation(M_points, point_array, self.connectivity.neighbors, self.energy,allowed_mask=allowed_mask)
                down_trace = OT_propagation(point_array, M_points, self.connectivity.neighbors, up_trace, allowed_mask=up_trace!=0)
                down_trace[M_points] = 1
                down_trace[point] = 1
                return reduce_dtype(np.array([visited_nodes[0]])), reduce_dtype(np.argwhere(down_trace!=0).flatten())
            else:
                return reduce_dtype(np.array([visited_nodes[0]])), np.empty(0,dtype=np.int_)
        else:
            return None, None

    def set_OT(self,OT_points = [None,None]):
        self.get_gnw()
        diff_OT = False
        if (OT_points[0] is not None):
            if(OT_points[0] >= 0 and OT_points[0] < self.energy.shape[0]):
                if (self.OT_points[0][0] != OT_points[0]):
                    diff_OT = True
                    self.OT_points[0] = np.array([OT_points[0],]).flatten()
                    node_ids = []
                    for p in self.OT_points[0]:
                        node_ids.append(np.empty(self.gnw_clusters[self.gnw_node_ids[p]].shape,dtype='uint64'))
                        node_ids[-1].fill(self.gnw_node_ids[p])
                        node_ids[-1] = reduce_dtype(node_ids[-1])
                    node_ids = pd.unique(np.concatenate(node_ids))
                    node_ids, self.OT_not_node_fix[0] = self.down_propagate_OT(self.OT_points[0],node_ids)
                    sampled_nodes = np.unique(self.gnw_minimum_clusters_location[self.OT_not_node_fix[0]])
                    if (sampled_nodes.shape[0] == 1 and sampled_nodes[0] != -1):
                        self.OT_not_node_fix[0] = None
                    self.OT_cluster_ids[0] = node_ids
            else:
                print("ERROR: Selected point index is not within the range of minimum and maximum point indices available. Ignoring origin modification.")
        if (OT_points[1] is not None):
            if(OT_points[1] >= 0 and OT_points[1] < self.energy.shape[0]):
                if(self.OT_points[1][0] != OT_points[1]):
                    diff_OT = True
                    self.OT_points[1] = np.array([OT_points[1],]).flatten()
                    node_ids = []
                    for p in self.OT_points[1]:
                        node_ids.append(np.empty(self.gnw_clusters[self.gnw_node_ids[p]].shape,dtype='uint64'))
                        node_ids[-1].fill(self.gnw_node_ids[p])
                        node_ids[-1] = reduce_dtype(node_ids[-1])
                    node_ids = pd.unique(np.concatenate(node_ids))
                    node_ids, self.OT_not_node_fix[1] = self.down_propagate_OT(self.OT_points[1],node_ids)
                    sampled_nodes = np.unique(self.gnw_minimum_clusters_location[self.OT_not_node_fix[1]])
                    if (sampled_nodes.shape[0] == 1 and sampled_nodes[0] != -1):
                        self.OT_not_node_fix[1] = None
                    self.OT_cluster_ids[1] = node_ids
            else:
                print("ERROR: Selected point index is not within the range of minimum and maximum point indices available. Ignoring target modification.")
        if(diff_OT == True):
            self.lv0trace = False
            self.lv1trace = False
            self.lv2trace = False
            self.lv3trace = False
            self.lv0path = False
            self.lv1path = False
            self.lv2path = False
            self.lv3path = False
            self.fpsr_done = False
            self.gnw_paths = []


    def get_minima_candidates_mask(self,coord_range):
        mask1 = np.ones(np.shape(self.coords)[0],dtype=np.bool_)
        mask2 = np.ones(np.shape(self.coords)[0],dtype=np.bool_)
        for c in range(np.shape(self.coords)[1]):
            mask_down = self.coords[:,c] >= coord_range[c,0]
            mask_up = self.coords[:,c] <= coord_range[c,1]
            np.logical_and(mask1,np.logical_and(mask_down,mask_up), out=mask2)
            np.copyto(mask1,mask2)
        return np.logical_and(mask1,self.gnw_minimum_clusters_location!=-1)

    def select_origin_by_range(self,coord_range, minimum = True):
        self.get_gnw()
        if(minimum):
            point = self.select_minima_in_range(coord_range)
        else:
            point = self.select_lowest_point_in_range(coord_range)
        if point > -1:
            self.set_OT([point,None])
#         print(point)
        return point

    def select_target_by_range(self,coord_range, minimum = True):
        self.get_gnw()
        if(minimum):
            point = self.select_minima_in_range(coord_range)
        else:
            point = self.select_lowest_point_in_range(coord_range)
        if point > -1:
            self.set_OT([None,point])
#         print(point)
        return point

    def select_point_by_minimum_id(self,index):
        candidate_minimum_coords = self.coords[self.gnw_clusters[index]]
        for i in np.argwhere(self.connectivity.periodicity > 0).flatten():
            candidate_minimum_coords[:,i] = candidate_minimum_coords[:,i] - np.min(candidate_minimum_coords[:,i])
            mod = np.floor(candidate_minimum_coords[:,i]/self.connectivity.periodicity[i]) * self.connectivity.periodicity[i]
            candidate_minimum_coords[:,i] = candidate_minimum_coords[:,i] - mod
        coords_mean = np.mean(candidate_minimum_coords, axis=0)
        dist_to_com = np.linalg.norm(candidate_minimum_coords-coords_mean, axis = 1)
        return (self.gnw_clusters[index])[np.argwhere(dist_to_com == np.min(dist_to_com)).flatten()[0]]

    def select_origin_by_minimum_id(self,index):
        self.get_gnw()
        self.set_OT([self.select_point_by_minimum_id(index),None])
    def select_target_by_minimum_id(self,index):
        self.get_gnw()
        self.set_OT([None,self.select_point_by_minimum_id(index)])

    def select_lowest_point_in_range(self,coord_range):
        mask1 = np.ones(np.shape(self.coords)[0],dtype=np.bool_)
        mask2 = np.ones(np.shape(self.coords)[0],dtype=np.bool_)
        for c in range(np.shape(self.coords)[1]):
            mask_down = self.coords[:,c] >= coord_range[c,0]
            mask_up = self.coords[:,c] <= coord_range[c,1]
            np.logical_and(mask1,np.logical_and(mask_down,mask_up), out=mask2)
            np.copyto(mask1,mask2)
        candidates = np.argwhere(mask1).flatten()
        if (candidates.shape[0] > 0):
            energy = self.energy[candidates]
            candidates = candidates[energy == np.min(energy)]
            if (candidates.shape[0] > 1):
                range_center = (coords_range[:,0] + coords_range[:,1])/2
                euc_dist = np.zeros(n_points)
                for candidate in candidates:
                    point_coords = self.coords[candidate,:]
                    euc_dist += np.linalg.norm(point_coords-range_center, axis = 1)
                candidates = candidates[euc_dist==np.min(euc_dist)]
                if (candidates.shape[0] > 1):
                    print("Warning: " + str(np.sum(candidates_mask)) + " equivalent lowest energy points detected. Selecting one arbitrarily.")
                    chosen_point = candidates[0]
                    print("\tThe index of the point selected is " + str(chosen_point) + "\n\tThe point coordinates are:\n\t" + str(self.coords[chosen_point]))
                else:
                    return candidates[0]
            else:
                return candidates[0]
        else:
            return -1

    def select_minima_in_range(self,coord_range):
        self.get_gnw()
        candidates_mask = self.get_minima_candidates_mask(coord_range)
        if (np.any(candidates_mask) == True):
            min_e = np.min(self.energy[candidates_mask])
            candidates_mask[self.energy != min_e] = False
            if(np.sum(candidates_mask.shape[0]) > 1):
                range_center_of_mass = coord_range[:,0]+((coord_range[:,1]-coord_range[:,0])/2)
                dists = np.zeros(self.energy.shape)
                dists.fill(np.inf)
                candidate_points = np.argwhere(candidates_mask).flatten()
                candidate_points_coords = self.coords[candidate_points,:]
                dists[candidate_points] = np.linalg.norm(candidate_points_coords-range_center_of_mass, axis=1)
                min_d = np.min(dists)
                candidates_mask[dists != min_d] = False
                if(np.sum(candidates_mask) > 1):
                    print("Warning: " + str(np.sum(candidates_mask)) + " equivalent minima points detected. Selecting one arbitrarily.")
                    chosen_point = np.argwhere(candidates_mask).flatten()[0]
                    print("\tThe index of the point selected is " + str(chosen_point) + "\n\tThe point coordinates are:\n\t" + str(self.coords[chosen_point]))
                    candidates_mask.fill(False)
                    candidates_mask[chosen_point] = True
            return (np.argwhere(candidates_mask).flatten())[0]
        else:
            return -1

    def get_network(self):

        clustered_points_mask = np.zeros(self.energy.shape,dtype = np.bool_)
        occupied=np.zeros(self.energy.shape[0], dtype = np.bool_)
        not_indexed=np.ones(self.energy.shape[0], dtype = np.bool_)
        points_to_occupy_mask=np.zeros(self.energy.shape[0], dtype = np.bool_)
        indices = reduce_dtype(np.arange(self.energy.shape[0]))
        poi_clusters = []

        print("\tDetecting minima clusters")
        nodes = node_detection(self.energy, self.connectivity.neighbors, flat_is_node = False)
        minimum_clusters_location = np.empty(self.energy.shape[0],dtype = np.int_)
        minimum_clusters_location.fill(-1)
        progress = progress_reporter(len(poi_clusters), 5, 0,indent="\t\tMinima cluster detection: ")
        for node in nodes[np.argsort(self.energy[nodes]).flatten()]:
            if (clustered_points_mask[node] == False):
                occupied.fill(False)
                not_indexed.fill(True)
                points_to_occupy_mask.fill(False)
                points, is_minimum = plateau_and_minima_evaluation(np.array([node]), self.connectivity.neighbors, self.energy, occupied, not_indexed, points_to_occupy_mask, indices)
                if (is_minimum == True):
                    poi_clusters.append(points)
                    minimum_clusters_location[points] = len(poi_clusters) - 1
                clustered_points_mask[points] = True
        del(clustered_points_mask)
        nw_node_propagation_clusters = []
        print("\t\t" + str(len(poi_clusters)))

        print("\tPropagating minima")

        progress = progress_reporter(len(poi_clusters), 0.1, 1,indent="\t\tMinima propagation: ")
        for i,node in enumerate(poi_clusters):
            occupied.fill(False)
            not_indexed.fill(True)
            points_to_occupy_mask.fill(False)
            nw_node_propagation_clusters.append(node_sampler_up_propagation(node, self.connectivity.neighbors, self.energy, occupied, not_indexed, points_to_occupy_mask, indices))
            progress.count(i)

        nw_node_propagation_clusters = tuple(nw_node_propagation_clusters)
        print("\n\tAnnotating minima propagation")

        nw_node_propagation_sum = np.zeros((self.energy.shape[0]), dtype = np.int_)
        #nw_node_contacts_mask = np.zeros(
        for c in range(len(nw_node_propagation_clusters)):
            nw_node_propagation_sum[nw_node_propagation_clusters[c]] += 1
        contact_points_mask=nw_node_propagation_sum>1
        total_contact_points = np.sum(contact_points_mask)
        #nw_node_contacts_list = [None] * np.sum(contact_points_mask)
        nw_node_contacts_list = []
        nw_node_contacts_list_ids = np.zeros(contact_points_mask.shape,dtype=reduce_dtype(np.arange(total_contact_points)).dtype)
        nw_node_contacts_list_ids[np.argwhere(contact_points_mask).flatten()] = reduce_dtype(np.arange(total_contact_points))
        nw_node_contacts_list_last = np.zeros(total_contact_points,dtype=reduce_dtype(np.array((0,np.max(nw_node_propagation_sum)),dtype='uint64')).dtype)
        for i in np.argwhere(contact_points_mask).flatten():
            nw_node_contacts_list.append(np.zeros(nw_node_propagation_sum[i],dtype=nw_node_contacts_list_ids.dtype))
        del(nw_node_propagation_sum)
        #dtype = reduce_dtype(np.arange(len(nw_node_propagation_clusters))).dtype
        progress = progress_reporter(len(nw_node_propagation_clusters), 0.1, 1,indent="\t\tMinima propagation annotation: ")
        for c in range(len(nw_node_propagation_clusters)):
            for point in nw_node_propagation_clusters[c]:
                if(contact_points_mask[point] and ( nw_node_contacts_list_last[nw_node_contacts_list_ids[point]] == 0 or ( not np.in1d(c,nw_node_contacts_list[nw_node_contacts_list_ids[point]], assume_unique=True) ) ) ):
                        nw_node_contacts_list[nw_node_contacts_list_ids[point]][nw_node_contacts_list_last[nw_node_contacts_list_ids[point]]] = c
            progress.count(c)
        del(nw_node_contacts_list_last)
        minima_cluster_indices = reduce_dtype(np.arange(len(poi_clusters)))
        nw_node_contacts_list = tuple(nw_node_contacts_list)
        print("\n\tDetecting barrier clusters")

        barrier_clusters_location = np.empty(self.energy.shape[0],dtype = np.int_)
        barrier_clusters_location.fill(-1)

        tmp_mask1 = np.zeros(self.energy.shape,dtype=np.bool_)
        tmp_mask2 = np.ones(self.energy.shape,dtype=np.bool_)

        blocking_energy = np.empty(self.energy.shape)
        b_e = np.max(self.energy) + 1
        blocking_energy.fill(b_e)
        allowed_mask = np.zeros(self.energy.shape, dtype=np.bool_)
        nw_neighbors = [[] for i in range(len(poi_clusters))]

        progress = progress_reporter(minima_cluster_indices.shape[0], 1, 1,indent="\t\tBarrier cluster detection: ")
        contacted_minima_mask = np.zeros(minima_cluster_indices.shape,dtype = np.bool_)
        for i in minima_cluster_indices:
            tmp_contact_points = (nw_node_propagation_clusters[i])[contact_points_mask[nw_node_propagation_clusters[i]]]
            if (tmp_contact_points.shape[0] > 0):
                contacted_minima_mask.fill(False)
                for contact_point in tmp_contact_points:
                    contacted_minima_mask[nw_node_contacts_list[nw_node_contacts_list_ids[contact_point]]] = True
                contacted_minima = np.argwhere(contacted_minima_mask).flatten()
                contacted_minima = contacted_minima[contacted_minima>i]
                for j in contacted_minima:
                    prop_1 = nw_node_propagation_clusters[i]
                    prop_2 = nw_node_propagation_clusters[j]
                    tmp_mask1[prop_1] = True
                    barrier_points = prop_2[tmp_mask1[prop_2]]
                    tmp_mask1[prop_1] = False

                    barrier_points_e = self.energy[barrier_points]
                    barrier_points = barrier_points[barrier_points_e==np.min(barrier_points_e)]
                    blocking_energy[barrier_points] = self.energy[barrier_points]
                    allowed_mask[barrier_points] = True
                    nodes = node_detection(blocking_energy, self.connectivity.neighbors, flat_is_node = True, allowed_mask = allowed_mask, indices=barrier_points)
                    allowed_mask[barrier_points] = False

                    nodes = nodes[np.argsort(self.energy[nodes]).flatten()]
                    while (nodes.shape[0] >0):
                        occupied.fill(False)
                        not_indexed.fill(True)
                        points_to_occupy_mask.fill(False)
                        points, is_minimum = plateau_and_minima_evaluation(np.array([nodes[0]]), self.connectivity.neighbors, blocking_energy, occupied, not_indexed, points_to_occupy_mask, indices)
                        if (is_minimum):
                            if(barrier_clusters_location[points[0]] == -1):
                                barrier_cluster_id = len(poi_clusters)
                                barrier_clusters_location[points] = barrier_cluster_id
                                poi_clusters.append(points)
                                nw_neighbors.append([i,j],)
                            else:
                                barrier_cluster_id = barrier_clusters_location[points[0]]
                                nw_neighbors[barrier_cluster_id].extend([i,j])
                            nw_neighbors[i].append(barrier_cluster_id)
                            nw_neighbors[j].append(barrier_cluster_id)
                        tmp_mask2[points]=False
                        nodes = nodes[tmp_mask2[nodes]]
                        tmp_mask2[points]=True
                    blocking_energy[barrier_points] = b_e
            progress.count(i)
        del(nw_node_contacts_list)
        del(nw_node_contacts_list_ids)
        del(indices)
        del(occupied)
        del(not_indexed)
        del(points_to_occupy_mask)
        poi_clusters = tuple(poi_clusters)
        barrier_cluster_indices = reduce_dtype(np.arange(minima_cluster_indices.shape[0], len(poi_clusters)))
        print("\n\tBuilding network connectivity")

        nw_energy = np.empty(len(poi_clusters))
        for i in range(len(poi_clusters)):
            nw_neighbors[i] = np.unique(np.array(nw_neighbors[i],dtype=np.int_))
            nw_energy[i] = self.energy[(poi_clusters[i])[0]]
        nw_neighbors = tuple(nw_neighbors)

        print("\tAnnotating global node ids")
        nw_is_node = np.zeros(self.energy.shape[0],dtype=np.bool_)
        for c, cluster in enumerate(poi_clusters):
            nw_is_node[cluster] = True
        nw_node_ids = np.zeros(self.energy.shape[0],dtype='int64')
        nw_node_ids.fill(-1)
        for c, cluster in enumerate(nw_node_propagation_clusters):
            nw_node_ids[cluster] = c

        occupied = np.zeros(nw_energy.shape[0], dtype = np.bool_)
        not_indexed = np.ones(nw_energy.shape[0], dtype = np.bool_)
        points_to_occupy_mask = np.zeros(nw_energy.shape[0], dtype = np.bool_)
        indices = reduce_dtype(np.arange(nw_energy.shape[0]))
        islands = []
        visited = np.zeros(nw_energy.shape[0], dtype = np.bool_)
        while(np.any(visited==False)):
            occupied.fill(False)
            not_indexed.fill(True)
            points_to_occupy_mask.fill(False)
            flat_propagation(np.array((np.argwhere(visited==False).flatten()[0],),dtype=np.int_),nw_neighbors, occupied, not_indexed, points_to_occupy_mask, indices)
            islands.append(np.argwhere(occupied).flatten())
            visited[islands[-1]] = True
        if (len(islands)>1):
            print("WARNING: " + str(len(islands)) + " disconnected regions detected. No paths can be found between points of two disconnected regions.")
            for c,island in enumerate(islands):
#                print("\tRegion " + str(c) + ": " + ",".join(island.astype(str)))
                print("\tRegion " + str(c))
                print("\t\tNumber of nodes: " + str(island.shape[0]))
                print("\t\tSummary of nodes: " + np.array2string(island))
        return poi_clusters, nw_neighbors, reduce_dtype(nw_energy), minima_cluster_indices, barrier_cluster_indices, nw_node_propagation_clusters, reduce_dtype(minimum_clusters_location), reduce_dtype(barrier_clusters_location), reduce_dtype(nw_node_ids), nw_is_node

    def get_global_path_fragments(self,visited_barriers):
        allowed_mask = np.zeros(self.energy.shape[0], dtype = np.bool_)
        indices = np.arange(self.energy.shape[0])
        propagation = np.zeros(self.energy.shape[0], dtype = np.bool_)
        not_indexed = np.ones(self.energy.shape[0], dtype = np.bool_)
        available = np.zeros(self.energy.shape[0], dtype = np.bool_)
        stop_points_mask = np.zeros(self.energy.shape[0], dtype = np.bool_)
        step_trace = np.zeros(self.energy.shape[0], dtype = np.int_)
        self.path_fragments = list(self.path_fragments)
        progress = progress_reporter(visited_barriers.shape[0], 1, 1,indent="\t\t\tGlobal path fragments: ")
        for i,barrier in enumerate(visited_barriers):
            if(self.path_fragments[barrier] is None):
                self.path_fragments[barrier] = []
                if (self.gnw_neighbors[barrier].shape[0] > 0):
                    B_points = self.gnw_clusters[barrier]
                    for minimum in self.gnw_neighbors[barrier]:
                        M_points = self.gnw_clusters[minimum]
                        #propagation.fill(False)
                        node_propagation_points = self.gnw_node_propagation_clusters[minimum]
                        allowed_mask.fill(False)
                        allowed_mask[node_propagation_points] = True
                        #not_indexed.fill(True)
                        #available.fill(False)
                        #down_propagation(B_points, self.connectivity.neighbors, self.energy, allowed_mask, propagation, not_indexed, available, indices, steepest = self.steepest)

                        not_indexed.fill(True)
                        available.fill(False)
                        stop_points_mask.fill(False)
                        step_trace.fill(0)

                        up_trace = OT_propagation(M_points, B_points, self.connectivity.neighbors, self.energy,allowed_mask=allowed_mask,not_indexed=not_indexed,step_trace=step_trace,stop_points_mask=stop_points_mask,available=available,indices=indices)

                        allowed_mask = up_trace!=0
                        not_indexed.fill(True)
                        available.fill(False)
                        stop_points_mask.fill(False)

                        down_trace = OT_propagation(B_points, M_points, self.connectivity.neighbors, up_trace, allowed_mask=allowed_mask, not_indexed=not_indexed,stop_points_mask=stop_points_mask,available=available,indices=indices)

                        down_trace[M_points] = 1
                        down_trace[B_points] = 1
                        self.path_fragments[barrier].append(reduce_dtype(indices[down_trace!=0]))
            progress.count(i)
        self.path_fragments = tuple(self.path_fragments)
    def get_gnw(self):
        if (self.gnw_done == False):
            self.connectivity.get_global_connectivity()
            print("Global network calculation")
            self.gnw_clusters, self.gnw_neighbors, self.gnw_energy, self.gnw_minimum_clusters_indices, self.gnw_barrier_clusters_indices, self.gnw_node_propagation_clusters, self.gnw_minimum_clusters_location, self.gnw_barrier_clusters_location, self.gnw_node_ids, self.is_gnw_node = self.get_network()
            self.path_fragments = [None] * len(self.gnw_clusters)
            self.gnw_done = True
            print("Done")


    def get_non_diag_fragmentwise_neighbors(self,path_fragments):

        total_points = 0

        mask = np.zeros(self.energy.shape,dtype=np.bool_)
        mask[np.concatenate(path_fragments)] = True

        fragmentwise_points = np.argwhere(np.logical_and(np.logical_or(self.gnw_minimum_clusters_location!=-1,self.gnw_barrier_clusters_location!=-1),mask)).flatten()

        mask.fill(False)
        mask[fragmentwise_points] = True

        fragmentwise_points_reference = np.empty(self.energy.shape,dtype='int64')
        fragmentwise_points_reference.fill(-1)
        fragmentwise_points_reference[fragmentwise_points] = np.arange(fragmentwise_points.shape[0])
        fragmentwise_neighbors = []
        for point in fragmentwise_points:
            fragmentwise_neighbors.append([fragmentwise_points_reference[(self.connectivity.neighbors[point])[mask[self.connectivity.neighbors[point]]]]])
        c = fragmentwise_points.shape[0]
        for fragment in path_fragments:
            fragment_points_to_add = fragment[mask[fragment]==False]
            fragmentwise_points_reference[fragment_points_to_add] = np.arange(fragment_points_to_add.shape[0]) + c
            c += fragment_points_to_add.shape[0]
            for point in fragment_points_to_add:
                neighbors_to_add = reduce_dtype(fragmentwise_points_reference[np.intersect1d(self.connectivity.neighbors[point],fragment).astype('uint64')])
                fragmentwise_neighbors.append([neighbors_to_add])
                min_barr_neighbors = neighbors_to_add[neighbors_to_add < fragmentwise_points.shape[0]]
                for min_barr_neighbor in min_barr_neighbors:
                    fragmentwise_neighbors[min_barr_neighbor].append(np.array([fragmentwise_points_reference[point]],dtype='int64'))
        for i in range(len(fragmentwise_neighbors)):
            fragmentwise_neighbors[i] = reduce_dtype(np.concatenate(fragmentwise_neighbors[i]).flatten())
        return tuple(fragmentwise_neighbors)

    def get_fragmentwise_propagation(self,path_fragments):
        total_points = 0

        mask = np.zeros(self.energy.shape,dtype=np.bool_)
        mask[np.concatenate(path_fragments)] = True

        fragmentwise_points = reduce_dtype(np.argwhere(np.logical_and(np.logical_or(self.gnw_minimum_clusters_location!=-1,self.gnw_barrier_clusters_location!=-1),mask)).flatten())

        mask.fill(False)
        mask[fragmentwise_points] = True

        fragmentwise_points_reference = np.empty(self.energy.shape,dtype='int64')
        fragmentwise_points_reference.fill(-1)
        fragmentwise_points_reference[fragmentwise_points] = np.arange(fragmentwise_points.shape[0])
        fragmentwise_neighbors = []
        for point in fragmentwise_points:
            if (self.connectivity.diagonals):
                fragmentwise_neighbors.append([reduce_dtype(fragmentwise_points_reference[(self.connectivity.diag_neighbors[point])[mask[self.connectivity.diag_neighbors[point]]]])])
            else:
                fragmentwise_neighbors.append([reduce_dtype(fragmentwise_points_reference[(self.connectivity.neighbors[point])[mask[self.connectivity.neighbors[point]]]])])
        c = fragmentwise_points.shape[0]
        points_to_add_list = [fragmentwise_points,]
        out_path_fragments = []
        for fragment in path_fragments:
            fragment_points_to_add = reduce_dtype(fragment[mask[fragment]==False])
            fragmentwise_points_reference[fragment_points_to_add] = np.arange(fragment_points_to_add.shape[0]) + c
            c += fragment_points_to_add.shape[0]
            points_to_add_list.append(fragment_points_to_add)
            out_path_fragments.append(reduce_dtype(fragmentwise_points_reference[fragment]))
            for point in fragment_points_to_add:
                if (self.connectivity.diagonals):
                    neighbors_to_add = fragmentwise_points_reference[np.intersect1d(self.connectivity.diag_neighbors[point],fragment).astype('uint64')]
                else:
                    neighbors_to_add = fragmentwise_points_reference[np.intersect1d(self.connectivity.neighbors[point],fragment).astype('uint64')]
                fragmentwise_neighbors.append([neighbors_to_add])
                min_barr_neighbors = neighbors_to_add[neighbors_to_add < fragmentwise_points.shape[0]]
                for min_barr_neighbor in min_barr_neighbors:
                    fragmentwise_neighbors[min_barr_neighbor].append(reduce_dtype(np.array([fragmentwise_points_reference[point]],dtype='int64')))
        for i in range(len(fragmentwise_neighbors)):
            fragmentwise_neighbors[i] = reduce_dtype(np.concatenate(fragmentwise_neighbors[i]).flatten())
        fragmentwise_neighbors = tuple(fragmentwise_neighbors)
        out_path_fragments = tuple(out_path_fragments)
        fragmentwise_points = np.concatenate(points_to_add_list)
        fragmentwise_energy = self.energy[fragmentwise_points]
        fragmentwise_coords = self.coords[fragmentwise_points,:]
        fragmentwise_O_points = []
        for O_point in self.OT_points[0]:
            fragmentwise_O_points.append(np.argwhere(fragmentwise_points==O_point).flatten())
        fragmentwise_O_points = reduce_dtype(np.concatenate(fragmentwise_O_points))

        fragmentwise_T_points = []
        for T_point in self.OT_points[1]:
            fragmentwise_T_points.append(np.argwhere(fragmentwise_points==T_point).flatten())
        fragmentwise_T_points = reduce_dtype(np.concatenate(fragmentwise_T_points))

        if (self.connectivity.diagonals):
            tmp_neighbors = self.get_non_diag_fragmentwise_neighbors(path_fragments)
            dumb_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, tmp_neighbors, fragmentwise_energy, allowed_mask = np.empty(0), blind_descent=True, plateau_check = True,dumb=True)
            descent_count_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, tmp_neighbors, fragmentwise_energy, allowed_mask = np.empty(0), blind_descent=True, plateau_check = True,descent_count=True)

        else:
        #dumb_bw_step_trace = OT_propagation(fragmentwise_T_points,fragmentwise_O_points, fragmentwise_neighbors, fragmentwise_energy, allowed_mask = np.empty(0), blind_descent=True, plateau_check = True)
            dumb_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, fragmentwise_energy, allowed_mask = np.empty(0), blind_descent=True, plateau_check = True,dumb=True)
            descent_count_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, fragmentwise_energy, allowed_mask = np.empty(0), blind_descent=True, plateau_check = True,descent_count=True)

        euclidean_bw_step_trace = OT_propagation(fragmentwise_T_points,fragmentwise_O_points, fragmentwise_neighbors, fragmentwise_coords, allowed_mask = np.empty(0), mode = "euclidean", blind_descent=False, plateau_check = True)
        euclidean_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, fragmentwise_coords, allowed_mask = euclidean_bw_step_trace!=0, mode = "euclidean", blind_descent=False, plateau_check = True, descent_count = True)
        euclidean_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, fragmentwise_energy, allowed_mask = euclidean_fw_step_trace!=0, blind_descent=True, plateau_check = True, descent_count = True, dumb=True)

        djk_bw_step_trace = dijkstra_like_OT_propagation(fragmentwise_T_points,fragmentwise_O_points, fragmentwise_neighbors, allowed_mask = euclidean_fw_step_trace!=0)
        djk_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, djk_bw_step_trace, allowed_mask = djk_bw_step_trace!=0, mode = "min_e", blind_descent=False, plateau_check = True, descent_count = True)

        simp_bw_step_trace = OT_propagation(fragmentwise_T_points,fragmentwise_O_points, fragmentwise_neighbors, fragmentwise_energy, allowed_mask = djk_fw_step_trace!=0, blind_descent=True, plateau_check = True,dumb=True)
        simp_fw_step_trace = OT_propagation(fragmentwise_O_points,fragmentwise_T_points, fragmentwise_neighbors, simp_bw_step_trace, allowed_mask = simp_bw_step_trace!=0, blind_descent=False, plateau_check = False,dumb=True)

        return dumb_fw_step_trace, descent_count_fw_step_trace, euclidean_fw_step_trace, djk_fw_step_trace, simp_fw_step_trace, out_path_fragments, fragmentwise_points

    def get_fragmentwise_gnw_path(self,input_trace):
        print ("\tComputing fragmentwise path")
        allowed_mask = np.zeros(self.energy.shape, dtype=np.bool_)

        input_mask = input_trace!=0
        visited_barriers = self.gnw_barrier_clusters_indices[input_mask[self.gnw_barrier_clusters_indices]]

        self.get_global_path_fragments(visited_barriers) #ensure the fragments are computed
        print ("\n\t\tComputing node pairs")
        node_pairs = []
        progress = progress_reporter(visited_barriers.shape[0], 1, 1,indent="\t\t\tNode pairs computation: ")
        for j, barrier in enumerate(visited_barriers):
            minima = self.gnw_neighbors[barrier]
            visited_neighbors_ids = np.argwhere(input_mask[minima]).flatten()
            for i in visited_neighbors_ids:
                node_pairs.append([barrier,i,minima[i]])
            progress.count(j)
        del(visited_barriers)
        node_pairs = np.array(node_pairs,dtype = np.int_)

        node_pairs_order = np.argsort(input_trace[node_pairs[:,0]]) # sort by barrier occupation step
        node_pairs = node_pairs[node_pairs_order,:]

        node_pairs_order = np.argsort(input_trace[node_pairs[:,2]]) # sort by minimum occupation step
        node_pairs = node_pairs[node_pairs_order,:]
        del(node_pairs_order)
        print ("\n\t\tComputing fragments")
        fragmentwise_gnw_path_nodes = []
        fragmentwise_gnw_path_fragments = []
        if (self.OT_not_node_fix[0] is not None):
            if(self.OT_not_node_fix[0].shape[0] > 0):
                fragmentwise_gnw_path_nodes.append([-1,-1])
                fragment = self.OT_not_node_fix[0]
                allowed_mask[fragment] = True
                fragment_trace = OT_propagation(self.OT_points[0],np.empty(0,dtype=np.int_), self.connectivity.neighbors, self.energy, allowed_mask = allowed_mask, blind_descent=True, plateau_check = True, dumb=True)
                allowed_mask[fragment] = False
                fragment_order = np.argsort(fragment_trace[fragment])
                fragment = fragment[fragment_order]
                fragmentwise_gnw_path_fragments.append(reduce_dtype(fragment))
        progress = progress_reporter(node_pairs.shape[0], 1, 1,indent="\t\t\tFragments computation: ")
        for i in np.arange(node_pairs.shape[0]):
            fragment = reduce_dtype((self.path_fragments[node_pairs[i,0]])[node_pairs[i,1]])
            allowed_mask[fragment] = True
            if (input_trace[node_pairs[i,0]] < input_trace[node_pairs[i,2]]):
                O = self.gnw_clusters[node_pairs[i,0]]
                nodes = [node_pairs[i,0],node_pairs[i,2]]
            else:
                O = self.gnw_clusters[node_pairs[i,2]]
                nodes = [node_pairs[i,2],node_pairs[i,0]]
            fragment_trace = OT_propagation(O,np.empty(0,dtype=np.int_), self.connectivity.neighbors, self.energy, allowed_mask = allowed_mask, blind_descent=True, plateau_check = True, dumb=True)

            allowed_mask[fragment] = False
            fragment_order = np.argsort(fragment_trace[fragment])
            fragment = fragment[fragment_order]
            fragment_trace = fragment_trace[fragment]
            if (fragment_trace[0]>1):
                fragment_trace -= (fragment_trace[0] - 1)
            fragmentwise_gnw_path_nodes.append(nodes)
            fragmentwise_gnw_path_fragments.append(reduce_dtype(fragment))
            progress.count(i)
        del(node_pairs)
        if (self.OT_not_node_fix[1] is not None):
            if(self.OT_not_node_fix[1].shape[0] > 0):
                fragmentwise_gnw_path_nodes.append([-2,-2])
                fragment = self.OT_not_node_fix[1]
                allowed_mask[fragment] = True
                fragment_trace = OT_propagation(self.OT_points[1],np.empty(0,dtype=np.int_), self.connectivity.neighbors, self.energy, allowed_mask = allowed_mask, blind_descent=True, plateau_check = True, dumb=True).astype('int64')
                trace_mask = fragment_trace!=0
                fragment_trace[trace_mask] -= np.max(fragment_trace[trace_mask])
                fragment_trace[trace_mask] *= -1
                allowed_mask[fragment] = False
                fragment_order = np.argsort(fragment_trace[fragment])
                fragment = fragment[fragment_order]
                fragmentwise_gnw_path_fragments.append(reduce_dtype(fragment))
        fragmentwise_gnw_path_nodes = reduce_dtype(np.array(fragmentwise_gnw_path_nodes))
        dumb_fragmentwise_fw_step_trace, descent_count_fragmentwise_fw_step_trace, euclidean_fragmentwise_fw_step_trace, djk_fragmentwise_fw_step_trace, simp_fragmentwise_fw_step_trace, fragmentwise_path_fragments, real_points = self.get_fragmentwise_propagation(fragmentwise_gnw_path_fragments)
        del(fragmentwise_gnw_path_fragments)
        input_trace_gnw_path_fragments = []
        input_trace_gnw_path_dumb_traces = []
        input_trace_gnw_path_descent_count_traces = []
        input_trace_gnw_path_euclidean_traces = []
        input_trace_gnw_path_djk_traces = []
        input_trace_gnw_path_simp_traces = []
        for fragment in fragmentwise_path_fragments:
            real_fragment_points = real_points[fragment]
            dumb_fragment_fw_step_trace = dumb_fragmentwise_fw_step_trace[fragment]
            fragment_order = np.argsort(dumb_fragment_fw_step_trace).flatten()
            real_fragment_points = real_fragment_points[fragment_order]
            input_trace_gnw_path_fragments.append(reduce_dtype(real_fragment_points))
            input_trace_gnw_path_dumb_traces.append(reduce_dtype(dumb_fragmentwise_fw_step_trace[fragment[fragment_order]]))
            input_trace_gnw_path_descent_count_traces.append(reduce_dtype(descent_count_fragmentwise_fw_step_trace[fragment[fragment_order]]))
            input_trace_gnw_path_euclidean_traces.append(reduce_dtype(euclidean_fragmentwise_fw_step_trace[fragment[fragment_order]]))
            input_trace_gnw_path_djk_traces.append(reduce_dtype(djk_fragmentwise_fw_step_trace[fragment[fragment_order]]))
            input_trace_gnw_path_simp_traces.append(reduce_dtype(simp_fragmentwise_fw_step_trace[fragment[fragment_order]]))
        trace = np.zeros(allowed_mask.shape,dtype=np.int_)
        for i,frag_trace in enumerate(input_trace_gnw_path_simp_traces):
            trace[input_trace_gnw_path_fragments[i]] += frag_trace

        if (self.connectivity.diagonals):
            bw_gnwpath_step_trace = OT_propagation(self.OT_points[1],self.OT_points[0], self.connectivity.diag_neighbors, self.energy, allowed_mask = trace!=0, mode = "min_e", blind_descent=False, plateau_check = False, dumb=True)
            unique_step_trace = OT_propagation(self.OT_points[0],self.OT_points[1], self.connectivity.diag_neighbors, bw_gnwpath_step_trace, allowed_mask = bw_gnwpath_step_trace!=0, mode = "min_e", blind_descent=False, plateau_check = False, dumb=True)
        else:
            bw_gnwpath_step_trace = OT_propagation(self.OT_points[1],self.OT_points[0], self.connectivity.neighbors, self.energy, allowed_mask = trace!=0, mode = "min_e", blind_descent=False, plateau_check = False, dumb=True)
            unique_step_trace = OT_propagation(self.OT_points[0],self.OT_points[1], self.connectivity.neighbors, bw_gnwpath_step_trace, allowed_mask = bw_gnwpath_step_trace!=0, mode = "min_e", blind_descent=False, plateau_check = False, dumb=True)
        print("\n\tDone")
        return tuple(input_trace_gnw_path_fragments), tuple(input_trace_gnw_path_dumb_traces), tuple(input_trace_gnw_path_descent_count_traces), tuple(input_trace_gnw_path_euclidean_traces), tuple(input_trace_gnw_path_djk_traces), tuple(input_trace_gnw_path_simp_traces), fragmentwise_gnw_path_nodes, unique_step_trace

    def save_simplified_path_to_txt(self,filename,n=0):
        if (n >= 0 and n < len(self.gnw_paths)):
            trace = (self.gnw_paths[n])[7]
            points = np.argwhere(trace != 0).flatten()
            order = np.argsort(trace[points]).flatten()
            points = points[order]
            fmt = ["%d"]
            for c in range(self.coords.shape[1] + 1):
                fmt.append("%g")
            out = np.column_stack((points, self.coords[points], self.energy[points]))
            np.savetxt(filename,out, fmt = fmt)
            return(out)
    def save_fragmentwise_path_to_txt(self,filename,n=0):
        if (n >= 0 and n < len(self.gnw_paths)):
            sio = io.StringIO()
            fragments = (self.gnw_paths[n])[0]
            dumb_traces = (self.gnw_paths[n])[1]
            descent_count_traces = (self.gnw_paths[n])[2]
            euclidean_traces = (self.gnw_paths[n])[3]
            djk_traces = (self.gnw_paths[n])[4]
            simp_traces = (self.gnw_paths[n])[5]
            fragmentwise_gnw_path_nodes = (self.gnw_paths[n])[6]

            points = np.concatenate(fragments)
            input_mask = np.zeros(self.energy.shape,dtype=np.bool_)
            input_mask[points] = True

            origin_steps=[]
            target_steps=[]
            contact_steps=[]

            for i,trace in enumerate(descent_count_traces):
                origin_steps.append(trace[0])
                target_steps.append(trace[-1])
                frag_energy = self.energy[fragments[i]]
                barrier = (np.argwhere(frag_energy==np.max(frag_energy)).flatten())[0]
                contact_steps.append(trace[barrier])

            origin_steps = np.array(origin_steps)
            target_steps = np.array(target_steps)
            contact_steps = np.array(contact_steps)

            step_counts = np.unique(np.concatenate((contact_steps,origin_steps,target_steps)))
            sort_reference = np.empty(np.max(step_counts)+1,dtype=np.int_)
            sort_reference[step_counts] = np.argsort(np.argsort(step_counts)) + 1

            origin_steps = sort_reference[origin_steps]
            target_steps = sort_reference[target_steps]
            contact_steps = sort_reference[contact_steps]
            f = open(filename,"w+")
            for c, pair in enumerate(fragmentwise_gnw_path_nodes):
                if (pair[0] >= 0):
                    if (pair[0] < self.gnw_minimum_clusters_indices.shape[0]):
                        sio.write("#From minimum " + str(pair[0]))
                    else:
                        sio.write("#From barrier " + str(pair[0]))
                    if (pair[1] < self.gnw_minimum_clusters_indices.shape[0]):
                        sio.write(" to minimum " + str(pair[1]) + "\n")
                    else:
                        sio.write(" to barrier " + str(pair[1]) + "\n")
                #" origin step: " + str(origin_steps[c]) +" barrier step: " + str(contact_steps[c]) + " target step: " + str(target_steps[c]) + "\n")
                elif(pair[0] == -1):
                    sio.write("#From abitrary point\n")
                elif(pair[0] == -2):
                    sio.write("#To abitrary point\n")
                np.savetxt(sio, np.column_stack((fragments[c],self.coords[fragments[c]], self.energy[fragments[c]], dumb_traces[c], descent_count_traces[c], euclidean_traces[c], djk_traces[c], simp_traces[c])), delimiter="\t", fmt='%g')
                f.write(sio.getvalue())
                sio = io.StringIO()
            f.close()

    def get_npaths(self, n=0, allowed_mask = np.empty(0), remove_full_barriers = False):
        self.get_gnw()
        if(self.OT_points[0][0] != self.OT_points[1][0] and self.OT_cluster_ids[0][0] != self.OT_cluster_ids[1][0]):
            if(n >= len(self.gnw_paths)):
                self.gnw_paths = list(self.gnw_paths)
                if(len(self.gnw_paths)==0):
                    self.alt_neighbors = list(self.gnw_neighbors)
                for x in range(len(self.gnw_paths),n+1):
                    if (x == 0):
                        print("Calculating minimum energy path")
                    else:
                        print("Calculating alternative path " + str(x))
                    bw_tmp_step_trace = OT_propagation(self.OT_cluster_ids[1],self.OT_cluster_ids[0], self.alt_neighbors, self.gnw_energy, allowed_mask = allowed_mask, mode = "min_e", blind_descent=True, plateau_check = True)
                    fw_tmp_step_trace = OT_propagation(self.OT_cluster_ids[0],self.OT_cluster_ids[1], self.alt_neighbors, self.gnw_energy, allowed_mask = bw_tmp_step_trace!=0, mode = "min_e", blind_descent=True, plateau_check = True)
                    bw_gnwpath_step_trace = OT_propagation(self.OT_cluster_ids[1],self.OT_cluster_ids[0], self.alt_neighbors, fw_tmp_step_trace, allowed_mask = fw_tmp_step_trace!=0, mode = "min_e", blind_descent=True, plateau_check = True)
                    fw_gnwpath_step_trace = OT_propagation(self.OT_cluster_ids[0],self.OT_cluster_ids[1], self.alt_neighbors, bw_tmp_step_trace, allowed_mask = bw_gnwpath_step_trace!=0, mode = "min_e", blind_descent=True, plateau_check = True)
                    mask = fw_gnwpath_step_trace!=0
                    if (np.any(mask)):
                        points = np.argwhere(mask).flatten()
                        points_energy = self.gnw_energy[points]
                        points = points[points_energy==np.max(points_energy)]
                        points_trace = fw_gnwpath_step_trace[points]
                        points = points[points_trace==np.max(points_trace)]


                        for point in points:
                            if (self.alt_neighbors[point].shape[0] > 0):
                                if (remove_full_barriers):
                                    neighbors_to_remove = self.alt_neighbors[point]
                                    self.alt_neighbors[point] = np.zeros(0,dtype=np.int_)
                                else:
                                    neighs_steps = fw_gnwpath_step_trace[self.alt_neighbors[point]]
                                    neighs_mask = neighs_steps==np.min(neighs_steps[neighs_steps!=0])
                                    neighbors_to_remove = (self.alt_neighbors[point])[neighs_mask==True]
                                    self.alt_neighbors[point] = (self.alt_neighbors[point])[neighs_mask==False]
                                for neighbor in neighbors_to_remove:
                                    self.alt_neighbors[neighbor] = (self.alt_neighbors[neighbor])[self.alt_neighbors[neighbor]!=point]
                        self.gnw_paths.append(self.get_fragmentwise_gnw_path(mask))
                        print("Done")
                    else:
                        print("No more alternative paths can be found.")
                        break

                self.gnw_paths = tuple(self.gnw_paths)

    def get_trace_mask_from_nw_mask(self,input_mask,max_e):
        points = []
        indices = np.argwhere(input_mask).flatten()
        indices = indices[indices<=self.gnw_minimum_clusters_indices[-1]]
        tmp_mask = np.zeros(self.energy.shape,dtype=np.bool_)
        for i in indices:
            tmp_mask[self.gnw_node_propagation_clusters[i]] = True
        return np.logical_and(self.energy<=max_e,tmp_mask)

    def get_well_sampling(self, allowed_mask = np.empty(0)):
        if (self.gnw_done == False):
            self.get_gnw()
        if(self.fpsr_done == False):
            print("Full path sampled network calculation")
            bw_gnwpath_step_trace = OT_propagation(self.OT_cluster_ids[1],self.OT_cluster_ids[0], self.gnw_neighbors, self.gnw_energy, allowed_mask = allowed_mask, mode = "min_e", blind_descent=False, plateau_check = True)
            fw_gnwpath_step_trace = OT_propagation(self.OT_cluster_ids[0],self.OT_cluster_ids[1], self.gnw_neighbors, self.gnw_energy, allowed_mask = allowed_mask, mode = "min_e", blind_descent=False, plateau_check = True)

            bw_trace_indices = np.argwhere(bw_gnwpath_step_trace!=0).flatten()

            max_e = np.max(self.gnw_energy[bw_trace_indices[bw_trace_indices>=self.gnw_barrier_clusters_indices[0]]])
            last_fw = np.max(fw_gnwpath_step_trace[self.gnw_energy==max_e])
            last_bw = np.max(bw_gnwpath_step_trace[self.gnw_energy==max_e])

            O_mask = np.logical_and(fw_gnwpath_step_trace<=last_fw,fw_gnwpath_step_trace!=0)
            T_mask = np.logical_and(bw_gnwpath_step_trace<=last_bw,bw_gnwpath_step_trace!=0)
            I_mask = np.logical_and(O_mask,T_mask)

            fpsr_O_mask = self.get_trace_mask_from_nw_mask(O_mask,max_e)
            fpsr_T_mask = self.get_trace_mask_from_nw_mask(T_mask,max_e)
            self.fpsr_annotation = fpsr_O_mask.astype(np.int_) + (fpsr_T_mask.astype(np.int_) * 2)

            self.fpsr_O_clusters = np.argwhere(O_mask).flatten()
            self.fpsr_T_clusters = np.argwhere(T_mask).flatten()
            self.fpsr_I_clusters = np.argwhere(I_mask).flatten()
            self.fpsr_done = True
            print("Done")

    def save_minbar_clusters_to_txt(self, filepath):
        sio = io.StringIO()
        for i in self.gnw_minimum_clusters_indices:
            sio.write("#Minimum " + str(i) + "\n")
            index_array = np.empty(self.coords[self.gnw_clusters[i]].shape[0],dtype=np.int_)
            index_array.fill(i)
            np.savetxt(sio,np.column_stack((index_array,self.coords[self.gnw_clusters[i]],self.energy[self.gnw_clusters[i]])), fmt='%g')
        for i in self.gnw_barrier_clusters_indices:
            sio.write("#Barrier " + str(i) + "\n")
            index_array = np.empty(self.coords[self.gnw_clusters[i]].shape[0],dtype=np.int_)
            index_array.fill(i)
            np.savetxt(sio,np.column_stack((index_array,self.coords[self.gnw_clusters[i]],self.energy[self.gnw_clusters[i]])), fmt='%g')
        f = open(filepath,"w+")
        f.write(sio.getvalue())
        f.close()

    def save_well_sampling_to_txt(self,filename):
        self.get_well_sampling()
        sio = io.StringIO()
        O_points = np.argwhere(self.fpsr_annotation==1).flatten()
        T_points = np.argwhere(self.fpsr_annotation==2).flatten()
        B_points = np.argwhere(self.fpsr_annotation==3).flatten()

        sio.write("#Origin well points\n")
        np.savetxt(sio, np.column_stack((O_points,self.coords[O_points], self.energy[O_points])), delimiter="\t")
        sio.write("#Target well points\n")
        np.savetxt(sio, np.column_stack((T_points,self.coords[T_points], self.energy[T_points])), delimiter="\t")
        sio.write("#Barrier or intermediate well points\n")
        np.savetxt(sio, np.column_stack((B_points,self.coords[B_points], self.energy[B_points])), delimiter="\t")

        f = open(filename,"w+")
        f.write(sio.getvalue())
        f.close()


def dijkstra_like_OT_propagation(O_points, T_points, connectivity, allowed_mask = np.empty(0), not_indexed=None, step_trace1=None, step_trace2=None,stop_points_mask=None,available=None,indices=None):
    n_points = len(connectivity)
    if (allowed_mask.shape[0] != n_points or np.sum(allowed_mask==False) == 0):
        allowed_mask = np.ones(n_points, dtype = np.bool_)
    allowed_O_points = O_points[allowed_mask[O_points]]
    allowed_T_points = T_points[allowed_mask[T_points]]
    if (step_trace1 is None):
        step_trace1 = np.zeros(n_points, dtype = reduce_dtype(np.array((0,n_points),dtype='uint64')).dtype)
    if (allowed_O_points.shape[0] == 0 or allowed_T_points.shape[0] == 0):
        return step_trace1
    if (step_trace2 is None):
        step_trace2 = np.zeros(n_points, dtype = reduce_dtype(np.array((0,n_points),dtype='uint64')).dtype)

    if (not_indexed is None):
        not_indexed = np.ones(n_points, dtype = np.bool_)
    if (available is None):
        available = np.zeros(n_points, dtype = np.bool_)
    if (stop_points_mask is None):
        stop_points_mask = np.zeros(n_points,dtype = np.bool_)
    stop_points_mask[T_points] = True
    stop_points_mask[O_points] = False
    available[allowed_O_points] = True
    not_indexed[allowed_O_points] = False
    stop_check = False

    if(indices is None):
        indices = np.arange(len(connectivity))
    while(stop_check == False):
        points_to_occupy = indices[available]
        if(np.any(stop_points_mask[points_to_occupy]) == True or points_to_occupy.shape[0] == 0):
            stop_check = True#no return here to allow the filling of equal energy points
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy] #look for neighbors)
            if (neighbors.shape[0]>0):
                allowed_neighbors = neighbors[allowed_mask[neighbors]]
                if (allowed_neighbors.shape[0] > 0):
                    occupied_neighbors = allowed_neighbors[step_trace1[allowed_neighbors]!=0]
                    if(occupied_neighbors.shape[0]>0):
                        step_trace2[point_to_occupy]=np.min(step_trace1[occupied_neighbors]) + 1
                    else:#this implies this is an origin point an, therefore, step trace should be 1
                        step_trace2[point_to_occupy]=1
                    new_available_points = allowed_neighbors[not_indexed[allowed_neighbors]]
                    available[new_available_points] = True
                    not_indexed[new_available_points] = False
        np.copyto(step_trace1,step_trace2)
        available[points_to_occupy] = False
    return reduce_dtype(step_trace1)

def OT_propagation(O_points, T_points, connectivity, metric, allowed_mask = np.empty(0), mode = "min_e", blind_descent=False, plateau_check = False, descent_count = False, dumb = False, not_indexed=None, step_trace=None,stop_points_mask=None,available=None,indices=None):
    n_points = len(connectivity)
    if (allowed_mask.shape[0] != n_points or np.sum(allowed_mask==False) == 0):
        allowed_mask = np.ones(n_points, dtype = np.bool_)
    allowed_O_points = O_points[allowed_mask[O_points]]

    if (step_trace is None):
        step_trace = np.zeros(n_points, dtype = reduce_dtype(np.array((0,n_points),dtype='uint64')).dtype)
    if (allowed_O_points.shape[0] == 0):
        return step_trace

    if(mode == "min_e"):
        energy = metric
    if (not_indexed is None):
        not_indexed = np.ones(n_points, dtype = np.bool_)

    if (available is None):
        available = np.zeros(n_points, dtype = np.bool_)

    if(stop_points_mask is None):
        stop_points_mask = np.zeros(n_points,dtype = np.bool_)
    stop_points_mask[T_points] = True
    stop_points_mask[O_points] = False
    if (mode == "euclidean"):
        if (O_points.shape[0] == 0):
            return step_trace
        coords = metric#if euclidean True metric must be coords

        O_energy = np.zeros(n_points)
        for O_point in O_points:
            O_coords = coords[O_point,:]
            O_energy += np.linalg.norm(coords-O_coords, axis = 1)
        O_energy /= O_points.shape[0]
        if (T_points.shape[0] == 0):
            return step_trace

        T_energy = np.zeros(n_points)
        for T_point in T_points:
            T_coords = coords[T_point,:]
            T_energy += np.linalg.norm(coords-T_coords, axis = 1)
        T_energy /= T_points.shape[0]

        energy = (O_energy + T_energy)/2

    available[allowed_O_points] = True
    not_indexed[allowed_O_points] = False
    step = 0
    pre_stop_check = False
    stop_check = False
    prev_e = np.min(energy[allowed_O_points])
    curr_e = prev_e
    if (indices is None):
        indices = np.arange(len(connectivity))
    if (plateau_check == True):
        was_plateau=False

    while(stop_check == False):
        avail_points = indices[available]
        if(avail_points.shape[0] > 0):
            if (plateau_check == True):
                points_to_occupy = avail_points[energy[avail_points] == prev_e]
                if (points_to_occupy.shape[0] == 0):
                    if (pre_stop_check):
                        stop_check = True
                    else:
                        curr_e = np.min(energy[avail_points])
                        if(blind_descent == False):
                            points_to_occupy = avail_points[energy[avail_points] == curr_e]
                            if(dumb==False):
                                step += 1
                        else:
                            if (curr_e < prev_e):
                                points_to_occupy = avail_points[energy[avail_points] <= prev_e]
                                if(dumb==False and (was_plateau==True or descent_count==True)):
                                    step += 1
                            else:
                                points_to_occupy = avail_points[energy[avail_points] == curr_e]
                                if(dumb==False):
                                    step += 1
                    was_plateau=False
                else:
                    was_plateau=True
            else:
                curr_e = np.min(energy[avail_points])
                if(blind_descent and curr_e < prev_e):
                    points_to_occupy = avail_points[energy[avail_points] <= prev_e]
                else:
                    points_to_occupy = avail_points[energy[avail_points] == curr_e]
                if(dumb==False):
                    step += 1
        else:
            stop_check = True
        if(step ==0):
            step=1
        prev_e = curr_e
        if(np.any(stop_points_mask[points_to_occupy]) or points_to_occupy.shape[0] == 0):
            if (plateau_check):
                pre_stop_check = True
            else:
                stop_check = True#no return here to allow the filling of equal energy points
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy] #look for neighbors)
            if (neighbors.shape[0]>0):
                allowed_neighbors = neighbors[allowed_mask[neighbors]]
                if (allowed_neighbors.shape[0] > 0):
                    new_available_points = allowed_neighbors[not_indexed[allowed_neighbors]]
                    available[new_available_points] = True
                    not_indexed[new_available_points] = False
        if (dumb==False):
            step_trace[points_to_occupy] = step #here to consider empty unoccupied all the points occupied in the same step
        else:
            step_trace[points_to_occupy] = indices[:points_to_occupy.shape[0]]+step
            step+=points_to_occupy.shape[0]
        available[points_to_occupy] = False
    return reduce_dtype(step_trace)

def down_propagation(O_points, connectivity, energy, allowed_mask, propagation, not_indexed, available, indices, steepest=False):
    allowed_O_points = O_points[allowed_mask[O_points]]
    if (allowed_O_points.shape[0] == 0):
        return propagation

    available[O_points] = True
    not_indexed[O_points] = False
    step = 1
    stop_check = False
    avail_points = indices[available]
    prev_e = np.min(energy[O_points])
    curr_e = prev_e
    points_to_occupy = avail_points[energy[avail_points] == prev_e]
    while(points_to_occupy.shape[0] != 0):
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy] #look for neighbors)
            if (neighbors.shape[0] > 0):
                allowed_neighbors = neighbors[np.logical_and(not_indexed[neighbors],allowed_mask[neighbors])]
                if (allowed_neighbors.shape[0] > 0):
                    new_available_points = allowed_neighbors[energy[allowed_neighbors] <= curr_e]
                    available[new_available_points] = True
                    not_indexed[new_available_points] = False
        propagation[points_to_occupy] = True #here to consider empty unoccupied all the points occupied in the same step
        available[points_to_occupy] = False

        avail_points = indices[available]
        if (steepest== True):
            if(avail_points.shape[0] != 0):
                curr_e = np.min(energy[avail_points])
                if(curr_e < prev_e):
                    points_to_occupy = avail_points[energy[avail_points] == curr_e]
                elif(curr_e == prev_e):
                    points_to_occupy = avail_points[energy[avail_points] == prev_e]
                else:
                    points_to_occupy = np.empty(0)
            else:
                points_to_occupy = np.empty(0)
        else:
            points_to_occupy = avail_points[energy[avail_points] == prev_e]
            if (points_to_occupy.shape[0] == 0 and avail_points.shape[0] != 0):
                curr_e = np.min(energy[avail_points])
                if(curr_e < prev_e):
                    points_to_occupy = avail_points[energy[avail_points] == curr_e]
        prev_e = curr_e

def node_sampler_up_propagation(start_points, connectivity, energy, occupied, not_indexed, points_to_occupy_mask, indices):
    points_to_occupy = start_points
    not_indexed[points_to_occupy] = False
    points_to_occupy_mask[points_to_occupy] = True
    while(points_to_occupy.shape[0] > 0):
        occupied[points_to_occupy] = True
        points_to_occupy_mask[points_to_occupy] = False
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy]
            if (neighbors.shape[0] > 0):
                allowed_neighbors = neighbors[not_indexed[neighbors]]
                if (allowed_neighbors.shape[0] > 0):
                    allowed_neighbors = allowed_neighbors[energy[allowed_neighbors] >= energy[point_to_occupy]]
                    if (allowed_neighbors.shape[0] > 0):
                        points_to_occupy_mask[allowed_neighbors] = True
                        not_indexed[allowed_neighbors] = False
        points_to_occupy = indices[points_to_occupy_mask]
    return reduce_dtype(indices[occupied])

def plateau_and_minima_evaluation(start_points, connectivity, energy, occupied, not_indexed, points_to_occupy_mask, indices, drop_on_non_minima=True):
    points_to_occupy = start_points
    not_indexed[points_to_occupy] = False
    points_to_occupy_mask[points_to_occupy] = True
    minimum = True
    while(points_to_occupy.shape[0] > 0):
        occupied[points_to_occupy] = True
        points_to_occupy_mask[points_to_occupy] = False
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy]
            if (neighbors.shape[0] > 0):
                allowed_neighbors = neighbors[not_indexed[neighbors]]
                if (allowed_neighbors.shape[0] > 0):
                    e_allowed_neighbors = allowed_neighbors[energy[allowed_neighbors] == energy[point_to_occupy]]
                    if (e_allowed_neighbors.shape[0] > 0):
                        points_to_occupy_mask[e_allowed_neighbors] = True
                        not_indexed[e_allowed_neighbors] = False
                    if(minimum==True):
                        if(np.any(allowed_neighbors[energy[allowed_neighbors] < energy[point_to_occupy]])):
                            minimum = False
                            if(drop_on_non_minima):
                                points_to_occupy_mask.fill(False)
        points_to_occupy = indices[points_to_occupy_mask]
    return reduce_dtype(indices[occupied]),minimum

def flat_propagation(start_points,connectivity, occupied, not_indexed, points_to_occupy_mask, indices):
    points_to_occupy=start_points
    not_indexed[points_to_occupy] = False
    points_to_occupy_mask[points_to_occupy] = True
    while(points_to_occupy.shape[0] > 0):
        occupied[points_to_occupy] = True
        points_to_occupy_mask[points_to_occupy] = False
        for point_to_occupy in points_to_occupy:
            neighbors = connectivity[point_to_occupy]
            if (neighbors.shape[0] > 0):
                allowed_neighbors = neighbors[not_indexed[neighbors]]
                if (allowed_neighbors.shape[0] > 0):
                    points_to_occupy_mask[allowed_neighbors] = True
                    not_indexed[allowed_neighbors] = False
        points_to_occupy = indices[points_to_occupy_mask]

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        return pickle.dump(obj, output, -1)

def load_object(filename):
    try:
        with open(filename, 'rb') as input:
            return pickle.load(input)
    except:
        return None


class mepsa_gui:
# MAIN WINDOW ACTIONS
    def load_surface(self):
        self.update_main_window_states()
        self.file_path = filedialog.askopenfilename(parent=self.root)
        if self.file_path is None or len(self.file_path) == 0:
            self.update_main_window_states()
            return
        if self.file_path != '' and self.file_path != 'none':
            self.load_surface_B.config(text='LOADING...')
            self.load_surface_B.grid()
            self.load_surface_B.update()
            self.surface = surface_handler(self.file_path)
            self.load_surface_B.config(text='Load surface')
            self.load_surface_B.grid()
            self.load_surface_B.update()
            if (self.surface.good == True):
                self.session_started = True
                headers = ['Lower cutoff','Upper cutoff']
                min_coords = np.min(self.surface.coords, axis = 0)
                max_coords = np.max(self.surface.coords, axis = 0)
                self.origin_coord_ranges = np.column_stack((min_coords,max_coords))
                self.target_coord_ranges = np.copy(self.origin_coord_ranges)
                cuts = np.column_stack((self.surface.connectivity.state_cutoffs,self.surface.connectivity.edge_cutoffs)).tolist()
                rows = list(cuts)
                df = pd.DataFrame(rows, columns=headers)
                text = df.to_string()
                msg = messagebox.showinfo( "Caution", "MEPSAnd can optionally use cutoffs to define connectivity.\nIf you intend to use cuttoff-based connetivity calculations instead of the default \"grid-like\" method, please check the detected cutoffs.\nIf you want to use different cutoffs that those listed here, please check the manual section 7.1.\n\n" + text)
            else:
                del(self.surface)
                messagebox.showinfo( "ERROR", "Surface file could not be loaded.")
        self.update_main_window_states()
    def load_session(self):
        self.force_main_window_states()
        session_path = filedialog.askopenfilename(parent=self.root)
        if session_path is None or len(session_path) == 0:
            self.update_main_window_states()
            return
        if session_path != '' and session_path != 'none':
            try:
                self.load_session_B.config(text='LOADING...')
                self.load_session_B.grid()
                self.load_session_B.update()
                self.surface = load_object(session_path)
                self.load_session_B.config(text='Load session')
                self.load_session_B.grid()
                self.load_session_B.update()
                if (self.surface.good == True):
                    self.session_started = True
                    min_coords = np.min(self.surface.coords, axis = 0)
                    max_coords = np.max(self.surface.coords, axis = 0)
                    self.origin_coord_ranges = np.column_stack((min_coords,max_coords))
                    self.target_coord_ranges = np.copy(self.origin_coord_ranges)
                else:
                    del(self.surface)
                    messagebox.showinfo( "ERROR", "Session file could not be loaded.")

            except:
                messagebox.showinfo( "ERROR", "Session file could not be loaded.")
        self.update_main_window_states()

    def save_session_main(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.root,mode='w', defaultextension=".mepsand")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.save_session_B.config(text='SAVING...')
                self.save_session_B.grid()
                self.save_session_B.update()
                save_object(self.surface, file_to_save.name)
                self.save_session_B.config(text='Save session')
                self.save_session_B.grid()
                self.save_session_B.update()
            except:
                self.save_session_B.config(text='Save session')
                self.save_session_B.grid()
                self.save_session_B.update()
                messagebox.showinfo( "ERROR", "Session could not be saved.")
        self.update_main_window_states()

    def save_session_pipe(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.pathfinding_pipeline_W,mode='w', defaultextension=".mepsand")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.save_session_pipe_B.config(text='SAVING...')
                self.save_session_pipe_B.grid()
                self.save_session_pipe_B.update()
                save_object(self.surface, file_to_save.name)
                self.save_session_pipe_B.config(text='Save session')
                self.save_session_pipe_B.grid()
                self.save_session_pipe_B.update()
            except:
                self.save_session_pipe_B.config(text='Save session')
                self.save_session_pipe_B.grid()
                self.save_session_pipe_B.update()
                messagebox.showinfo( "ERROR", "Session could not be saved.")
        self.update_main_window_states()
    def save_session_nw(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.network_projections_W,mode='w', defaultextension=".mepsand")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.save_session_nw_B.config(text='SAVING...')
                self.save_session_nw_B.grid()
                self.save_session_nw_B.update()
                save_object(self.surface, file_to_save.name)
                self.save_session_nw_B.config(text='Save session')
                self.save_session_nw_B.grid()
                self.save_session_nw_B.update()
            except:
                self.save_session_nw_B.config(text='Save session')
                self.save_session_nw_B.grid()
                self.save_session_nw_B.update()
                messagebox.showinfo( "ERROR", "Session could not be saved.")
        self.update_main_window_states()

    def load_cutoffs(self):
        self.force_main_window_states()
        cut_path = filedialog.askopenfilename(parent=self.root)
        if cut_path is None or len(cut_path) == 0:
            self.update_main_window_states()
            return
        if cut_path != '' and cut_path != 'none':
            try:
                self.surface.connectivity.load_cutoffs(cut_path)
            except:
                messagebox.showinfo( "ERROR", "Custom cutoffs file could not be interpreted.")
        self.update_main_window_states()

    def save_cutoffs(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.root,mode='w', defaultextension=".cutoffs")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.surface.connectivity.save_cutoffs(file_to_save.name)
            except:
                messagebox.showinfo( "ERROR", "Cutoffs could not be saved.")
        self.update_main_window_states()
# CONNECTIVITY BUTTONS CALLBACKS
    def ask_connectivity_options(self):
        self.ask_connectivity_options_W = tki.Toplevel(master=self.root)
        self.ask_connectivity_options_W.resizable(width=FALSE, height=FALSE)
        self.ask_connectivity_options_W.protocol("WM_DELETE_WINDOW", self.ask_connectivity_options_W_exit_handler)
        self.ask_connectivity_options_W.title("Connectivity options")
        self.ask_connectivity_options_W.wait_visibility()
        self.ask_connectivity_options_W.grab_set()
        self.ask_diagonals_S = tki.StringVar(self.ask_connectivity_options_W)
        self.ask_diagonals_S.set("No (Default)")
        self.ask_assumegrid_S = tki.StringVar(self.ask_connectivity_options_W)
        self.ask_assumegrid_S.set("Yes (Default)")
        i = 0
        spacer_L = tki.Label(master = self.ask_connectivity_options_W, text = "Allow diagonals (steps in more than one dimension at once) for path reduction?\n")
        spacer_L.grid(row = i, column = 0, sticky=W)
        options = ["Yes","No (Default)"]
        for text in options:
            i += 1
            self.ask_connectivity_options_B = tki.Radiobutton(self.ask_connectivity_options_W, text=text, variable=self.ask_diagonals_S, value=text)
            self.ask_connectivity_options_B.grid(row =i, column = 0,sticky=W)
        i += 1
        spacer_L = tki.Label(master = self.ask_connectivity_options_W, text = "")
        spacer_L.grid(row = i, column = 0, columnspan = 1,sticky=W+E)
        i += 1
        spacer_L = tki.Label(master = self.ask_connectivity_options_W, text = "Assume grid-like connectivity?\n")
        spacer_L.grid(row = i, column = 0,sticky=W)
        options = ["Yes (Default)","No (Use cutoffs)"]
        for text in options:
            i += 1
            self.ask_assumegrid_B = tki.Radiobutton(self.ask_connectivity_options_W, text=text, variable=self.ask_assumegrid_S, value=text)
            self.ask_assumegrid_B.grid(row =i, column = 0,sticky=W)
        i += 1
        spacer_L = tki.Label(master = self.ask_connectivity_options_W, text = "")
        spacer_L.grid(row = i, column = 0, columnspan = 1,sticky=W+E)
        i += 1
        set_origin_man_B = tki.Button(self.ask_connectivity_options_W, text = "Select", command = self.compute_connectivity_call)
        set_origin_man_B.grid(row = i, column = 0, columnspan = 3,sticky=W+E)

    def compute_connectivity_call(self):
        self.force_main_window_states()
        self.surface.connectivity.diagonals = self.ask_diagonals_S.get() == "Yes"
        self.surface.connectivity.assume_grid = self.ask_assumegrid_S.get() == "Yes (Default)"
        self.ask_connectivity_options_W_exit_handler()
        self.compute_connectivity_B.config(text="RUNNING...")
        self.compute_connectivity_B.grid()
        self.compute_connectivity_B.update()
        self.surface.get_global_network()
        self.compute_connectivity_B.config(text="Compute connectivity")
        self.compute_connectivity_B.grid()
        self.compute_connectivity_B.update()
        self.update_main_window_states()

    def load_periodicity_call(self):
        self.force_main_window_states()
        path = filedialog.askopenfilename(parent=self.root)
        if path is None or len(path) == 0:
            self.update_main_window_states()
            return
        if path != '' and path != 'none':
            try:
                self.surface.connectivity.load_periodicity(path)
            except:
                messagebox.showinfo( "ERROR", "Periodicity file could not be interpreted.")
        self.update_main_window_states()
    def save_periodicity_call(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.root,mode='w', defaultextension=".periodicity")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.surface.connectivity.save_periodicity(file_to_save.name)
            except:
                messagebox.showinfo( "ERROR", "Periodicity could not be saved.")
        self.update_main_window_states()

    def load_connectivity_call(self):
        self.force_main_window_states()
        path = filedialog.askopenfilename(parent=self.root)
        if path is None or len(path) == 0:
            self.update_main_window_states()
            return
        if path != '' and path != 'none':
            try:
                self.surface.connectivity.load_connectivity(path)
                if(self.surface.connectivity.connectivity_done):
                    self.surface.get_global_network()
                else:
                    messagebox.showinfo("ERROR", "Conectivity file could not be interpreted.")
            except:
                messagebox.showinfo( "ERROR", "Conectivity file could not be interpreted.")
        self.update_main_window_states()
        self.load_diag_connectivity_call()

    def load_diag_connectivity_call(self):
        self.force_main_window_states()
        diag = tki.messagebox.askquestion ('Load diagonal connectivity','Load a diagonal connectivity file for path reduction?', default=messagebox.NO)
        if diag == 'yes':
            path = filedialog.askopenfilename(parent=self.root)
            if path is None or len(path) == 0:
                self.update_main_window_states()
                return
            if path != '' and path != 'none':
                try:
                    self.surface.connectivity.load_diag_connectivity(path)
                    if(self.surface.connectivity.diagonals == False):
                        messagebox.showinfo("ERROR", "Diagonal conectivity file could not be interpreted.")
                        self.load_diag_connectivity_call()
                except:
                    messagebox.showinfo( "ERROR", "Diagonal conectivity file could not be interpreted.")
                    self.load_diag_connectivity_call()
        self.update_main_window_states()

    def save_connectivity_call(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.root,mode='w', defaultextension=".connectivity")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.surface.connectivity.save_connectivity(file_to_save.name)
            except:
                messagebox.showinfo( "ERROR", "Conectivity could not be saved.")
        self.update_main_window_states()

# SAVE NODES
    def save_nodes_call(self):
        self.force_main_window_states()
        file_to_save = filedialog.asksaveasfile(parent=self.root,mode='w', defaultextension=".minbar")
        if file_to_save is None:
            self.update_main_window_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.surface.propagation.save_minbar_clusters_to_txt(file_to_save.name)
            except:
                messagebox.showinfo( "ERROR", "Minima and barriers could not be saved.")
        self.update_main_window_states()


# PATHFINDING PIPELINE ACTIONS
    #SET ORIGIN
    def set_origin_id(self, input_index = -1):
        if (input_index == -1):
            input_index = self.origin_id_E.get()
            good = False
            try:
                index = int(input_index)
                good = True
            except:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_origin_W)
        else:
            index = input_index
            good = True
        if(good):
            if (index >= 0):
                if (index < self.surface.coords.shape[0]):
                    self.surface.propagation.set_OT([np.array([index]),None])
                    self.origin_id_S.set(index)
                    self.update_pathfinding_pipeline_states()
                    self.set_origin_W.destroy()
                else:
                    messagebox.showinfo( "ERROR", "The maximum index available is: " + str(self.surface.energy.shape[0] -1),parent=self.set_origin_W)
            else:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_origin_W)

    def set_origin_min_id(self, input_index = -1):
        if (input_index == -1):
            input_index = self.origin_id_E.get()
            good = False
            try:
                index = int(input_index)
                good = True
            except:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_origin_W)
        else:
            index = input_index
            good = True
        if(good):
            if (index >= 0):
                if (index <= self.surface.propagation.gnw_minimum_clusters_indices.shape[0]):
                    self.surface.propagation.select_origin_by_minimum_id(index)
                    self.origin_id_S.set(index)
                    self.update_pathfinding_pipeline_states()
                    self.set_origin_W.destroy()
                else:
                    messagebox.showinfo( "ERROR", "The maximum index available is: " + str(self.surface.propagation.gnw_minimum_clusters_indices.shape[0] -1),parent=self.set_origin_W)
            else:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_origin_W)


    def load_origin_ranges(self):
        ranges_path = filedialog.askopenfilename(parent=self.set_origin_ranges_W)
        if ranges_path is None or len(ranges_path) == 0:
            return
        if ranges_path != '' and ranges_path != 'none':
            try:
                df = pd.read_csv(ranges_path, sep='\s+', header = None)
                ranges = df.values
                if (np.all(ranges.shape == self.origin_coord_ranges.shape) == True):
                    self.origin_coord_ranges = ranges
                    self.set_origin_ranges_W.destroy()
                    self.set_origin_ranges()
                else:
                    messagebox.showinfo( "ERROR", "Provided origin ranges do not match the expected shape.",parent=self.set_origin_W)

            except:
                messagebox.showinfo( "ERROR", "Origin ranges file could not be parsed.",parent=self.set_origin_W)

    def save_origin_ranges(self):
        self.read_origin_ranges()
        file_to_save = filedialog.asksaveasfile(parent=self.set_origin_ranges_W,mode='w', defaultextension=".ranges")
        if file_to_save is None:
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                np.savetxt(file_to_save.name, self.origin_coord_ranges, delimiter = "\t", fmt='%g')
            except:
                messagebox.showinfo( "ERROR", "Ranges could not be saved.")

    def read_origin_ranges(self):
        try:
            df = pd.read_csv(io.StringIO(self.text_box.get("1.0",END)), sep='\s+')
        except:
            messagebox.showinfo( "ERROR", "Origin ranges could not be parsed.",parent=self.set_origin_W)
        ranges = df.values[:,1:]
        if (np.all(ranges.shape == self.origin_coord_ranges.shape) == True):
            self.origin_coord_ranges = ranges
            headers = ['Dimension_id','Range_lower_limit', 'Range_upper_limit']
            ranges_down = self.origin_coord_ranges[:,0].tolist()
            ranges_up = self.origin_coord_ranges[:,1].tolist()
            rows = list(zip(range(self.origin_coord_ranges.shape[0]), ranges_down, ranges_up))
            df = pd.DataFrame(rows, columns=headers)
            text = "" + df.to_string(index = False)
            self.text_box.delete("1.0",END)
            self.text_box.insert(END,text)
        else:
            messagebox.showinfo( "ERROR", "Provided origin ranges do not match the expected shape.",parent=self.set_origin_W)

    def set_origin_ranges(self):
        self.set_origin_W.grab_release()
        self.set_origin_ranges_W = tki.Toplevel(master=self.set_origin_W)
        self.set_origin_ranges_W.resizable(width=FALSE, height=FALSE)
        self.set_origin_ranges_W.protocol("WM_DELETE_WINDOW", self.set_origin_ranges_W_exit_handler)
        self.set_origin_ranges_W.title("Minima search coordinates range")
        self.set_origin_ranges_W.wait_visibility()
        self.set_origin_ranges_W.grab_set()
        set_origin_man_B = tki.Button(self.set_origin_ranges_W, text = "Load coordinate ranges", command = self.load_origin_ranges)
        set_origin_man_B.grid(row = 0, column = 0, columnspan = 1,sticky=W+E)
        set_origin_man_B = tki.Button(self.set_origin_ranges_W, text = "Save coordinate ranges", command = self.save_origin_ranges)
        set_origin_man_B.grid(row = 0, column = 1, columnspan = 1,sticky=W+E)
        set_read_origin_ranges_B = tki.Button(self.set_origin_ranges_W, text = "Read origin coordinate ranges", command = self.read_origin_ranges)
        set_read_origin_ranges_B.grid(row = 1, column = 0, columnspan = 1,sticky=W+E)
        set_origin_man_B = tki.Button(self.set_origin_ranges_W, text = "Set origin as the lowest minimum in the coordinate ranges", command = self.set_origin_from_ranges)
        set_origin_man_B.grid(row = 1, column = 1, columnspan = 1,sticky=W+E)

        self.text_box = tki.Text(self.set_origin_ranges_W, borderwidth=3, relief="sunken")
        self.text_box.config(font=("consolas", 12), undo=True, wrap='word')
        self.text_box.grid(row=2, column=0, sticky=N+S+E+W, columnspan=2)
        scrollbar = tki.Scrollbar(self.set_origin_ranges_W, command=self.text_box.yview)
        scrollbar.grid(row=2, column=2, sticky=N+S+E+W)
        self.text_box['yscrollcommand'] = scrollbar.set
        headers = ['Dimension_id','Range_lower_limit', 'Range_upper_limit']
        ranges_down = self.origin_coord_ranges[:,0].tolist()
        ranges_up = self.origin_coord_ranges[:,1].tolist()
        rows = list(zip(range(self.origin_coord_ranges.shape[0]), ranges_down, ranges_up))
        df = pd.DataFrame(rows, columns=headers)
        text = "" + df.to_string(index = False)
        self.text_box.delete("1.0",END)
        self.text_box.insert(END,text)

    def set_origin_from_ranges(self):
        self.read_origin_ranges()
        point = -1
        if (np.all(self.origin_coord_ranges[:,0]<=self.origin_coord_ranges[:,1])):
            point = self.surface.propagation.select_origin_by_range(self.origin_coord_ranges, minimum=True)
            if(point == -1):
                continue_bool = messagebox.askokcancel(title='Minima search warning', message="WARNING: No valid points could be found with the current ranges. Do you want to take a point with the lowest energy in this range?",parent=self.set_origin_W)
                if (continue_bool):
                    point = self.surface.propagation.select_origin_by_range(self.origin_coord_ranges, minimum=False)
                    if(point==-1):
                        messagebox.showinfo( "ERROR", "No valid points could be found with the current ranges.",parent=self.set_origin_W)
        else:
            messagebox.showinfo( "ERROR", "Current origin ranges present at least one lower limit greater than its corresponding upper limit.",parent=self.set_origin_W)
        if (point == self.surface.propagation.OT_points[0][0] and point != -1):
            self.set_origin_id(point)

    def set_origin_call(self):
        self.set_origin_W = tki.Toplevel(master=self.pathfinding_pipeline_W)
        self.set_origin_W.resizable(width=FALSE, height=FALSE)
        self.set_origin_W.protocol("WM_DELETE_WINDOW", self.set_origin_W_exit_handler)
        self.set_origin_W.title("Set origin")
        self.set_origin_W.wait_visibility()
        self.set_origin_W.grab_set()
        self.set_origin_B['state']='disabled'
        origin_id_L = tki.Label(master = self.set_origin_W, text = "Point/Minimum ID:")
        origin_id_L.grid(row = 0, column = 0, columnspan = 1,sticky=W+E)
        self.origin_id_S = tki.StringVar()
        self.origin_id_E = tki.Entry(master = self.set_origin_W,textvariable=self.origin_id_S)
        self.origin_id_E.grid(row = 0, column = 1, columnspan = 1,sticky=W+E)
        set_origin_id_B = tki.Button(self.set_origin_W, text = "Set origin by point index", command = self.set_origin_id)
        set_origin_id_B.grid(row = 1, column = 0, columnspan = 2,sticky=W+E)
        set_origin_min_id_B = tki.Button(self.set_origin_W, text = "Set origin by minimum index", command = self.set_origin_min_id)
        set_origin_min_id_B.grid(row = 2, column = 0, columnspan = 2,sticky=W+E)
        set_origin_man_B = tki.Button(self.set_origin_W, text = "Find origin by coordinate ranges", command = self.set_origin_ranges)
        set_origin_man_B.grid(row = 3, column = 0, columnspan = 2,sticky=W+E)

    def show_origin_coords(self, trash):
        self.force_pathfinding_pipeline_states()
        if (self.set_origin_L['text'] != 'None'):
            self.set_origin_L_W = tki.Toplevel(master=self.pathfinding_pipeline_W)
            self.set_origin_L_W.protocol("WM_DELETE_WINDOW", self.set_origin_L_W_exit_handler)
            self.set_origin_L_W.resizable(width=FALSE, height=FALSE)
            self.set_origin_L_W.title("Origin coordinates")
            text_box = tki.Text(self.set_origin_L_W, borderwidth=3, relief="sunken")
            text_box.config(font=("consolas", 12), undo=True, wrap='word')
            text_box.grid(row=1, column=0, sticky=N+S+E+W, padx=2, pady=2)
            scrollbar = tki.Scrollbar(self.set_origin_L_W, command=text_box.yview)
            scrollbar.grid(row=1, column=2, sticky=N+S+E+W)
            text_box['yscrollcommand'] = scrollbar.set
            headers = ['Dimension_id','Value']
            coords = self.surface.coords[self.surface.propagation.OT_points[0][0]]
            rows = list(zip(range(coords.shape[0]), coords))
            df = pd.DataFrame(rows, columns=headers)
            text = df.to_string(index = False)
            text_box.delete("1.0",END)
            text_box.insert(END,text)
            text_box['state']="disabled"
            self.set_origin_L_W.wait_visibility()
            self.set_origin_L_W.grab_set()
        else:
            self.update_pathfinding_pipeline_states()

    #SET TARGET
    def set_target_id(self, input_index = -1):
        if (input_index == -1):
            input_index = self.target_id_E.get()
            good = False
            try:
                index = int(input_index)
                good = True
            except:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_target_W)
        else:
            index = input_index
            good = True
        if(good):
            if (index >= 0):
                if (index < self.surface.coords.shape[0]):
                    self.surface.propagation.set_OT([None,np.array([index])])
                    self.target_id_S.set(index)
                    self.update_pathfinding_pipeline_states()
                    self.set_target_W.destroy()
                else:
                    messagebox.showinfo( "ERROR", "The maximum index available is: " + str(self.surface.energy.shape[0] -1),parent=self.set_target_W)
            else:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0, not 1.",parent=self.set_target_W)


    def set_target_min_id(self, input_index = -1):
        if (input_index == -1):
            input_index = self.target_id_E.get()
            good = False
            try:
                index = int(input_index)
                good = True
            except:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_target_W)
        else:
            index = input_index
            good = True
        if(good):
            if (index >= 0):
                if (index < self.surface.propagation.gnw_minimum_clusters_indices.shape[0]):
                    self.surface.propagation.select_target_by_minimum_id(index)
                    self.target_id_S.set(index)
                    self.update_pathfinding_pipeline_states()
                    self.set_target_W.destroy()
                else:
                    messagebox.showinfo( "ERROR", "The maximum index available is: " + str(self.surface.propagation.gnw_minimum_clusters_indices.shape[0] -1),parent=self.set_target_W)
            else:
                messagebox.showinfo( "ERROR", "Index could not be read. Only positive integers are allowed.\nTIP: The first point in the file has index 0.",parent=self.set_target_W)


    def load_target_ranges(self):
        ranges_path = filedialog.askopenfilename(parent=self.set_target_ranges_W)
        if ranges_path is None or len(ranges_path) == 0:
            return
        if ranges_path != '' and ranges_path != 'none':
            try:
                df = pd.read_csv(ranges_path, sep='\s+', header = None)
                ranges = df.values
                if (np.all(ranges.shape == self.target_coord_ranges.shape) == True):
                    self.target_coord_ranges = ranges
                    self.set_target_ranges_W.destroy()
                    self.set_target_ranges()
                else:
                    messagebox.showinfo( "ERROR", "Provided target ranges do not match the expected shape.",parent=self.set_target_W)

            except:
                messagebox.showinfo( "ERROR", "target ranges file could not be parsed.",parent=self.set_target_W)

    def save_target_ranges(self):
        self.read_target_ranges()
        file_to_save = filedialog.asksaveasfile(parent=self.set_target_ranges_W,mode='w', defaultextension=".ranges")
        if file_to_save is None:
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                np.savetxt(file_to_save.name, self.target_coord_ranges, delimiter = "\t", fmt='%g')
            except:
                messagebox.showinfo( "ERROR", "Ranges could not be saved.")

    def read_target_ranges(self):
        try:
            df = pd.read_csv(io.StringIO(self.text_box.get("1.0",END)), sep='\s+')
        except:
            messagebox.showinfo( "ERROR", "Target ranges could not be parsed.",parent=self.set_target_W)
        ranges = df.values[:,1:]
        if (np.all(ranges.shape == self.target_coord_ranges.shape) == True):
            self.target_coord_ranges = ranges
            headers = ['Dimension_id','Range_lower_limit', 'Range_upper_limit']
            ranges_down = self.target_coord_ranges[:,0].tolist()
            ranges_up = self.target_coord_ranges[:,1].tolist()
            rows = list(zip(range(self.target_coord_ranges.shape[0]), ranges_down, ranges_up))
            df = pd.DataFrame(rows, columns=headers)
            text = "" + df.to_string(index = False)
            self.text_box.delete("1.0",END)
            self.text_box.insert(END,text)
        else:
            messagebox.showinfo( "ERROR", "Provided target ranges do not match the expected shape.",parent=self.set_target_W)

    def set_target_ranges(self):
        self.set_target_W.grab_release()
        self.set_target_ranges_W = tki.Toplevel(master=self.set_target_W)
        self.set_target_ranges_W.resizable(width=FALSE, height=FALSE)
        self.set_target_ranges_W.protocol("WM_DELETE_WINDOW", self.set_target_ranges_W_exit_handler)
        self.set_target_ranges_W.title("Minima search coordinates range")
        self.set_target_ranges_W.wait_visibility()
        self.set_target_ranges_W.grab_set()
        set_target_man_B = tki.Button(self.set_target_ranges_W, text = "Load coordinate ranges", command = self.load_target_ranges)
        set_target_man_B.grid(row = 0, column = 0, columnspan = 1,sticky=W+E)
        set_target_man_B = tki.Button(self.set_target_ranges_W, text = "Save coordinate ranges", command = self.save_target_ranges)
        set_target_man_B.grid(row = 0, column = 1, columnspan = 1,sticky=W+E)
        set_read_target_ranges_B = tki.Button(self.set_target_ranges_W, text = "Read target coordinate ranges", command = self.read_target_ranges)
        set_read_target_ranges_B.grid(row = 1, column = 0, columnspan = 1,sticky=W+E)
        set_target_man_B = tki.Button(self.set_target_ranges_W, text = "Set target as the lowest minimum in the coordinate ranges", command = self.set_target_from_ranges)
        set_target_man_B.grid(row = 1, column = 1, columnspan = 1,sticky=W+E)

        self.text_box = tki.Text(self.set_target_ranges_W, borderwidth=3, relief="sunken")
        self.text_box.config(font=("consolas", 12), undo=True, wrap='word')
        self.text_box.grid(row=2, column=0, sticky=N+S+E+W, columnspan=2)
        scrollbar = tki.Scrollbar(self.set_target_ranges_W, command=self.text_box.yview)
        scrollbar.grid(row=2, column=2, sticky=N+S+E+W)
        self.text_box['yscrollcommand'] = scrollbar.set
        headers = ['Dimension_id','Range_lower_limit', 'Range_upper_limit']
        ranges_down = self.target_coord_ranges[:,0].tolist()
        ranges_up = self.target_coord_ranges[:,1].tolist()
        rows = list(zip(range(self.target_coord_ranges.shape[0]), ranges_down, ranges_up))
        df = pd.DataFrame(rows, columns=headers)
        text = "" + df.to_string(index = False)
        self.text_box.delete("1.0",END)
        self.text_box.insert(END,text)

    def set_target_from_ranges(self):
        self.read_target_ranges()
        point = -1
        if (np.all(self.target_coord_ranges[:,0]<=self.target_coord_ranges[:,1])):
            point = self.surface.propagation.select_target_by_range(self.target_coord_ranges, minimum=True)
            if(point == -1):
                continue_bool = messagebox.askokcancel(title='Minima search warning', message="WARNING: No valid points could be found with the current ranges. Do you want to take a point with the lowest energy in this range?",parent=self.set_target_W)
                if (continue_bool):
                    point = self.surface.propagation.select_target_by_range(self.target_coord_ranges, minimum=False)
                    if(point==-1):
                        messagebox.showinfo( "ERROR", "No valid points could be found with the current ranges.",parent=self.set_target_W)
        else:
            messagebox.showinfo( "ERROR", "Current target ranges present at least one lower limit greater than its corresponding upper limit.",parent=self.set_target_W)
        if (point == self.surface.propagation.OT_points[1][0] and point != -1):
            self.set_target_id(point)

    def set_target_call(self):
        self.set_target_W = tki.Toplevel(master=self.pathfinding_pipeline_W)
        self.set_target_W.resizable(width=FALSE, height=FALSE)
        self.set_target_W.protocol("WM_DELETE_WINDOW", self.set_target_W_exit_handler)
        self.set_target_W.title("Set target")
        self.set_target_W.wait_visibility()
        self.set_target_W.grab_set()
        self.set_target_B['state']='disabled'
        target_id_L = tki.Label(master = self.set_target_W, text = "Point/Minima ID:")
        target_id_L.grid(row = 0, column = 0, columnspan = 1,sticky=W+E)
        self.target_id_S = tki.StringVar()
        self.target_id_E = tki.Entry(master = self.set_target_W,textvariable=self.target_id_S)
        self.target_id_E.grid(row = 0, column = 1, columnspan = 1,sticky=W+E)
        set_target_id_B = tki.Button(self.set_target_W, text = "Set target by point index", command = self.set_target_id)
        set_target_id_B.grid(row = 1, column = 0, columnspan = 2,sticky=W+E)
        set_target_min_id_B = tki.Button(self.set_target_W, text = "Set target by minimum index", command = self.set_target_min_id)
        set_target_min_id_B.grid(row = 2, column = 0, columnspan = 2,sticky=W+E)
        set_target_man_B = tki.Button(self.set_target_W, text = "Find target by coordinate ranges", command = self.set_target_ranges)
        set_target_man_B.grid(row = 3, column = 0, columnspan = 2,sticky=W+E)

    def show_target_coords(self, trash):
        self.force_pathfinding_pipeline_states()
        if (self.set_target_L['text'] != 'None'):
            self.set_target_L_W = tki.Toplevel(master=self.pathfinding_pipeline_W)
            self.set_target_L_W.protocol("WM_DELETE_WINDOW", self.set_target_L_W_exit_handler)
            self.set_target_L_W.resizable(width=FALSE, height=FALSE)
            self.set_target_L_W.title("Target coordinates")

            text_box = tki.Text(self.set_target_L_W, borderwidth=3, relief="sunken")
            text_box.config(font=("consolas", 12), undo=True, wrap='word')
            text_box.grid(row=1, column=0, sticky=N+S+E+W, padx=2, pady=2)
            scrollbar = tki.Scrollbar(self.set_target_L_W, command=text_box.yview)
            scrollbar.grid(row=1, column=2, sticky=N+S+E+W)
            text_box['yscrollcommand'] = scrollbar.set
            headers = ['Dimension_id','Value']
            coords = self.surface.coords[self.surface.propagation.OT_points[1][0]]
            rows = list(zip(range(coords.shape[0]), coords))
            df = pd.DataFrame(rows, columns=headers)
            text = df.to_string(index = False)
            text_box.delete("1.0",END)
            text_box.insert(END,text)
            text_box['state']="disabled"
            self.set_target_L_W.wait_visibility()
            self.set_target_L_W.grab_set()
        else:
            self.update_pathfinding_pipeline_states()

    #CALCULATIONS
    def calculate_path(self):
        self.force_pathfinding_pipeline_states()
        self.path_B.config(text="RUNNING...")
        self.path_B.grid()
        self.path_B.update()
        self.surface.propagation.get_npaths(n=0)
        self.path_B.config(text="Minimum Energy Path")
        self.path_B.grid()
        self.path_B.update()
        self.update_pathfinding_pipeline_states()


    def calculate_fpsr(self):
        self.force_pathfinding_pipeline_states()
        self.fpsr_B.config(text="RUNNING...")
        self.fpsr_B.grid()
        self.fpsr_B.update()
        self.surface.propagation.get_well_sampling()
        self.fpsr_B.config(text="Well sampling")
        self.fpsr_B.grid()
        self.fpsr_B.update()
        self.update_pathfinding_pipeline_states()

    def calculate_alternative_paths(self):
        self.force_pathfinding_pipeline_states()
        self.alternative_paths_B.config(text="RUNNING...")
        self.alternative_paths_B.grid()
        self.alternative_paths_B.update()
        n = self.alternative_paths_S.get()
        if(n.isdigit):
            n = int(n)+ len(self.surface.propagation.gnw_paths)
            if (len(self.surface.propagation.gnw_paths) > 0):
                n-=1
            self.surface.propagation.get_npaths(n)
        self.alternative_paths_B.config(text="Alternative paths")
        self.alternative_paths_B.grid()
        self.alternative_paths_B.update()
        self.update_pathfinding_pipeline_states()


    #PLOTS
    def get_points_and_trace(self,mode,n=0):
        if (mode == "Raw path"):
            trace = np.concatenate((self.surface.propagation.gnw_paths[n])[1])
        elif (mode == "Hide plateaus"):
            trace = np.concatenate((self.surface.propagation.gnw_paths[n])[2])
        elif (mode == "Lvl1 reduction path"):
            trace = np.concatenate((self.surface.propagation.gnw_paths[n])[3])
        elif (mode == "Lvl2 reduction path"):
            trace = np.concatenate((self.surface.propagation.gnw_paths[n])[4])
        elif (mode == "Lvl3 reduction path"):
            trace = np.concatenate((self.surface.propagation.gnw_paths[n])[5])
        elif (mode == "Simplified path"):
            trace = (self.surface.propagation.gnw_paths[n])[7]
        mask = trace!=0
        if (mode == "Simplified path"):
            points = np.argwhere(mask).flatten()
            trace = trace[points]
        else:
            points = np.concatenate((self.surface.propagation.gnw_paths[n])[0])
            points = points[mask]
            trace = trace[mask]
            unique_indices = reduce_dtype(np.unique(points, return_index =True)[1])
            points = points[unique_indices]
            trace = trace[unique_indices]
        order = np.argsort(trace).flatten()
        return reduce_dtype(points[order]), reduce_dtype(trace[order])

    def parse_selection(self,in_selection):
        selection=in_selection.strip()
        selection_filt=selection.replace("  ", " ")
        while(selection != selection_filt):
            selection=selection_filt
            selection_filt=selection.replace("  ", " ")
        if (selection == ""):
            selection = np.zeros(1,dtype=np.int_)
        else:
            selection_filt=selection.replace(" -", "-")
            selection=selection_filt
            selection_filt=selection.replace("- ", "-")
            selection=selection_filt
            selection_split = selection.split(" ")
            ids = []
            for element in selection_split:
                if (element.isdigit()):
                    ids.append(np.array([int(element),],dtype=np.int_))
                else:
                    element_split = element.split("-")
                    if (len(element_split)!=2):
                        ids =[np.array([-1],dtype=np.int_)]
                        break
                    if (element_split[0].isdigit()==False or element_split[1].isdigit()==False):
                        ids =[np.array([-1],dtype=np.int_)]
                        break
                    id_from = int(element_split[0])
                    id_to = int(element_split[1])
                    if (id_from>id_to):
                        id_from,id_to=id_to,id_from
                    indices = reduce_dtype(np.arange(id_from,id_to+1))
                    if (id_from==int(element_split[0])):
                        indices=np.flip(indices)
                    ids.append(indices)
            if ((ids[0])[0]!=-1):
                selection = np.unique(np.concatenate(ids))
            else:
                selection = np.empty(0,dtype=np.int_)
        return selection

    def plot_profile(self,mode):
        self.force_pathfinding_pipeline_states()
        selection = 0
        if (mode == "None"):
            return
        elif (mode == "Select minima and barriers by id"):
            string_var = "Select which minima to plot by index.\nTIP 1: First id is 0, not 1.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23."
            string_var = string_var + "\n\n\tMinima range = " + str(self.surface.propagation.gnw_minimum_clusters_indices[0]) + "-" + str(self.surface.propagation.gnw_minimum_clusters_indices[-1])
            string_var = string_var + "\n\tBarriers range = " + str(self.surface.propagation.gnw_barrier_clusters_indices[0]) + "-" + str(self.surface.propagation.gnw_barrier_clusters_indices[-1])
            selection = simpledialog.askstring("Minima and barrier selection",string_var,parent=self.pathfinding_pipeline_W,initialvalue="all")
            #self.pathfinding_pipeline_W.wait_visibility()
            self.pathfinding_pipeline_W.grab_set()
            if (selection is None):
                self.update_pathfinding_pipeline_states()
                return
            if ("all" in selection.lower()):
                selection = np.arange(self.surface.propagation.gnw_barrier_clusters_indices[-1]+1)
            else:
                selection = self.parse_selection(selection)
                selection = selection[selection<=self.surface.propagation.gnw_barrier_clusters_indices[-1]]
            if (selection.shape[0]==0):
                messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the minima or barriers indices available.",parent=self.pathfinding_pipeline_W)
                self.update_pathfinding_pipeline_states()
                return
            trace = selection
            energy = self.surface.propagation.gnw_energy[selection]
            plt.scatter(trace,energy)
        else:
            if (len(self.surface.propagation.gnw_paths) != 1):
                selection = simpledialog.askstring("Paths selection", "Select which paths to plot by index.\nTIP 1: Minimum energy path has index 0, first alternative path has index 1 and so on.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23.",parent=self.pathfinding_pipeline_W,initialvalue="0")
                #self.pathfinding_pipeline_W.wait_visibility()
                self.pathfinding_pipeline_W.grab_set()
                if (selection is None):
                    self.update_pathfinding_pipeline_states()
                    return
                if ("all" in selection.lower()):
                    selection = np.arange(len(self.surface.propagation.gnw_paths))
                else:
                    selection = self.parse_selection(selection)
                    selection = selection[selection<len(self.surface.propagation.gnw_paths)]
            else:
                selection = np.zeros(1,dtype=np.int_)
            selection = selection[selection<len(self.surface.propagation.gnw_paths)]
            if (selection.shape[0]==0):
                messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the paths calculated.",parent=self.pathfinding_pipeline_W)
                self.update_pathfinding_pipeline_states()
                return
            for index in selection:
                points, trace = self.get_points_and_trace(mode,n=index)
                energy = self.surface.energy[points]
                if (selection.shape[0] > 1):
                    plt.plot(trace,energy,label=str(index), alpha=0.5)
                else:
                    plt.plot(trace,energy)
        if (mode == "Select minima and barriers by id"):
            plt.xlabel('Minimum/barrier index')
        else:
            plt.xlabel('Propagation step')
        plt.ylabel('Energy')
        if (selection.shape[0] > 1 and selection.shape[0] <= 10 and mode != "Select minima and barriers by id"):
            leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            for lh in leg.legendHandles:
                lh.set_alpha(1.0)
        plt.show()
        self.update_pathfinding_pipeline_states()

    def evaluate_size_numbers(self, strings_list):
        out = []
        for string in strings_list:
            if(string.isdigit()):
                point_size=int(string)
                out.append(point_size)
            else:
                try:
                    point_size=float(string)
                    if(point_size >0):
                        out.append(point_size)
                except:
                    pass
        if (len(out)!=len(strings_list)):
            return []
        else:
            return out

    def evaluate_colors(self, strings_list):
        out = []
        for string in strings_list:
            if(self.evaluate_graph_argument(string,"color")):
                out.append(string)
            else:
                break
        if (len(out)!=len(strings_list)):
            return []
        else:
            return out

    def get_coord_by_name(self,points,coord_name):
        if (coord_name.isdigit()):
            return self.surface.coords[points,int(coord_name)]
        elif (coord_name == "Energy"):
            return self.surface.energy[points]

    def get_coord_str_by_name(self,coord_name):
        if (coord_name.isdigit()):
            return 'Coordinate ' + coord_name
        else:
            return coord_name

    def multi_coord_projection(self,mode,selection,input_sizes,input_colors,contour=False):
        coord1 = self.coord_1_S.get()
        coord2 = self.coord_2_S.get()
        coord3 = self.coord_3_S.get()
        fig = plt.figure()
        if(contour==True and coord3 == "None"):
            norm = cm.colors.Normalize(vmax=np.max(self.surface.energy),vmin=np.min(self.surface.energy))
            cmap = cm.rainbow
            total_points = np.arange(self.surface.energy.shape[0])
            plt.tricontour(self.get_coord_by_name(total_points,coord1),self.get_coord_by_name(total_points,coord2),self.surface.energy[total_points],colors='k', levels=int(0.5 +cmap.N/50), linewidths=0.25)
            tcf = plt.tricontourf(self.get_coord_by_name(total_points,coord1),self.get_coord_by_name(total_points,coord2),self.surface.energy[total_points], cmap=cmap, levels=cmap.N)
            plt.colorbar(tcf)
        if (coord3 != "None"):
            ax = fig.add_subplot(projection='3d')
        for path_id in selection:
            points, trace = self.get_points_and_trace(mode,path_id)
            size_array=np.empty(points.shape[0])
            size_array.fill(input_sizes[0])
            coords1 = self.get_coord_by_name(points,coord1)
            coords2 = self.get_coord_by_name(points,coord2)
            if (coord3 == "None"):
                if(contour==True):
                    plt.scatter(coords1,coords2, s=size_array, c=["black"] * points.shape[0], alpha = 0.35)
                else:
                    plt.scatter(coords1,coords2, s=size_array, alpha = 0.25, label=str(path_id))
                if (input_sizes[3]>0):
                    for i,point in enumerate(points):
                        if (self.surface.propagation.is_gnw_node[point]):
                            cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                            if cluster_id > -1:
                                s = str(point) + "("+str(cluster_id)+")"
                            plt.annotate(s, (coords1[i],coords2[i]),size=input_sizes[3],color=input_colors[3])
            else:
                coords3 = self.get_coord_by_name(points,coord3)
                ax = fig.gca()
                ax.scatter(coords1,coords2,coords3, s=size_array, alpha = 0.25, label=str(path_id))
                if (input_sizes[3]>0):
                    for i,point in enumerate(points):
                        if (self.surface.propagation.is_gnw_node[point]):
                            cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                            if cluster_id > -1:
                                s = str(point) + "("+str(cluster_id)+")"
                            ax.text(coords1[i],coords2[i],coords3[i],s,size=input_sizes[3],color=input_colors[3])

        if (coord3 == "None"):
            plt.xlabel(self.get_coord_str_by_name(coord1))
            plt.ylabel(self.get_coord_str_by_name(coord2))
        else:
            ax.set_xlabel(self.get_coord_str_by_name(coord1))
            ax.set_ylabel(self.get_coord_str_by_name(coord2))
            ax.set_zlabel(self.get_coord_str_by_name(coord3))
        if (selection.shape[0] > 1 and selection.shape[0] <= 10):
            leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            for lh in leg.legendHandles:
                lh.set_alpha(1.0)
        plt.show()
        if (contour == True):
            self.coord_projection_contour_plot_S.set("Plot")
        else:
            self.coord_projection_plot_S.set("Plot")
        self.update_coordinate_projections_states()

    def coord_projection_plot(self,mode,contour=False):
        if((self.coord_1_S.get() != self.coord_2_S.get()) and (self.coord_2_S.get() != self.coord_3_S.get()) and (self.coord_2_S.get() != self.coord_3_S.get())):
            self.force_coordinate_projections_states()
            input_sizes = self.evaluate_size_numbers([self.point_size_S.get(),self.O_point_size_S.get(),self.T_point_size_S.get(),self.label_size_S.get()])
            input_colors = self.evaluate_colors([self.point_color_S.get(),self.O_point_color_S.get(),self.T_point_color_S.get(),self.label_color_S.get()])
            if (len(input_sizes) != 4):
                messagebox.showinfo( "ERROR", "Size must be a positive integer or float.",parent=self.coordinate_projections_W)
                self.update_coordinate_projections_states()
                return
            if (len(input_colors) != 4):
                messagebox.showinfo( "ERROR", "Color format could not be understood.",parent=self.coordinate_projections_W)
                self.update_coordinate_projections_states()
                return
            if ("Well sampling" not in mode):
                selection = np.empty(0,dtype='uint64')
                if (mode == "Select minima and barriers by id"):
                    string_var = "Select which minima and/or barriers to plot by index.\nTIP 1: First id is 0, not 1.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23."
                    string_var = string_var + "\n\n\tMinima range = " + str(self.surface.propagation.gnw_minimum_clusters_indices[0]) + "-" + str(self.surface.propagation.gnw_minimum_clusters_indices[-1])
                    string_var = string_var + "\n\tBarriers range = " + str(self.surface.propagation.gnw_barrier_clusters_indices[0]) + "-" + str(self.surface.propagation.gnw_barrier_clusters_indices[-1])
                    selection = simpledialog.askstring("Minima and barrier selection", string_var,parent=self.coordinate_projections_W,initialvalue="all")
                    #self.coordinate_projections_W.wait_visibility()
                    self.coordinate_projections_W.grab_set()
                    if (selection is None):
                        self.update_coordinate_projections_states()
                        return
                    if ("all" in selection.lower()):
                        selection = np.arange(self.surface.propagation.gnw_barrier_clusters_indices[-1]+1,dtype='uint64')
                    else:
                        selection = self.parse_selection(selection)
                        selection = selection[selection<=self.surface.propagation.gnw_barrier_clusters_indices[-1]]

                    if (selection.shape[0]==0):
                        messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the minima or barriers indices available.",parent=self.pathfinding_pipeline_W)
                        self.update_coordinate_projections_states()
                        return
                elif(mode != "None"):
                    if (len(self.surface.propagation.gnw_paths) != 1):
                        selection = simpledialog.askstring("Paths selection", "Select which paths to plot by index.\nTIP 1: Minimum energy path has index 0, first alternative path has index 1 and so on.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23.",parent=self.coordinate_projections_W,initialvalue="0")
                        #self.coordinate_projections_W.wait_visibility()
                        self.coordinate_projections_W.grab_set()
                        if (selection is None):
                            self.update_coordinate_projections_states()
                            return
                        if ("all" in selection.lower()):
                            selection = np.arange(len(self.surface.propagation.gnw_paths))
                        else:
                            selection = self.parse_selection(selection)
                            selection = selection[selection<len(self.surface.propagation.gnw_paths)]
                    else:
                        selection = np.zeros(1,dtype=np.int_)
                    selection = selection[selection<len(self.surface.propagation.gnw_paths)]
                    if (selection.shape[0]==0 and mode != "None"):
                        messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the paths calculated.",parent=self.pathfinding_pipeline_W)
                        self.update_coordinate_projections_states()
                        return
                if ((selection.shape[0] > 1 and mode != "Select minima and barriers by id") or mode == "None"):
                    self.multi_coord_projection(mode,selection,input_sizes,input_colors,contour=contour)
                    self.update_coordinate_projections_states()
                    return
                else:
                    if (mode == "Select minima and barriers by id"):
                        points_list = []
                        for node_id in selection:
                            points = self.surface.propagation.gnw_clusters[node_id]
                            points_list.append(points)
                        points =  np.unique(np.concatenate(points_list))
                        energy = self.surface.propagation.energy[points]
                        order = np.argsort(-energy)
                        points = points[order]
                        energy = energy[order]
                        colors = [input_colors[0]] * points.shape[0]
                        O_points = np.empty(0,dtype=np.int_)
                        T_points = np.empty(0,dtype=np.int_)
                        size_array=np.empty(points.shape[0])
                        size_array.fill(input_sizes[0])

                    else:
                        points, trace = self.get_points_and_trace(mode,selection[0])
                        O_points = np.argwhere(trace==trace[0]).flatten()
                        T_points = np.argwhere(trace==trace[-1]).flatten()
                        sorter = np.zeros(points.shape,dtype=np.int_)
                        sorter[O_points] = 1
                        sorter[T_points] = 2
                        order = np.argsort(sorter).flatten()
                        points=points[order]
                        sorter=sorter[order]
                        O_points = np.argwhere(sorter==1).flatten()
                        T_points = np.argwhere(sorter==2).flatten()

                        size_array=np.empty(points.shape[0])
                        size_array.fill(input_sizes[0])
                        colors = [input_colors[0]] * points.shape[0]

                        for p in O_points:
                            colors[p] = input_colors[1]
                        size_array[O_points] = input_sizes[1]

                        for p in T_points:
                            colors[p] = input_colors[2]
                        size_array[T_points] = input_sizes[2]
            else:
                well_sampling_annotation = self.surface.propagation.fpsr_annotation
                if (mode == "Well sampling: all"):
                    points = np.argwhere(well_sampling_annotation!=0).flatten()
                    order = np.argsort(well_sampling_annotation[points]).flatten()
                    points=points[order]

                    size_array=np.empty(points.shape[0])
                    size_array.fill(input_sizes[0])
                    annotation = well_sampling_annotation[points]
                    colors = [input_colors[0]] * points.shape[0]
                    O_points = np.argwhere(annotation==1).flatten()
                    for p in O_points:
                        colors[p] = input_colors[1]
                    T_points = np.argwhere(annotation==2).flatten()
                    for p in T_points:
                        colors[p] = input_colors[2]
                    size_array[O_points] = input_sizes[1]
                    size_array[T_points] = input_sizes[2]
                elif (mode == "Well sampling: origin well region"):
                    points = np.argwhere(well_sampling_annotation==1).flatten()
                    order = np.argsort(well_sampling_annotation[points]).flatten()
                    points=points[order]

                    colors = [input_colors[1]] * points.shape[0]
                    size_array=np.empty(points.shape[0])
                    size_array.fill(input_sizes[1])
                elif (mode == "Well sampling: target well region"):
                    points = np.argwhere(well_sampling_annotation==2).flatten()
                    order = np.argsort(well_sampling_annotation[points]).flatten()
                    points=points[order]

                    colors = [input_colors[2]] * points.shape[0]
                    size_array=np.empty(points.shape[0])
                    size_array.fill(input_sizes[2])
                elif (mode == "Well sampling: barrier/intermediate region"):
                    points = np.argwhere(well_sampling_annotation==3).flatten()
                    order = np.argsort(well_sampling_annotation[points]).flatten()
                    points=points[order]

                    colors = [input_colors[0]] * points.shape[0]
                    size_array=np.empty(points.shape[0])
                    size_array.fill(input_sizes[0])
            coord1 = self.coord_1_S.get()
            coord2 = self.coord_2_S.get()
            coord3 = self.coord_3_S.get()
            coords1 = self.get_coord_by_name(points,coord1)
            coords2 = self.get_coord_by_name(points,coord2)

            if (coord3 == "None"):
                if(contour==True):
                    norm = cm.colors.Normalize(vmax=np.max(self.surface.energy),vmin=np.min(self.surface.energy))
                    cmap = cm.rainbow
                    total_points = np.arange(self.surface.energy.shape[0])
                    plt.tricontour(self.get_coord_by_name(total_points,coord1),self.get_coord_by_name(total_points,coord2),self.surface.energy[total_points],colors='k', levels=int(0.5 +cmap.N/50), linewidths=0.25)
                    tcf = plt.tricontourf(self.get_coord_by_name(total_points,coord1),self.get_coord_by_name(total_points,coord2),self.surface.energy[total_points],cmap=cmap, levels=cmap.N)
                    plt.colorbar(tcf)
                    plt.scatter(coords1,coords2, s=size_array, c=colors, alpha = 1)
                else:
                    plt.scatter(coords1,coords2, s=size_array, c=colors, alpha = 1)
                plt.xlabel(self.get_coord_str_by_name(coord1))
                plt.ylabel(self.get_coord_str_by_name(coord2))
                if (input_sizes[3]>0):
                    if (mode == "Select minima and barriers by id"):
                        for i,point in enumerate(points):
                            if (self.surface.propagation.is_gnw_node[point]):
                                cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                                if cluster_id > -1:
                                    s = str(point) + "("+str(cluster_id)+")"
                                    plt.annotate(s, (coords1[i],coords2[i]),size=input_sizes[3],color=input_colors[3])
                    else:
                        for i,point in enumerate(points):
                            if (self.surface.propagation.is_gnw_node[point]):
                                cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                                if cluster_id > -1:
                                    s = str(point) + "("+str(cluster_id)+")"
                                    plt.annotate(s, (coords1[i],coords2[i]),size=input_sizes[3],color=input_colors[3])
                plt.show()
            else:
                coords3 = self.get_coord_by_name(points,coord3)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(coords1,coords2,coords3, s=size_array, c=colors, cmap="viridis", alpha = 1)
                ax.set_xlabel(self.get_coord_str_by_name(coord1))
                ax.set_ylabel(self.get_coord_str_by_name(coord2))
                ax.set_zlabel(self.get_coord_str_by_name(coord3))
                if (input_sizes[3]>0):
                    if (mode == "Select minima and barriers by id"):
                        for i,point in enumerate(points):
                            if (self.surface.propagation.is_gnw_node[point]):
                                cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                                if cluster_id > -1:
                                    s = str(point) + "("+str(cluster_id)+")"
                                    ax.text(coords1[i],coords2[i],coords3[i],s,size=input_sizes[3],color=input_colors[3])
                    else:
                        for i,point in enumerate(points):
                            if (self.surface.propagation.is_gnw_node[point]):
                                cluster_id = max(self.surface.propagation.gnw_minimum_clusters_location[point], self.surface.propagation.gnw_barrier_clusters_location[point])
                                if cluster_id > -1:
                                    s = str(point) + "("+str(cluster_id)+")"
                                    ax.text(coords1[i],coords2[i],coords3[i],s,size=input_sizes[3],color=input_colors[3])
                plt.show()
        if (contour == True):
            self.coord_projection_contour_plot_S.set("Contour plot")
        else:
            self.coord_projection_plot_S.set("Plot")
        self.update_coordinate_projections_states()

    def coord_projection_contour_plot(self,mode):
        self.coord_projection_plot(mode,contour=True)

    #PATHFINDING WINDOWS

    def get_points_from_selection(self,selection):
        if (selection == "All"):
            return None
        elif (selection == "Select minima and barriers by id"):
            ids_selection = self.node_id_E.get()
            if ("all" in ids_selection.lower()):
                node_ids = np.arange(self.surface.propagation.gnw_barrier_clusters_indices[-1]+1)
            else:
                node_ids = self.parse_selection(self.node_id_E.get())
                node_ids = node_ids[node_ids<=self.surface.propagation.gnw_barrier_clusters_indices[-1]]
            if (node_ids.shape[0] == 0):
                return np.empty(0,dtype=np.int_)
            else:
                points_list = []
                for node_id in node_ids:
                    points = self.surface.propagation.gnw_clusters[node_id]
                    points_list.append(points)
                return np.unique(np.concatenate(points_list))
            return np.empty(0,dtype=np.int_)
        elif (selection == "Origin minimum"):
            return self.surface.propagation.gnw_clusters[self.surface.propagation.OT_cluster_ids[0][0]]
        elif (selection == "Target minimum"):
            return self.surface.propagation.gnw_clusters[self.surface.propagation.OT_cluster_ids[1][0]]
        elif ("Well sampling" not in selection):
            if (len(self.surface.propagation.gnw_paths) != 1):
                path_ids_selection = self.node_id_E.get()
                if (path_ids_selection is None):
                    return
                if ("all" in path_ids_selection.lower()):
                    path_ids_selection = np.arange(len(self.surface.propagation.gnw_paths))
                else:
                    path_ids_selection = self.parse_selection(path_ids_selection)
                    path_ids_selection = path_ids_selection[path_ids_selection<len(self.surface.propagation.gnw_paths)]
            else:
                path_ids_selection = np.zeros(1,dtype=np.int_)
            if (path_ids_selection.shape[0] == 0):
                return np.empty(0,dtype=np.int_)
            else:
                points_list = []
                for path_id in path_ids_selection:
                    points,trace = self.get_points_and_trace(selection,n=path_id)
                    points_list.append(points)
                return np.unique(np.concatenate(points_list))
        else:
            if (selection == "Well sampling: all"):
                    out = np.argwhere(self.surface.propagation.fpsr_annotation!=0).flatten()
                    if (self.surface.propagation.OT_not_node_fix[0] is not None):
                        if(self.surface.propagation.OT_not_node_fix[0].shape[0] > 0):
                            out = np.unique(np.concatenate((out,self.surface.propagation.OT_not_node_fix[0])))
                    if (self.surface.propagation.OT_not_node_fix[1] is not None):
                        if(self.surface.propagation.OT_not_node_fix[1].shape[0] > 0):
                            out = np.unique(np.concatenate((out,self.surface.propagation.OT_not_node_fix[1])))
                    return out
            elif (selection == "Well sampling: origin well region"):
                    out = np.argwhere(self.surface.propagation.fpsr_annotation==1).flatten()
                    if (self.surface.propagation.OT_not_node_fix[0] is not None):
                        if(self.surface.propagation.OT_not_node_fix[0].shape[0] > 0):
                            out = np.unique(np.concatenate((out,self.surface.propagation.OT_not_node_fix[0])))
                    return out
            elif (selection == "Well sampling: target well region"):
                    out = np.argwhere(self.surface.propagation.fpsr_annotation==2).flatten()
                    if (self.surface.propagation.OT_not_node_fix[1] is not None):
                        if(self.surface.propagation.OT_not_node_fix[1].shape[0] > 0):
                            out = np.unique(np.concatenate((out,self.surface.propagation.OT_not_node_fix[1])))
                    return out
            elif (selection == "Well sampling: barrier/intermediate region"):
                    return np.argwhere(self.surface.propagation.fpsr_annotation==3).flatten()

    def reset_graph(self, ask = True):
        self.force_network_projections_states()
        if (ask == True):
            restart_bool = messagebox.askokcancel(title='Graph reset warning', message='Are you sure you want to reset to default graph parameters?',parent=self.network_projections_W)
            #self.network_projections_W.wait_visibility()
            self.network_projections_W.grab_set()
        else:
            restart_bool = True
        if (restart_bool == True):
            print("Loading default graph parameters")
            points = None
            color = None
            size = 20
            alpha = 0.25
            frame_color = 'black'
            frame_width = 0
            frame_alpha = 1
            shape = 'circle'
            label = None
            label_dist = 0
            label_angle = 0
            label_color = 'black'
            label_size = 0
            bring_to = 'front'
            add_tags = None
            remove_tags = None
            self.surface.gnw_graph.set_node_attributes(points = points, minima=True,barriers=True, color = color, size = size, alpha = alpha, frame_color = frame_color, frame_width = frame_width, frame_alpha = frame_alpha, shape = shape, label = label, label_dist = label_dist, label_angle = label_angle, label_color = label_color, label_size = label_size, bring_to = bring_to, add_tags = add_tags, remove_tags = remove_tags)
            alpha = 0.25
            width = 10
            self.surface.gnw_graph.set_edge_attributes(points = points, color = color, width = width, alpha = alpha, bring_to = bring_to, add_tags = add_tags, remove_tags = remove_tags)
            self.surface.gnw_graph.assign_cmap_colors()
            self.surface.gnw_graph.set_node_indices_as_label(points=points,minima=True,barriers=True)
            if (self.surface.barriers_as_edges == False):
                self.surface.gnw_graph.set_node_attributes(points = points, minima=False, barriers=True, shape="triangle-up")
        self.update_network_projections_states()

    def customize_graph(self):
        self.force_network_projections_states()
        selection = self.customization_selection_S.get()
        minima_barriers = self.customization_minima_or_edges_S.get()
        action = self.action_S.get()
        value = self.action_value_S.get()
        points = self.get_points_from_selection(selection)
        if points is not None:
            points = np.unique(points)
        if (points is not None):
            if(points.shape[0] == 0):
                messagebox.showinfo( "ERROR", "The selection represents 0 nodes.",parent=self.network_projections_W)
                self.update_network_projections_states()
                return
        good = False
        nodes = False
        edges = False
        minima = False
        barriers = False
        if (minima_barriers == "Minimum vertices" or minima_barriers == "Both" or minima_barriers == "Barrier vertices" or minima_barriers == "All vertices" or minima_barriers == "All vertices+edges"):
            nodes = True
        if (minima_barriers == "Barrier edges" or minima_barriers == "Both" or minima_barriers == "All vertices+edges"):
            edges = True
        if (minima_barriers == "Minimum vertices" or minima_barriers == "All vertices" or minima_barriers == "All vertices+edges"):
            minima = True
        if (minima_barriers == "Barrier vertices" or minima_barriers == "All vertices" or minima_barriers == "All vertices+edges"):
            barriers = True
        if (action == "Assign color map"):
            if (value == ""):
                cmap_name = 'rainbow'
            else:
                cmap_name = value
            if (self.evaluate_graph_argument(cmap_name,"cmap name")):
                self.surface.gnw_graph.assign_cmap_colors(points = points, minima = minima, barriers = barriers, cmap_name = cmap_name, nodes = nodes, edges = edges, independent = False)
                good = True
        elif (action == "Set cluster indices as labels"):
            self.surface.gnw_graph.set_node_indices_as_label(points, minima = minima, barriers = barriers)
            good = True
        elif (action == "Set cluster energy as labels"):
            self.surface.gnw_graph.set_node_energy_as_label(points, minima = minima, barriers = barriers)
            good = True
        else:
            color = None
            size = None
            alpha = None
            frame_color = None
            frame_width = None
            frame_alpha = None
            shape = None
            label = None
            label_dist = None
            label_angle = None
            label_color = None
            label_size = None
            bring_to = None
            add_tags = None
            remove_tags = None
            width = None
            if (action == "Set color"):
                if (self.evaluate_graph_argument(value,"color")):
                    color = value
                    good = True
            elif (action == "Set size"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    size = float(value)
                    good = True
            elif (action == "Set alpha"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    alpha = float(value)
                    if (alpha >=0 and alpha <= 1):
                        good = True
            elif (action == "Set frame color"):
                if (self.evaluate_graph_argument(value,"color")):
                    frame_color = value
                    good = True
            elif (action == "Set frame width"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    frame_width = float(value)
                    good = True
            elif (action == "Set frame alpha"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    frame_alpha = float(value)
                    if (frame_alpha >=0 and frame_alpha <= 1):
                        good = True
            elif (action == "Set shape"):
                if (self.evaluate_graph_argument(value,"shape")):
                    shape = value
                    good = True
            elif (action == "Set label"):
                label = value
                good = True
            elif (action == "Set label distance"):
                if (self.evaluate_graph_argument(value,"number")):
                    label_dist = float(value)
                    good = True
            elif (action == "Set label angle"):
                if (self.evaluate_graph_argument(value,"number")):
                    label_angle = math.radians(float(value))
                    good = True
            elif (action == "Set label color"):
                if (self.evaluate_graph_argument(value,"color")):
                    label_color = value
                    good = True
            elif (action == "Set label size"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    label_size = float(value)
                    good = True
            elif (action == "Bring to front"):
                bring_to = "front"
                good = True
            elif (action == "Bring to back"):
                bring_to = "back"
                good = True
            elif (action == "Add tag"):
                if (value != ""):
                    add_tags = [value,]
                    good = True
            elif (action == "Remove tag"):
                if (value != ""):
                    remove_tags = [value,]
                    good = True
            elif (action == "Set width"):
                if (self.evaluate_graph_argument(value,"positive number")):
                    width = float(value)
                    good = True
            if (good == True):
                if (nodes == True):
                    self.surface.gnw_graph.set_node_attributes(points = points, minima = minima, barriers = barriers, color = color, size = size, alpha = alpha, frame_color = frame_color, frame_width = frame_width, frame_alpha = frame_alpha, shape = shape, label = label, label_dist = label_dist, label_angle = label_angle, label_color = label_color, label_size = label_size, bring_to = bring_to, add_tags = add_tags, remove_tags = remove_tags)
                if (edges == True):
                    self.surface.gnw_graph.set_edge_attributes(points = points, color = color, width = width, alpha = alpha, bring_to = bring_to, add_tags = add_tags, remove_tags = remove_tags)
        self.update_network_projections_states()
        if (good == False):
            messagebox.showinfo( "ERROR", "The value provided does not fit the action selected.",parent=self.network_projections_W)

    def set_barriers_as_edges(self):
        self.surface.barriers_as_edges = self.ask_barriers_as_edges_S.get() == "No"
        self.update_network_projections_states()

        self.create_network_graph()

    def ask_barriers_as_edges(self):
        self.force_network_projections_states()
        self.ask_barriers_as_edges_W = tki.Toplevel(master=self.network_projections_W)
        self.ask_barriers_as_edges_W.resizable(width=FALSE, height=FALSE)
        seln.ask_barriers_as_edges_W.protocol("WM_DELETE_WINDOW", self.ask_barriers_as_edges_W_exit_handler)
        self.ask_barriers_as_edges_W.title("Represent barriers as vertices")
        self.ask_barriers_as_edges_W.wait_visibility()
        self.ask_barriers_as_edges_W.grab_set()
        self.ask_barriers_as_edges_S = tki.StringVar(self.ask_barriers_as_edges_W)
        self.ask_barriers_as_edges_S.set("Yes")
        spacer_L = tki.Label(master = self.ask_barriers_as_edges_W, text = "Represent barriers as vertices?\n")
        spacer_L.grid(row = 0, column = 0, columnspan = 3,sticky=W+E)
        options = ["Yes","No"]
        for i,text in enumerate(options):
            self.ask_barriers_as_edges_B = tki.Radiobutton(self.ask_barriers_as_edges_W, text=text, variable=self.ask_barriers_as_edges_S, value=text)
            self.ask_barriers_as_edges_B.grid(row =i+1, column = 1, columnspan = 1,sticky=W)
        spacer_L = tki.Label(master = self.ask_barriers_as_edges_W, text = "")
        spacer_L.grid(row = i+2, column = 0, columnspan = 1,sticky=W+E)
        set_barriers_as_edges_B = tki.Button(self.ask_barriers_as_edges_W, text = "Select", command = self.set_barriers_as_edges)
        set_barriers_as_edges_B.grid(row = i+3, column = 0, columnspan = 3,sticky=W+E)

    def create_network_graph(self):
        if (network_dependencies == False):
            return
        self.ask_barriers_as_edges_W_exit_handler()
        self.force_network_projections_states()
        self.create_nw_graph_B.config(text="RUNNING...")
        self.create_nw_graph_B.grid()
        self.create_nw_graph_B.update()
        self.surface.create_gnw_graph()
        self.reset_graph(ask=False)
        self.create_nw_graph_B.config(text="Create network graph")
        self.create_nw_graph_B.grid()
        self.create_nw_graph_B.update()
        self.update_network_projections_states()
        self.network_projections_W_exit_handler()
        self.network_projections_call()


    def evaluate_graph_argument(self, argument, kind):
        if (kind == "cmap name"):
            try:
                cm.get_cmap(argument)
            except:
                return False
        elif (kind == "color"):
            try:
                rgba_color = mpl.colors.to_rgba(argument)
                mpl.colors.to_hex(rgba_color,keep_alpha = True)
            except:
                return False
        elif (kind == "positive number"):
            try:
                if(float(argument) <0):
                    return False
            except:
                return False
        elif (kind == "number"):
            try:
                float(argument)
            except:
                return False
        elif (kind == "shape"):
            shapes = {"rectangle":1, "circle":1, "hidden":1, "triangle-up":1, "triangle-down":1}
            if argument not in shapes:
                return False
        else:
            print("Warning: Argument type not understood.")
            return False
        return True


    def set_action_list(self):
        mode = self.customization_minima_or_edges_S.get()
        if (mode == "Minimum vertices" or mode == "Barrier vertices" or mode == "All vertices"):
            actions = [
            "Set color",
            "Assign color map",
            "Set size",
            "Set alpha",
            "Set frame width",
            "Set frame color",
            "Set frame alpha",
            "Set shape",
            "Set label",
            "Set label size",
            "Set label color",
            "Set label distance",
            "Set label angle",
            "Set cluster indices as labels",
            "Set cluster energy as labels",
            "Bring to front",
            "Bring to back",
            "Add tag",
            "Remove tag"
            ]
        elif (mode == "Barrier edges"):
            actions = [
            "Set color",
            "Assign color map",
            "Set width",
            "Set alpha",
            "Bring to front",
            "Bring to back",
            "Add tag",
            "Remove tag"
            ]
        else:
            actions = [
            "Set color",
            "Assign color map",
            "Set alpha",
            "Bring to front",
            "Bring to back",
            "Add tag",
            "Remove tag"
            ]
        self.action_B['menu'].delete(0, 'end')
        for action in actions:
                self.action_B['menu'].add_command(label=action, command=tki._setit(self.action_S, action))
        self.action_S.set(actions[0])


    def run_forceatlas2(self):
        self.force_network_projections_states()
        steps = self.forceatlas2_steps_S.get()
        edgeWeightInfluence = self.forceatlas2_weight_S.get()
        scalingRatio = self.forceatlas2_scaling_S.get()
        gravity = self.forceatlas2_gravity_S.get()
        outboundAttractionDistribution = bool(self.forceatlas2_outbound_I.get())
        resume = bool(self.forceatlas2_restart_I.get())

        good = True
        print("Start")
        if (not self.evaluate_graph_argument(steps,"positive number")):
            good = False
        steps = int(steps)
        if (steps <= 0):
            good = False
        if (not self.evaluate_graph_argument(edgeWeightInfluence,"positive number")):
            good = False
        edgeWeightInfluence = float(edgeWeightInfluence)
        if (not self.evaluate_graph_argument(scalingRatio,"positive number")):
            good = False
        scalingRatio = float(scalingRatio)
        if (not self.evaluate_graph_argument(gravity,"positive number")):
            good = False
        gravity = float(gravity)
        if (good==True):
            self.forceatlas2_B.config(text="RUNNING...")
            self.forceatlas2_B.grid()
            self.forceatlas2_B.update()
            self.surface.gnw_graph.compute_layout(edgeWeightInfluence=edgeWeightInfluence,outboundAttractionDistribution=outboundAttractionDistribution,scalingRatio=scalingRatio,gravity=gravity,verbose=True,steps = steps,resume=resume)
            self.forceatlas2_B.config(text="Compute layout")
            self.forceatlas2_B.grid()
            self.forceatlas2_B.update()
            self.update_network_projections_states()
        else:
            messagebox.showinfo( "ERROR", "The option values provided are no valid.",parent=self.network_projections_W)
    def graph_plot(self):
        self.force_network_projections_states()
        width = self.graph_width_S.get()
        height = self.graph_height_S.get()
        background = self.graph_background_S.get()
        good = True
        if (not self.evaluate_graph_argument(width,"positive number")):
            good = False
        width = float(width)
        if (not self.evaluate_graph_argument(height,"positive number")):
            good = False
        height = float(height)
        if (background=="None" or background == "none" or background == ""):
            background = None
        elif (not self.evaluate_graph_argument(background,"color")):
            good = False
        if (good):
            try:
                self.surface.gnw_graph.view_plot(width = width, height = height, background = background)
            except:
                good=False
        if (good==False):
            messagebox.showinfo( "ERROR", "The option values provided are no valid.",parent=self.network_projections_W)
        self.update_network_projections_states()

    def export_graph(self):
        self.force_network_projections_states()
        file_to_save = filedialog.asksaveasfile(parent=self.network_projections_W, mode='w', defaultextension=".graphml")
        if file_to_save is None:
            self.update_network_projections_states()
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.export_nw_graph_B.config(text="EXPORTING...")
                self.export_nw_graph_B.grid()
                self.export_nw_graph_B.update()
                self.surface.gnw_graph.save_graph_to_graphml(file_name = file_to_save.name)
                self.export_nw_graph_B.config(text="Export to graphml")
                self.export_nw_graph_B.grid()
                self.export_nw_graph_B.update()
            except:
                self.export_nw_graph_B.config(text="Export to graphml")
                self.export_nw_graph_B.grid()
                self.export_nw_graph_B.update()
                messagebox.showinfo( "ERROR", "Graph could not be exported to graphml format.")
        self.update_network_projections_states()

    def save_graph(self):
        self.force_network_projections_states()
        width = self.graph_width_S.get()
        height = self.graph_height_S.get()
        background = self.graph_background_S.get()
        good = True
        if (not self.evaluate_graph_argument(width,"positive number")):
            good = False
        width = float(width)
        if (not self.evaluate_graph_argument(height,"positive number")):
            good = False
        height = float(height)
        if (background=="None" or background == "none" or background == ""):
            background = None
        elif (not self.evaluate_graph_argument(background,"color")):
            good = False
        if (good):
            file_to_save = filedialog.asksaveasfile(parent=self.network_projections_W, mode='w', defaultextension=".svg")
            if file_to_save is None:
                self.update_network_projections_states()
                return
            if file_to_save != '' and file_to_save != 'none':
                try:
                    self.graph_save_B.config(text="SAVING...")
                    self.graph_save_B.grid()
                    self.graph_save_B.update()
                    self.surface.gnw_graph.save_plot_to_file(file_name = file_to_save.name, width = width, height = height, background = background)
                    self.graph_save_B.config(text="Save plot")
                    self.graph_save_B.grid()
                    self.graph_save_B.update()
                except:
                    self.graph_save_B.config(text="Save plot")
                    self.graph_save_B.grid()
                    self.graph_save_B.update()
                    messagebox.showinfo( "ERROR", "Graph could not be saved.")
        else:
            messagebox.showinfo( "ERROR", "The option values provided are no valid.",parent=self.network_projections_W)
        self.update_network_projections_states()

    def create_network_from_selection(self):
        if (network_dependencies == False):
            return
        self.force_network_projections_states()
        self.create_sele_nw_B.config(text="CREATING...")
        self.create_sele_nw_B.grid()
        self.create_sele_nw_B.update()
        selection = self.customization_selection_S.get()
        continue_bool = messagebox.askokcancel(title='Network creation warning', message='Creating a new network will create a completely new surface and close the current one. Thus, any unsaved progress will be lost. Continue?',parent=self.network_projections_W)
        if(continue_bool):
            if (selection == "All"):
                points = np.arange(self.surface.energy.shape[0])
            else:
                points = self.get_points_from_selection(selection)
            print("Number of points in new surface: " + str(points.shape[0]))
            if (points.shape[0] >=3):
                new_indices = reduce_dtype(np.arange(points.shape[0]))
                new_OT_points = np.copy(self.surface.propagation.OT_points)
                id_reference = np.empty(self.surface.energy.shape, dtype=np.int_)
                id_reference.fill(-1)
                id_reference[points]=new_indices
                for x in range(2):
                    new_OT_points[x] = id_reference[new_OT_points[x]][0]
                self.surface.coords = self.surface.coords[points]
                self.surface.energy = self.surface.energy[points]
                new_neighbors = []
                for p in points:
                    new_neighbor_points = id_reference[self.surface.connectivity.neighbors[p]]
                    new_neighbors.append(new_neighbor_points[new_neighbor_points!=-1])
                self.surface.connectivity.neighbors = new_neighbors

                self.surface.propagation = propagation_handler(self.surface.coords, self.surface.energy, self.surface.connectivity)
                self.surface.propagation.get_gnw()
                self.surface.propagation.set_OT(new_OT_points)
                self.gnw_graph = None
                self.surface.gnw_graph_done=False

        self.create_sele_nw_B.config(text="Create network from selection")
        self.create_sele_nw_B.grid()
        self.create_sele_nw_B.update()
        self.update_pathfinding_pipeline_states()
        if (continue_bool):
            self.network_projections_W_exit_handler()
        else:
            self.update_network_projections_states()
    def network_projections_call(self):
        self.update_pathfinding_pipeline_states() #this here to update self.combined_options_list
        self.network_projections_W = tki.Toplevel(master=self.pathfinding_pipeline_W, takefocus=True)
        #self.network_projections_W.wait_visibility()
        self.network_projections_W.grab_set()
        self.network_projections_W.resizable(width=FALSE, height=FALSE)
        self.network_projections_W.protocol("WM_DELETE_WINDOW", self.network_projections_W_exit_handler)
        self.network_projections_W.title("Network projections")
        self.create_nw_graph_B = tki.Button(self.network_projections_W, text = "Create network graph", command = self.ask_barriers_as_edges)
        self.create_nw_graph_B.grid(row = 0, column = 0, columnspan = 3,sticky=W+E)
        self.export_nw_graph_B = tki.Button(self.network_projections_W, text = "Export to graphml", command = self.export_graph)
        self.export_nw_graph_B.grid(row = 0, column = 3, columnspan = 3,sticky=W+E)
        self.layaout_L = tki.Label(self.network_projections_W, text = "\nFORCE ATLAS 2 LAYOUT\n")
        self.layaout_L.grid(row =1, column = 0, columnspan = 6,sticky=W+E)

        self.forceatlas2_steps_S = tki.StringVar(self.network_projections_W)
        self.forceatlas2_steps_S.set("2000")

        self.forceatlas2_weight_S = tki.StringVar(self.network_projections_W)
        self.forceatlas2_weight_S.set("0")

        self.forceatlas2_scaling_S = tki.StringVar(self.network_projections_W)
        self.forceatlas2_scaling_S.set("100")

        self.forceatlas2_gravity_S = tki.StringVar(self.network_projections_W)
        self.forceatlas2_gravity_S.set("1")

        self.forceatlas2_outbound_I = tki.IntVar(self.network_projections_W)
        self.forceatlas2_outbound_I.set(1)

        self.forceatlas2_restart_I = tki.IntVar(self.network_projections_W)
        self.forceatlas2_restart_I.set(1)


        self.forceatlas2_steps_L = tki.Label(self.network_projections_W, text = "Steps: ")
        self.forceatlas2_steps_L.grid(row = 2, column = 0, columnspan = 1,sticky=W)

        self.forceatlas2_scaling_L = tki.Label(self.network_projections_W, text = "Scaling ratio: ")
        self.forceatlas2_scaling_L.grid(row = 2, column = 1, columnspan = 1,sticky=W)

        self.forceatlas2_gravity_L = tki.Label(self.network_projections_W, text = "Gravity: ")
        self.forceatlas2_gravity_L.grid(row = 2, column = 2, columnspan = 1,sticky=W)

        self.forceatlas2_outbound_L = tki.Label(self.network_projections_W, text = "Outbound attraction")
        self.forceatlas2_outbound_L.grid(row = 2, column = 4, columnspan = 1,sticky=W)

        self.forceatlas2_restart_L = tki.Label(self.network_projections_W, text = "Start from previous layout")
        self.forceatlas2_restart_L.grid(row = 2, column = 5, columnspan = 1,sticky=W)

        self.forceatlas2_steps_E = tki.Entry(master = self.network_projections_W,textvariable=self.forceatlas2_steps_S)
        self.forceatlas2_steps_E.grid(row = 3, column = 0, columnspan = 1,sticky=W)

        self.forceatlas2_scaling_E = tki.Entry(master = self.network_projections_W,textvariable=self.forceatlas2_scaling_S)
        self.forceatlas2_scaling_E.grid(row = 3, column = 1, columnspan = 1,sticky=W)

        self.forceatlas2_gravity_E = tki.Entry(master = self.network_projections_W,textvariable=self.forceatlas2_gravity_S)
        self.forceatlas2_gravity_E.grid(row = 3, column = 2, columnspan = 1,sticky=W)

        self.forceatlas2_outbound_C = tki.Checkbutton(master = self.network_projections_W,variable=self.forceatlas2_outbound_I)
        self.forceatlas2_outbound_C.grid(row = 3, column = 4, columnspan = 1,sticky=W+E)

        self.forceatlas2_restart_C = tki.Checkbutton(master = self.network_projections_W,variable=self.forceatlas2_restart_I)
        self.forceatlas2_restart_C.grid(row = 3, column = 5, columnspan = 1,sticky=W+E)

        self.forceatlas2_B = tki.Button(self.network_projections_W, text = "Compute layout", command = self.run_forceatlas2)
        self.forceatlas2_B.grid(row = 4, column = 0, columnspan = 4,sticky=W+E)

        self.customization_L = tki.Label(self.network_projections_W, text = "\nGRAPH CUSTOMIZATION\n")
        self.customization_L.grid(row =5, column = 0, columnspan = 6,sticky=W+E)
        self.customization_selection_L = tki.Label(self.network_projections_W, text = "Network region:")
        self.customization_selection_L.grid(row =6, column = 0, columnspan = 3,sticky=W)
        self.customization_minima_or_edges_L = tki.Label(self.network_projections_W, text = "Vertices/edges:")
        self.customization_minima_or_edges_L.grid(row =6, column = 3, columnspan = 3,sticky=W)

        next_row = 7

        self.customization_selection_S = tki.StringVar(self.network_projections_W)
        self.customization_selection_S.set("All")
        radio_options = ["All",] + self.combined_options_list
        self.OT_cluster_ids = [-1,-1]
        if (self.surface.propagation.OT_cluster_ids[0] != -1):
            radio_options += ["Origin minimum"]
        if (self.surface.propagation.OT_cluster_ids[1] != -1):
            radio_options += ["Target minimum"]
        for i,text in enumerate(radio_options):
            self.customization_selection_B = tki.Radiobutton(self.network_projections_W, text=text, variable=self.customization_selection_S, value=text)
            self.customization_selection_B.grid(row =next_row+i, column = 0, columnspan = 3,sticky=W)
        row_mod = len(radio_options)
        self.customization_minima_or_edges_S = tki.StringVar(self.network_projections_W)
        if (self.surface.barriers_as_edges):
            self.customization_minima_or_edges_S.set("Both")
        else:
            self.customization_minima_or_edges_S.set("All vertices+edges")
        if (self.surface.barriers_as_edges):
            radio_options = ["Minimum vertices","Barrier edges","Both"]
        else:
            radio_options = ["Minimum vertices","Barrier edges", "Barrier vertices", "All vertices", "All vertices+edges"]

        if (len(radio_options) > row_mod):
            row_mod = len(radio_options)
        row_mod += next_row

        self.action_S = tki.StringVar(self.network_projections_W)
        self.action_S.set("Set color")
        self.action_B = tki.OptionMenu(self.network_projections_W, self.action_S, *["Place","Holder"])
        self.action_B.grid(row = row_mod+1, column = 0, columnspan = 2,sticky=W)

        for i,text in enumerate(radio_options):
            self.customization_minima_or_edges_B = tki.Radiobutton(self.network_projections_W, text=text, variable=self.customization_minima_or_edges_S, value=text,command=self.set_action_list)
            self.customization_minima_or_edges_B.grid(row =next_row+i, column = 3, columnspan = 3,sticky=W)

        self.action_L = tki.Label(self.network_projections_W, text = "Action:")
        self.action_L.grid(row =row_mod, column = 0, columnspan = 2,sticky=W)
        self.action_value_L = tki.Label(self.network_projections_W, text = "Value:")
        self.action_value_L.grid(row = row_mod, column = 2, columnspan = 2,sticky=W)
        self.node_id_L = tki.Label(self.network_projections_W, text = "Path or cluster id selection:")
        self.node_id_L.grid(row = row_mod, column = 4, columnspan = 2,sticky=W)

        self.set_action_list()

        self.action_value_S = tki.StringVar(self.network_projections_W)
        self.action_value_S.set("")
        self.action_value_E = tki.Entry(master = self.network_projections_W,textvariable=self.action_value_S)
        self.action_value_E.grid(row = row_mod+1, column = 2, columnspan = 2,sticky=W+E)

        self.node_id_S = tki.StringVar(self.network_projections_W)
        self.node_id_S.set("0")
        self.node_id_E = tki.Entry(master = self.network_projections_W,textvariable=self.node_id_S)
        self.node_id_E.grid(row = row_mod+1, column = 4, columnspan = 2,sticky=W+E)

        self.reset_graph_B = tki.Button(self.network_projections_W, text = "Reset", command = self.reset_graph)
        self.reset_graph_B.grid(row = row_mod+2, column = 0, columnspan = 1,sticky=W+E)

        self.apply_action_B = tki.Button(self.network_projections_W, text = "Apply", command = self.customize_graph)
        self.apply_action_B.grid(row = row_mod+2, column = 2, columnspan = 3,sticky=W+E)


        self.graph_width_S = tki.StringVar(self.network_projections_W)
        self.graph_width_S.set("2000")
        self.graph_height_S = tki.StringVar(self.network_projections_W)
        self.graph_height_S.set("2000")
        self.graph_background_S = tki.StringVar(self.network_projections_W)
        self.graph_background_S.set("white")

        self.graph_plot_L = tki.Label(self.network_projections_W, text = "\nPLOT\n")
        self.graph_plot_L.grid(row = row_mod+3, column = 0, columnspan = 6,sticky=W+E)

        self.graph_width_L = tki.Label(self.network_projections_W, text = "Width: ")
        self.graph_width_L.grid(row = row_mod+4, column = 0, columnspan = 2,sticky=W)

        self.graph_height_L = tki.Label(self.network_projections_W, text = "Height: ")
        self.graph_height_L.grid(row = row_mod+4, column = 2, columnspan = 2,sticky=W)

        self.graph_background_L = tki.Label(self.network_projections_W, text = "Background color: ")
        self.graph_background_L.grid(row = row_mod+4, column = 4, columnspan = 2,sticky=W)


        self.graph_width_E = tki.Entry(master = self.network_projections_W,textvariable=self.graph_width_S)
        self.graph_width_E.grid(row = row_mod+5, column = 0, columnspan = 2,sticky=W)

        self.graph_height_E = tki.Entry(master = self.network_projections_W,textvariable=self.graph_height_S)
        self.graph_height_E.grid(row = row_mod+5, column = 2, columnspan = 2,sticky=W)

        self.graph_background_E = tki.Entry(master = self.network_projections_W,textvariable=self.graph_background_S)
        self.graph_background_E.grid(row = row_mod+5, column = 4, columnspan = 2,sticky=W)


        self.graph_plot_B = tki.Button(self.network_projections_W, text = "Plot", command = self.graph_plot)
        self.graph_plot_B.grid(row = row_mod+6, column = 0, columnspan = 2,sticky=W+E)

        self.graph_save_B = tki.Button(self.network_projections_W, text = "Save plot", command = self.save_graph)
        self.graph_save_B.grid(row = row_mod+6    , column = 3, columnspan = 3,sticky=W+E)

        self.complementary_L = tki.Label(self.network_projections_W, text = "\nCOMPLEMENTARY OPTIONS\n")
        self.complementary_L.grid(row =row_mod+7, column = 0, columnspan = 6,sticky=W+E)

        self.save_session_nw_B = tki.Button(self.network_projections_W, text = "Save session", command = self.save_session_nw)
        self.save_session_nw_B.grid(row = row_mod+8, column = 0, columnspan = 2,sticky=W+E)

        self.create_sele_nw_B = tki.Button(self.network_projections_W, text = "Create network from selection", command = self.create_network_from_selection)
        self.create_sele_nw_B.grid(row = row_mod+8, column = 3, columnspan = 3,sticky=W+E)

        self.update_network_projections_states()

    def coordinate_projections_call(self):
        self.coordinate_projections_W = tki.Toplevel(master=self.pathfinding_pipeline_W, takefocus=True)
        #self.coordinate_projections_W.wait_visibility()
        self.coordinate_projections_W.grab_set()
        self.coordinate_projections_W.resizable(width=FALSE, height=FALSE)
        self.coordinate_projections_W.protocol("WM_DELETE_WINDOW", self.coordinate_projections_W_exit_handler)
        self.coordinate_projections_W.title("Coordinate projections")

        coordinates_list = np.arange(self.surface.coords.shape[1]).tolist()
        for i, coord in enumerate(coordinates_list):
            coordinates_list[i] = str(coord)
        coordinates_list += ["Energy"]
        coordinates_list_w_none = ["None"] + coordinates_list

        self.coord_1_S = tki.StringVar(self.coordinate_projections_W)
        self.coord_1_S.set("0")
        self.coord_2_S = tki.StringVar(self.coordinate_projections_W)
        self.coord_2_S.set("0")
        self.coord_3_S = tki.StringVar(self.coordinate_projections_W)
        self.coord_3_S.set("None")
        self.coord_1_L = tki.Label(self.coordinate_projections_W, text = "X coordinate: ")
        self.coord_1_L.grid(row = 0, column = 0, columnspan = 2,sticky=W+E)
        self.coord_1_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_1_S, *coordinates_list)
        self.coord_1_B.grid(row = 0, column = 2, columnspan = 2,sticky=W+E)
        self.coord_1_L = tki.Label(self.coordinate_projections_W, text = "Y coordinate: ")
        self.coord_1_L.grid(row = 1, column = 0, columnspan = 2,sticky=W+E)
        self.coord_2_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_2_S, *coordinates_list)
        self.coord_2_B.grid(row = 1, column = 2, columnspan = 2,sticky=W+E)
        self.coord_1_L = tki.Label(self.coordinate_projections_W, text = "Z coordinate: ")
        self.coord_1_L.grid(row = 2, column = 0, columnspan = 2,sticky=W+E)
        self.coord_3_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_3_S, *coordinates_list_w_none)
        self.coord_3_B.grid(row = 2, column = 2, columnspan = 2,sticky=W+E)

        self.point_size_L = tki.Label(self.coordinate_projections_W, text = "Points size: ")
        self.point_size_L.grid(row = 3, column = 0, columnspan = 1,sticky=W+E)
        self.point_size_S = tki.StringVar(self.coordinate_projections_W)
        self.point_size_S.set("1")
        self.point_size_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.point_size_S)
        self.point_size_E.grid(row = 3, column = 1, columnspan = 1,sticky=W+E)

        self.point_color_L = tki.Label(self.coordinate_projections_W, text = "Points color: ")
        self.point_color_L.grid(row = 3, column = 2, columnspan = 1,sticky=W+E)
        self.point_color_S = tki.StringVar(self.coordinate_projections_W)
        #self.point_color_S.set("C0")
        self.point_color_S.set("black")
        self.point_color_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.point_color_S)
        self.point_color_E.grid(row = 3, column = 3, columnspan = 1,sticky=W+E)


        self.O_point_size_L = tki.Label(self.coordinate_projections_W, text = "Origin points size: ")
        self.O_point_size_L.grid(row = 4, column = 0, columnspan = 1,sticky=W+E)
        self.O_point_size_S = tki.StringVar(self.coordinate_projections_W)
        self.O_point_size_S.set("1")
        self.O_point_size_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.O_point_size_S)
        self.O_point_size_E.grid(row = 4, column = 1, columnspan = 1,sticky=W+E)

        self.O_point_color_L = tki.Label(self.coordinate_projections_W, text = "Origin points color: ")
        self.O_point_color_L.grid(row = 4, column = 2, columnspan = 1,sticky=W+E)
        self.O_point_color_S = tki.StringVar(self.coordinate_projections_W)
        self.O_point_color_S.set("C2")
        self.O_point_color_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.O_point_color_S)
        self.O_point_color_E.grid(row = 4, column = 3, columnspan = 1,sticky=W+E)


        self.T_point_size_L = tki.Label(self.coordinate_projections_W, text = "Target points size: ")
        self.T_point_size_L.grid(row = 5, column = 0, columnspan = 1,sticky=W+E)
        self.T_point_size_S = tki.StringVar(self.coordinate_projections_W)
        self.T_point_size_S.set("1")
        self.T_point_size_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.T_point_size_S)
        self.T_point_size_E.grid(row = 5, column = 1, columnspan = 1,sticky=W+E)

        self.T_point_color_L = tki.Label(self.coordinate_projections_W, text = "Target points color: ")
        self.T_point_color_L.grid(row = 5, column = 2, columnspan = 1,sticky=W+E)
        self.T_point_color_S = tki.StringVar(self.coordinate_projections_W)
        self.T_point_color_S.set("C3")
        self.T_point_color_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.T_point_color_S)
        self.T_point_color_E.grid(row = 5, column = 3, columnspan = 1,sticky=W+E)

        self.label_size_L = tki.Label(self.coordinate_projections_W, text = "Index labels size: ")
        self.label_size_L.grid(row = 6, column = 0, columnspan = 1,sticky=W+E)
        self.label_size_S = tki.StringVar(self.coordinate_projections_W)
        self.label_size_S.set("0")
        self.label_size_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.label_size_S)
        self.label_size_E.grid(row = 6, column = 1, columnspan = 1,sticky=W+E)

        self.label_color_L = tki.Label(self.coordinate_projections_W, text = "Index labels color: ")
        self.label_color_L.grid(row = 6, column = 2, columnspan = 1,sticky=W+E)
        self.label_color_S = tki.StringVar(self.coordinate_projections_W)
        self.label_color_S.set("black")
        self.label_color_E = tki.Entry(master = self.coordinate_projections_W,textvariable=self.label_color_S)
        self.label_color_E.grid(row = 6, column = 3, columnspan = 1,sticky=W+E)

        if (self.surface.coords.shape[1] == 2):
            self.coord_projection_plot_S = tki.StringVar(self.coordinate_projections_W)
            self.coord_projection_plot_S.set("Plot")
            self.coord_projection_plot_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_projection_plot_S, *self.combined_options_list, command=self.coord_projection_plot)
            self.coord_projection_plot_B.grid(row = 7, column = 0, columnspan = 2,sticky=W+E)

            self.coord_projection_contour_plot_S = tki.StringVar(self.coordinate_projections_W)
            self.coord_projection_contour_plot_S.set("Contour plot")
            self.coord_projection_contour_plot_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_projection_contour_plot_S, *self.combined_options_list, command=self.coord_projection_contour_plot)
            self.coord_projection_contour_plot_B.grid(row = 7, column = 2, columnspan = 2,sticky=W+E)

        else:
            self.coord_projection_plot_S = tki.StringVar(self.coordinate_projections_W)
            self.coord_projection_plot_S.set("Plot")
            self.coord_projection_plot_B = tki.OptionMenu(self.coordinate_projections_W, self.coord_projection_plot_S, *self.combined_options_list, command=self.coord_projection_plot)
            self.coord_projection_plot_B.grid(row = 7, column = 0, columnspan = 4,sticky=W+E)
        self.update_coordinate_projections_states()

    def save_simplified_path(self):
        selection = np.zeros(1,dtype=np.int_)
        if (len(self.surface.propagation.gnw_paths) != 1):
            selection = simpledialog.askstring("Paths selection", "Select which paths to plot by index.\nTIP 1: Minimum energy path has index 0, first alternative path has index 1 and so on.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23.",parent=self.pathfinding_pipeline_W,initialvalue="0")
            #self.pathfinding_pipeline_W.wait_visibility()
            self.pathfinding_pipeline_W.grab_set()
            if (selection is None):
                return
            if ("all" in selection.lower()):
                selection = np.arange(len(self.surface.propagation.gnw_paths))
            else:
                selection = self.parse_selection(selection)
                selection = selection[selection<len(self.surface.propagation.gnw_paths)]

            selection = selection[selection<len(self.surface.propagation.gnw_paths)]
            if (selection.shape[0]==0):
                messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the paths calculated.",parent=self.pathfinding_pipeline_W)
                return
            elif (selection.shape[0] > 1):
                file_to_save = filedialog.asksaveasfilename(parent=self.pathfinding_pipeline_W)
                #self.pathfinding_pipeline_W.wait_visibility()
                self.pathfinding_pipeline_W.grab_set()
                if file_to_save is None:
                    return
                if file_to_save != '' and file_to_save != 'none':
                    try:
                        for index in selection:
                            self.surface.propagation.save_simplified_path_to_txt(file_to_save + "_" + str(index) + ".spath",n=index)
                    except:
                        messagebox.showinfo( "ERROR", "There was an error while saving multiple paths.")

        if (selection.shape[0] == 1):
            file_to_save = filedialog.asksaveasfile(parent=self.pathfinding_pipeline_W,mode='w', defaultextension=".spath")
            #self.pathfinding_pipeline_W.wait_visibility()
            self.pathfinding_pipeline_W.grab_set()
            if file_to_save is None:
                return
            if file_to_save != '' and file_to_save != 'none':
                try:
                    self.surface.propagation.save_simplified_path_to_txt(file_to_save.name,n=selection[0])
                except:
                    messagebox.showinfo( "ERROR", "Path could not be saved.")


    def save_path(self):
        selection = np.zeros(1,dtype=np.int_)
        if (len(self.surface.propagation.gnw_paths) != 1):
            selection = simpledialog.askstring("Paths selection", "Select which paths to plot by index.\nTIP 1: Minimum energy path has index 0, first alternative path has index 1 and so on.\nTIP 2: Separate individual indices with \" \" and define ranges with \"-\"(with no spaces).\ne.g. \"0 10-13 21 23\" selects indices 0 10 11 12 13 and 23.",parent=self.pathfinding_pipeline_W,initialvalue="0")
            #self.pathfinding_pipeline_W.wait_visibility()
            self.pathfinding_pipeline_W.grab_set()
            if (selection is None):
                return
            if ("all" in selection.lower()):
                selection = np.arange(len(self.surface.propagation.gnw_paths))
            else:
                selection = self.parse_selection(selection)
                selection = selection[selection<len(self.surface.propagation.gnw_paths)]

            selection = selection[selection<len(self.surface.propagation.gnw_paths)]
            if (selection.shape[0]==0):
                messagebox.showinfo( "ERROR", "The provided indices are not in the range of any of the paths calculated.",parent=self.pathfinding_pipeline_W)
                return
            elif (selection.shape[0] > 1):
                file_to_save = filedialog.asksaveasfilename(parent=self.pathfinding_pipeline_W)
                #self.pathfinding_pipeline_W.wait_visibility()
                self.pathfinding_pipeline_W.grab_set()
                if file_to_save is None:
                    return
                if file_to_save != '' and file_to_save != 'none':
                    try:
                        for index in selection:
                            self.surface.propagation.save_fragmentwise_path_to_txt(file_to_save + "_" + str(index) + ".path",n=index)
                    except:
                        messagebox.showinfo( "ERROR", "There was an error while saving multiple paths.")

        if (selection.shape[0] == 1):
            file_to_save = filedialog.asksaveasfile(parent=self.pathfinding_pipeline_W,mode='w', defaultextension=".path")
            #self.pathfinding_pipeline_W.wait_visibility()
            self.pathfinding_pipeline_W.grab_set()
            if file_to_save is None:
                return
            if file_to_save != '' and file_to_save != 'none':
                try:
                    self.surface.propagation.save_fragmentwise_path_to_txt(file_to_save.name,n=selection[0])
                except:
                    messagebox.showinfo( "ERROR", "Path could not be saved.")

    def save_well_sampling_to_txt(self):
        file_to_save = filedialog.asksaveasfile(parent=self.pathfinding_pipeline_W,mode='w', defaultextension=".ws")
        if file_to_save is None:
            return
        if file_to_save != '' and file_to_save != 'none':
            try:
                self.surface.propagation.save_well_sampling_to_txt(file_to_save.name)
            except:
                messagebox.showinfo( "ERROR", "Well sampling could not be saved.")

    def pathfinding_pipeline_call(self):
        self.pathfinding_pipeline_W = tki.Toplevel(master=self.root, takefocus=True)
        #self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()
        self.pathfinding_pipeline_W.resizable(width=FALSE, height=FALSE)
        self.pathfinding_pipeline_W.protocol("WM_DELETE_WINDOW", self.pathfinding_pipeline_W_exit_handler)
        self.pathfinding_pipeline_W.title("Path-finding pipeline")
        self.set_origin_B = tki.Button(self.pathfinding_pipeline_W, text = "Set Origin", command = self.set_origin_call)
        self.set_origin_B.grid(row = 0, column = 0, columnspan = 6,sticky=W+E)
        self.set_origin_L = tki.Label(self.pathfinding_pipeline_W, text = "None")
        self.set_origin_L.grid(row = 0, column = 6, columnspan = 6,sticky=W+E)
        self.set_origin_L.bind("<Button-1>",self.show_origin_coords)
        self.set_target_B = tki.Button(self.pathfinding_pipeline_W, text = "Set Target", command = self.set_target_call)
        self.set_target_B.grid(row = 1, column = 0, columnspan = 6,sticky=W+E)
        self.set_target_L = tki.Label(self.pathfinding_pipeline_W, text = "None")
        self.set_target_L.grid(row = 1, column = 6, columnspan = 6,sticky=W+E)
        self.set_target_L.bind("<Button-1>",self.show_target_coords)
        self.calculations_L = tki.Label(self.pathfinding_pipeline_W, text = "CALCULATIONS")
        self.calculations_L.grid(row = 2, column = 0, columnspan = 12,sticky=W+E)
        self.path_B = tki.Button(self.pathfinding_pipeline_W, text = "Minimum Energy Path", command = self.calculate_path)
        self.path_B.grid(row = 3, column = 0, columnspan = 6,sticky=W+E)
        self.fpsr_B = tki.Button(self.pathfinding_pipeline_W, text = "Well sampling", command = self.calculate_fpsr)
        self.fpsr_B.grid(row = 3, column = 6, columnspan = 6,sticky=W+E)

        self.alternative_paths_B = tki.Button(self.pathfinding_pipeline_W, text = "Alternative paths", command = self.calculate_alternative_paths)
        self.alternative_paths_B.grid(row = 4, column = 0, columnspan = 6,sticky=W+E)

        self.alternative_paths_S = tki.StringVar(self.pathfinding_pipeline_W)
        self.alternative_paths_S.set("1")
        self.alternative_paths_E = tki.Entry(master = self.pathfinding_pipeline_W,textvariable=self.alternative_paths_S)
        self.alternative_paths_E.grid(row = 4, column = 6, columnspan = 6,sticky=W+E)

        self.total_paths_L = tki.Label(self.pathfinding_pipeline_W, text = "Total paths calculated: ")
        self.total_paths_L.grid(row = 5, column = 0, columnspan = 6,sticky=W+E)
        self.total_paths_count_L = tki.Label(self.pathfinding_pipeline_W, text = "None")
        self.total_paths_count_L.grid(row = 5, column = 6, columnspan = 6,sticky=W+E)

        self.plots_L = tki.Label(self.pathfinding_pipeline_W, text = "PLOTS")
        self.plots_L.grid(row = 6, column = 0, columnspan = 12,sticky=W+E)
        self.path_energy_S = tki.StringVar(self.pathfinding_pipeline_W)
        self.path_energy_S.set("Energy profiles")
        self.profile_plot_options_list = ["Select minima and barriers by id",]
        self.path_energy_B = tki.OptionMenu(self.pathfinding_pipeline_W, self.path_energy_S, *self.profile_plot_options_list, command=self.plot_profile)
        self.path_energy_B.grid(row = 7, column = 0, columnspan = 4,sticky=W+E)
        self.coord_projections_B = tki.Button(self.pathfinding_pipeline_W, text = "Coordinate projections", command = self.coordinate_projections_call)
        self.coord_projections_B.grid(row = 7, column = 4, columnspan = 4,sticky=W+E)
        self.nw_projections_B = tki.Button(self.pathfinding_pipeline_W, text = "Network projections", command = self.network_projections_call)
        self.nw_projections_B.grid(row = 7, column = 8, columnspan = 4,sticky=W+E)
        self.save_L = tki.Label(self.pathfinding_pipeline_W, text = "SAVE TO DISK")
        self.save_L.grid(row = 8, column = 0, columnspan = 12,sticky=W+E)
        self.path_save_B = tki.Button(self.pathfinding_pipeline_W, text = "Fragmentwise path", command=self.save_path) # meter Draft
        self.path_save_B.grid(row = 9, column = 0, columnspan = 3,sticky=W+E)
        self.simp_path_save_B = tki.Button(self.pathfinding_pipeline_W, text = "Simplified path", command=self.save_simplified_path) # meter Draft
        self.simp_path_save_B.grid(row = 9, column = 3, columnspan = 3,sticky=W+E)
        self.fpsr_save_B = tki.Button(self.pathfinding_pipeline_W, text = "Well sampling", command = self.save_well_sampling_to_txt)
        self.fpsr_save_B.grid(row = 9, column = 6, columnspan = 3,sticky=W+E)
        self.save_session_pipe_B = tki.Button(self.pathfinding_pipeline_W, text = "Save session", command = self.save_session_pipe)
        self.save_session_pipe_B.grid(row = 9, column = 9, columnspan = 3,sticky=W+E)
        self.update_pathfinding_pipeline_states()

    def force_network_projections_states(self,state='disabled'):
        self.create_nw_graph_B['state']=state
        self.export_nw_graph_B['state']=state
        self.forceatlas2_B['state']=state
        self.reset_graph_B['state']=state
        self.apply_action_B['state']=state
        self.graph_plot_B['state']=state
        self.graph_save_B['state']=state
        self.save_session_nw_B['state']=state
        self.create_sele_nw_B['state']=state
        if (state=='disabled'):
            self.force_pathfinding_pipeline_states()

    def force_coordinate_projections_states(self,state='disabled'):
        self.coord_1_B['state']=state
        self.coord_2_B['state']=state
        self.coord_3_B['state']=state
        self.coord_projection_plot_B['state']=state
        if (self.surface.coords.shape[1] == 2):
            self.coord_projection_contour_plot_B['state']=state
        if (state=='disabled'):
            self.force_pathfinding_pipeline_states()

    def force_pathfinding_pipeline_states(self,state='disabled'):
        self.set_origin_B['state']=state
        self.set_target_B['state']=state
        self.path_B['state']=state
        self.fpsr_B['state']=state
        self.alternative_paths_B['state']=state
        self.path_energy_B['state']=state
        self.coord_projections_B['state']=state
        self.nw_projections_B['state']=state
        self.path_save_B['state']=state
        self.fpsr_save_B['state']=state
        if (state=='disabled'):
            self.force_main_window_states()

    def force_main_window_states(self,state='disabled'):
        self.load_surface_B['state']=state
        self.load_session_B['state']=state
        self.save_session_B['state']=state
        self.load_cutoffs_B['state']=state
        self.save_cutoffs_B['state']=state
        self.compute_connectivity_B['state']=state
        self.load_connectivity_B['state']=state
        self.save_connectivity_B['state']=state
        self.load_periodicity_B['state']=state
        self.save_periodicity_B['state']=state
        self.save_nodes_B['state']=state
        self.pathfinding_pipeline_B['state']=state

    def update_network_projections_states(self):
        self.force_network_projections_states('normal')
        self.create_nw_graph_B['state']='disabled'
        self.export_nw_graph_B['state']='disabled'
        self.forceatlas2_B['state']='disabled'
        self.action_B['state']='disabled'
        self.reset_graph_B['state']='disabled'
        self.apply_action_B['state']='disabled'
        self.graph_plot_B['state']='disabled'
        self.graph_save_B['state']='disabled'
        if (self.surface.gnw_graph_done==False):
            self.create_nw_graph_B['state']='normal'
        else:
            self.export_nw_graph_B['state']='normal'
            self.forceatlas2_B['state']='normal'
            self.forceatlas2_B['state']='normal'
            self.action_B['state']='normal'
            self.reset_graph_B['state']='normal'
            self.apply_action_B['state']='normal'
            if(self.surface.gnw_graph.layout_done):
                self.graph_plot_B['state']='normal'
                self.graph_save_B['state']='normal'

        self.update_pathfinding_pipeline_states()

    def update_coordinate_projections_states(self):
        self.force_coordinate_projections_states('normal')
        self.update_pathfinding_pipeline_states()

    def update_pathfinding_pipeline_states(self):
        self.force_pathfinding_pipeline_states('normal')
        self.combined_options_list=["None",]
        self.profile_plot_options_list=[]
        self.set_origin_L['text'] = 'None'
        self.set_target_L['text'] = 'None'
        self.set_origin_B['state']='normal'
        self.set_target_B['state']='disabled'
        self.path_B['state']='disabled'
        self.fpsr_B['state']='disabled'
        self.path_save_B['state']='disabled'
        self.simp_path_save_B['state']='disabled'
        self.fpsr_save_B['state']='disabled'
        self.path_energy_B['state']='disabled'
        self.coord_projections_B['state']='disabled'
        self.nw_projections_B['state']='disabled'
        self.alternative_paths_B['state']='disabled'
        self.path_energy_S.set("Energy profiles")

        if (self.surface.propagation.gnw_done == True):
            if (network_dependencies):
                self.nw_projections_B['state']='normal'
            self.path_energy_B['state']='normal'
            self.coord_projections_B['state']='normal'

        if (self.surface.propagation.OT_points[0][0] != -1):
            self.set_target_B['state']='normal'
            self.set_origin_L['text'] = str(self.surface.propagation.OT_points[0][0])+"("+str(self.surface.propagation.OT_cluster_ids[0][0])+")"
            if (self.surface.propagation.OT_points[1][0] != -1):
                if(self.surface.propagation.OT_points[0][0] != self.surface.propagation.OT_points[1][0] and self.surface.propagation.OT_cluster_ids[0][0] != self.surface.propagation.OT_cluster_ids[1][0]):
                    self.path_B['state']='normal'
                    self.fpsr_B['state']='normal'
                    self.alternative_paths_B['state']='normal'
        if(len(self.surface.propagation.gnw_paths)==0):
            self.total_paths_count_L['text'] = "None"
        else:
            self.total_paths_count_L['text'] = (str(len(self.surface.propagation.gnw_paths)))
        if (self.surface.propagation.OT_points[1][0] != -1):
            self.set_target_L['text'] = str(self.surface.propagation.OT_points[1][0])+"("+str(self.surface.propagation.OT_cluster_ids[1][0])+")"

        if(len(self.surface.propagation.gnw_paths)>0):
            self.profile_plot_options_list = ["Raw path", "Hide plateaus", "Lvl1 reduction path", "Lvl2 reduction path", "Lvl3 reduction path","Simplified path"]
            self.combined_options_list += ["Raw path", "Lvl1 reduction path", "Lvl2 reduction path", "Lvl3 reduction path","Simplified path"]
            self.path_save_B['state']='normal'
            self.simp_path_save_B['state']='normal'
        self.profile_plot_options_list += ["Select minima and barriers by id"]
        self.combined_options_list += ["Select minima and barriers by id"]
        if(self.surface.propagation.fpsr_done == True):
            O_clusters = self.surface.propagation.fpsr_O_clusters
            T_clusters = self.surface.propagation.fpsr_T_clusters
            I_clusters = self.surface.propagation.fpsr_I_clusters
            if (self.surface.barriers_as_edges):
                O_clusters = O_clusters[O_clusters <= self.surface.propagation.gnw_minimum_clusters_indices[-1]]
                T_clusters = T_clusters[T_clusters <= self.surface.propagation.gnw_minimum_clusters_indices[-1]]
                I_clusters = I_clusters[I_clusters <= self.surface.propagation.gnw_minimum_clusters_indices[-1]]
            if (O_clusters.shape[0] > 0 or T_clusters.shape[0] > 0 or I_clusters.shape[0] > 0):
                self.combined_options_list += ["Well sampling: all"]
                if (O_clusters.shape[0] > 0):
                    self.combined_options_list += ["Well sampling: origin well region"]
                if (T_clusters.shape[0] > 0):
                    self.combined_options_list += ["Well sampling: target well region"]
                if (I_clusters.shape[0] > 0):
                    self.combined_options_list += ["Well sampling: barrier/intermediate region"]
            self.coord_projections_B['state']='normal'
            self.fpsr_save_B['state']='normal'

        menu = self.path_energy_B["menu"]
        menu.delete(0, "end")
        for string in self.profile_plot_options_list:
            menu.add_command(label=string, command=lambda value=string: self.plot_profile(value))

        self.update_main_window_states()

    def update_main_window_states(self):
        self.force_main_window_states('normal')
        if(self.session_started == True):
            self.load_surface_B['state']='disabled'
            self.load_session_B['state']='disabled'
            self.save_session_B['state']='normal'
            self.load_cutoffs_B['state']='normal'
            self.save_cutoffs_B['state']='normal'
            self.load_periodicity_B['state']='normal'
            self.save_periodicity_B['state']='normal'
            if (self.surface.propagation.gnw_done == False):
                self.compute_connectivity_B['state']='normal'
                self.pathfinding_pipeline_B['state']='disabled'
                self.load_connectivity_B['state']='normal'
                self.save_connectivity_B['state']='disabled'
                self.save_nodes_B['state']='disabled'
            else:
                self.compute_connectivity_B['state']='disabled'
                self.pathfinding_pipeline_B['state']='normal'
                self.load_connectivity_B['state']='disabled'
                self.save_connectivity_B['state']='normal'
                self.save_nodes_B['state']='normal'
                self.load_periodicity_B['state']='disabled'
                self.load_cutoffs_B['state']='disabled'
        else:
            self.load_surface_B['state']='normal'
            self.load_session_B['state']='normal'
            self.save_session_B['state']='disabled'
            self.load_cutoffs_B['state']='disabled'
            self.save_cutoffs_B['state']='disabled'
            self.compute_connectivity_B['state']='disabled'
            self.pathfinding_pipeline_B['state']='disabled'
            self.load_periodicity_B['state']='disabled'
            self.save_periodicity_B['state']='disabled'
            self.load_connectivity_B['state']='disabled'
            self.save_connectivity_B['state']='disabled'
            self.save_nodes_B['state']='disabled'

    def ask_connectivity_options_W_exit_handler(self):
        self.ask_connectivity_options_W.grab_release()
        self.ask_connectivity_options_W.destroy()
        self.update_main_window_states()

    def ask_barriers_as_edges_W_exit_handler(self):
        self.ask_barriers_as_edges_W.grab_release()
        self.ask_barriers_as_edges_W.destroy()
        self.update_network_projections_states()

    def set_origin_L_W_exit_handler(self):
        self.set_origin_L_W.grab_release()
        self.set_origin_L_W.destroy()
        self.update_pathfinding_pipeline_states()
        #self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()

    def set_target_L_W_exit_handler(self):
        self.set_target_L_W.grab_release()
        self.set_target_L_W.destroy()
        self.update_pathfinding_pipeline_states()
        #self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()

    def pathfinding_pipeline_W_exit_handler(self):
        self.pathfinding_pipeline_B['state']='normal'
        self.pathfinding_pipeline_W.grab_release()
        self.pathfinding_pipeline_W.destroy()
        self.update_main_window_states()

    def set_origin_W_exit_handler(self):
        self.set_origin_B['state']='normal'
        self.set_origin_W.grab_release()
        self.set_origin_W.destroy()
        self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()
        self.update_pathfinding_pipeline_states()

    def set_origin_ranges_W_exit_handler(self):
        self.set_origin_ranges_W.grab_release()
        self.set_origin_ranges_W.destroy()
        self.set_origin_W.wait_visibility()
        self.set_origin_W.grab_set()

    def set_target_W_exit_handler(self):
        self.set_target_B['state']='normal'
        self.set_target_W.grab_release()
        self.set_target_W.destroy()
        self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()
        self.update_pathfinding_pipeline_states()

    def set_target_ranges_W_exit_handler(self):
        self.set_target_ranges_W.grab_release()
        self.set_target_ranges_W.destroy()
        self.set_target_W.wait_visibility()
        self.set_target_W.grab_set()

    def coordinate_projections_W_exit_handler(self):
        self.coord_projections_B['state']='normal'
        self.coordinate_projections_W.grab_release()
        #self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()
        self.coordinate_projections_W.destroy()
        self.update_pathfinding_pipeline_states()

    def network_projections_W_exit_handler(self):
        self.nw_projections_B['state']='normal'
        self.network_projections_W.grab_release()
        self.network_projections_W.destroy()
        #self.pathfinding_pipeline_W.wait_visibility()
        self.pathfinding_pipeline_W.grab_set()
        self.network_projections_W.destroy()
        self.update_pathfinding_pipeline_states()
# INIT
    def __init__(self, root):
        self.session_started = False
        self.root = root
        self.root.title("MEPSAnd GUI")
        self.load_surface_B = tki.Button(self.root, text = "Load surface", command = self.load_surface)
        self.load_surface_B.grid(row = 0, column = 0, columnspan = 2,sticky=W+E)
        self.load_session_B = tki.Button(self.root, text = "Load session", command = self.load_session)
        self.load_session_B.grid(row = 0, column = 2, columnspan = 2,sticky=W+E)
        self.save_session_B = tki.Button(self.root, text = "Save session", command = self.save_session_main)
        self.save_session_B.grid(row = 1, column = 0, columnspan = 4,sticky=W+E)
        self.load_cutoffs_B = tki.Button(self.root, text = "Load custom cutoffs", command = self.load_cutoffs)
        self.load_cutoffs_B.grid(row = 2, column = 0, columnspan = 2,sticky=W+E)
        self.save_cutoffs_B = tki.Button(self.root, text = "Save current cutoffs", command = self.save_cutoffs)
        self.save_cutoffs_B.grid(row = 2, column = 2, columnspan = 2,sticky=W+E)
        self.compute_connectivity_B = tki.Button(self.root, text = "Compute connectivity", command = self.ask_connectivity_options)
        self.compute_connectivity_B.grid(row = 3, column = 0, columnspan = 4,sticky=W+E)
        self.load_connectivity_B = tki.Button(self.root, text = "Load connectivity", command = self.load_connectivity_call)
        self.load_connectivity_B.grid(row = 4, column = 0, columnspan = 2,sticky=W+E)
        self.save_connectivity_B = tki.Button(self.root, text = "Save connectivity", command = self.save_connectivity_call)
        self.save_connectivity_B.grid(row = 4, column = 2, columnspan = 2,sticky=W+E)
        self.load_periodicity_B = tki.Button(self.root, text = "Load periodicity", command = self.load_periodicity_call)
        self.load_periodicity_B.grid(row = 5, column = 0, columnspan = 2,sticky=W+E)
        self.save_periodicity_B = tki.Button(self.root, text = "Save periodicity", command = self.save_periodicity_call)
        self.save_periodicity_B.grid(row = 5, column = 2, columnspan = 2,sticky=W+E)
        self.save_nodes_B = tki.Button(self.root, text = "Save minimum and barrier clusters", command = self.save_nodes_call)
        self.save_nodes_B.grid(row = 6, column = 0, columnspan = 4,sticky=W+E)
        self.pathfinding_pipeline_B = tki.Button(self.root, text = "Path-finding pipeline", command = self.pathfinding_pipeline_call)
        self.pathfinding_pipeline_B.grid(row = 7, column = 0, columnspan = 4,sticky=W+E)
        self.update_main_window_states()

version = '1.6'

# print("\n\n#####################################################################\nMEPSAnd (Minimum Energy Path Surface Analysis over n-dimensional surfaces) \nVersion: " + version + "\n#####################################################################\nCopyright (C) 2019, Íñigo Marcos Alcalde\n#####################################################################\n#                            LICENSE                                #\n#####################################################################        \nMEPSAnd is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 3 of the License, or\n(at your option) any later version.\n\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\nGNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License\nalong with this program. If not, see <http://www.gnu.org/licenses/>.\n#####################################################################\n#####################################################################\nContact: pagomez@cbm.csic.es, eduardo.lopez@ufv.es and imarcos@cbm.csic.es\n#####################################################################\nCitation: \nMarcos-Alcalde, I., Lopez-Viñas, E. & Gómez-Puertas, P. (2020). MEPSAnd: Minimum Energy Path Surface Analysis over n-dimensional surfaces. Bioinformatics 36, 956–958.\n#####################################################################\n\n")


if __name__ == '__main__':
    root = tki.Tk()
    root.resizable(width=FALSE, height=FALSE)
    gui = mepsa_gui(root)
    root.mainloop()