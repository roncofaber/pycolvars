#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:02:38 2023

@author: roncoroni
"""

import pycolvars._utilities as aux
import numpy as np
import copy

# plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import seaborn as sns
sns.set_style("white")

import colorcet
from matplotlib.colors import LightSource
#%%

def plot_colvars_data(pmf_obj,
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

    # wrapper function to generate the 2D slice from data and transpose it
    def generate_2d_slice(data, axes, vmax, fixed_axis=None):
        

        pivotted = aux.slice_ndarray(data, axes[0], axes[1], fixed_axis)

        # make NaN stuff that has maximum V
        pivotted[pivotted >= vmax] = np.nan

        if axes[1] < axes[0]:
            return pivotted

        return pivotted.T # transpose to match plot

    if pmf_data is None:
        pmf_data = copy.deepcopy(pmf_obj.main_pmf)
    else:
        pmf_data = copy.deepcopy(pmf_data)

    # check function arguments
    if vmax is None:
        vmax = np.max(pmf_data)

    # just take first two axes
    if axes is None:
        axes = [0,1]

    if pmf_data.ndim > 2:
        if fixed_axis is None:
            fixed_axis = [0]*(pmf_data.ndim-2)

    # generate data
    pivotted = generate_2d_slice(pmf_data, axes, vmax, fixed_axis)
    


    
    if hillshade:
        # Fill NaNs with the max (or min) value of the valid data
        valid_mask = ~np.isnan(pivotted)
        fill_value = np.nanmax(pivotted)  # or np.nanmin(pivotted)
        pivotted_filled = np.where(valid_mask, pivotted, fill_value)
        normed = (pivotted_filled - np.nanmin(pivotted_filled)) / (np.nanmax(pivotted_filled) - np.nanmin(pivotted_filled))
        normed = np.clip(normed, 0, 1)
        ls = LightSource(azdeg=315, altdeg=45)
        shaded = ls.shade(normed, cmap=plt.get_cmap(cmap), blend_mode='overlay')
        shaded[~valid_mask] = 1.0  # White (or set to [1,1,1,1] if RGBA)
    else:
        shaded = pivotted
    
    
    
    # parameters
    xrange = pmf_obj.ax_bounds[axes[0]]
    yrange = pmf_obj.ax_bounds[axes[1]]
    fs     = 16

    ###
    # generate figure
    if squared:
        figs = (6,5)
    else:
        figs = (8,5)
    
    fig, ax1 = plt.subplots(figsize=figs,
                            )
                            # layout='constrained')

    # adjust axis position
    ax1.set_position([0.1, 0.15, 0.9, 0.8])

    # global pos
    # global contour

    global contour # make it global for the moment
    
    # Prepare colormap and set NaN color to white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')

    # plot energy surface
    pos = ax1.imshow(shaded,
                     aspect        = "auto",
                     origin        = "lower",
                     extent        = (xrange[0], xrange[-1],
                                      yrange[0]+yshift, yrange[-1]+yshift),
                     vmin          = 0,
                     vmax          = vmax,
                     interpolation = "none",
                     cmap          = cmap_obj)# if not hillshade else None)

    if contours:
        levels = np.arange(0, vmax, con_steps)
        contour = ax1.contour(
            pivotted,
            levels = levels,
            extent=(xrange[0], xrange[-1],
                    yrange[0]+yshift, yrange[-1]+yshift),
            colors='black',
            linewidths=0.75
            )


    # add limits
    ax1.set_xlim(xrange[0], xrange[-1])
    ax1.set_ylim(yrange[0]+yshift, yrange[-1]+yshift)

    #generate sliders
    if slider:

        # get third axis
        other_axes = [axis for axis in range(pmf_data.ndim) if axis not in axes]

        slider_objs = []
        zranges     = []
        for cc, axis3 in enumerate(other_axes):

            index = fixed_axis[cc]

            slider_range = np.shape(pmf_data)[axis3]-1
            zrange = np.linspace(*pmf_obj.ax_bounds[axis3],
                                  np.shape(pmf_data)[axis3]+1)
            zrange = (zrange[:-1] + zrange[1:])/2

            zranges.append(zrange)


            ax_slider = plt.axes([0.20, 0.015+cc*0.05, 0.6, 0.03])

            sliderobj = Slider(ax_slider, pmf_obj.colvars_name[axis3],
                                0, slider_range,
                                valinit = index,
                                valstep = 1,
                                # valfmt="%.2f"
                                )

            sliderobj.valtext.set_text("{:.2f}".format(zrange[index]))

            slider_objs.append(sliderobj)

        def update(val):
            global contour

            vals = [so.val for so in slider_objs]

            # calculate new values
            pivotted = generate_2d_slice(pmf_data, axes, vmax, vals)

            # update heatmap
            pos.set_data(pivotted)

            if contours:

                # remove previous plot
                contour.remove()

                levels = np.arange(0, vmax, con_steps)
                contour = ax1.contour(
                    pivotted,
                    levels = levels,
                    extent=(xrange[0], xrange[-1],
                            yrange[0]+yshift, yrange[-1]+yshift),
                    colors='black',
                    linewidths=0.75
                    )

            # update slider and draw
            for cc, so in enumerate(slider_objs):
                so.valtext.set_text("{:.2f}".format(zranges[cc][vals[cc]]))
            fig.canvas.draw_idle()

            return


        for sliderobj in slider_objs:
            sliderobj.on_changed(update)



    # labels and such
    if xlabel is None:
        xlabel = pmf_obj.colvars_name[axes[0]]
    if ylabel is None:
        ylabel = pmf_obj.colvars_name[axes[1]]

    # add grid
    ax1.grid(linestyle="--", color="#888888")

    # colorbar
    cbar = fig.colorbar(pos, ax=ax1)
    cbar.set_label(r"$\Delta$G [kT]", fontsize=fs+1, fontweight="bold")

    # titles and such
    ax1.set_xlabel(xlabel, fontsize=fs, fontweight="bold")
    ax1.set_ylabel(ylabel, fontsize=fs, fontweight="bold")

    if title is not None:
        ax1.set_title(title, fontsize=fs+1, fontweight="bold")

    if path is not None:
        ax1.scatter(path[:,1], path[:,2], color="r", s=6)
    
    if squared:
        ax1.set_aspect('equal', 'box')
        
    fig.tight_layout()

    if savename is not None:
        plt.savefig(savename + ".png", dpi=300)
    
    fig.show()
    
    return

def plot_colvars_slice(self,
                       coord,
                       axis   = 0,
                       xlabel = None,
                       title  = None,
                       label  = None,
                       ):

    xrange = self.axes[-1-axis]
    slices, values = self.get_1D_slice(coord, axis)

    fs = 13

    fig, ax1 = plt.subplots(figsize=(8,5))

    for cc, sl in enumerate(slices):

        if label is not None:
            ax1.plot(xrange, sl, label=label.format(values[cc]))
        else:
            ax1.plot(xrange, sl)

    # labels and such
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=fs, fontweight="bold")

    ax1.set_ylabel(r"$\Delta$G [kT]", fontsize=fs, fontweight="bold")
    if title is not None:
        ax1.set_title(title, fontsize=fs+1, fontweight="bold")

    # add grid
    ax1.grid(linestyle="--")

    ax1.set_xlim(left=0)

    if label is not None:
        ax1.legend()
    # various
    fig.tight_layout()
    fig.show()

    return

# def plot_isosurface(self):

#     # unpack 3D data
#     X, Y, Z, values = self.raw_data.T


#     fig = go.Figure(data=go.Isosurface(
#         x=X.flatten(),
#         y=Y.flatten(),
#         z=Z.flatten(),
#         value=values.flatten(),
#         isomin=0,
#         isomax=20,
#         surface_count=5, # number of isosurfaces, 2 by default: only min and max
#         colorbar_nticks=5, # colorbar ticks correspond to isosurface values
#         caps=dict(x_show=False, y_show=False)
#         ))

#     fig.update_traces(opacity=0.3, selector=dict(type='isosurface'))
#     plt.show()


#     return fig


