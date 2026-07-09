#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:37:50 2025

@author: roncofaber
"""

# numpy as always
import numpy as np

# ase
from ase import units

# os stuff
import os
import glob
import subprocess
from typing import List, Tuple, Optional

#%%



class UmbrellaSampling:
    """
    Class to process umbrella sampling data and run WHAM analysis.
    """
    def __init__(
        self,
        pathlist: List[str],
        filename: str,
        harm_k: float,
        temperature: float,
        cols_of_interest: Optional[List[str]] = None
    ):
        """
        Initialize the UmbrellaSampling object.

        Args:
            pathlist (List[str]): List of directories to search for data files.
            filename (str): Name of the data file to look for in each directory.
            harm_k (float): Harmonic force constant.
            temperature (float): Temperature for WHAM.
            cols_of_interest (List[str], optional): Columns to extract from data files.
        """
        self._whamdata = None
        self._whamlog = None
        self.harm_k = harm_k
        self.temperature = temperature

        self.cols_of_interest = cols_of_interest or ['time', 'dZ.z']

        self._filelist = self._gather_files(pathlist, filename)
        self._datas, self._x0s = self._process_data(self._filelist)
        
        self._whamdata, self._whamlog = self._run_wham(harm_k, temperature)

    def _gather_files(self, pathlist: List[str], filename: str) -> List[str]:
        """
        Collect all matching files from the provided paths.
        """
        filelist = []
        for path in pathlist:
            tfiles = glob.glob(os.path.join(path, "*/", filename))
            filelist.extend(tfiles)
        if not filelist:
            raise FileNotFoundError(f"No files found for {filename} in {pathlist}")
        return filelist

    def _process_data(self, filelist: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Read and sort the data files.
        """
        datas = []
        x0s = []
        for filename in filelist:
            odata, x0 = self._read_data(filename)
            datas.append(odata)
            x0s.append(x0)
        # Sort by x0
        sorted_pairs = sorted(zip(x0s, datas), key=lambda pair: pair[0])
        x0s_sorted, datas_sorted = zip(*sorted_pairs)
        return list(datas_sorted), list(x0s_sorted)

    def _read_data(self, filename: str) -> Tuple[np.ndarray, float]:
        """
        Read a single data file and extract relevant columns and x0.
        """
        try:
            x0 = float(os.path.basename(os.path.dirname(filename)).split("_")[-1])
        except Exception as e:
            raise ValueError(f"Could not parse x0 from filename '{filename}': {e}")

        with open(filename) as f:
            first_line = f.readline().strip()

        columns = first_line.replace('#! FIELDS', '').split()
        try:
            indices = [columns.index(col) for col in self.cols_of_interest]
        except ValueError as e:
            raise ValueError(f"Column of interest not found in {filename}: {e}")

        data = np.genfromtxt(filename, skip_header=1, invalid_raise=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        data = data[~np.isnan(data).any(axis=1)]
        odata = data[:, indices]
        return odata, x0

    def _run_wham(
        self,
        harm_k: Optional[float] = None,
        temperature: Optional[float] = None,
        wham_conf: str = "whamfile.in",
        wham_out: str = "whamfile.out",
        wham_bin: str = "wham"
    ):
        """
        Prepare input files and run WHAM analysis.

        Args:
            harm_k (float, optional): Harmonic force constant (overrides init).
            temperature (float, optional): Temperature for WHAM (overrides init).
            wham_conf (str): WHAM input file name.
            wham_out (str): WHAM output file name.
            wham_bin (str): WHAM executable name.
        """
        harm_k = harm_k if harm_k is not None else self.harm_k
        temperature = temperature if temperature is not None else self.temperature

        # Write WHAM configuration and data files
        with open(wham_conf, "w") as fout:
            for cc, odata in enumerate(self._datas):
                fname = f"umbrella_{str(cc).zfill(2)}.dat"
                x0 = self._x0s[cc]
                np.savetxt(fname, odata)
                fout.write(f"{fname}  {x0}  {harm_k}  25\n")

        # Run WHAM
        cmd = [
            wham_bin, "units", "real",
            "0.0", "18", "145",
            "1e-6", f"{temperature}", "0",
            wham_conf,
            wham_out,
            "10",
            "42"
        ]
        try:
            whamlog = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"WHAM failed: {e.stderr}")

        whamdata = np.genfromtxt(wham_out, invalid_raise=False)

        return whamdata, whamlog

    @property
    def whamdata(self):
        return self._whamdata

    @property
    def whamlog(self):
        return self._whamlog