# pycolvars

Python tools to post-process and analyze collective variable (CV) data from
metadynamics and umbrella sampling simulations: reading colvars/PLUMED-style
trajectory files, reconstructing free energy surfaces (PMF), and finding
minimum energy paths across them.

This package was originally developed alongside
[`sea_urchin`](https://gitlab.com/electrolyte-machine/sea_urchin) and later
split out as a standalone tool.

## Installation

```bash
pip install -e .
```

## Minimum energy path algorithm

`pycolvars.min_energy_path` finds the path between two basins on an energy
surface that minimises the maximum energy barrier crossed (the minimax / bottleneck
path). It uses the Minimum Spanning Tree of the grid connectivity graph (edge
weight = max endpoint energy), which provably yields the optimal minimax path.
k-th alternative paths are found by iteratively removing the bottleneck edge
of the previous best path and recomputing.

This is a native implementation using `networkx`. It replaces an earlier
vendored copy of [MEPSAnd](https://doi.org/10.1093/bioinformatics/btz684) by
Marcos-Alcalde et al.
