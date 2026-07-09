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

## Note on `min_energy_path`

`pycolvars/min_energy_path/MEPSAnd_ASM.py` is a vendored copy of
[MEPSAnd](https://doi.org/10.1093/bioinformatics/btz684) by Marcos-Alcalde et
al., distributed under the GPLv3 license (see `LICENSE-MEPSAND`), separate
from the MIT license covering the rest of this repository.
