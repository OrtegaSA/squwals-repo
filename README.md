<div align="center">    
 
# SQUWALS


[![arXiv](http://img.shields.io/badge/arXiv-2307.14314-B31B1B.svg)](https://arxiv.org/abs/2307.14314)
[![Journal](http://img.shields.io/badge/Advanced_Quantum_Technologies-2024-4b44ce.svg)](https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202400022)

</div>
 
## Description   
A Szegedy QUantum WALks Simulator.

## Installation  
Open a system's console or an Anaconda Prompt depending on your python installation.

First, clone the repository.
```bash
git clone https://github.com/OrtegaSA/squwals-repo
```
This creates a folder called squwals-repo. Change the directory to it.
```bash
cd squwals-repo
```
Install the package using pip.
```bash
pip install .
```

Alternativelly, you can download the folder squwals and copy it in your python working directory, or in some directory included in PYTHONPATH.

<!--  
Then we create a conda environment
```
conda create -n squwals python=3.6
conda activate squwals
```
-->  

## Applications

This package includes some high-level applications:

- Quantum PageRank
- Semiclassical Szegedy Walk
- Quantum, Semiclassical and Randomized SearchRank

## Tutorials
There is a tutorial for using SQUWALS in the folder Tutorials, as well as examples of the high-level applications. 

## v2.0
This version allows the introduction of complex-phase extensions, as local arbitrary phase rotations, in order to simulate the graph-phased Szegedy quantum walk. A tutorial is included.

### Citation 
<!---
```
@article{ortega2023squwals,
  title={SQUWALS: A Szegedy QUantum WALks Simulator},
  author={Ortega, Sergio A. and Martin-Delgado, Miguel Angel},
  journal={arXiv:2307.14314},
  year={2023},
}
```
-->
```
@article{ortega2024squwals,
	title={SQUWALS: A Szegedy QUantum WALks Simulator},
	author={Ortega, S. A. and Martin-Delgado, M. A.},
	journal={Advanced Quantum Technologies},
	pages = "2400022",
	year={2024}
}
```
<!---
For the graph-phased Szegdy quantum walk:
```
@article{ortega2024graph-phased,
  title={Complex-Phase Extensions of Szegedy Quantum Walk on Graphs},
  author={Ortega, Sergio A. and Martin-Delgado, Miguel Angel},
  journal={arXiv:...},
  year={2024},
}
```
-->
