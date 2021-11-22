# SpikeShip
## Fast, unsupervised discovery of high-dimensional neural spiking patterns based on optimal transport 
## theory
A Python 3 module called `spikeship` which implements the Fast, unsupervised discovery of high-dimensional 
neural spiking patterns based on optimal transport theory described in Sotomayor-Gomez, B., L., Battaglia, 
F. and Vinck, M. (2020). A geometry of spike sequences: Fast, unsupervised discovery of high-dimensional 
neural spiking patterns based on optimal transport theory. *Bioarxiv.*

## Setup
The dependencies can be installed by running `./env_setup.sh <ENV_NAME>` with the optional argument 
specifying the target environment (which must be source-able). To setup the module, run `python setup.py 
install`. A jupyter notebook is available in `notebooks/`, along with a demo dataset, showing an example 
workflow for the SPOTDisClust method.

### Linux (Anaconda)
1) `conda create -n spikeship python=3` 1) `./env_setup.sh spikeship` 1) `source activate spikeship` 1) 
`python setup.py install`

### Windows (Anaconda)
1) `conda create -n spikeship python=3` 1) `conda activate spikeship` 1) `conda install python=3.6.5` 1) 
`conda install -c conda-forge hdbscan=0.8.13=py36_0` 1) `conda install numba` 1) `conda install ipykernel` 
1) `conda install matplotlib` 1) `python setup.py install`

## Further notes
The software requirements/dependencies are the same from the work ![SPOTDis](https://github.com/LGro/spot), 
the implementatoin of the Spike Pattern Optimal Transport Dissimilarity described in Grossberger, L., 
Battaglia, F. and Vinck, M. (2018). Unsupervised clustering of temporal patterns in high-dimensional 
neuronal ensembles using a novel dissimilarity measure. *PLoS Comput. Biol.*
