# modality
Statistical tests for evaluating unimodality, in particular:
- Hartigan's dip test (equivalent to excess mass test).
- Silverman's bandwidth test.
- Calibrated versions of the above.
Estimation of excess mass modes

## Version 1.1
Added flow cytometry interface, for integrated preprocessing of flow
cytometry data. See tests/test_fc_interface.py

### References
K. Johnsson and M. Fontes (2016): What is a `unimodal' cell
population? Using statistical tests as criteria for unimodality in
automated gating and quality control (manuscript).

Hartigan and Hartigan (1985): The dip test of unimodality.
The Annals of Statistics. 13(1).

Silverman (1981): Using kernel density estimates to
investigate multimodality. Journal of the Royal Statistical
Society. Series B. 43(1).

M.-Y. Cheng and P. Hall (1999): Mode testing in difficult cases.
The Annals of Statistics. 27(4).

Müller and Sawitski (1991): Excess Mass Estimates and Tests for
Multimodality. JASA. 86(415).

Cheng and Hall (1998): On mode testing and empirical
approximations to distributions. Statistics & Probability
Letters 39.

## Usage
Basic usage of the package is showcased in the test files in the
directory `tests/`. Each test file can be run directly or with mpi,
for example
```
mpirun -n N python test_modality.py
```
where `N` is the number of cores to be used.

### Calling modality from R
After the package has been installed (see below) it can be used from R
through the R package XRPython.

NB! In your R session, before XRPython is loaded you have to set the 
PYTHONPATH environment variable so that it points to the directory where
your Python packages are installed (e.g. "/usr/local/lib/python2.7/site-packages").
This can be done with:
```
Sys.setenv(PYTHONPATH="`path to packages`")
```
If you don't know where your packages are installed, start a Python
session and write:
```
import os
print os.environ['PYTHONPATH']
```
Some basic usage is showcased in the file tests/test_R.R.

## Dependencies
The package has the following dependencies:
- Python 2.7, including packages scipy, numpy (>= 1.11), matplotlib, mpi4py, (rpy2), scikit-learn
- OpenMPI

rpy2 is necessary for the uncalibrated version of Hartigan's dip test,
as well as R and the R package diptest (see Installation).

### Ubuntu
Python, OpenMPI and the required Python packages can be installed by:
```
sudo apt-get install python, openmpi  
sudo pip install numpy, scipy, matplotlib, mpi4py, rpy2, sklearn
```

### Mac
Python and OpenMPI can be installed using homebrew.
First install homebrew following the instructions at http://brew.sh.
Then install Python and OpenMPI:
```
brew install python, openmpi
```
Then Python packages can be installed using
```
pip install numpy, scipy, matplotlib, mpi4py, rpy2, sklearn
```

## Installation
Download modality-1.1.tar.gz at

https://github.com/kjohnsson/modality/blob/master/dist/modality-1.1.tar.gz

and unpack it. In the terminal, go to the extracted directory modality-1.1 and write
```
python setup.py install
```
If installation for only one user is wanted, you can add the option
`--user`.

During installation, tablulated p-values for Hartigan's dip test
are obtained from the R package 'diptest'. If the package fails to load,
rpy2 tries to install it. Another way to install the 'diptest' R package
is to write "install.packages('diptest')" within R.

## Using MPI
When using parallel execution with MPI, it is recommended to add the
following code to your script, so that the process is aborted if only
one thread fails.

```
from mpi4py import MPI
import traceback
import sys

if MPI.COMM_WORLD.Get_size() > 1:
    # Ensure that all threads are terminated if one fails

    def mpiexceptabort(type, value, tb):
        traceback.print_exception(type, value, tb)
        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = mpiexceptabort
```