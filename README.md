![build](https://github.com/mir-group/flare_pp/actions/workflows/main.yml/badge.svg)

# flare++
Documentation can be accessed [here](https://mir-group.github.io/flare_pp/). An introductory tutorial in Google Colab is available [here](https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv).

## Installation

```
pip install flare_pp
```

If you're installing on Harvard's compute cluster, load the following modules first:
```
module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01
```

### MPI compilation
To install the MPI version of sparse GP, on Harvard's compute cluster, load the following modules first:
```
module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01
module load intel-mkl/2017.2.174-fasrc01 openmpi/4.0.5-fasrc01
```

Then clone the repository
```
git clone https://github.com/mir-group/flare_pp.git
git checkout mpi_distmat
```

Create a directory for building the library
```
mkdir build
cd build
```

Compile with `cmake` and `make`. Here we need to specify the MPI compiler with 
`CC=mpicc CXX=mpic++ FC=mpif90`. We also set the option `-DSCALAPACK_LIB=NOTFOUND`,
such that we download and compile our own static Scalapack library to support 
python binding with MPI parallelized sparse GP
```
CC=mpicc CXX=mpic++ FC=mpif90 cmake .. -DSCALAPACK_LIB=NOTFOUND
make -j
```

Copy the python binding library to the folder of `flare_pp`
```
cp _C_flare.cpython-36m-x86_64-linux-gnu.so ../flare_pp
```
