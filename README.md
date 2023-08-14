# gpu-pcg-tracking

## Setting up for experiments
Some basic setup is required to prepare for test execution. After cloning the repo, initialize the submodules with:

```
git submodule update --init --recursive
```

Then build ![qdldl](https://github.com/osqp/qdldl):

```
cd qdldl
mkdir build
cd build
cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false ..
cmake —build .
```
Then add the `qdldl/build/out` directory which gets created to the LD_LIBRARY_PATH environment variable.

## Performing experiments
There are two separate types of experiment this repository is setup to perform - timing the linear system solve time, and comparing this solve time to the solve
times from QDLDL, and timing the full SQP loop using GPU-PCG and QDLDL as the two underlying linear system solvers, and comparing performance and runtime of SQP in each configuration.
