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
Also add the `qdldl/build/out` directory which gets created to the LD_LIBRARY_PATH environment variable.

## Performing experiments
The perform_experiment.sh script has a section at the top where values for different parameters can be specified. Execute this script and save the output to a file. Set time_linsys to 1 in this section to record linear system solve times. Set it to 0 to record number of sqp iterations. In both cases, the tracking error will also be recorded.

While most configuration options are set in this one section of the script, there are a couple of other places for a few particular options to be aware of. The number of test iterations is set via the TEST_ITERS macro in the settings.cuh file. The exit tolerances to try for each number of knot point are set in the runme.cu file, in the pcg_exit_vals array.
