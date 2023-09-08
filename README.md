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

To enable testing with Crocoddyl, we also need to get that installed. Start by creating a fresh conda environment using the conda-env.yml file here (taken from the pinocchio github action workflow, with the python version hardcoded to 3.8, because more recent versions of python have given me issues):


```
conda env create -f conda-env.yml
```

We also need to install some other dependencies into the conda environment:

```
conda activate pinocchio
conda install example-robot-data -c conda-forge --no-deps
conda install cmake ccache llvm-openmp compilers=1.4.2 -c conda-forge
```

Then we need to build pinocchio. Can also try installing it via conda, but if we build from source we can use the devel branch. Clone the repo, switch to the devel branch, and build it:

```
git clone --recursive https://github.com/stack-of-tasks/pinocchio
cd pinocchio
git checkout devel
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j4
sudo make install
```

Then we install crocoddyl, also from the devel branch:
```
git clone --recursive https://github.com/loco-3d/crocoddyl.git
cd crocoddyl
git checkout devel
mkdir build && cd build
cmake .. -DBUILD_WITH_MULTITHREADS=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make
sudo make install
```

## Performing experiments
The perform_experiment.sh script has a section at the top where values for different parameters can be specified. Execute this script and save the output to a file. Set time_linsys to 1 in this section to record linear system solve times. Set it to 0 to record number of sqp iterations. In both cases, the tracking error will also be recorded.

While most configuration options are set in this one section of the script, there are a couple of other places for a few particular options to be aware of. The number of test iterations is set via the TEST_ITERS macro in the settings.cuh file. The exit tolerances to try for each number of knot point are set in the runme.cu file, in the pcg_exit_vals array.
