
#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#define Q_COST          (.10)
#define QD_COST          (.10)
#define R_COST          (0.0001)

#include "iiwa_eepos_plant.cuh"

/*
Test script to validate the end effector positions computed by grid
Compile with:
nvcc -arch=sm_86 -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl -I/home/a2rlab/anaconda3/envs/crocoddyl/include -I/home/a2rlab/anaconda3/envs/crocoddyl/include/eigen3  -L/home/a2rlab/anaconda3/envs/crocoddyl/lib -lcrocoddyl -o kin_test.out test_ee_positions.cu
*/

template <typename T>
__global__ void simpleTestKernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu, T *d_eePos_traj, T *d_temp, grid::robotModel<T> *d_robotModel) {
    __shared__ float s_temp[144];  // This allocates 144 * 8 bytes of shared memory

    // Initialize shared memory to zero if needed (This step is optional)
    for (int i = 0; i < 144; ++i) {
        s_temp[i] = 0.0;
    }
    __syncthreads();
    gato_plant::test_ee_pos<T>(state_size, control_size, knot_points, d_xu, d_eePos_traj, s_temp, d_robotModel);
	__syncthreads();
}

int main() {
  uint32_t state_size = 14;
  uint32_t control_size = 7;
  uint32_t knot_points = 32;
  
  float h_xu0[] = {2.01871455, -0.44240776, 1.67994461, 1.25010269, 2.4427646, -1.26689386, -1.00656691};
  float h_xu1[] = {2.32800806,  0.14724287,  2.11142166,  0.6768961,   2.05601061, -1.59882133, -1.04659708};
  float h_xu2[] = {1.91128841, -0.93231821,  2.80165631,  0.66113484,  2.56988238, -0.79271947, -1.37765013};
  float h_xu3[] = {2.38766759, -0.40455493,  2.01021502,  1.32428586,  2.89747593, -1.19397084, -0.06813061};

  // Allocate device memory
  float *d_xu;
  cudaMalloc((void**)&d_xu, state_size * sizeof(float));

  // Copy h_xu from host to device
  cudaMemcpy(d_xu, h_xu3, state_size * sizeof(float), cudaMemcpyHostToDevice);

  // Initialize h_eePos_traj to zeros on the host
  float h_eePos_traj[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  // Allocate device memory for d_eePos_traj
  float *d_eePos_traj;
  cudaMalloc((void**)&d_eePos_traj, 6 * sizeof(float));
  // Copy h_eePos_traj from host to device (all zeros)
  cudaMemcpy(d_eePos_traj, h_eePos_traj, 6 * sizeof(float), cudaMemcpyHostToDevice);

  float *d_temp;  // Allocated and initialized elsewhere

  void *d_dynmem = gato_plant::initializeDynamicsConstMem<float>();

  simpleTestKernel<float><<<1, 1>>>(state_size, control_size, knot_points, d_xu, d_eePos_traj, d_temp, (grid::robotModel<float> *)d_dynmem);

  cudaDeviceSynchronize();
  
  cudaFree(d_xu);
  gato_plant::freeDynamicsConstMem<float>(d_dynmem);

  return 0;
}