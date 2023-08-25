#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include "gpuassert.cuh"
#include "file_utils.cuh"
#include "types.cuh"
#include "integrator.cuh"
#include "rbd_plant.cuh"


int main(){

    float time = 0.015625;

    std::vector<std::vector<float>> traj2d = readCSVToVecVec("./testfiles/6_traj.csv");
    std::vector<float> h_traj;
    for (const auto& vec : traj2d) {
        h_traj.insert(h_traj.end(), vec.begin(), vec.end());
    }
    float *d_traj, *d_xs;
    gpuErrchk(cudaMalloc(&d_traj, h_traj.size()*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_xs, 14*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_traj, h_traj.data(), h_traj.size()*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_xs, h_traj.data(), 14*sizeof(float), cudaMemcpyHostToDevice));
    
    void *d_dynmem = gato_plant::initializeDynamicsConstMem<float>();

    float h_temp[14];

    for(int i = 0; i < 100; i++){
        // gpuErrchk(cudaMemcpy(d_xs, &d_traj[i*21], 14*sizeof(float), cudaMemcpyDeviceToDevice));
        simple_integrator<float>(14, 7, 50, d_xs, d_traj, d_dynmem, time, i * time, time);
        gpuErrchk(cudaMemcpy(h_temp, d_xs, 14*sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 14; i++){
            std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_temp[i] << " ";
        }
        std::cout <<"\n\n";
    }
    std::cout << "state 10 should be:\n";
    for (int i = 0; i < 14; i++){
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_traj.data()[21*100 + i] << " ";
    }
    std::cout << std::endl;

    gato_plant::freeDynamicsConstMem<float>(d_dynmem);

    gpuErrchk(cudaFree(d_traj));
    gpuErrchk(cudaFree(d_xs));

    return 0;
}