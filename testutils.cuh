#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include "gpuassert.cuh"
#include "settings.cuh"


template <typename T>
__global__
void compute_tracking_error_kernel(float *d_tracking_error, uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu_goal, T *d_xu){
    
    float err;

    for(int ind = threadIdx.x; ind < (state_size+control_size)*knot_points - control_size; ind += blockDim.x){
        if (ind % (state_size+control_size) < state_size){
            err = abs(d_xu[ind] - d_xu_goal[ind]);
            atomicAdd(d_tracking_error, err);
        }
    }
}



template <typename T>
float compute_tracking_error(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu_goal, T *d_xu){

    float h_tracking_error = 0.0f;
    float *d_tracking_error;
    gpuErrchk(cudaMalloc(&d_tracking_error, sizeof(float)));
    gpuErrchk(cudaMemcpy(d_tracking_error, &h_tracking_error, sizeof(float), cudaMemcpyHostToDevice));

    compute_tracking_error_kernel<T><<<1,1024>>>(d_tracking_error, state_size, control_size, knot_points, d_xu_goal, d_xu);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&h_tracking_error, d_tracking_error, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_tracking_error));
    return h_tracking_error;
}


template <typename T>
void dump_tracking_data(std::vector<int> *pcg_iters, std::vector<double> *linsys_times, std::vector<double> *sqp_times, std::vector<uint32_t> *sqp_iters, std::vector<bool> *sqp_exits, std::vector<float> *tracking_errors, std::vector<std::vector<T>> *tracking_path, uint32_t timesteps_taken, uint32_t control_updates_taken, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter){
    // Helper function to create file names
    auto createFileName = [&](const std::string& data_type) {
        std::string filename = RESULTS_DIRECTORY + std::to_string(start_state_ind) + "_" + std::to_string(goal_state_ind) + "_" + std::to_string(test_iter) + "_" + data_type + ".result";
        return filename;
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");


    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    // printStatsToFile<double>(&linsys_times, )
    
    statsfile.close();
}