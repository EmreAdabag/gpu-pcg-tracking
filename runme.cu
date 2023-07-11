#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include "toplevel.cuh"
#include "qdldl.h"
#include "rbd_plant.cuh"
#include "settings.cuh"






template <typename T>
std::vector<std::vector<T>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<T>> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;


        while (std::getline(infile, line)) {
            std::vector<T> row;
            std::stringstream ss(line);
            std::string val;

            while (std::getline(ss, val, ',')) {
                row.push_back(std::stof(val));
            }

            data.push_back(row);
        }
    }

    infile.close();
    return data;
}




int main(){

    const uint32_t state_size = grid::NUM_JOINTS*2;
    const uint32_t control_size = grid::NUM_JOINTS;
    const uint32_t knot_points = KNOT_POINTS;
    const pcg_t timestep = .015625;

    const uint32_t traj_test_iters = 1;

    // checks GPU space for pcg
    checkPcgOccupancy<pcg_t>((void *) pcg<pcg_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);    
    if(!std::is_same<QDLDL_float, pcg_t>::value){ std::cout << "GPU-PCG QDLDL type mismatch" << std::endl; exit(1); }

    print_test_config();

    const uint32_t recorded_states = 5;
    const uint32_t start_goal_combinations = recorded_states*recorded_states;

    char eePos_traj_file_name[100];
    char xu_traj_file_name[100];

    int start_state, goal_state;
    pcg_t *d_eePos_traj, *d_xu_traj, *d_xs;

    for(uint32_t ind = 0; ind < start_goal_combinations; ind++){

        start_state = ind % recorded_states;
        goal_state = ind / recorded_states;
        if(start_state == goal_state && start_state != 0){ continue; }
        std::cout << "start: " << start_state << " goal: " << goal_state << std::endl;

        for (uint32_t single_traj_test_iter = 0; single_traj_test_iter < traj_test_iters; single_traj_test_iter++){

            // read in traj
            snprintf(eePos_traj_file_name, sizeof(eePos_traj_file_name), "testfiles/%d_%d_eepos.traj", start_state, goal_state);
            std::vector<std::vector<pcg_t>> eePos_traj2d = readCSVToVecVec<pcg_t>(eePos_traj_file_name);
            
            snprintf(xu_traj_file_name, sizeof(xu_traj_file_name), "testfiles/%d_%d_traj.csv", start_state, goal_state);
            std::vector<std::vector<pcg_t>> xu_traj2d = readCSVToVecVec<pcg_t>(xu_traj_file_name);
            
            if(eePos_traj2d.size() < knot_points){std::cout << "precomputed traj length < knotpoints, not implemented\n"; continue; }


            std::vector<pcg_t> h_eePos_traj;
            for (const auto& vec : eePos_traj2d) {
                h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
            }
            std::vector<pcg_t> h_xu_traj;
            for (const auto& xu_vec : xu_traj2d) {
                h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
            }

            gpuErrchk(cudaMalloc(&d_eePos_traj, h_eePos_traj.size()*sizeof(pcg_t)));
            gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(pcg_t), cudaMemcpyHostToDevice));
            
            gpuErrchk(cudaMalloc(&d_xu_traj, h_xu_traj.size()*sizeof(pcg_t)));
            gpuErrchk(cudaMemcpy(d_xu_traj, h_xu_traj.data(), h_xu_traj.size()*sizeof(pcg_t), cudaMemcpyHostToDevice));
            
            gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(pcg_t)));
            gpuErrchk(cudaMemcpy(d_xs, h_xu_traj.data(), state_size*sizeof(pcg_t), cudaMemcpyHostToDevice));


            track<pcg_t>(state_size, control_size, knot_points, static_cast<uint32_t>(eePos_traj2d.size()), timestep, d_eePos_traj, d_xu_traj, d_xs, start_state, goal_state, single_traj_test_iter);

            gpuErrchk(cudaPeekAtLastError());
            
        }
        break;
    }



    gpuErrchk(cudaFree(d_xu_traj));
    gpuErrchk(cudaFree(d_eePos_traj));
    gpuErrchk(cudaFree(d_xs));

    return 0;
}