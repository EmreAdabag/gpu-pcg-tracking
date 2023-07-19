#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include "toplevel.cuh"
#include "qdldl.h"
#include "rbd_plant.cuh"
#include "settings.cuh"







int main(){

    const uint32_t state_size = grid::NUM_JOINTS*2;
    const uint32_t control_size = grid::NUM_JOINTS;
    const uint32_t knot_points = KNOT_POINTS;
    const pcg_t timestep = .015625;

    const uint32_t traj_test_iters = TEST_ITERS;

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

        uint32_t num_exit_vals = 10;
        float pcg_exit_vals[num_exit_vals] = {1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8};


        for (uint32_t pcg_exit_ind = 0; pcg_exit_ind < num_exit_vals; pcg_exit_ind++){
            // if(pcg_exit_ind < 2){continue;}

            float pcg_exit_tol = pcg_exit_vals[pcg_exit_ind];
            double tot_avg_sqp_iters = 0;
            double tot_avg_tracking_err = 0;
            double tot_final_tracking_err = 0;


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


                auto trackingstats = track<pcg_t>(state_size, control_size, knot_points, static_cast<uint32_t>(eePos_traj2d.size()), timestep, d_eePos_traj, d_xu_traj, d_xs, start_state, goal_state, single_traj_test_iter, pcg_exit_tol);
                tot_avg_sqp_iters += std::get<0>(trackingstats);
                tot_avg_tracking_err += std::get<1>(trackingstats);
                tot_final_tracking_err += std::get<2>(trackingstats);

                if (std::get<1>(trackingstats) > 1 || std::get<2>(trackingstats) > .1){
                    std::cout << "error condition violation, ignore this result set\n";
                    break;
                }

                gpuErrchk(cudaFree(d_xu_traj));
                gpuErrchk(cudaFree(d_eePos_traj));
                gpuErrchk(cudaFree(d_xs));
                gpuErrchk(cudaPeekAtLastError());
                
            }

            std::cout << "\nRESULTS\n";
            std::cout << "exit tol: " << pcg_exit_tol << std::endl;
            std::cout << "avg avg tracking err: " << tot_avg_tracking_err / traj_test_iters << " avg final tracking err " << tot_final_tracking_err / traj_test_iters << std::endl;
            std::cout << "avg avg sqp iters: " << tot_avg_sqp_iters / traj_test_iters << std::endl;
            std::cout << "\n\n";
        }
        break;
    }




    return 0;
}