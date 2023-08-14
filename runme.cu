#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include "toplevel.cuh"
#include "qdldl.h"
#include "rbd_plant.cuh"
#include "settings.cuh"
#include "experiment_helpers.cuh"






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
        // std::cout << "start: " << start_state << " goal: " << goal_state << std::endl;

#if PCG_SOLVE
        uint32_t num_exit_vals = 9;
        float pcg_exit_vals[num_exit_vals];
        if(knot_points==32){
            pcg_exit_vals[0] = 1e-7;
            pcg_exit_vals[1] = 7.5e-6;
            pcg_exit_vals[0] = 5e-6;
            pcg_exit_vals[1] = 2.5e-6;
            pcg_exit_vals[2] = 1e-6;
            pcg_exit_vals[3] = 7.5e-5;
            pcg_exit_vals[4] = 5e-5;
            pcg_exit_vals[3] = 2.5e-5;
            pcg_exit_vals[4] = 1e-5;
        }
        else if(knot_points==64){
            pcg_exit_vals[0] = 1e-6;
            pcg_exit_vals[1] = 7.5e-5;
            pcg_exit_vals[0] = 5e-5;
            pcg_exit_vals[1] = 2.5e-5;
            pcg_exit_vals[2] = 1e-5;
            pcg_exit_vals[3] = 7.5e-4;
            pcg_exit_vals[4] = 5e-4;
            pcg_exit_vals[3] = 2.5e-4;
            pcg_exit_vals[4] = 1e-4;
        }
        else{
            pcg_exit_vals[0] = 5e-5;
            pcg_exit_vals[1] = 2.5e-5;
            pcg_exit_vals[0] = 1e-5;
            pcg_exit_vals[1] = 7.5e-4;
            pcg_exit_vals[2] = 5e-4;
            pcg_exit_vals[3] = 2.5e-4;
            pcg_exit_vals[4] = 1e-4;
            pcg_exit_vals[3] = 7.5e-3;
            pcg_exit_vals[4] = 5e-3;
        }
#else
        uint32_t num_exit_vals = 1;
        float pcg_exit_vals[num_exit_vals] = {-1};
#endif

        for (uint32_t pcg_exit_ind = 0; pcg_exit_ind < num_exit_vals; pcg_exit_ind++){

            float pcg_exit_tol = pcg_exit_vals[pcg_exit_ind];
            std::vector<double> linsys_times;
            std::vector<uint32_t> sqp_iters;
            std::vector<toplevel_return_type> current_results;
            std::vector<float> tracking_errs;
            std::vector<float> cur_tracking_errs;
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

                std::tuple<std::vector<toplevel_return_type>, std::vector<pcg_t>, pcg_t> trackingstats = track<pcg_t, toplevel_return_type>(state_size, control_size, knot_points, static_cast<uint32_t>(eePos_traj2d.size()), timestep, d_eePos_traj, d_xu_traj, d_xs, start_state, goal_state, single_traj_test_iter, pcg_exit_tol);
                
                current_results = std::get<0>(trackingstats);
                if (TIME_LINSYS == 1) {
                    linsys_times.insert(linsys_times.end(), current_results.begin(), current_results.end());
                } else {
                    sqp_iters.insert(sqp_iters.end(), current_results.begin(), current_results.end());
                }

                cur_tracking_errs = std::get<1>(trackingstats);
                tracking_errs.insert(tracking_errs.end(), cur_tracking_errs.begin(), cur_tracking_errs.end());

                tot_final_tracking_err += std::get<2>(trackingstats);
                


                gpuErrchk(cudaFree(d_xu_traj));
                gpuErrchk(cudaFree(d_eePos_traj));
                gpuErrchk(cudaFree(d_xs));
                gpuErrchk(cudaPeekAtLastError());
                
            }

            std::cout << "\nRESULTS*************************************\n";
            std::cout << "exit tol: " << pcg_exit_tol << std::endl;
            std::cout << "tracking err\n";
            printStats<float>(&tracking_errs);
            std::cout << tot_final_tracking_err / traj_test_iters << std::endl;
            if (TIME_LINSYS == 1)
            {
                std::cout << "linsys times\n";
                printStats<double>(&linsys_times);
            }
            else
            {
                std::cout << "sqp iters\n";
                printStats<uint32_t>(&sqp_iters);
            }
            std::cout << "************************************************\n\n";
        }
        break;
    }




    return 0;
}