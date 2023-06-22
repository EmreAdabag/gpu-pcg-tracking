#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include "toplevel.cuh"

#include "iiwa_plant.cuh"

#include "settings.cuh"






std::vector<std::vector<float>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;


        while (std::getline(infile, line)) {
            std::vector<float> row;
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
    const uint32_t state_size = 14;
    const uint32_t control_size = 7;
    const uint32_t knot_points = 64;
    const float timestep = .015625;


    const uint32_t recorded_states = 5;
    const uint32_t start_goal_combinations = recorded_states*recorded_states;

    char traj_file_name[100];
    char lambda_file_name[100];

    int start_state, goal_state;
    float *d_traj, *d_traj_lambdas, *d_xs;

    for(int ind = 0; ind < start_goal_combinations; ind++){

        start_state = ind % recorded_states;
        goal_state = ind / recorded_states;
        if(start_state == goal_state){ continue; }
        std::cout << "start: " << start_state << " goal: " << goal_state << std::endl;

        for (int single_traj_test_iter = 0; single_traj_test_iter < 2; single_traj_test_iter++){

            // read in traj, lambdas
            snprintf(traj_file_name, sizeof(traj_file_name), "testfiles/%d_%d_traj.csv", start_state, goal_state);
            snprintf(lambda_file_name, sizeof(lambda_file_name), "testfiles/%d_%d_lambdas.csv", start_state, goal_state);
            std::vector<std::vector<float>> traj2d = readCSVToVecVec(traj_file_name);
            std::vector<std::vector<float>> lambdas2d = readCSVToVecVec(lambda_file_name);

            std::vector<float> h_traj;
            for (const auto& vec : traj2d) {
                h_traj.insert(h_traj.end(), vec.begin(), vec.end());
            }
            std::vector<float> h_lambdas;
            for (const auto& vec : lambdas2d) {
                h_lambdas.insert(h_lambdas.end(), vec.begin(), vec.end());
            }

            gpuErrchk(cudaMalloc(&d_traj, h_traj.size()*sizeof(float)));
            gpuErrchk(cudaMemcpy(d_traj, h_traj.data(), h_traj.size()*sizeof(float), cudaMemcpyHostToDevice));
            
            gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(float)));
            gpuErrchk(cudaMemcpy(d_xs, h_traj.data(), state_size*sizeof(float), cudaMemcpyHostToDevice));
            
            gpuErrchk(cudaMalloc(&d_traj_lambdas, h_lambdas.size() *sizeof(float)));
            gpuErrchk(cudaMemcpy(d_traj_lambdas, h_lambdas.data(), h_lambdas.size()*sizeof(float), cudaMemcpyHostToDevice));


            track<float>(state_size, control_size, knot_points, traj2d.size(), timestep, d_traj, d_traj_lambdas, d_xs, start_state, goal_state, single_traj_test_iter);

            gpuErrchk(cudaPeekAtLastError());
        }
    }



    gpuErrchk(cudaFree(d_traj));
    gpuErrchk(cudaFree(d_xs));
    gpuErrchk(cudaFree(d_traj_lambdas))

    return 0;
}