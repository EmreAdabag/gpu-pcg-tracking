#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include "toplevel.cuh"

#include "iiwa_plant.cuh"








std::vector<std::vector<float>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File could not be opened!\n";
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
    const uint32_t knot_points = 20;


    std::vector<std::vector<float>> traj2d = readCSVToVecVec("./testfiles/0_traj.csv");
    std::vector<std::vector<float>> lambdas2d = readCSVToVecVec("./testfiles/0_lambdas.csv");

    std::vector<float> h_traj;
    for (const auto& vec : traj2d) {
        h_traj.insert(h_traj.end(), vec.begin(), vec.end());
    }
    std::vector<float> h_lambdas;
    for (const auto& vec : lambdas2d) {
        h_lambdas.insert(h_lambdas.end(), vec.begin(), vec.end());
    }

    float *d_traj, *d_traj_lambdas, *d_xs;
    gpuErrchk(cudaMalloc(&d_traj, h_traj.size()*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_traj, h_traj.data(), h_traj.size()*sizeof(float), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc(&d_xs, state_size*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_xs, h_traj.data(), state_size*sizeof(float), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc(&d_traj_lambdas, h_lambdas.size() *sizeof(float)));
    gpuErrchk(cudaMemcpy(d_traj_lambdas, h_lambdas.data(), h_lambdas.size()*sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "traj steps: " << traj2d.size() << std::endl;
    track<float>(state_size, control_size, knot_points, traj2d.size(), .01, d_traj, d_traj_lambdas, d_xs);
    gpuErrchk(cudaPeekAtLastError());
   


    gpuErrchk(cudaFree(d_traj));
    gpuErrchk(cudaFree(d_xs));
    gpuErrchk(cudaFree(d_traj_lambdas))

    return 0;
}