#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include "iiwa_eepos_grid.cuh"
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


void write_device_to_file(float* d_matrix, int rows, int cols, const char* filename, int filesuffix = 0) {
    
    char fname[100];
    snprintf(fname, sizeof(fname), "%s%d.traj", filename, filesuffix);
    
    // Allocate host memory for the matrix
    float* h_matrix = new float[rows * cols];

    // Copy the data from the device to the host memory
    size_t pitch = cols * sizeof(float);
    cudaMemcpy2D(h_matrix, pitch, d_matrix, pitch, pitch, rows, cudaMemcpyDeviceToHost);

    // Write the data to a file in column-major order
    std::ofstream outfile(fname);
    if (outfile.is_open()) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                outfile << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_matrix[col * rows + row] << (col==cols-1 ? "" : ",");
            }
            outfile << std::endl;
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << fname << std::endl;
    }

    // Deallocate host memory
    delete[] h_matrix;
}


int main(){

    const uint32_t state_size = grid::NUM_JOINTS*2;
    const uint32_t control_size = grid::NUM_JOINTS;

    char traj_file_name[100];

    pcg_t *d_traj, *d_eePos;

    snprintf(traj_file_name, sizeof(traj_file_name), "testfiles/0_0_traj.csv");
    std::vector<std::vector<pcg_t>> traj2d = readCSVToVecVec<pcg_t>(traj_file_name);


    std::vector<pcg_t> h_traj;
    for (const auto& vec : traj2d) {
        h_traj.insert(h_traj.end(), vec.begin(), vec.end());
    }

    gpuErrchk(cudaMalloc(&d_traj, h_traj.size()*sizeof(pcg_t)));
    gpuErrchk(cudaMemcpy(d_traj, h_traj.data(), h_traj.size()*sizeof(pcg_t), cudaMemcpyHostToDevice));    
    gpuErrchk(cudaMalloc(&d_eePos, 6*sizeof(pcg_t)));
    
    std::cout << traj2d.size() << std::endl;
    const grid::robotModel<float> *d_robotModel = grid::init_robotModel<float>();

    float h_eePos[6];

    for(int i = 0; i < (int) traj2d.size(); i++){
        grid::end_effector_positions_kernel<float><<<1,128>>>(d_eePos, &d_traj[i*(state_size+control_size)], grid::NUM_JOINTS, d_robotModel, 1);
        gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < 6; j++){
            std::cout << h_eePos[j] << (j==5 ? "\n" : ",");
        }
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

            
    // grid::free_robotModel<float>(d_robotModel);

    gpuErrchk(cudaFree(d_traj));
    gpuErrchk(cudaFree(d_eePos));

    return 0;
}