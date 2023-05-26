#include <iostream>
#include "toplevel.cuh"

#include "iiwa_plant.cuh"







/*

pre-compute trajectory

loop:
    shift trajectory with new state vector
    solve from current state vector
    simulate for solve duration


-------------------------------------------

realtime

pre-compute trajectory

concurrently:
    track traj:
        shift traj with new state vector
        solve QP
    move:
        follow traj
        on traj update: 

*/





int main(){
    const uint32_t state_size = 14;
    const uint32_t control_size = 7;
    const uint32_t knot_points = 3;

    float h_traj[56] = {-0.7072,    0.7312,   -1.4039,    0.2646,    1.4471,    1.575 ,    0.5363,    0.215 , 0.0392,   -4.8188,    2.9574,    1.3779,    0.7255,    1.0542, -112.371 ,  162.7095, -21.7513,  -35.1634,   -0.0134,    3.246 ,    0.1966, -0.2716,    0.4929,   -0.9196,    0.0426,    0.7934,    1.5672,    0.0077,   -1.5472, 0.3976,    0.3557,    1.5639,    0.5326,    0.9623,    1.0188, -112.371 ,  162.7095, -21.7513,  -35.1634,   -0.0134,    3.246 ,    0.1966,  -0.4264,   0.5326,  -0.884 ,   0.199 ,   0.8466,   1.6634,   0.1096, -28.0848,   9.7976, -42.5967,  33.0413,  57.6764,  23.3162,  18.1502};
    float h_xu[56] = {-0.63498, 0.76645, -1.3094, 0.29421, 1.52768, 1.61919, 0.54144, 0.30833, 0.08429, -4.7648, 3.02619, 1.45485, 0.72795, 1.09755, -112.27, 162.742, -21.678, -35.113, 0.00713, 3.26368, 0.19843, -0.2607, 0.58379, -0.8371, 0.10687, 0.86277, 1.64689, 0.07752, -1.5099, 0.42137, 0.42680, 1.59879, 0.57870, 1.05100, 1.03166, -112.33, 162.733, -21.728, -35.125, 0.03702, 3.30164, 0.28213, -0.3508, 0.59656, -0.8614, 0.25919, 0.92122, 1.70438, 0.18908, -28.054, 9.85011, -42.577, 33.0433, 57.7261, 23.4036, 18.1905};
    float h_lambda[42] = {-5.0946,    61.5739,  -191.5559,   361.1493,     5.8803,    -7.007 ,     0.7601,    -2.05  ,     2.5814,   -17.9229,    34.9526,     0.6137,    -0.6097,     0.0788,    -5.0341,    79.4689,  -172.678 ,   333.9097,     6.2139,    -5.4741,     0.7087,    -0.6474,    -5.1649,     0.3811,    -0.1262,    -0.0873,    -0.0163,    -0.    ,    -4.9748,   142.6363,    21.6536,   -46.603 ,     2.3691,     1.1642,     0.6466,    -1.1772,   -18.8751,    -0.1228,     3.4542,    -0.2771,    -0.0852,     0.0039};

    std::cout << "INITIAL XU\n";
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 21; j++){
            if(i==2 && j > 13){ continue; }
            std::cout << h_xu[21 * i + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "INITIAL TRAJ\n";
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 21; j++){
            if(i==2 && j > 13){ continue; }
            std::cout << h_traj[21 * i + j] << " ";
        }
        std::cout << std::endl;
    }

    float *d_traj, *d_lambda, *d_xu;
    gpuErrchk(cudaMalloc(&d_traj, 56 * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_traj, h_traj, 56*sizeof(float), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc(&d_xu, 56*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_xu, h_xu, 56*sizeof(float), cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc(&d_lambda, 42*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, 42*sizeof(float), cudaMemcpyHostToDevice));
    

    track<float>(state_size, control_size, knot_points, .1, d_xu, d_traj, d_lambda);
    gpuErrchk(cudaPeekAtLastError());
   
    std::cout << "FINAL xu\n";
    float h_xu_copy[56];
    gpuErrchk(cudaMemcpy(h_xu_copy, d_xu, 56*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 21; j++){
            if(i==2 && j > 13){ continue; }
            std::cout << h_xu_copy[21 * i + j] << " ";
        }
        std::cout << std::endl;
    }


    gpuErrchk(cudaFree(d_traj));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_lambda));

    return 0;
}