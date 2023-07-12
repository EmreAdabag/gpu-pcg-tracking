#include <iostream>


#include "cooperative_groups.h"
#include "gpuassert.cuh"
#include "iiwa_eepos_grid.cuh"





namespace cgrps = cooperative_groups;






__global__
void test_kernel(float *d_xu, float *d_temp, grid::robotModel<float> *d_robotModel){
    const int state_size = 14;

    // truth
    // gato_plant::forwardDynamicsAndGradient<float>(d_temp, &d_temp[state_size/2], d_xu, &d_xu[state_size/2], &d_xu[state_size],  &d_temp[state_size], (void *)d_robotModel, cgrps::this_thread_block());


    
    float *s_q = d_xu;
    float *s_qd = &d_xu[state_size/2];
    float *s_u = &d_xu[state_size];
    float *s_dqdd = d_temp;
    float *s_qdd = &d_temp[state_size/2];

    float *s_XImats = &d_temp[state_size]; float *s_vaf = &d_temp[504+state_size]; float *s_dc_du = &s_vaf[126]; float *s_Minv = &s_dc_du[98]; float *s_temp = &s_Minv[49];
    grid::load_update_XImats_helpers<float>(s_XImats, s_q, d_robotModel, s_temp); __syncthreads();
    //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
    grid::direct_minv_inner<float>(s_Minv, s_q, s_XImats, s_temp); __syncthreads();
    float *s_c = s_temp;
    grid::inverse_dynamics_inner<float>(s_c, s_vaf, s_q, s_qd, s_XImats, &s_temp[7], 0); __syncthreads();
    grid::forward_dynamics_finish<float>(s_qdd, s_u, s_c, s_Minv); __syncthreads();
    grid::inverse_dynamics_inner_vaf<float>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, 0); __syncthreads();
    grid::inverse_dynamics_gradient_inner<float>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, 0); __syncthreads();
    for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
        int row = ind % 7; int dc_col_offset = ind - row;
        // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
        float val = static_cast<float>(0);
        for(int col = 0; col < 7; col++) {
            int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
            val += s_Minv[index] * s_dc_du[dc_col_offset + col];
        }
        s_dqdd[ind] = -val;
        if (1 && ind < 49){
            int col = ind / 7; int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
            s_dqdd[ind + 98] = s_Minv[index];
        }
    }
}




int main(){

    const int state_size = 14;
    const int control_size = 7;


    float h_xu[] = {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.9,0.9,0.9,0.9,0.9,0.9};
    float *d_xu, *d_temp;
    gpuErrchk(cudaMalloc(&d_xu, (state_size+control_size)*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_xu, h_xu, (state_size+control_size)*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_temp, 10000*sizeof(float)));


    grid::robotModel<float> *d_robotModel = grid::init_robotModel<float>();

    test_kernel<<<1,128>>>(d_xu, d_temp, d_robotModel);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    float h_temp[state_size];
    gpuErrchk(cudaMemcpy(h_temp, d_temp, state_size*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < state_size; i++){
        std::cout << h_temp[i] << " ";
    }
    std::cout << std::endl;

    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_temp));

    return 0;
}