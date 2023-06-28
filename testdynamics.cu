#include <iostream>
#include "twoiiwa_plant.cuh"









__global__
void test_dynamics_kernel(float *d_xu, grid::robotModel<float> *dynmem){

    extern __shared__ float s_mem[];
    __shared__ int s_topology_helpers[71];

    float *s_q = s_mem; 					
    float *s_qd = s_q + STATE_SIZE/2; 				
    float *s_u = s_qd + STATE_SIZE/2;
    float *s_qdd = s_u + STATE_SIZE/2;
    float *s_XITemp = s_qdd + STATE_SIZE/2; 					


    for (int i = threadIdx.x; i < 42; i+= blockDim.x){
        s_mem[i] = d_xu[i];
    }

    __syncthreads();
    
    if(threadIdx.x==0){
        for(int i = 0; i < 42; i++){
            if(i%(STATE_SIZE/2) == 0){ printf("\n");}
            printf("%f ", s_mem[i]);
        }
        printf("\n");
    }
    __syncthreads();

    float *s_XImats = s_XITemp; float *s_temp = &s_XITemp[1008];
    grid::load_update_XImats_helpers<float>(s_XImats, s_q, s_topology_helpers, dynmem, s_temp);
    __syncthreads();

    grid::forward_dynamics_inner<float>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, static_cast<float>(9.81));
    __syncthreads();
    
    for(int i = threadIdx.x; i < STATE_SIZE / 2; i+=blockDim.x){
        d_xu[3*(STATE_SIZE/2) + i] = s_qdd[i];
    }


}









int main(){

    float h_xu[] = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f,0.9f};
    
    std::cout << STATE_SIZE << " " << STATE_SIZE/2 << std::endl;
    std::cout << sizeof(h_xu) / sizeof(h_xu[0]) << std::endl;
    for(int i = 0; i < STATE_SIZE; i++){
        std::cout << h_xu[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < STATE_SIZE/2; i++){
        std::cout << h_xu[STATE_SIZE+i] << " ";
    }
    std::cout << std::endl;



    float *d_xu;
    gpuErrchk(cudaMalloc(&d_xu, (STATE_SIZE+(STATE_SIZE/2)+STATE_SIZE)*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_xu, h_xu, (STATE_SIZE+(STATE_SIZE/2))*sizeof(float), cudaMemcpyHostToDevice));
    float *d_qdd = d_xu + 3 * (STATE_SIZE/2);


    // void *dynmem = gato_plant::initializeDynamicsConstMem<float>();
    grid::robotModel<float> *robomodel = grid::init_robotModel<float>();

    test_dynamics_kernel<<<1, 1, sizeof(float)*gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared()>>>(d_xu, robomodel);
    // grid::forward_dynamics_kernel<float><<<1,grid::SUGGESTED_THREADS, grid::FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(float)>>>(d_qdd, d_xu, 3*(STATE_SIZE/2), robomodel, 9.81, 1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    
    
    gpuErrchk(cudaMemcpy(h_xu, d_qdd,(STATE_SIZE/2)*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i <(STATE_SIZE/2); i++){
        if (i==7){std::cout << std::endl;}
        std::cout << h_xu[i] << " ";
    }
    std::cout << std::endl;

    gpuErrchk(cudaFree(d_xu));

    return 0;
}