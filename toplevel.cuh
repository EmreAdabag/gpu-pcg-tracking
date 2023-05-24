#pragma once

#include <cstdint>

#include "schur.cuh"




template <typename T>
void interpolate_traj(uint32_t knot_points, T *d_traj){

}



template <typename T>
void sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_traj, T *d_lambda, T *d_xu, void *d_dynMem_const){
    
    const uint32_t merit_smem_size = get_merit_smem_size<T>();


    cudaStream_t streams[4];
    for(int str = 0; str < 4; str++){
        cudaStreamCreate(&streams[str]);
    }


    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(1); }

    float h_merit_initial;
    float delta_merit_iter = 0;
    float delta_merit_total = 0;
    float h_merit_news[8];
    unsigned line_search_step = 0;
    unsigned sqp_iters;

    T *d_merit_initial, *d_merit_news, *d_merit_temp,
          *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
          *d_S, *d_Pinv, *d_gamma,
          *d_dz;

    float alphafinal, min_merit;
    
    float drho = 1.0;
    float rho = 1e-3;
    float rho_factor = 4;
    float rho_max = 1e3;
    float rho_min = 1e-3;

    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp, *d_r_tilde, *d_upsilon;
    unsigned *d_pcgiters;
    


    /*  STREAM 3    */
    gpuErrchk(cudaMallocAsync(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_g,        KKT_g_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_c,        KKT_c_SIZE_BYTES, streams[3]));
    d_Ginv_dense = d_G_dense;

    /*  STREAM 2    */
    gpuErrchk(cudaMallocAsync(&d_S, 3*STATES_SQ*knot_points*sizeof(T), streams[2]));
    gpuErrchk(cudaMallocAsync(&d_Pinv, 3*STATES_SQ*knot_points*sizeof(T), streams[2]));
    gpuErrchk(cudaMallocAsync(&d_gamma, state_size*knot_points*sizeof(T), streams[2]));

    
    /*   STREAM 1   */
    gpuErrchk(cudaMallocAsync(&d_dz,       DZ_SIZE_BYTES, streams[1]));
    gpuErrchk(cudaMallocAsync(&d_xs, state_size*sizeof(T), streams[1]));
    gpuErrchk(cudaMemcpyAsync(d_xs, d_xu, state_size*sizeof(T), cudaMemcpyDeviceToDevice, streams[1]));
    gpuErrchk(cudaMallocAsync(&d_merit_news, config->max_alphas*sizeof(T), streams[1]));     
    gpuErrchk(cudaMallocAsync(&d_merit_temp, config->max_alphas*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_p, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_v_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_eta_new_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r_tilde, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_upsilon, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_pcgiters, sizeof(int), streams[1]));


    /*   STREAM 0   */
    gpuErrchk(cudaMallocAsync(&d_merit_initial, sizeof(T), streams[0]));
    gpuErrchk(cudaMemsetAsync(d_merit_initial, 0, sizeof(T), streams[0]));

    


    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<<<knot_points, 64, merit_smem_size, streams[0]>>>(
        d_xu, 
        d_traj, 
        10, 
        timestep, 
        d_dynMem_const, 
        d_merit_initial
    );
    

    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost, streams[0]));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    //
    //      SQP LOOP
    //
    // for(sqp_iters = 0; sqp_iters < config->sqp_max_iters; sqp_iters++){

    //     gato_form_kkt<_INTEGRATOR_TYPE, _ANGLE_WRAP><<<config->kkt_grid, config->kkt_block, get_kkt_smem_size()>>>(
    //         d_xu,
    //         d_xg,
    //         d_xs,
    //         d_G_dense, 
    //         d_C_dense, 
    //         d_g, 
    //         d_c,
    //         d_dynMem_const,
    //         config->dt
    //     );
        
        
    //     ///TODO: if line search fails don't have to reform KKT, just reform schur with updated rho
    //     // form schur
    //     form_schur(d_G_dense, d_C_dense, d_g, d_c,
    //                d_S, d_Pinv, d_gamma,
    //                rho,
    //                config
    //     );

    //     // pcg
    //     int pcg_iters;
    //     solve_pcg<float>(d_S, d_Pinv, d_gamma, d_lambda, d_r, d_p, d_v_temp, d_eta_new_temp, d_r_tilde, d_upsilon, d_pcgiters, config);
    //     gpuErrchk(cudaMemcpy(&pcg_iters, d_pcgiters, sizeof(int), cudaMemcpyDeviceToHost));

    //     // recover dz
    //     compute_dz(
    //         d_Ginv_dense, 
    //         d_C_dense, 
    //         d_g, 
    //         d_lambda, 
    //         d_dz,
    //         config
    //     );

        

    //     // line search
    //     parallel_line_search(d_xu, d_xg, d_dynMem_const, d_dz, d_merit_news, d_merit_temp, config);
    //     cudaMemcpy(h_merit_news, d_merit_news, config->max_alphas*sizeof(T), cudaMemcpyDeviceToHost);


    //     line_search_step = 0;
    //     min_merit = h_merit_initial;
    //     for(int i = 0; i < config->max_alphas; i++){
    //         ///TODO: reduction ratio
    //         if(h_merit_news[i] < min_merit){
    //             min_merit = h_merit_news[i];
    //             line_search_step = i;
    //         }
    //     }
        
    //     if(min_merit == h_merit_initial){
    //         // line search failure
    //         drho = max(drho*rho_factor, rho_factor);
    //         rho = max(rho*drho, rho_min);
    //         if(rho > rho_max){ /* std::cout << "exiting SQP for max rho\n"; */ break; }
    //         continue;
    //     }

    //     alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

    //     drho = min(drho/rho_factor, 1/rho_factor);
    //     rho = max(rho*drho, rho_min);
        
    //     // add the update
    //     cublasSaxpy(
    //         handle, 
    //         DZ_SIZE_BYTES / sizeof(T),
    //         &alphafinal,
    //         d_dz, 1,
    //         d_xu, 1
    //     );

    //     delta_merit_iter = h_merit_initial - min_merit;
    //     delta_merit_total += delta_merit_iter;
        
    //     if( delta_merit_iter < config->sqp_exit_tolerance){
    //         // std::cout << "exiting SQP for exit tolerance\n";
    //         break;
    //     }

    //     h_merit_initial = min_merit;
    
    // }

    cublasDestroy(handle);

    for(int st=0; st < 4; st++){
        gpuErrchk(cudaStreamDestroy(streams[st]));
    }


    gpuErrchk(cudaFreeAsync(d_r, 0));
    gpuErrchk(cudaFreeAsync(d_p, 0));
    gpuErrchk(cudaFreeAsync(d_v_temp, 0));
    gpuErrchk(cudaFreeAsync(d_eta_new_temp, 0));
    gpuErrchk(cudaFreeAsync(d_r_tilde, 0));
    gpuErrchk(cudaFreeAsync(d_upsilon, 0));    
    gpuErrchk(cudaFreeAsync(d_pcgiters, 0));


    gpuErrchk(cudaFreeAsync(d_merit_initial, 0));
    gpuErrchk(cudaFreeAsync(d_merit_news, 0));
    gpuErrchk(cudaFreeAsync(d_merit_temp, 0));
    gpuErrchk(cudaFreeAsync(d_G_dense, 0));
    gpuErrchk(cudaFreeAsync(d_C_dense, 0));
    gpuErrchk(cudaFreeAsync(d_g, 0));
    gpuErrchk(cudaFreeAsync(d_c, 0));
    gpuErrchk(cudaFreeAsync(d_S, 0));
    gpuErrchk(cudaFreeAsync(d_Pinv, 0));
    gpuErrchk(cudaFreeAsync(d_gamma, 0));
    gpuErrchk(cudaFreeAsync(d_dz, 0));
}


template <typename T>
void track(uint32_t state_size, 
           uint32_t control_size, 
           uint32_t knot_points,
           float timestep,
           T *d_traj,
           T *d_lambda,
           T *d_xu)
{

    sqp_solve<T>(state_size, control_size, knot_points, d_traj, d_lambda, d_xu);

}