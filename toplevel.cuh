#pragma once

#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include "schur.cuh"
#include "merit.cuh"
#include "gpu_pcg.cuh"

template <typename T>
void interpolate_traj(uint32_t knot_points, T *d_traj){

}



template <typename T>
void sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_traj, T *d_lambda, T *d_xu, void *d_dynMem_const){
    
    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES (((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES ((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       (((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES         ((states_p_controls)*sizeof(T));     
    const uint32_t LAMBDA_SIZE_BYTES        ((states_p_controls*sizeof(T)));
    const uint32_t DZ_SIZE_BYTES            ((states_s_controls*knot_points-control_size)*sizeof(T));

    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);

    gpuErrchk(cudaPeekAtLastError());
    cudaStream_t streams[4];
    for(int str = 0; str < 4; str++){
        cudaStreamCreate(&streams[str]);
    }

    gpuErrchk(cudaPeekAtLastError());

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(1); }
    gpuErrchk(cudaPeekAtLastError());

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
    
    gpuErrchk(cudaPeekAtLastError());


    /*  STREAM 3    */
    gpuErrchk(cudaMallocAsync(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_g,        KKT_g_SIZE_BYTES, streams[3]));
    gpuErrchk(cudaMallocAsync(&d_c,        KKT_c_SIZE_BYTES, streams[3]));
    d_Ginv_dense = d_G_dense;

    /*  STREAM 2    */
    gpuErrchk(cudaMallocAsync(&d_S, 3*states_sq*knot_points*sizeof(T), streams[2]));
    gpuErrchk(cudaMallocAsync(&d_Pinv, 3*states_sq*knot_points*sizeof(T), streams[2]));
    gpuErrchk(cudaMallocAsync(&d_gamma, state_size*knot_points*sizeof(T), streams[2]));
    gpuErrchk(cudaPeekAtLastError());

    
    /*   STREAM 1   */
    gpuErrchk(cudaMallocAsync(&d_dz,       DZ_SIZE_BYTES, streams[1]));
    gpuErrchk(cudaMallocAsync(&d_merit_news, 8*sizeof(T), streams[1]));     
    gpuErrchk(cudaMallocAsync(&d_merit_temp, 8*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_p, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_v_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_eta_new_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r_tilde, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_upsilon, state_size*knot_points*sizeof(T), streams[1]));


    /*   STREAM 0   */
    gpuErrchk(cudaMallocAsync(&d_merit_initial, sizeof(T), streams[0]));
    gpuErrchk(cudaMemsetAsync(d_merit_initial, 0, sizeof(T), streams[0]));
    gpuErrchk(cudaPeekAtLastError());

    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<float><<<knot_points, 64, merit_smem_size*5>>>(
        state_size, control_size, knot_points,
        d_xu, 
        d_traj, 
        static_cast<T>(10), 
        timestep, 
        d_dynMem_const, 
        d_merit_initial
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost, streams[0]));

    
    //
    //      SQP LOOP
    //
    for(sqp_iters = 0; sqp_iters < 5; sqp_iters++){

        std::cout << "xu\n";
        T h_xu_copy[56];
        gpuErrchk(cudaMemcpy(h_xu_copy, d_xu, 56*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 21; j++){
                if(i==2 && j > 13){ continue; }
                std::cout << h_xu_copy[21 * i + j] << " ";
            }
            std::cout << std::endl;
        }

        gato_form_kkt<float><<<knot_points, 64, oldschur::get_kkt_smem_size<T>(state_size, control_size)>>>(
            state_size,
            control_size,
            knot_points,
            d_G_dense, 
            d_C_dense, 
            d_g, 
            d_c,
            d_dynMem_const,
            timestep,
            d_xu,
            d_traj
        );
        gpuErrchk(cudaPeekAtLastError());
        

        // form schur
        form_schur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho);
        gpuErrchk(cudaPeekAtLastError());

        // pcg
        pcg_config config;
        uint32_t pcg_iters = solvePCG(state_size, knot_points, d_S, d_Pinv, d_gamma, d_lambda, &config);
        gpuErrchk(cudaPeekAtLastError());

        // // recover dz
        compute_dz(
            state_size,
            control_size,
            knot_points,
            d_Ginv_dense, 
            d_C_dense, 
            d_g, 
            d_lambda, 
            d_dz
        );

        // line search
        parallel_line_search(state_size, control_size, knot_points, d_xu, d_traj, d_dynMem_const, d_dz, timestep, d_merit_news, d_merit_temp);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);

        line_search_step = 0;
        min_merit = h_merit_initial;
        for(int i = 0; i < 8; i++){
            ///TODO: reduction ratio
            if(h_merit_news[i] < min_merit){
                min_merit = h_merit_news[i];
                line_search_step = i;
            }
        }
        
        if(min_merit == h_merit_initial){
            // line search failure
            drho = max(drho*rho_factor, rho_factor);
            rho = max(rho*drho, rho_min);
            if(rho > rho_max){ /* std::cout << "exiting SQP for max rho\n"; */ break; }
            continue;
        }

        alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

        drho = min(drho/rho_factor, 1/rho_factor);
        rho = max(rho*drho, rho_min);
        
        // add the update
        cublasSaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        
        if( delta_merit_iter < 1e-6){
            // std::cout << "exiting SQP for exit tolerance\n";
            break;
        }

        h_merit_initial = min_merit;
    
    }

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
void track(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_xu, T *d_traj, T *d_lambda){

    void *d_dynmem = gato_plant::initializeDynamicsConstMem<float>();
    gpuErrchk(cudaPeekAtLastError());

    sqpSolve<float>(state_size, control_size, knot_points, .1, d_traj, d_lambda, d_xu, d_dynmem);    

    gato_plant::freeDynamicsConstMem<float>(d_dynmem);
}


template <typename T>
void interpolate_knotpoints_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, float time_mod_timestep, T *d_xs, T *d_xu, T *d_xu_temp){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t grid_dim = gridDim.x;
    const uint32_t states_s_controls = state_size + control_size;
    float new_timestep = (timestep * knot_points - time_mod_timestep) / knot_points;    // redistributed knot points for new time-to-goal

    float new_time, time_diff;
    T prev, next, diff;

    for(uint32_t knot = block_id; knot < knot_points; knot += grid_dim){

        new_time = knot * new_timestep;
        uint32_t knot_before_new_time = static_cast<uint32_t>(floorf(new_time / timestep));
        float time_diff = new_time - timestep * knot_before_new_time;

        for (uint32_t ind = thread_id; ind < states_s_controls; ind += block_dim){
            
            prev = d_xu[knot_before_new_time*states_s_controls+ind];
            next = d_xu[(knot_before_new_time+1)*states_s_controls+ind];
            diff = next - prev;

            d_xu_temp[block_id * (states_s_controls) + ind] = prev + (time_diff/timestep) * diff;
        }
    }
}

template <typename T>
float interpolate_knotpoints(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, float time_mod_timestep, T *d_xs, T *d_xu){

    float new_timestep = (timestep * knot_points - time_mod_timestep) / knot_points;    // redistributed knot points for new time-to-goal
    T d_xu_temp;
    gpuErrchk(cudaMalloc(&d_xu_temp, (state_size+control_size)*knot_points*sizeof(T)));

    interpolate_knotpoints_kernel<T><<<knot_points, 64>>>(state_size, control_size, knot_points, timestep, time_mod_timestep, d_xs, d_xu, d_xu_temp);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(d_xu, d_xu_temp, (state_size+control_size)*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaPeekAtLastError());

    return new_timestep;
}