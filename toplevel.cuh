#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <cublas_v2.h>
#include <math.h>
#include "schur.cuh"
#include "merit.cuh"
#include "gpu_pcg.cuh"
#include "integrator.cuh"

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
    const uint32_t KKT_c_SIZE_BYTES         ((state_size*knot_points)*sizeof(T));     
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

        // std::cout << "xu\n";
        // T h_xu_copy[56];
        // gpuErrchk(cudaMemcpy(h_xu_copy, d_xu, 56*sizeof(float), cudaMemcpyDeviceToHost));
        // for(int i = 0; i < 3; i++){
        //     for(int j = 0; j < 21; j++){
        //         if(i==2 && j > 13){ continue; }
        //         std::cout << h_xu_copy[21 * i + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

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
        gpuErrchk(cudaDeviceSynchronize());

        // form schur
        form_schur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // pcg
        pcg_config config;
        uint32_t pcg_iters = solvePCG(state_size, knot_points, d_S, d_Pinv, d_gamma, d_lambda, &config);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

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
        gpuErrchk(cudaDeviceSynchronize());

        // line search
        parallel_line_search(state_size, control_size, knot_points, d_xu, d_traj, d_dynMem_const, d_dz, timestep, d_merit_news, d_merit_temp);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
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
        gpuErrchk(cudaDeviceSynchronize());


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
__global__
void interpolate_knotpoints_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, float traj_timestep, float time_since_update, T *d_traj_lambdas, T *d_lambda, T *d_xu, T *d_xu_temp, T *d_traj){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t grid_dim = gridDim.x;
    const uint32_t states_s_controls = state_size + control_size;
    float new_timestep = (traj_timestep * traj_steps - time_since_update) / knot_points;    // redistributed knot points for new time-to-goal

    float new_time, time_diff;
    T prev, next, diff;
    uint32_t prev_step, threads_needed;

    for(uint32_t knot = block_id; knot < knot_points; knot += grid_dim){
        ///TODO: handle case where solve takes longer than traj
        new_time = knot * new_timestep + time_since_update;
        ///TODO: make sure on first iter floating point error isn't making this mess up
        prev_step = static_cast<uint32_t>(floorf(new_time / traj_timestep));
        time_diff = new_time - traj_timestep * prev_step;
        threads_needed = state_size + (knot < knot_points-1)*control_size;

        for (uint32_t ind = thread_id; ind < threads_needed; ind += block_dim){
            
            if(ind < state_size){
                d_lambda[knot*state_size + ind] = d_traj_lambdas[prev_step*state_size+ind];
            }

            // if (knot==0){
            //     d_xu_temp[ind] = d_xu[ind];    
            // }
            // else{
                prev = d_traj[prev_step*states_s_controls+ind];
                next = d_traj[(prev_step+1)*states_s_controls+ind];
                diff = next - prev;

                d_xu_temp[knot * (states_s_controls) + ind] = prev + (time_diff/traj_timestep) * diff;
            // }
        }
    }
}

template <typename T>
float interpolate_knotpoints(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, const float traj_timestep, float time_since_update, T *d_traj_lambdas, T *d_lambda, T *d_xu_goal,T *d_traj){

    float new_timestep = (traj_timestep * traj_steps - time_since_update) / knot_points;    // redistributed knot points for new time-to-goal
    T *d_xu_temp;
    gpuErrchk(cudaMalloc(&d_xu_temp, ((state_size+control_size)*knot_points-control_size)*sizeof(T)));

    interpolate_knotpoints_kernel<T><<<knot_points, 64>>>(state_size, control_size, knot_points, traj_steps, traj_timestep, time_since_update, d_traj_lambdas, d_lambda, d_xu_goal, d_xu_temp, d_traj);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(d_xu_goal, d_xu_temp, ((state_size+control_size)*knot_points-control_size)*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaPeekAtLastError());

    return new_timestep;
}

template <typename T>
void track(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, float traj_timestep, T *d_traj, T *d_traj_lambdas, T *d_xs){

    const size_t traj_len = (state_size+control_size)*knot_points-control_size;
    float cur_timestep = traj_timestep;
    float old_timestep = traj_timestep;
    double time_since_start, iter_solve_time, prev_iter_solve_time;
    time_since_start = 0;
    prev_iter_solve_time = 0;
    std::chrono::time_point<std::chrono::steady_clock> iter_start, iter_end;
    std::chrono::duration<double> iter_diff;
    std::vector<std::vector<T>> tracking_path;

    T *d_lambda, *d_xu_goal, *d_xu, *d_xu_old;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_goal, traj_len*sizeof(T)));

    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    T h_xs[state_size];
    T h_xg[state_size];
    gpuErrchk(cudaMemcpy(h_xg, &d_traj[(traj_steps-1)*(state_size+control_size)], state_size*sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "goal state" << std::endl;
    for (int i = 0; i < state_size; i++){
        std::cout << h_xg[i] << " ";
    }
    std::cout <<"\n\n\n";

    int iter = 0;

    while(1){

        // store xs in final trajectory vector
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));

        // exit if low error
        T tot_err = 0;
        for (int i = 0; i < state_size; i++){
            std::cout << h_xs[i] << " ";
            tot_err += abs(h_xs[i] - h_xg[i]);
        }
        std::cout << std::endl;
        if(tot_err < .1){ std::cout << "exiting for proximity\n"; }


        iter_start = std::chrono::steady_clock::now();
        
        // interpolate knot points on pre-computed trajectory, fill xu_goal
        cur_timestep = interpolate_knotpoints<T>(state_size, 
                                                 control_size, 
                                                 knot_points, 
                                                 traj_steps, 
                                                 traj_timestep, 
                                                 time_since_start, 
                                                 d_traj_lambdas, 
                                                 d_lambda, 
                                                 d_xu_goal, 
                                                 d_traj);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpy(d_xu, d_xu_goal, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));

        if (iter==0){
            gpuErrchk(cudaMemcpy(d_xu_old, d_xu_goal, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        
        sqpSolve<T>(state_size, control_size, knot_points, cur_timestep, d_xu_goal, d_lambda, d_xu, d_dynmem);

        iter_end = std::chrono::steady_clock::now();


        iter_diff = iter_end - iter_start;
        // iter_solve_time = iter_diff.count();
        iter_solve_time = .01; // EMRE 
        time_since_start += iter_solve_time;

        
        // std::cout << "solve took " << iter_solve_time << " seconds\n";
        // simulate for iter_solve_time seconds using old traj
        integrator_shift<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, old_timestep, prev_iter_solve_time, iter_solve_time);


        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
        old_timestep = cur_timestep;
        prev_iter_solve_time = iter_solve_time;

        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));

        // break;
        iter++;

    }

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_xu_goal));
    gpuErrchk(cudaFree(d_xu_old));
}


__device__ __forceinline__
void gato_fmemset(float *dst, float val, unsigned num_floats){
    for(int i = GATO_THREAD_ID; i < num_floats; i+=GATO_THREADS_PER_BLOCK){
        dst[i] = val;
    }
}

__global__
void gato_fmemset_kernel(float *dst, float val, unsigned num_floats){
    gato_fmemset(dst, val, num_floats);
}


        // std::cout << "goal\n";
        // gpuErrchk(cudaMemcpy(h_xs, d_xu_goal, 14*sizeof(float), cudaMemcpyDeviceToHost));
        // for(int i = 0; i < 14; i++){
        //     std::cout << h_xs[i] << " ";
        // }
        // std::cout << std::endl;

// template <typename T>
// std::vector<std::vector<T>> plan_traj(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *h_xs, T *h_xg, ){
    
//     std::vector<std::vector<T>> final_traj;

//     const uint32_t states_s_controls = state_size + control_size;

//     const uint32_t xu_len = (states_s_controls*KNOT_POINTS-CONTROL_SIZE);
//     const uint32_t lambda_len = (state_size * knot_points);
//     const uint32_t xg_len = state_size;

//     // Allocate space for host xu and goal
//     T h_xu[states_s_controls*KNOT_POINTS-CONTROL_SIZE];
//     T h_xg[STATE_SIZE];
//     T h_xs[STATE_SIZE];

//     // Alocate space for them d_xu, d_xg, d_lambda, d_dynMem, config
//     T *d_xu;
//     T *d_xg;
//     T *d_lambda;
    
//     gpuErrchk(cudaMalloc(&d_xu, xu_len * sizeof(T)));
//     gpuErrchk(cudaMalloc(&d_xg, xg_len * sizeof(T)));
//     gpuErrchk(cudaMalloc(&d_lambda, lambda_len * sizeof(T)));
//     gato_fmemset_kernel<<<1, 64>>>(d_lambda, 0.0, lambda_len);

//     for (uint32_t ind = 0; ind < xu_len; ind++){
//         h_xu[ind] = h_xs[ind % states_s_controls];
//     }

//     final_path.push_back(std::vector<T>(h_xs, &h_xs[STATE_SIZE]));

//     gpuErrchk(cudaMemcpy(d_xu, h_xu, xu_size*sizeof(T), cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(d_xg, h_xg, xg_size*sizeof(T), cudaMemcpyHostToDevice));


    
    
//     while(true){

//         // Call SQP with d_xu, d_xg, d_lambda, d_dynMem, config
    
//         SQP_PCG(d_xu, d_xg, d_lambda, d_dynMem_const, input_config, mpc_iters);

//         // Get x, u
//         gpuErrchk(cudaMemcpy(h_xu, d_xu, xu_size, cudaMemcpyDeviceToHost));
        
//         // Integrate and Shift trajectory
//         integrator_shift<T>(h_xs, h_xu, d_dynMem_const, input_config);
        
//         // add to trajectory
//         final_path.push_back(std::vector<T>(h_xs, &h_xs[STATE_SIZE]));
        
//         // calculate error
//         err = error(h_xs, h_xg);
//         delta_err = abs(last_err - err);

//         if(err < input_config->mpc_exit_tolerance){
//             std::cout << "converged in " << mpc_iters << " iterations with final error " << err << std::endl;
//             break;
//         }
        
//         std::cout <<  "iter " <<  mpc_iters << " distance to goal: " << err << "\n";
//         //Copy shifted trajectory back
//     #if DYNAMIC_RHO
//         input_config->rho_init = min(err*DYNAMIC_RHO_CONST, DYNAMIC_RHO_CONST);
//         // if(err < 3.5){
//         //     input_config->rho_init *= 4;
//         // }
//     #endif /* DYNAMIC_RHO*/
       
//         gpuErrchk(cudaMemcpy(d_xu, h_xu, xu_size, cudaMemcpyHostToDevice));
//         last_err = err;
        
//         mpc_iters++;
//         if(mpc_iters >= MAX_MPC_ITERS){
//             std::cout << "breaking for lots of mpc iters\n";
//             break;
//         }
//     }

//     //Free stuff
//     gpuErrchk(cudaFree(d_xu));
//     gpuErrchk(cudaFree(d_xg));
//     gpuErrchk(cudaFree(d_lambda));
//     gato_plant::freeDynamicsConstMem<T>(d_dynMem_const);

//     return final_path;

// }