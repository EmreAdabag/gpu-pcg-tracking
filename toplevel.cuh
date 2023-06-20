#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <time.h>
#include "schur.cuh"
#include "merit.cuh"
#include "gpu_pcg.cuh"
#include "integrator.cuh"
#include "settings.cuh"


void write_device_matrix_to_file(float* d_matrix, int rows, int cols, const char* filename, int filesuffix = 0) {
    
    char fname[100];
    snprintf(fname, sizeof(fname), "%s%d.txt", filename, filesuffix);
    
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
                outfile << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_matrix[col * rows + row] << "\t";
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

template <typename T>
std::vector<int> sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_traj, T *d_lambda, T *d_xu, void *d_dynMem_const){
    
    std::vector<int> thisthing;

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
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(13); }
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
          *d_dz,
          *d_xs;

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
    gpuErrchk(cudaMallocAsync(&d_xs,       state_size*sizeof(T), streams[1]));
    //EMRE should this be passed in instead?
    gpuErrchk(cudaMemcpyAsync(d_xs, d_xu,  state_size*sizeof(T), cudaMemcpyDeviceToDevice, streams[1]));
    gpuErrchk(cudaMallocAsync(&d_merit_news, 8*sizeof(T), streams[1]));     
    gpuErrchk(cudaMallocAsync(&d_merit_temp, 8*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_p, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_v_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_eta_new_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_r_tilde, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_upsilon, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaPeekAtLastError());


    /*   STREAM 0   */
    gpuErrchk(cudaMallocAsync(&d_merit_initial, sizeof(T), streams[0]));
    gpuErrchk(cudaMemsetAsync(d_merit_initial, 0, sizeof(T), streams[0]));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    

    
    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<float><<<knot_points, 64, merit_smem_size*5, streams[0]>>>(
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

    ///EMRE YOU ADDED THIS
    // struct timespec start, end;
    
    
    



    //
    //      SQP LOOP
    //
    for(sqp_iters = 0; sqp_iters < MAX_SQP_ITERS; sqp_iters++){
        // std::cout << "sqp iter: " << sqp_iters << std::endl;

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
            d_traj,
            d_xs,
            d_xu
        );
        gpuErrchk(cudaPeekAtLastError());



        // form schur
        form_schur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho);
        gpuErrchk(cudaPeekAtLastError());
        

    
#if ZERO_LAMBDA
        gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(T)));
#endif


        // clock_gettime(CLOCK_MONOTONIC,&start);

        // pcg
        pcg_config config;
        uint32_t pcg_iters = solvePCG(14, 50, d_S, d_Pinv, d_gamma, d_lambda, &config);
        thisthing.push_back(pcg_iters);

        // clock_gettime(CLOCK_MONOTONIC,&end);
        // double time = time_delta_us_timespec(start,end);

        // std::cout << "pcg time " << time << std::endl;

#if PRINT_PCG_ITERS
        std::cout << "pcg iters: " << pcg_iters << std::endl;
#endif
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
        gpuErrchk(cudaPeekAtLastError());
        
        // write_device_matrix_to_file(d_dz, (control_size+state_size)*knot_points-control_size, 1, "dz", 0);
        // gpuErrchk(cudaPeekAtLastError());


        // line search
        parallel_line_search(state_size, control_size, knot_points, d_xs, d_xu, d_traj, d_dynMem_const, d_dz, timestep, d_merit_news, d_merit_temp);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);


        line_search_step = 0;
        min_merit = h_merit_initial;
#if PRINT_LINE_SEARCH
        std::cout << "merit to beat: " << min_merit << " line search merits:\n";
        for(int i = 0; i < 8; i++){
            std::cout << "[ " << 1.0 / (1 << i) << " ] [ " << h_merit_news[i] << " ]\t";
        }
        std::cout << std::endl;
#endif
        for(int i = 0; i < 8; i++){
            ///TODO: reduction ratio
            if(h_merit_news[i] < min_merit){
                min_merit = h_merit_news[i];
                line_search_step = i;
            }
        }

        gpuErrchk(cudaPeekAtLastError());

        if(min_merit == h_merit_initial){
#if PRINT_LINE_SEARCH
            std::cout << "line search failed\n";
#endif
            // line search failure
            drho = max(drho*rho_factor, rho_factor);
            rho = max(rho*drho, rho_min);
            if(rho > rho_max){ 
                // std::cout << "exiting SQP for max rho\n"; 
                break; 
            }
            continue;
        }

#if PRINT_LINE_SEARCH
        std::cout << "line search accepted\n";
#endif
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
        gpuErrchk(cudaPeekAtLastError());


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
    gpuErrchk(cudaFreeAsync(d_xs, 0));

    return thisthing;
}
/*
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
        prev_step = static_cast<uint32_t>(floorf(new_time / traj_timestep + .000001));
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
__global__
void simple_interpolate_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, float goal_time_offset, float xu_time_offset, T *d_xu_goal, T *d_xu, T *d_xu_temp, uint32_t traj_steps, T *d_traj){

    T prev, next, diff;
    const float epsilon = .00001; // this is to prevent float error from messing up floor
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t grid_dim = gridDim.x;
    const uint32_t states_s_controls = state_size + control_size;
    const float goal_time_offset_mod_timestep = fmodf(goal_time_offset, timestep);
    const float xu_time_offset_mod_timestep = fmodf(xu_time_offset, timestep);
    uint32_t goal_prev_traj_step, xu_prev_traj_step;
    float goal_new_time, xu_new_time;

    for (uint32_t knot = block_id; knot < knot_points; knot += grid_dim){

        const uint32_t threads_needed = state_size + (knot < knot_points-1)*control_size;
        
        goal_new_time = goal_time_offset + knot * timestep;
        goal_prev_traj_step = static_cast<uint32_t>(floorf(goal_new_time / timestep + epsilon));
        xu_new_time = xu_time_offset + knot * timestep;
        xu_prev_traj_step = static_cast<uint32_t>(floorf(xu_new_time / timestep + epsilon));

        for (uint32_t ind = thread_id; ind < threads_needed; ind += block_dim){
             
            if (goal_prev_traj_step > traj_steps - 2){
                d_xu_goal[knot*states_s_controls + ind] = d_traj[(traj_steps-1)*states_s_controls + ind];
            }
            else{
                prev = d_traj[goal_prev_traj_step*states_s_controls+ind];
                next = d_traj[(goal_prev_traj_step+1)*states_s_controls+ind];
                diff = next - prev;

                d_xu_goal[knot*states_s_controls + ind] = prev + (goal_time_offset_mod_timestep/timestep)*diff;
            }

            // EMRE VERYFI  
            if (xu_prev_traj_step > knot_points - 3){
                if (goal_prev_traj_step > traj_steps - 2){
                    d_xu_temp[knot*states_s_controls + ind] = d_traj[(traj_steps-1)*states_s_controls + ind];
                }
                else{
                    prev = d_traj[goal_prev_traj_step*states_s_controls+ind];
                    next = d_traj[(goal_prev_traj_step+1)*states_s_controls+ind];
                    diff = next - prev;

                    d_xu_temp[knot*states_s_controls + ind] = prev + (goal_time_offset_mod_timestep/timestep)*diff;
                }
            }
            else{
                prev = d_xu[xu_prev_traj_step*states_s_controls+ind];
                next = d_xu[(xu_prev_traj_step+1)*states_s_controls+ind];
                diff = next - prev;

                d_xu_temp[knot*states_s_controls+ind] = prev + (xu_time_offset_mod_timestep/timestep)*diff;
            }
        }
    }
}


// given a time offset from the beginning of d_traj, interpolate between d_traj and set d_xu, d_xu_goal
template <typename T>
void simple_interpolate(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, float goal_time_offset, float xu_time_offset, T *d_xu_goal, T *d_xu, T *d_xu_temp, uint32_t traj_steps, T *d_traj){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    simple_interpolate_kernel<T><<<knot_points, 32>>>(state_size, control_size, knot_points, timestep, goal_time_offset, xu_time_offset, d_xu_goal, d_xu, d_xu_temp, traj_steps, d_traj);

    gpuErrchk(cudaMemcpy(d_xu, d_xu_temp, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));

}
*/

__global__
void addnoise(float *thingy, int num, float rand){


    for (int i = threadIdx.x; i < num; i += blockDim.x){
        thingy[i] += rand;
    }
}

template <typename T>
void track(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, float timestep, T *d_traj, T *d_traj_lambdas, T *d_xs){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    const float shift_threshold = SHIFT_THRESHOLD;
    float iter_solve_time = 0;
    float prev_iter_solve_time = 0;
    float time_since_timestep = 0;
    bool shifted = false;
    int traj_offset = 0;
    std::vector<std::vector<T>> tracking_path;
    std::vector<int> pcg_iters_steps;
    std::chrono::time_point<std::chrono::steady_clock> iter_start, iter_end;
    std::chrono::duration<double> iter_diff;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 0.002f);
    std::uniform_real_distribution<float> dis1(-.01f, .01f);
    // std::uniform_real_distribution<float> dis2(-10.0f, 10.0f);

    T *d_lambda, *d_xu_goal, *d_xu, *d_xu_old, *d_xu_temp;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_goal, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_temp, traj_len*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_lambda, d_traj_lambdas, state_size*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_goal, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));

    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    T h_xs[state_size];
    T h_xg[state_size];
    T h_temp[state_size];
    gpuErrchk(cudaMemcpy(h_xg, &d_traj[(traj_steps-1)*(state_size+control_size)], state_size*sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "goal state" << std::endl;
    for (int i = 0; i < state_size; i++){
        std::cout << h_xg[i] << " ";
    }
    std::cout <<"\n\n\n";
    
    gpuErrchk(cudaPeekAtLastError());
    
    for(int iter = 0; iter < 50000; iter++){

        if(iter == 100){
            break;
        }

        
        // store xs in final trajectory vector
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));
        

        if (iter % 100 == 0){
            int sum = std::accumulate(pcg_iters_steps.begin(), pcg_iters_steps.end(), 0);

            // Calculate the average
            double average = static_cast<double>(sum) / pcg_iters_steps.size();

            std::cout << "pcg averaging: " << average << " iters" << std::endl;
        }

        // exit if low error
        T tot_err = 0;
        std::cout << iter << "| ";
        for (int i = 0; i < state_size; i++){
            std::cout << h_xs[i] << " ";
            tot_err += abs(h_xs[i] - h_xg[i]);
        }
        std::cout << std::endl;
        if(tot_err < .1){ 
            std::cout << "close proximity\n"; 

            int sum = std::accumulate(pcg_iters_steps.begin(), pcg_iters_steps.end(), 0);

            // Calculate the average
            double average = static_cast<double>(sum) / pcg_iters_steps.size();

            std::cout << "pcg averaging: " << average << " iters" << std::endl;

            break;
        }
        


        gpuErrchk(cudaPeekAtLastError());
        iter_start = std::chrono::steady_clock::now();

        // solve for better xu
        std::vector<int> pcg_iters = sqpSolve<T>(state_size, control_size, knot_points, timestep, d_xu_goal, d_lambda, d_xu, d_dynmem);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        iter_end = std::chrono::steady_clock::now();

        pcg_iters_steps.insert(pcg_iters_steps.end(), pcg_iters.begin(), pcg_iters.end());



#if CONSTANT_SOLVE_TIME
        iter_solve_time = SOLVE_TIME;
#else
        // iter_solve_time = dis(gen); // EMRE 
        iter_diff = iter_end - iter_start;
        iter_solve_time = iter_diff.count();
        if(iter < 50){ iter_solve_time = .02;}
        std::cout << "solve time: " << iter_solve_time << std::endl;
#endif
        time_since_timestep += iter_solve_time;


        // // simulate traj for prev solve time
        // simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_iter_solve_time, iter_solve_time);

        // gpuErrchk(cudaPeekAtLastError());

        // // old xu = new xu
        // gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        // // shift xu if over halfway through knot
        // if (!shifted && time_since_timestep > shift_threshold){
        //     just_shift<T>(state_size, control_size, knot_points, d_xu);
        //     gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size+control_size)], &d_traj[(state_size+control_size)*(traj_offset+1)-control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));
        //     shifted = true;
        // }

        // // shift goal if through knot
        // if (time_since_timestep > timestep){
        //     std::cout << "shifted to offset: " << traj_offset + 1 << std::endl;
        //     shifted = false;
        //     traj_offset++;
        //     gpuErrchk(cudaMemcpy(d_xu_goal, &d_traj[traj_offset * (state_size + control_size)], traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
        //     gpuErrchk(cudaMemcpy(d_lambda, &d_traj_lambdas[traj_offset * (state_size*knot_points)], (state_size*knot_points)*sizeof(T), cudaMemcpyDeviceToDevice));
        //     time_since_timestep = 0.0f;
        // }

        // gpuErrchk(cudaPeekAtLastError());

        // gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));

        prev_iter_solve_time = iter_solve_time;

    }

    // ignore
        // simulate for iter_solve_time seconds using old traj starting at prev_iter_solve_time offset
        // integrator_shift<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, .01, prev_iter_solve_time, .01);

        // gpuErrchk(cudaMemcpy(h_temp, d_xu_goal, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < state_size; i++){
        //     std::cout << h_temp[i] << " ";
        // }
        // std::cout <<"\n\n";


    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_xu_goal));
    gpuErrchk(cudaFree(d_xu_old));
    gpuErrchk(cudaFree(d_xu_temp));
}


__device__ __forceinline__
void gato_fmemset(float *dst, float val, unsigned num_floats){
    for(int i = threadIdx.x; i < num_floats; i+=blockDim.x){
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

//     const uint32_t xu_len = (states_s_controls*50-7);
//     const uint32_t lambda_len = (state_size * knot_points);
//     const uint32_t xg_len = state_size;

//     // Allocate space for host xu and goal
//     T h_xu[states_s_controls*50-7];
//     T h_xg[14];
//     T h_xs[14];

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

//     final_path.push_back(std::vector<T>(h_xs, &h_xs[14]));

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
//         final_path.push_back(std::vector<T>(h_xs, &h_xs[14]));
        
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