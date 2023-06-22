#pragma once
#include <vector>
#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>
#include "schur.cuh"
#include "merit.cuh"
#include "gpu_pcg.cuh"
#include "integrator.cuh"
#include "settings.cuh"
#include "testutils.cuh"
#include "experiment_helpers.cuh"


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
std::tuple<std::vector<int>, std::vector<double>, uint32_t, bool> sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, struct timespec sqp_start){
    

    std::vector<int> pcg_iter_vec;
    std::vector<double> pcg_time_vec;

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
    uint32_t line_search_step = 0;
    uint32_t sqp_iter = 0;
    bool sqp_time_exit = 1;     // for data recording, not a flag


    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_start,sqp_cur) > SQP_MAX_TIME_US;
    };

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
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    
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
    // pcg iterates
    gpuErrchk(cudaMallocAsync(&d_r, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_p, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_v_temp, knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaMallocAsync(&d_eta_new_temp, knot_points*sizeof(T), streams[1]));
    // gpuErrchk(cudaMallocAsync(&d_r_tilde, state_size*knot_points*sizeof(T), streams[1]));
    // gpuErrchk(cudaMallocAsync(&d_upsilon, state_size*knot_points*sizeof(T), streams[1]));
    gpuErrchk(cudaPeekAtLastError());


    /*   STREAM 0   */
    gpuErrchk(cudaMallocAsync(&d_merit_initial, sizeof(T), streams[0]));
    gpuErrchk(cudaMemsetAsync(d_merit_initial, 0, sizeof(T), streams[0]));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    

    
    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<float><<<knot_points, MERIT_THREADS, merit_smem_size*5, streams[0]>>>(
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

    struct timespec start, end;
    

    pcg_config config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = PCG_EXIT_TOL;
    config.pcg_max_iter = PCG_MAX_ITER;
    int pcg_iters;
    double pcg_time;


    //
    //      SQP LOOP
    //
    while(1){
        
        gato_form_kkt<float><<<knot_points, KKT_THREADS, oldschur::get_kkt_smem_size<T>(state_size, control_size)>>>(
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
        if (sqpTimecheck()){ break; }


        // form schur
        form_schur(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }
        

        // TIME PCG SOLVE
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&start);

        pcg_iters = solvePCG(state_size, knot_points, d_S, d_Pinv, d_gamma, d_lambda, d_r, d_p, d_v_temp, d_eta_new_temp, &config);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        clock_gettime(CLOCK_MONOTONIC,&end);
        
        // PCG time vs. qdl is solve time + preconditioner time
        pcg_time = time_delta_us_timespec(start,end);


        pcg_time_vec.push_back(pcg_time);
        pcg_iter_vec.push_back(pcg_iters);
        if (sqpTimecheck()){ break; }


        // recover dz
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
        if (sqpTimecheck()){ break; }
        

        // line search
        parallel_line_search(state_size, control_size, knot_points, d_xs, d_xu, d_traj, d_dynMem_const, d_dz, timestep, d_merit_news, d_merit_temp);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }
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
            if(rho > rho_max){
                sqp_time_exit = 0;
                break; 
            }
            sqp_iter++;
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
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }

        // if success increment after update
        sqp_iter++;


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        
        // if( delta_merit_iter < 1e-6){
        //     // std::cout << "exiting SQP for exit tolerance\n";
        //     tolerance_exits++;
        //     break;
        // }

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
    // gpuErrchk(cudaFreeAsync(d_r_tilde, 0));
    // gpuErrchk(cudaFreeAsync(d_upsilon, 0));    


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

    return std::make_tuple(pcg_iter_vec, pcg_time_vec, sqp_iter, sqp_time_exit);
}



template <typename T>
void track(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, float timestep, T *d_traj, T *d_traj_lambdas, T *d_xs, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    const float shift_threshold = SHIFT_THRESHOLD;
    const int max_control_updates = 10000;
    
    
    struct timespec solve_start, solve_end;
    double iter_solve_time_us = 0;              // current linear system solve time
    double prev_iter_solve_time = 0;            // last linear system solve time
    double time_since_timestep = 0;             // time since last timestep of original trajectory
    bool shifted = false;                       // has xu been shifted
    int traj_offset = 0;                        // current goal states of original trajectory


    // vars for recording data
    std::vector<std::vector<T>> tracking_path;      // list of traversed traj
    std::vector<int> pcg_iters;
    std::vector<double> linsys_times;
    std::vector<double> sqp_times;
    std::vector<uint32_t> sqp_iters;
    std::vector<bool> sqp_exits;
    std::vector<float> tracking_errors;
    std::vector<int> cur_pcg_iters;
    std::vector<double> cur_linsys_times;
    std::tuple<std::vector<int>, std::vector<double>, uint32_t, bool> sqp_stats;
    uint32_t cur_sqp_iters;
    float cur_tracking_error;
    float tot_err;
    int control_update_step;

    // noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-.01f, .01f);

    // mpc iterates
    T *d_lambda, *d_xu_goal, *d_xu, *d_xu_old;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_goal, traj_len*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_lambda, d_traj_lambdas, state_size*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_goal, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();


    // temp host memory
    T h_xs[state_size];
    T h_xg[state_size];
    gpuErrchk(cudaMemcpy(h_xg, &d_traj[(traj_steps-1)*(state_size+control_size)], state_size*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
    tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));    
    gpuErrchk(cudaPeekAtLastError());
    

#if REMOVE_JITTERS
    for(int j = 0; j < 100; j++){
        clock_gettime(CLOCK_MONOTONIC, &solve_start);
        sqpSolve<T>(state_size, control_size, knot_points, timestep, d_xu_goal, d_lambda, d_xu, d_dynmem, solve_start);
        gpuErrchk(cudaMemcpy(d_lambda, d_traj_lambdas, state_size*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_xu, d_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
#endif // #if REMOVE_JITTERS

    //
    // MPC
    //
    for(control_update_step = 0; control_update_step < max_control_updates; control_update_step++){


        tot_err = 0;
        for (int i = 0; i < state_size; i++){
#if LIVE_PRINT_PATH
            std::cout << h_xs[i] << (i < state_size-1 ? " " : "\n");
#endif // #if LIVE_PRINT_PATH
            tot_err += abs(h_xs[i] - h_xg[i]);
        }
        if(tot_err < TRACKING_EXIT_TOL){ 
            break;
        }
        

        
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &solve_start);

        sqp_stats = sqpSolve<T>(state_size, control_size, knot_points, timestep, d_xu_goal, d_lambda, d_xu, d_dynmem, solve_start);


        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        clock_gettime(CLOCK_MONOTONIC, &solve_end);

        cur_pcg_iters = std::get<0>(sqp_stats);
        cur_linsys_times = std::get<1>(sqp_stats);
        cur_sqp_iters = std::get<2>(sqp_stats);
        sqp_exits.push_back(std::get<3>(sqp_stats));
        iter_solve_time_us = time_delta_us_timespec(solve_start, solve_end);


        cur_tracking_error = compute_tracking_error<T>(state_size, control_size, knot_points, d_xu_goal, d_xu);
        

        // simulate traj for current solve time, offset by previous solve time
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_iter_solve_time, iter_solve_time_us);


        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        time_since_timestep += iter_solve_time_us * 1e-6;

        // shift xu and goal if shift_threshold% through timestep
        if (!shifted && time_since_timestep > shift_threshold){
            just_shift<T>(state_size, control_size, knot_points, d_xu);             // shift everything over one
            if (traj_offset + 1 + knot_points < traj_steps){
                // if within precomputed traj, fill in last state, control with precompute
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_traj[(state_size+control_size)*(traj_offset+1) - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            }
            else{
                // fill in last state with goal position, zero velocity, last control with zero control
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - state_size], &d_traj[(traj_steps-1)*(state_size+control_size)], (state_size/2)*sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemset(&d_xu[traj_len - state_size / 2], 0, (state_size/2) * sizeof(T)));
                gpuErrchk(cudaMemset(&d_xu[traj_len - (state_size+control_size)], 0, control_size * sizeof(T)));
            }
            traj_offset++;
            just_shift(state_size, control_size, knot_points, d_xu_goal);
            if (traj_offset + knot_points < traj_steps){
                gpuErrchk(cudaMemcpy(&d_xu_goal[(knot_points-1)*(state_size+control_size)], &d_traj[(traj_offset+knot_points-1) * (state_size + control_size)], state_size*sizeof(T), cudaMemcpyDeviceToDevice));
                
                // gpuErrchk(cudaMemcpy(d_lambda, &d_traj_lambdas[traj_offset * (state_size*knot_points)], (state_size*knot_points)*sizeof(T), cudaMemcpyDeviceToDevice));          // copy lambdas from precompute?

            }
            else{
                // fill in last goal state with goal state and zero velocity
                gpuErrchk(cudaMemcpy(&d_xu_goal[(knot_points-1)*(state_size+control_size)], &d_traj[(traj_steps-1)*(state_size+control_size)], (state_size/2)*sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemset(&d_xu_goal[(knot_points-1)*(state_size+control_size) + state_size / 2], 0, (state_size/2) * sizeof(T)));
            }
            shifted = true;
        }


        if (time_since_timestep > timestep){
            // std::cout << "shifted to offset: " << traj_offset + 1 << std::endl;
            shifted = false;
            time_since_timestep = std::fmod(time_since_timestep, timestep);
        }
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));


        
        prev_iter_solve_time = iter_solve_time_us;

        gpuErrchk(cudaPeekAtLastError());


        // record data
        pcg_iters.insert(pcg_iters.end(), cur_pcg_iters.begin(), cur_pcg_iters.end());                      // pcg iters
        linsys_times.insert(linsys_times.end(), cur_linsys_times.begin(), cur_linsys_times.end());          // linsys times
        sqp_times.push_back(iter_solve_time_us);                                                            // sqp time
        sqp_iters.push_back(cur_sqp_iters);                                                                 // sqp iters
        tracking_errors.push_back(cur_tracking_error);                                                      // tracking error
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));                                   // next state


    }

    dump_tracking_data(&pcg_iters, &linsys_times, &sqp_times, &sqp_iters, &sqp_exits, &tracking_errors, &tracking_path, traj_offset, control_update_step, start_state_ind, goal_state_ind, test_iter);

    std::cout << "linear system solve time:" << std::endl;
    printStats<double>(&linsys_times);
    // printStats<int>(&pcg_iters);
    // printStats<double>(&sqp_times);
    // printStats<float>(&tracking_errors);
    std::cout << "converged in " << traj_offset << " timesteps with " << control_update_step << " updates and original trajectory of length " << traj_steps << " timesteps\n";
    std::cout << "total tracking error: " << std::accumulate(tracking_errors.begin(), tracking_errors.end(), 0) << "\n\n";

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_xu_goal));
    gpuErrchk(cudaFree(d_xu_old));
}















//TODO  if timeout

    // ignore
        // simulate for iter_solve_time seconds using old traj starting at prev_iter_solve_time offset
        // integrator_shift<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, .01, prev_iter_solve_time, .01);
// periodically print averages
        // if (iter % 100 == 0){
        //     int sum = std::accumulate(pcg_iters_steps.begin(), pcg_iters_steps.end(), 0);

        //     // Calculate the average
        //     double average = static_cast<double>(sum) / pcg_iters_steps.size();

        //     std::cout << "pcg averaging: " << average << " iters" << std::endl;
        // }

    // previous end fills
            // gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_traj[(state_size+control_size)*(traj_offset+1) - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            // gpuErrchk(cudaMemcpy(d_xu_goal, &d_traj[traj_offset * (state_size + control_size)], traj_len*sizeof(T), cudaMemcpyDeviceToDevice));