#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
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
#include "qdldl_helper.cuh"
#include "settings.cuh"
#include "testutils.cuh"
#include "experiment_helpers.cuh"
#include "crocoddyl_helpers.cuh"


template <typename T>
auto sqpSolve(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, pcg_config& config, T &rho, T rho_reset){
    
    // data storage
    std::vector<int> pcg_iter_vec;
    std::vector<bool> pcg_exit_vec;
    std::vector<double> linsys_time_vec;
    bool sqp_time_exit = 1;     // for data recording, not a flag
    


    // sqp timing
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);



    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       =   static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));


    // line search things
    const float mu = 10.0f;
    const uint32_t num_alphas = 1;
    T h_merit_news[num_alphas];
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit;
    T alphafinal;
    T delta_merit_iter = 0;
    T delta_merit_total = 0;
    uint32_t line_search_step = 0;


    // streams n cublas init
    cudaStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        cudaStreamCreate(&streams[str]);
    }
    gpuErrchk(cudaPeekAtLastError());

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(13); }
    gpuErrchk(cudaPeekAtLastError());


    uint32_t sqp_iter = 0;



    T *d_merit_initial, *d_merit_news, *d_merit_temp,
          *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
          *d_S, *d_gamma,
          *d_dz,
          *d_xs;

    
    T drho = 1.0;
    T rho_factor = RHO_FACTOR;
    T rho_max = RHO_MAX;
    T rho_min = RHO_MIN;

    


    gpuErrchk(cudaMalloc(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_g,        KKT_g_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_c,        KKT_c_SIZE_BYTES));
    d_Ginv_dense = d_G_dense;

    gpuErrchk(cudaMalloc(&d_S, 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaPeekAtLastError());

    
    gpuErrchk(cudaMalloc(&d_dz,       DZ_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_xs,       state_size*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xs, d_xu,  state_size*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMalloc(&d_merit_news, 8*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_temp, 8*knot_points*sizeof(T)));
    // pcg iterates

    gpuErrchk(cudaMalloc(&d_merit_initial, sizeof(T)));
    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T)));
    


#if PCG_SOLVE

    // pcg things
    T *d_Pinv;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*states_sq*knot_points*sizeof(T)));
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    gpuErrchk(cudaMalloc(&d_r, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points*sizeof(T)));
    
    
    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t pcg_iters;
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool pcg_exit;
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));
    
    void *pcgKernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
        (void *)&d_gamma, 
        (void *)&d_lambda,
        (void *)&d_r,
        (void *)&d_p,
        (void *)&d_v_temp,
        (void *)&d_eta_new_temp,
        (void *)&d_pcg_iters,
        (void *)&d_pcg_exit,
        (void *)&config.pcg_max_iter,
        (void *)&config.pcg_exit_tol
    };
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


#else // #if PCG_SOLVE

    const int nnz = (knot_points-1)*states_sq + knot_points*((state_size+1)*state_size/2);

    QDLDL_float h_lambda[state_size*knot_points];
    QDLDL_float h_gamma[state_size*knot_points];
    QDLDL_int h_col_ptr[state_size*knot_points+1];
    QDLDL_int h_row_ind[nnz];
    QDLDL_float h_val[nnz];
    
    QDLDL_int *d_row_ind, *d_col_ptr;
    QDLDL_float *d_val, *d_lambda_double;
    gpuErrchk(cudaMalloc(&d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int)));
    gpuErrchk(cudaMalloc(&d_row_ind, nnz*sizeof(QDLDL_int)));
	gpuErrchk(cudaMalloc(&d_val, nnz*sizeof(QDLDL_float)));
	gpuErrchk(cudaMalloc(&d_lambda_double, (state_size*knot_points)*sizeof(QDLDL_float)));
    
    // fill col ptr and row ind, these won't change 
    prep_csr<<<knot_points, 64>>>(state_size, knot_points, d_col_ptr, d_row_ind);
    gpuErrchk(cudaMemcpy(h_col_ptr, d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_row_ind, d_row_ind, (nnz)*sizeof(QDLDL_int), cudaMemcpyDeviceToHost));

    
    const QDLDL_int An = state_size*knot_points;

    // Q things
    QDLDL_int  sumLnz;
    QDLDL_int *etree;
	QDLDL_int *Lnz;
    etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
	Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);

    QDLDL_int *Lp;
	QDLDL_float *D;
	QDLDL_float *Dinv;
    Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
	D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
	Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //working data for factorisation
	QDLDL_int   *iwork;
	QDLDL_bool  *bwork;
	QDLDL_float *fwork;
    iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
	bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
	fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    sumLnz = QDLDL_etree(An,h_col_ptr,h_row_ind,iwork,Lnz,etree);
    
    QDLDL_int *Li;
	QDLDL_float *Lx;
    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
	Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);


#endif // #if PCG_SOLVE

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#if TIME_LINSYS
    struct timespec linsys_start, linsys_end;
    double linsys_time;
#endif
#if CONST_UPDATE_FREQ
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif


    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points,
        d_xu, 
        d_eePos_traj, 
        static_cast<T>(10), 
        timestep, 
        d_dynMem_const, 
        d_merit_initial
    );
    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    // gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "initial merit " << h_merit_initial << std::endl;
    // exit(0);

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){
        
        gato_form_kkt<T><<<knot_points, KKT_THREADS, 2 * oldschur::get_kkt_smem_size<T>(state_size, control_size)>>>(
            state_size,
            control_size,
            knot_points,
            d_G_dense, 
            d_C_dense, 
            d_g, 
            d_c,
            d_dynMem_const,
            timestep,
            d_eePos_traj,
            d_xs,
            d_xu
        );
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }

    // write_device_matrix_to_file(d_g, 1, 2*(state_size+control_size), "g", 0);
    // write_device_matrix_to_file(d_c, 1, 2*(state_size+control_size), "c", 0);
    // exit(2);

#if PCG_SOLVE

        form_schur<T>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }
        

    #if TIME_LINSYS    
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        clock_gettime(CLOCK_MONOTONIC,&linsys_start);
    #endif // #if TIME_LINSYS

        gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));    
        gpuErrchk(cudaMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaPeekAtLastError());

    #if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&linsys_end);
        
        linsys_time = time_delta_us_timespec(linsys_start,linsys_end);
        linsys_time_vec.push_back(linsys_time);
    #endif // #if TIME_LINSYS

        pcg_iter_vec.push_back(pcg_iters);
        pcg_exit_vec.push_back(pcg_exit);

#else // #if PCG_SOLVE

        form_schur_qdl<T>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_val, d_gamma, rho);
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }

    #if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        clock_gettime(CLOCK_MONOTONIC, &linsys_start);
    #endif // #if TIME_LINSYS


        gpuErrchk(cudaMemcpy(h_val, d_val, (nnz)*sizeof(T), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_gamma, d_gamma, (state_size*knot_points)*sizeof(T), cudaMemcpyDeviceToHost))

        qdl::qdldl_solve_schur(An, h_col_ptr, h_row_ind, h_val, h_gamma, h_lambda, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
        
        gpuErrchk(cudaMemcpy(d_lambda, h_lambda, (state_size*knot_points)*sizeof(T), cudaMemcpyHostToDevice));


    #if TIME_LINSYS
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &linsys_end);
        
        linsys_time = time_delta_us_timespec(linsys_start, linsys_end);
        linsys_time_vec.push_back(linsys_time);
    #endif // #if TIME_LINSYS

#endif // #if PCG_SOLVE
        
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
        
        // // Allocate space for host variables
        // const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;
        // T *h_xs = new T[state_size];
        // T *h_xu = new T[traj_len];
        // T *h_eePos_traj = new T[6];

        // line search
        for(uint32_t p = 0; p < num_alphas; p++){
            void *kernelArgs[] = {
                (void *)&state_size,
                (void *)&control_size,
                (void *)&knot_points,
                (void *)&d_xs,
                (void *)&d_xu,
                (void *)&d_eePos_traj,
                (void *)&mu, 
                (void *)&timestep,
                (void *)&d_dynMem_const,
                (void *)&d_dz,
                (void *)&p,
                (void *)&d_merit_news,
                (void *)&d_merit_temp
            };
                // if(p == 0) {
                //     // Copy data from device to host
                //     gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
                //     gpuErrchk(cudaMemcpy(h_xu, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToHost));
                //     gpuErrchk(cudaMemcpy(h_eePos_traj, d_eePos_traj, 6*sizeof(T), cudaMemcpyDeviceToHost));

                    // printf("Printing inputs to merit computation from host\n");

                    // // Print the data in human-readable format
                    // for(int i = 0; i < state_size; ++i) {
                    //     std::cout << "h_xs[" << i << "] = " << h_xs[i] << std::endl;
                    // }
                    
                    // for(int i = 0; i < traj_len; ++i) {
                    //     std::cout << "h_xu[" << i << "] = " << h_xu[i] << std::endl;
                    // }
                    
                    // for(int i = 0; i < 6; ++i) {
                    //     std::cout << "h_eePos_traj[" << i << "] = " << h_eePos_traj[i] << std::endl;
                    // }
                // }
            gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs, get_merit_smem_size<T>(state_size, knot_points), streams[p]));
        }
        if (sqpTimecheck()){ break; }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // //print out message and end the program
        // printf("Exiting after computing the merit function value, so we can use it for comparison\n");
        // // exit the program
        // exit(0);

        
        
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);
        if (sqpTimecheck()){ break; }


        line_search_step = 0;
        min_merit = h_merit_initial;
        for(int i = 0; i < 8; i++){
        //     std::cout << h_merit_news[i] << (i == 7 ? "\n" : " ");
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
            sqp_iter++;
            if(rho > rho_max){
                sqp_time_exit = 0;
                rho = rho_reset;
                break; 
            }
            continue;
        }
        // std::cout << "line search accepted\n";
        alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

        drho = min(drho/rho_factor, 1/rho_factor);
        rho = max(rho*drho, rho_min);
        

#if USE_DOUBLES
        cublasDaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#else
        cublasSaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#endif

        gpuErrchk(cudaPeekAtLastError());
        // if success increment after update
        sqp_iter++;

        if (sqpTimecheck()){ break; }


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        

        h_merit_initial = min_merit;
    
    }
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);

    cublasDestroy(handle);

    for(uint32_t st=0; st < num_alphas; st++){
        gpuErrchk(cudaStreamDestroy(streams[st]));
    }




    gpuErrchk(cudaFree(d_merit_initial));
    gpuErrchk(cudaFree(d_merit_news));
    gpuErrchk(cudaFree(d_merit_temp));
    gpuErrchk(cudaFree(d_G_dense));
    gpuErrchk(cudaFree(d_C_dense));
    gpuErrchk(cudaFree(d_g));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_dz));
    gpuErrchk(cudaFree(d_xs));

#if PCG_SOLVE
    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));
    gpuErrchk(cudaFree(d_Pinv));
    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_v_temp));
    gpuErrchk(cudaFree(d_eta_new_temp));
#else
    gpuErrchk(cudaFree(d_col_ptr));
    gpuErrchk(cudaFree(d_row_ind));
    gpuErrchk(cudaFree(d_val));
    gpuErrchk(cudaFree(d_lambda_double));
	free(etree);
	free(Lnz);
    free(Lp);
	free(D);
	free(Dinv);
	free(iwork);
	free(bwork);
	free(fwork);
	free(Li);
	free(Lx);

#endif


    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);

    return std::make_tuple(pcg_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, pcg_exit_vec);
}



template <typename T, typename return_type>
std::tuple<std::vector<toplevel_return_type>, std::vector<pcg_t>, pcg_t> track(uint32_t state_size, uint32_t control_size, uint32_t knot_points, const uint32_t traj_steps, 
            float timestep, T *d_eePos_traj, T *d_xu_traj, T *d_xs, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter, T pcg_exit_tol,
            std::string test_output_prefix, std::vector<std::vector<pcg_t>> eePos_traj2d, std::vector<std::vector<pcg_t>> xu_traj2d){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    const T shift_threshold = SHIFT_THRESHOLD;
    const int max_control_updates = 100000;
    
    
    // struct timespec solve_start, solve_end;
    double sqp_solve_time_us = 0;               // current sqp solve time
    double simulation_time = 0;                 // current simulation time
    double prev_simulation_time = 0;            // last simulation time
    double time_since_timestep = 0;             // time since last timestep of original trajectory
    bool shifted = false;                       // has xu been shifted
    uint32_t traj_offset = 0;                        // current goal states of original trajectory


    // vars for recording data
    std::vector<std::vector<T>> tracking_path;      // list of traversed traj
    std::vector<int> pcg_iters;
    std::vector<double> linsys_times;
    std::vector<double> sqp_times;
    std::vector<uint32_t> sqp_iters;
    std::vector<bool> sqp_exits;
    std::vector<bool> pcg_exits;
    std::vector<T> tracking_errors;
    std::vector<int> cur_pcg_iters;
    std::vector<bool> cur_pcg_exits;
    std::vector<double> cur_linsys_times;
    std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>> sqp_stats;
    double ddp_solve_time;
    std::size_t ddp_solve_iters;
    double ddp_cost;
    std::tuple<double, std::size_t, double> ddp_stats;
    uint32_t cur_sqp_iters;
    T cur_tracking_error;
    int control_update_step;


    // mpc iterates
    T *d_lambda, *d_eePos_goal, *d_xu, *d_xu_old;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eePos_goal, 6*knot_points*sizeof(T)));
    gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_eePos_goal, d_eePos_traj, 6*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();


    // temp host memory
    T h_xs[state_size];
    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
    tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));    
    gpuErrchk(cudaPeekAtLastError());
    T h_eePos[6];
    T h_eePos_goal[6];


    // temp device memory
    T *d_eePos;
    gpuErrchk(cudaMalloc(&d_eePos, 6*sizeof(T)));

    pcg_config config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = pcg_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;

    T rho = 1e-3;
    T rho_reset = 1e-3;

    // Crocoddyl Setup
#if CROCODDYL_SOLVE

    pinocchio::Model robot_model = initialize_pinocchio_robot();
    boost::shared_ptr<crocoddyl::StateMultibody> state = initialize_crocoddyl_state_multibody(robot_model);
    boost::shared_ptr<crocoddyl::ActuationModelFull> actuation = initialize_crocoddyl_actuation_model_full(state);
	
    Eigen::VectorXd Q_vec(state_size);
	Q_vec.fill(Q_COST);
	
	Eigen::VectorXd R_vec(control_size);
	R_vec.fill(R_COST);
	
	Eigen::VectorXd EE_penalty_vec(EE_DIM_POS);
	EE_penalty_vec.fill(EE_COST);

    const int ee_joint_frame_id = robot_model.getFrameId("iiwa_joint_7");

    // Allocate space for host variables
    uint32_t ee_state_size = 6;
    T *h_xu = new T[traj_len];
    T *h_eePos_traj = new T[ee_state_size*knot_points];

#endif // #if CROCODDYL_SOLVE

#if REMOVE_JITTERS
    std::cout << "Removing jitters" << std::endl;
    config.pcg_exit_tol = 1e-11;
    config.pcg_max_iter = 10000;
    
    for(int j = 0; j < 100; j++){
        sqpSolve<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, 1e-3);
        gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    rho = 1e-3;
    config.pcg_exit_tol = pcg_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;
    std::cout << "Done removing jitters" << std::endl;
#endif // #if REMOVE_JITTERS



    //
    // MPC tracking loop
    //
    for(control_update_step = 0; control_update_step < max_control_updates; control_update_step++){
        

        if (traj_offset == traj_steps){ break; }



#if LIVE_PRINT_PATH
        grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
        gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < 6; i++){
            std::cout << h_eePos[i] << (i < 5 ? " " : "\n");
        }
#endif // #if LIVE_PRINT_PATH
        

        #if CROCODDYL_SOLVE // Use the Crocoddyl DDP solver, rather than SQP, to solve the trajopt problem
        
            // step 1: move what we need from the GPU to the CPU

            // copy data from device to host, from d_xu into h_xu
            gpuErrchk(cudaMemcpy(h_xu, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToHost));
            // print the first and last 21 elements of h_xu
            std::cout << "h_xu start: ";
            for(int i = 0; i < 21; i++){     
                std::cout << h_xu[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "h_xu end: ";
            for(int i = traj_len-21; i < traj_len; i++){     
                std::cout << h_xu[i] << " ";
            }
            std::cout << std::endl;
            gpuErrchk(cudaMemcpy(h_eePos_traj, d_eePos_traj, ee_state_size*knot_points*sizeof(T), cudaMemcpyDeviceToHost));
            
            // TODO: fix the way I am dealing with end effector for DDP
            // current I'm passing in the eePos_traj, but I should be passing in the eePos_goal I think instead,
            // and using the same goal when computing the residual for each knot point. Need to confirm this though.
            // Also need to make sure I am initializing the problem correctly. I was previously passing in the same initial
            // state to the ddp solver each time. I have now fixed that, but I am likely making the same mistake with 
            // end effector.
            gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));

            // step 2: solve the problem
            ddp_stats = crocoddylSolve<T>(state_size, control_size, ee_state_size, knot_points, 
                timestep,eePos_traj2d, xu_traj2d,Q_vec, R_vec, EE_penalty_vec, state, actuation, 
                ee_joint_frame_id, h_xu);

            // print the h_xu start and end again, to see if they changed
            std::cout << "h_xu start: ";
            for(int i = 0; i < 21; i++){     
                std::cout << h_xu[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "h_xu end: ";
            for(int i = traj_len-21; i < traj_len; i++){     
                std::cout << h_xu[i] << " ";
            }
            std::cout << std::endl;

            ddp_solve_time = std::get<0>(ddp_stats);
            ddp_solve_iters = std::get<1>(ddp_stats);
            ddp_cost = std::get<2>(ddp_stats);
            // print the solve time
            std::cout << "DDP solve time: " << ddp_solve_time << std::endl;
            // also print number of ddp iters
            std::cout << "DDP iters: " << ddp_solve_iters << std::endl;
            // also print cost
            std::cout << "DDP cost: " << ddp_cost << std::endl;

            // // add small sleep for readability
            // usleep(1000000);


            // step 3: copy what we need back onto the GPU, so we can continue using the same MPC control loop
            
            // copy h_xu back onto the GPU, into d_xu
            gpuErrchk(cudaMemcpy(d_xu, h_xu, traj_len*sizeof(T), cudaMemcpyHostToDevice));

        #else // #if CROCODDYL_SOLVE
            sqp_stats = sqpSolve<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, rho_reset);


            cur_pcg_iters = std::get<0>(sqp_stats);
            cur_linsys_times = std::get<1>(sqp_stats);
            sqp_solve_time_us = std::get<2>(sqp_stats);
            cur_sqp_iters = std::get<3>(sqp_stats);
            sqp_exits.push_back(std::get<4>(sqp_stats));
            cur_pcg_exits = std::get<5>(sqp_stats);
        #endif

        std::cout << "done solve, starting simulation " << std::endl;


#if CONST_UPDATE_FREQ
        simulation_time = SIMULATION_PERIOD;
#else
        simulation_time = ddp_solve_time;
#endif

        std::cout << "simulating for " << simulation_time << " us\nxu:";
        float h_temp[21];
        gpuErrchk(cudaMemcpy(h_temp, d_xu_old, 21*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < 21; i++){
            std::cout << h_temp[i] << " ";
        }
        std::cout << std::endl << "xs:";
        gpuErrchk(cudaMemcpy(h_temp, d_xs, 14*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < 14; i++){
            std::cout << h_temp[i] << " ";
        }
        std::cout << std::endl;
        




        // simulate traj for current solve time, offset by previous solve time
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_simulation_time, simulation_time);

        
        std::cout << "result\n";
        gpuErrchk(cudaMemcpy(h_temp, d_xs, 14*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < 14; i++){
            std::cout << h_temp[i] << " ";
        }
        std::cout << std::endl;
        // exit(12);


        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        time_since_timestep += simulation_time * 1e-6;

        // if shift_threshold% through timestep
        if (!shifted && time_since_timestep > shift_threshold){
            
            // record tracking error
            grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
            gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
            cur_tracking_error = 0.0;
            for(uint32_t i=0; i < 3; i++){
                cur_tracking_error += abs(h_eePos[i] - h_eePos_goal[i]);
            }
            std ::cout << "tracking error: " << cur_tracking_error << std::endl;
            tracking_errors.push_back(cur_tracking_error);                                            
            
            traj_offset++;

            // shift xu
            just_shift<T>(state_size, control_size, knot_points, d_xu);             // shift everything over one
            if (traj_offset + knot_points < traj_steps){
                // if within precomputed traj, fill in last state, control with precompute
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_xu_traj[(state_size+control_size)*traj_offset - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            }
            else{
                // fill in last state with goal position, zero velocity, last control with zero control
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - state_size], &d_xu_traj[(traj_steps-1)*(state_size+control_size)], (state_size/2)*sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemset(&d_xu[traj_len - state_size / 2], 0, (state_size/2) * sizeof(T)));
                gpuErrchk(cudaMemset(&d_xu[traj_len - (state_size+control_size)], 0, control_size * sizeof(T)));
            }

            std::cout << "shifted to offset: " << traj_offset << std::endl;
            
            // shift goal
            just_shift(6, 0, knot_points, d_eePos_goal);
            if (traj_offset + knot_points < traj_steps){
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_offset+knot_points-1) * (6)], 6*sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else{
                // fill in last goal state with goal state and zero velocity
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_steps-1)*(6)], (6)*sizeof(T), cudaMemcpyDeviceToDevice));
                // gpuErrchk(cudaMemset(&d_eePos_goal[(knot_points-1)*(6) + state_size / 2], 0, (state_size/2) * sizeof(T)));
            }

            std::cout << "shifted goal to offset: " << 6 * (traj_offset + knot_points - 1) << std::endl;
            
            // shift lambda
            just_shift(state_size, 0, knot_points, d_lambda);
                // gpuErrchk(cudaMemset(&lambdas[i][state_size*(knot_points-1)], 0, state_size*sizeof(T)));
            
            shifted = true;
        }

#if ADD_NOISE
        addNoise<T>(state_size, d_xs, 1, .0001, .001);
#endif

        if (time_since_timestep > timestep){
            // std::cout << "shifted to offset: " << traj_offset + 1 << std::endl;
            shifted = false;
            time_since_timestep = std::fmod(time_since_timestep, timestep);
        }
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));


        
        prev_simulation_time = simulation_time;

        gpuErrchk(cudaPeekAtLastError());

        
        // record data
        pcg_iters.insert(pcg_iters.end(), cur_pcg_iters.begin(), cur_pcg_iters.end());                      // pcg iters
        linsys_times.insert(linsys_times.end(), cur_linsys_times.begin(), cur_linsys_times.end());          // linsys times
        pcg_exits.insert(pcg_exits.end(), cur_pcg_exits.begin(), cur_pcg_exits.end());
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));                                   // next state
        sqp_times.push_back(sqp_solve_time_us);
        sqp_iters.push_back(cur_sqp_iters);


#if LIVE_PRINT_STATS
        if (control_update_step % 1000 == 50){
            for (uint32_t i = 0; i < state_size; i++){
                std::cout << h_xs[i] << (i < state_size-1 ? " " : "\n");
            }
    #if TIME_LINSYS
            std::cout << "linear system solve time:" << std::endl;
            printStats<double>(&linsys_times);
    #endif // #if TIME_LINSYS
            std::cout << "goal offset [" << traj_offset << "]\n";
            std::cout << "sqp iters" << std::endl;
            printStats<uint32_t>(&sqp_iters);
            std::cout << "sqp times" << std::endl;
            printStats<double>(&sqp_times);
            
            int totalOnes = std::accumulate(pcg_exits.begin(), pcg_exits.end(), 0);
            double max_iter_pct = (static_cast<double>(totalOnes) / pcg_exits.size());
            std::cout << "pcg exits for max iter: " << max_iter_pct * 100 << "% of the time\n";
            if (max_iter_pct > 0.5) {
               std::cout << "WARNING: PCG exiting for max iter over 50% of the time" << std::endl;
            }
            
            std::cout << "avg tracking error: " << std::accumulate(tracking_errors.begin(), tracking_errors.end(), 0.0f) / traj_offset << " current error: " << cur_tracking_error << "\n";
            std::cout << std::endl;

        }

#endif


    }
#if SAVE_DATA
    dump_tracking_data(&pcg_iters, &pcg_exits, &linsys_times, &sqp_times, &sqp_iters, &sqp_exits, &tracking_errors, &tracking_path, 
            traj_offset, control_update_step, start_state_ind, goal_state_ind, test_iter, test_output_prefix);
#endif
    
#if TIME_LINSYS
    // std::cout << "\n\nlinear system solve time:" << std::endl;
    // printStats<double>(&linsys_times);
#endif
    // std::cout << "sqp iters" << std::endl;
    // printStats<uint32_t>(&sqp_iters);
    // printStats<int>(&pcg_iters);
    // printStats<double>(&sqp_times);
    // printStats<float>(&tracking_errors);
    // std::cout << "control updates: " << control_update_step << "\n";

    grid::end_effector_positions_kernel<T><<<1,128>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
    gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
    cur_tracking_error = 0.0;
    for(uint32_t i=0; i < 3; i++){
        cur_tracking_error += abs(h_eePos[i] - h_eePos_goal[i]);
    }
    // std::cout << "avg tracking error: " << std::accumulate(tracking_errors.begin(), tracking_errors.end(), 0.0f) / traj_steps << " final error: " << cur_tracking_error << "\n\n";

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_eePos_goal));
    gpuErrchk(cudaFree(d_xu_old));

    gpuErrchk(cudaFree(d_eePos));


    // auto ivecAvg = [](const std::vector<uint32_t>& v){
    //     return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    // };
    
    // auto tvecAvg = [](const std::vector<T>& v){
    //     return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    // };

    #if TIME_LINSYS 
        return std::make_tuple(linsys_times, tracking_errors, cur_tracking_error);
    #else
        return std::make_tuple(sqp_iters, tracking_errors, cur_tracking_error);
    #endif
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
            // gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_eePos_traj[(state_size+control_size)*(traj_offset+1) - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            // gpuErrchk(cudaMemcpy(d_xu_goal, &d_eePos_traj[traj_offset * (state_size + control_size)], traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
