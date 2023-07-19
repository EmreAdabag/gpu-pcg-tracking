#pragma once
#include <toplevel.cuh>

#define NUM_ALPHAS 8

template <typename T>
struct MPC_variables {
    int64_t utime;
    T *h_xu;
    T *h_x0;

    
    T *h_xs;
    double timestamp;
    double timestep;

    T rho_reset;
    uint32_t pcg_max_iter;
    T pcg_exit_tol;
    T *rho_ptr;


    T *d_xu;
    T *d_lambda;
    T *d_eePos_traj;
    T *d_eePos_full_traj;

    T *d_merit_initial, *d_merit_news, *d_merit_temp,
    *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
    *d_S, *d_gamma,
    *d_dz,
    *d_xs;
    
    T *d_Pinv;
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    
    uint32_t *d_pcg_iters;
    bool *d_pcg_exit;


    cudaStream_t *streams;
    T *h_merit_news;

    cublasHandle_t handle;

    void *d_dynmem;

};



template <typename T>
void setupTracking_pcg(struct MPC_variables<T> *mpcvars){

    const uint32_t state_size = grid::NUM_JOINTS*2;
    const uint32_t control_size = grid::NUM_JOINTS;
    const uint32_t knot_points = KNOT_POINTS;
    const uint32_t states_sq = state_size*state_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       =   static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));



    mpcvars->h_x0 = (T *) malloc(grid::NUM_JOINTS*sizeof(T));
    mpcvars->h_xu = (T *) malloc(STEPS_IN_TRAJ*(3 * grid::NUM_JOINTS)*sizeof(T));
    mpcvars->h_xs = (T *) malloc(grid::NUM_JOINTS*2*sizeof(T));

    mpcvars->rho_ptr = (T *) malloc(sizeof(T));

    mpcvars->streams = (cudaStream_t *) malloc(NUM_ALPHAS*sizeof(cudaStream_t));
    mpcvars->h_merit_news = (T *) malloc(NUM_ALPHAS*sizeof(T));

    // read in traj
    std::vector<std::vector<T>> eePos_traj2d = readCSVToVecVec<T>("testfiles/0_0_eepos.traj");
    std::vector<std::vector<T>> xu_traj2d = readCSVToVecVec<T>("testfiles/0_0_traj.csv");
    if(eePos_traj2d.size() < knot_points){std::cout << "precomputed traj length < knotpoints, not implemented\n"; exit(1); }
    std::vector<T> h_eePos_traj;
    for (const auto& vec : eePos_traj2d) {
        h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
    }
    std::vector<T> h_xu_traj;
    for (const auto& xu_vec : xu_traj2d) {
        h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
    }
    gpuErrchk(cudaMalloc(&(mpcvars->d_eePos_full_traj), h_eePos_traj.size()*sizeof(T)));
    gpuErrchk(cudaMemcpy(mpcvars->d_eePos_full_traj, h_eePos_traj.data(), h_eePos_traj.size()*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&(mpcvars->d_xu), ((states_s_controls*knot_points)-control_size)*sizeof(T)));
    gpuErrchk(cudaMemcpy(mpcvars->d_xu, h_xu_traj.data(), ((states_s_controls*knot_points)-control_size)*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&(mpcvars->d_xs), state_size*sizeof(T)));
    gpuErrchk(cudaMemcpy(mpcvars->d_xs, h_xu_traj.data(), state_size*sizeof(T), cudaMemcpyHostToDevice));


    mpcvars->timestamp = 0;
    mpcvars->timestep = .015625;
    mpcvars->pcg_max_iter = 200;
    mpcvars->pcg_exit_tol = 1e-6;
    mpcvars->rho_ptr[0] = 1e-3;

    gpuErrchk(cudaMalloc(&(mpcvars->d_lambda), state_size*knot_points*sizeof(T)))
    
    gpuErrchk(cudaMalloc(&(mpcvars->d_G_dense),  KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&(mpcvars->d_C_dense),  KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&(mpcvars->d_g),        KKT_g_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&(mpcvars->d_c),        KKT_c_SIZE_BYTES));
    
    
    gpuErrchk(cudaMalloc(&(mpcvars->d_S), 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_Pinv), 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_gamma), state_size*knot_points*sizeof(T)));
    
    
    gpuErrchk(cudaMalloc(&(mpcvars->d_dz),       DZ_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&(mpcvars->d_merit_news), 8*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_merit_temp), 8*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_merit_initial), sizeof(T)));
    
    
    
    gpuErrchk(cudaMalloc(&(mpcvars->d_r), state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_p), state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_v_temp), knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_eta_new_temp), knot_points*sizeof(T)));
    
    gpuErrchk(cudaMalloc(&(mpcvars->d_pcg_iters), sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&(mpcvars->d_pcg_exit), sizeof(bool)));


    for(uint32_t str = 0; str < NUM_ALPHAS; str++){
        cudaStreamCreate(&(mpcvars->streams[str]));
    }
    gpuErrchk(cudaPeekAtLastError());
    

    if (cublasCreate(&(mpcvars->handle)) != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(13); }
    gpuErrchk(cudaPeekAtLastError());


    mpcvars->d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

}

template <typename T>
void cleanupTracking_pcg(struct MPC_variables<T> *mpcvars){

    free(mpcvars->h_x0);
    free(mpcvars->h_xu);
    free(mpcvars->h_xs);
    free(mpcvars->rho_ptr);

    gpuErrchk(cudaFree(mpcvars->d_xu));
    gpuErrchk(cudaFree(mpcvars->d_lambda));
    
    cublasDestroy(mpcvars->handle);

    for(uint32_t st=0; st < NUM_ALPHAS; st++){
        gpuErrchk(cudaStreamDestroy(mpcvars->streams[st]));
    }

    gpuErrchk(cudaFree(mpcvars->d_merit_initial));
    gpuErrchk(cudaFree(mpcvars->d_merit_news));
    gpuErrchk(cudaFree(mpcvars->d_merit_temp));
    gpuErrchk(cudaFree(mpcvars->d_G_dense));
    gpuErrchk(cudaFree(mpcvars->d_C_dense));
    gpuErrchk(cudaFree(mpcvars->d_g));
    gpuErrchk(cudaFree(mpcvars->d_c));
    gpuErrchk(cudaFree(mpcvars->d_S));
    gpuErrchk(cudaFree(mpcvars->d_gamma));
    gpuErrchk(cudaFree(mpcvars->d_dz));
    gpuErrchk(cudaFree(mpcvars->d_xs));

    gpuErrchk(cudaFree(mpcvars->d_pcg_iters));
    gpuErrchk(cudaFree(mpcvars->d_pcg_exit));
    gpuErrchk(cudaFree(mpcvars->d_Pinv));
    gpuErrchk(cudaFree(mpcvars->d_r));
    gpuErrchk(cudaFree(mpcvars->d_p));
    gpuErrchk(cudaFree(mpcvars->d_v_temp));
    gpuErrchk(cudaFree(mpcvars->d_eta_new_temp));
    
    
    gato_plant::freeDynamicsConstMem<T>(mpcvars->d_dynmem);
    
}





// expecting h_xs and timestamp to be filled and everything initialized
template <typename T>
int updateTrajectory(struct MPC_variables<T> *mpcvars){
    
    struct timespec sqp_solve_start;
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);
    

    
    T *h_xs = mpcvars->h_xs;

    double timestamp = mpcvars->timestamp;
    double timestep = mpcvars->timestep;
    T rho_reset = mpcvars->rho_reset;
    uint32_t pcg_max_iter = mpcvars->pcg_max_iter;
    T pcg_exit_tol = mpcvars->pcg_exit_tol;
    T *rho_ptr = mpcvars->rho_ptr;
    
    void *d_dynMem_const = mpcvars->d_dynmem;
    
    T *d_xu = mpcvars->d_xu;
    T *d_lambda = mpcvars->d_lambda;
    T *d_eePos_traj = mpcvars->d_eePos_traj;
    T *d_eePos_full_traj = mpcvars->d_eePos_full_traj;
    
    T *d_merit_initial = mpcvars->d_merit_initial;
    T *d_merit_news = mpcvars->d_merit_news;
    T *d_merit_temp = mpcvars->d_merit_temp;
    T *d_G_dense = mpcvars->d_G_dense; 
    T *d_C_dense = mpcvars->d_C_dense; 
    T *d_g = mpcvars->d_g; 
    T *d_c = mpcvars->d_c; 
    T *d_Ginv_dense = mpcvars->d_Ginv_dense;
    T *d_S = mpcvars->d_S; 
    T *d_gamma = mpcvars->d_gamma;
    T *d_dz = mpcvars->d_dz;
    T *d_xs = mpcvars->d_xs;
    
    T *d_Pinv = mpcvars->d_Pinv;
    T *d_r = mpcvars->d_r; 
    T *d_p = mpcvars->d_p; 
    T *d_v_temp = mpcvars->d_v_temp; 
    T *d_eta_new_temp = mpcvars->d_eta_new_temp;    
    uint32_t *d_pcg_iters = mpcvars->d_pcg_iters;
    bool *d_pcg_exit = mpcvars->d_pcg_exit;
    
    
    const uint32_t state_size = grid::NUM_JOINTS*2;
    const uint32_t control_size = grid::NUM_JOINTS;
    const uint32_t knot_points = KNOT_POINTS;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));
    
    


    // copy in xs
    gpuErrchk(cudaMemcpy(d_xs, h_xs, state_size*sizeof(T), cudaMemcpyHostToDevice));
    // get goal pointer
    d_eePos_traj = &d_eePos_full_traj[static_cast<uint32_t>(std::fmod(timestamp, timestep)) * 6];

    gpuErrchk(cudaMemset(d_merit_initial, 0, sizeof(T)));
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    
    // line search things
    const float mu = 10.0f;
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit;
    T alphafinal;
    T delta_merit_iter = 0;
    T delta_merit_total = 0;
    uint32_t line_search_step = 0;
    
    
    cudaStream_t *streams = mpcvars->streams;
    T *h_merit_news = mpcvars->h_merit_news;

    cublasHandle_t handle = mpcvars->handle;

    
    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
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
        (void *)&pcg_max_iter,
        (void *)&pcg_exit_tol
    };
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    
    T drho = 1.0;
    T rho_factor = RHO_FACTOR;
    T rho_max = RHO_MAX;
    T rho_min = rho_reset;
    
    
    d_Ginv_dense = d_G_dense;
    
    
    
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
    
    

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


    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

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





        form_schur<T>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c,
                   d_S, d_Pinv, d_gamma,
                   rho_ptr[0]);
        gpuErrchk(cudaPeekAtLastError());


        if (sqpTimecheck()){ break; }
        

        gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));    

        
        gpuErrchk(cudaPeekAtLastError());        
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
        for(uint32_t p = 0; p < NUM_ALPHAS; p++){
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
            gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs, get_merit_smem_size<T>(state_size, knot_points), streams[p]));
        }
        if (sqpTimecheck()){ break; }

        
        
        cudaMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), cudaMemcpyDeviceToHost);
        if (sqpTimecheck()){ break; }


        line_search_step = 0;
        min_merit = h_merit_initial;
        for(int i = 0; i < 8; i++){
        //     std::cout << h_merit_news[i] << (i == 7 ? "\n" : " ");
            if(h_merit_news[i] < min_merit){
                min_merit = h_merit_news[i];
                line_search_step = i;
            }
        }


        if(min_merit == h_merit_initial){
            // line search failure
            drho = max(drho*rho_factor, rho_factor);
            rho_ptr[0] = max(rho_ptr[0]*drho, rho_min);
            if(rho_ptr[0] > rho_max){
                rho_ptr[0] = rho_reset;
                break; 
            }
            continue;
        }
        // std::cout << "line search accepted\n";
        alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

        drho = min(drho/rho_factor, 1/rho_factor);
        rho_ptr[0] = max(rho_ptr[0]*drho, rho_min);
        


        cublasSaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
        
        gpuErrchk(cudaPeekAtLastError());



        if (sqpTimecheck()){ break; }


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        

        h_merit_initial = min_merit;
    
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(mpcvars->h_xu, d_xu, STEPS_IN_TRAJ*(states_s_controls)*sizeof(T), cudaMemcpyDeviceToHost));


    return 0;
}


