#include "parallel-DDP/PDDP_config_ICRA_2024.cuh"

template <typename T>
__host__
void loadGoalICRA24(GPUVars<T> *gvars, trajVars<T> *tvars, matDimms *dimms, T *d_xGoal, int knot_points){
	gpuErrchk(cudaMemcpy(gvars->d_xGoal, d_xGoal, 6*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
__host__
void loadTrajICRA24(GPUVars<T> *gvars, trajVars<T> *tvars, matDimms *dimms, T *d_xu, int knot_points){
	// unfortunately we need to unzip the traj
	for (int k = 0; k < knot_points; k++){
		T *d_xuk = &d_xu[k*(dimms->ld_x + dimms->ld_u)];
		gpuErrchk(cudaMemcpy(&(tvars->x[k*(dimms->ld_x)]), d_xuk, (dimms->ld_x)*sizeof(T), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&(tvars->u[k*(dimms->ld_x)]), &d_xuk[dimms->ld_u], (dimms->ld_u)*sizeof(T), cudaMemcpyDeviceToHost));
	}
	memcpy(gvars->xActual, tvars->x, STATE_SIZE*sizeof(T));
	for (int i = 0; i < NUM_ALPHA; i++){
		gpuErrchk(cudaMemcpy(gvars->h_d_x[i], tvars->x, (dimms->ld_x)*knot_points*sizeof(T), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(gvars->h_d_u[i], tvars->u, (dimms->ld_u)*knot_points*sizeof(T), cudaMemcpyHostToDevice));
	}

	// memsets from old code re-included
	memset(tvars->KT, 0, (dimms->ld_KT)*DIM_KT_c*knot_points*sizeof(T));
	gpuErrchk(cudaMemsetAsync(gvars->d_du,0,(dimms->ld_du)*knot_points*sizeof(T),(gvars->streams)[0]));
    gpuErrchk(cudaMemsetAsync(gvars->d_err,0,M_BLOCKS_B*sizeof(int),(gvars->streams)[1]));
    gpuErrchk(cudaMemsetAsync(gvars->d_dT,0,NUM_ALPHA*sizeof(T),(gvars->streams)[2]));
    T *ABN = gvars->d_AB + (dimms->ld_AB)*DIM_AB_c*(knot_points-2);
    gpuErrchk(cudaMemsetAsync(ABN,0,(dimms->ld_AB)*DIM_AB_c*sizeof(T),(gvars->streams)[3]));
    // save the shifted into tvars in case all iters fail
    gpuErrchk(cudaMemcpyAsync(gvars->d_x_old, gvars->d_xp, (dimms->ld_x)*knot_points*sizeof(T), cudaMemcpyDeviceToHost, (gvars->streams)[4]));
    gpuErrchk(cudaMemcpyAsync(gvars->d_u_old, gvars->d_up, (dimms->ld_u)*knot_points*sizeof(T), cudaMemcpyDeviceToHost, (gvars->streams)[5]));
    gpuErrchk(cudaMemcpyAsync(gvars->d_KT_old, gvars->d_KT, (dimms->ld_KT)*DIM_KT_c*knot_points*sizeof(T), cudaMemcpyDeviceToHost, (gvars->streams)[6]));
    gpuErrchk(cudaDeviceSynchronize()); // sync and exit
}

template <typename T>
__host__
void storeTrajICRA24(GPUVars<T> *gvars, trajVars<T> *tvars, matDimms *dimms, T *d_xu, int knot_points){
	if (tvars->last_successful_solve != 1){
        gpuErrchk(cudaMemcpyAsync(gvars->h_d_x[*(gvars->alphaIndex)], gvars->d_x_old, (dimms->ld_x)*knot_points*sizeof(T), cudaMemcpyDeviceToDevice, (gvars->streams)[0]));
        gpuErrchk(cudaMemcpyAsync(gvars->h_d_u[*(gvars->alphaIndex)], gvars->d_u_old, (dimms->ld_u)*knot_points*sizeof(T), cudaMemcpyDeviceToDevice, (gvars->streams)[1]));
    }
    gpuErrchk(cudaDeviceSynchronize()); // sync to be done
	gpuErrchk(cudaMemcpyAsync(tvars->x, gvars->h_d_x[*(gvars->alphaIndex)], (dimms->ld_x)*knot_points*sizeof(T), cudaMemcpyDeviceToHost, (gvars->streams)[0]));
    gpuErrchk(cudaMemcpyAsync(tvars->u, gvars->h_d_u[*(gvars->alphaIndex)], (dimms->ld_u)*knot_points*sizeof(T), cudaMemcpyDeviceToHost, (gvars->streams)[1]));
    gpuErrchk(cudaDeviceSynchronize()); // sync to be done
	// unfortunately we need to unzip the traj
	for (int k = 0; k < knot_points; k++){
		T *d_xuk = &d_xu[k*(dimms->ld_x + dimms->ld_u)];
		gpuErrchk(cudaMemcpy(d_xuk, &(tvars->x[k*(dimms->ld_x)]), (dimms->ld_x)*sizeof(T), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(&d_xuk[dimms->ld_u], &(tvars->u[k*(dimms->ld_x)]), (dimms->ld_u)*sizeof(T), cudaMemcpyHostToDevice));
	}
}

template <typename T>
__host__ 
auto runiLQR_MPC_GPU_ICRA24(GPUVars<T> *gvars, trajVars<T> *tvars, matDimms *dimms, algTrace<T> *atrace, costParams<T> *cst, int knot_points){
	struct timespec pddp_solve_start, pddp_solve_end;
	// run solver
	clock_gettime(CLOCK_MONOTONIC, &pddp_solve_start);
	runiLQR_MPC_GPU<T>(tvars,gvars,dimms,atrace,cst,0,0,1,knot_points);
	clock_gettime(CLOCK_MONOTONIC, &pddp_solve_end);

	// package data for exit
	double pddp_solve_time = time_delta_us_timespec(pddp_solve_start, pddp_solve_end);
    return std::make_tuple(pddp_solve_time, (atrace->J).size(), (atrace->J).back());
}