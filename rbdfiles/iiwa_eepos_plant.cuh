#pragma once
// // values assumed coming from an instance of grid
// namespace grid{
// 	//
// 	// TODO do I need all of these?
// 	//

// 	const int NUM_JOINTS = 30;
//     const int ID_DYNAMIC_SHARED_MEM_COUNT = 2340;
//     const int MINV_DYNAMIC_SHARED_MEM_COUNT = 9210;
//     const int FD_DYNAMIC_SHARED_MEM_COUNT = 10110;
//     const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 10980;
//     const int ID_DU_MAX_SHARED_MEM_COUNT = 13410;
//     const int FD_DU_MAX_SHARED_MEM_COUNT = 16140;
//     const int SUGGESTED_THREADS = 512;

// 	template <typename T>
//     struct robotModel {
//         T *d_XImats;
//         int *d_topology_helpers;
//     };
// }

#include <stdio.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "iiwa_eepos_grid.cuh"
#include "settings.cuh"

#include "glass.cuh"

// #include <random>
// #define RANDOM_MEAN 0
// #define RANDOM_STDEV 0.001
// std::default_random_engine randEng(time(0)); //seed
// std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

namespace gato_plant{


	const unsigned SUGGESTED_THREADS = grid::SUGGESTED_THREADS;

	template<class T>
	__host__ __device__
	constexpr T PI() {return static_cast<T>(3.14159);}
	template<class T>
	__host__ __device__
	constexpr T GRAVITY() {return static_cast<T>(0.0);}


	template<class T>
	__host__ __device__
	constexpr T COST_Q1() {return static_cast<T>(Q_COST);}
	
	template<class T>
	__host__ __device__
	constexpr T COST_QD() {return static_cast<T>(QD_COST);}

	template<class T>
	__host__ __device__
	constexpr T COST_R() {return static_cast<T>(R_COST);}

	template <typename T>
	void *initializeDynamicsConstMem(){
		grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
		return (void *)d_robotModel;
	}
	template <typename T>
	void freeDynamicsConstMem(void *d_dynMem_const){
		grid::free_robotModel((grid::robotModel<T>*) d_dynMem_const);
	}

	// Start at q = [0,0,-0.25*PI,0,0.25*PI,0.5*PI,0] with small random for qd, u, lambda
	// template <typename T>
	// __host__
	// void loadInitialState(T *x){
	// 	T q[7] = {PI<T>(),0.25*PI<T>(),0.167*PI<T>(),-0.167*PI<T>(),PI<T>(),0.167*PI<T>(),0.5*PI<T>()};
	// 	for (int i = 0; i < 7; i++){
	// 		x[i] = q[i]; x[i + 7] = 0;
	// 	}
	// }

	// template <typename T>
	// __host__
	// void loadInitialControl(T *u){for (int i = 0; i < 7; i++){u[i] = 0;}}

	// // goal at q = [-0.5*PI,0.25*PI,0.167*PI,-0.167*PI,0.125*PI,0.167*PI,0.5*PI] with 0 for qd, u, lambda
	// template <typename T>
	// __host__
	// void loadGoalState(T *xg){
	// 	T q[7] = {0,0,-0.25*PI<T>(),0,0.25*PI<T>(),0.5*PI<T>(),0};
	// 	for (int i = 0; i < 7; i++){
	// 		xg[i] = q[i]; xg[i + 7] = static_cast<T>(0);
	// 	}
	// }

	template <typename T>
	__device__
	void forwardDynamics(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
		grid::forward_dynamics_device<T>(s_qdd,s_q,s_qd,s_u,s_temp,(grid::robotModel<T>*)d_dynMem_const,GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamics_TempMemSize_Shared(){return grid::FD_DYNAMIC_SHARED_MEM_COUNT;}

	template <typename T>
	__device__
	void forwardDynamicsGradient( T *s_dqdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
		grid::forward_dynamics_gradient_device<T,true>(s_dqdd, s_q, s_qd, s_u, s_temp, (grid::robotModel<T> *)d_dynMem_const,GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamicsGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT;}

	template <typename T>
	__device__
    void forwardDynamicsAndGradient(T *s_dqdd, T *s_qdd, T *s_q, T *s_qd, T *s_u,  T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
        grid::forward_dynamics_and_gradient_device<T,true>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_temp, (grid::robotModel<T> *)d_dynMem_const,GRAVITY<T>());
    }


	__host__ __device__
	constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT;}


	///TODO: get rid of divergence
	template <typename T>
	__device__
	T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *s_xu, T *s_eePos_traj, T *s_temp, const grid::robotModel<T> *d_robotModel){
		
        const T Q_cost = COST_Q1<T>();
		const T QD_cost = COST_QD<T>();
		const T R_cost = COST_R<T>();
        
        T err;
        T val = 0;
		
        // QD and R penalty
		const uint32_t threadsNeeded = state_size/2 + control_size * (blockIdx.x != knot_points - 1);
        
		T *s_eePos_cost = s_temp;
        T *s_qd_r_cost = s_eePos_cost + 6;
		T *s_scratch = s_qd_r_cost + threadsNeeded;

        grid::end_effector_positions_device<T>(s_eePos_cost, s_xu, d_robotModel);
        __syncthreads();
        

        for(int i = 0; i < 3; i++){
            err = s_eePos_cost[i] - s_eePos_traj[i];
            s_eePos_cost[i+3] = static_cast<T>(0.5) * Q_COST * err * err;
        }



        for(int i = threadIdx.x; i < threadsNeeded; i += blockDim.x){
			if(i < state_size/2){
                err = s_xu[i + state_size/2];
                val = QD_cost * err * err;
			}
			else{
				err = s_xu[i+state_size];
				val = R_cost * err * err;
			}
			s_qd_r_cost[i] = static_cast<T>(0.5) * val;
		}

		__syncthreads();
		glass::reduce<T>(3 + threadsNeeded, &s_eePos_cost[3]);
		__syncthreads();
		
        return s_eePos_cost[3];
	}


	///TODO: costgradientandhessian could be much faster with no divergence
	// not last block
	template <typename T, bool computeR=true>
	__device__
	void trackingCostGradientAndHessian(uint32_t state_size, 
										uint32_t control_size, 
										T *s_xu, 
										T *s_eePos_traj, 
										T *s_Qk, 
										T *s_qk, 
										T *s_Rk, 
										T *s_rk,
										T *s_temp)
	{	
		const T Q_cost = COST_Q1<T>();
		const T QD_cost = COST_QD<T>();
		const T R_cost = COST_R<T>();

		T *s_eePos = s_temp;
		T *s_eePos_grad = s_eePos + 6;
		// s_end = s_eePos_grad + 6 * state_size/2;

		const uint32_t threads_needed = state_size + control_size*computeR;
		uint32_t offset;
		T err;
		
		grid::end_effector_positions_device<T>(s_eePos, s_xu, d_robotModel);
		grid::end_effector_positions_gradient_device<T>(s_eePos_grad, s_xu, d_robotModel);
        __syncthreads();

		for (int i = threadIdx.x; i < threads_needed; i += blockDim.x){
			
			if(i < state_size){
				// sum x, y, z error
				err = (s_eePos[0] - s_eePos_traj[0]) +
					  (s_eePos[1] - s_eePos_traj[1]) +
					  (s_eePos[2] - s_eePos_traj[2]);

				//gradient
				if (i < state_size / 2){
					s_qk[i] = Q_cost * ( s_eePos_grad[6 * i + 0] + s_eePos_grad[6 * i + 1] + s_eePos_grad[6 * i + 2] ) * err;
				}
				else{
					err = s_xu[i];
					s_qk[i] = QD_cost * err;
				}
				
			}
			else{
				err = s_xu[i];
				offset = i - state_size;
				
				//gradient
				s_rk[offset] = R_cost * err;
			}
		}

		__syncthreads();

		for (int i = threadIdx.x; i < threads_needed; i += blockDim.x){
			if (i < state_size){
				//hessian
				for(int j = 0; j < state_size; j++){
					if(j < state_size / 2){
						s_Qk[i*state_size + j] = s_qk[i] * s_qk[j];
					}
					else{
						s_Qk[i*state_size + j] = (i == j) ? QD_cost : static_cast<T>(0);
					}
				}
			}
			else{
				//hessian
				for(int j = 0; j < control_size; j++){
					s_Rk[offset*control_size+j] = (offset == j) ? R_cost : static_cast<T>(0);
				}
			}
		}
	}

	// last block
	template <typename T>
	__device__
	void trackingCostGradientAndHessian_lastblock(uint32_t state_size, 
							    				  uint32_t control_size, 
							    				  T *s_xux, 
							    				  T *s_eePos_traj, 
							    				  T *s_Qk, 
							    				  T *s_qk, 
							    				  T *s_Rk, 
							    				  T *s_rk, 
							    				  T *s_Qkp1, 
							    				  T *s_qkp1,
							    				  T *s_temp)
	{
		trackingCostGradientAndHessian<T>(state_size, control_size, s_xu, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp);
		__syncthreads();
		trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xu, s_eePos_traj, s_Qkp1, s_qkp1, nullptr, nullptr, s_temp);
	}

	__host__ __device__
	constexpr unsigned costGradientAndHessian_TempMemSize_Shared(){return 0;}
}

