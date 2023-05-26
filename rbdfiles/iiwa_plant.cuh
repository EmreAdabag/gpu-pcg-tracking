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
#include "iiwa_grid.cuh"

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
	constexpr T COST_Q1() {return static_cast<T>(0.1);}
	// template<class T>
	// __host__ __device__
	// constexpr T COST_Q2() {return static_cast<T>(0.01);}
	// template<class T>
	// __host__ __device__
	// constexpr T COST_QF1() {return static_cast<T>(100);}		
	// template<class T>
	// __host__ __device__
	// constexpr T COST_QF2() {return static_cast<T>(100);}	
	// template<class T>
	// __host__ __device__
	// constexpr T COST_R() {return static_cast<T>(0.0001);}

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
	template <typename T>
	__host__
	void loadInitialState(T *x){
		T q[7] = {PI<T>(),0.25*PI<T>(),0.167*PI<T>(),-0.167*PI<T>(),PI<T>(),0.167*PI<T>(),0.5*PI<T>()};
		for (int i = 0; i < 7; i++){
			x[i] = q[i]; x[i + 7] = 0;
		}
	}

	template <typename T>
	__host__
	void loadInitialControl(T *u){for (int i = 0; i < 7; i++){u[i] = 0;}}

	// goal at q = [-0.5*PI,0.25*PI,0.167*PI,-0.167*PI,0.125*PI,0.167*PI,0.5*PI] with 0 for qd, u, lambda
	template <typename T>
	__host__
	void loadGoalState(T *xg){
		T q[7] = {0,0,-0.25*PI<T>(),0,0.25*PI<T>(),0.5*PI<T>(),0};
		for (int i = 0; i < 7; i++){
			xg[i] = q[i]; xg[i + 7] = static_cast<T>(0);
		}
	}

	template <typename T>
	__device__
	void forwardDynamics(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
		grid::forward_dynamics_device(s_qdd,s_q,s_qd,s_u,s_temp,(grid::robotModel<T>*)d_dynMem_const,GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamics_TempMemSize_Shared(){return grid::FD_DYNAMIC_SHARED_MEM_COUNT;}

	template <typename T>
	__device__
	void forwardDynamicsGradient( T *s_dqdd, T *s_q, T *s_qd, T *s_u, T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
		grid::forward_dynamics_gradient_device<T,true>(s_dqdd, s_q, s_qd, s_u, s_temp, (grid::robotModel<T> *)d_dynMem_const,GRAVITY<T>());
	}

	__host__ __device__
	constexpr unsigned forwardDynamicsGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT_new_version;}

	template <typename T>
	__device__
    void forwardDynamicsAndGradient(T *s_dqdd, T *s_qdd, T *s_q, T *s_qd, T *s_u,  T *s_temp, void *d_dynMem_const, cooperative_groups::thread_block block){
        grid::forward_dynamics_and_gradient_device<T,true>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_temp, (grid::robotModel<T> *)d_dynMem_const,GRAVITY<T>());
    }


	__host__ __device__
	constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared(){return grid::FD_DU_MAX_SHARED_MEM_COUNT_new_version;}


	///TODO: get rid of divergence
	template <typename T>
	__device__
	T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *s_xux, T *s_xux_traj, T *s_temp, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
		

		const uint32_t threadsNeeded = state_size + control_size * (blockIdx.x != knot_points - 1);
		const T Q_cost = static_cast<T>(.1);
		const T R_cost = static_cast<T>(0.0001);

		T err, val;


		for(int i = blockIdx.x; i < threadsNeeded; i += blockDim.x){
			if(i < state_size){
				err = s_xux[i] - s_xux_traj[i];
				val = Q_cost * err * err;
			}
			else{
				err = s_xux[state_size + i];
				val = R_cost * err * err;
			}
			s_temp[i] = val;
		}

		g.sync();
		glass::reduce<T>(threadsNeeded, s_temp, g);
		g.sync();
		return s_temp[0];
	}


	///TODO: costgradientandhessian could be much faster with no divergence
	// not last block
	template <typename T>
	__device__
	void trackingCostGradientAndHessian(uint32_t state_size, 
										uint32_t control_size, 
										T *s_xu, 
										T *s_xu_traj, 
										T *s_Qk, 
										T *s_qk, 
										T *s_Rk, 
										T *s_rk,
										uint32_t block_id, 
										cooperative_groups::thread_group g)
	{	
		const uint32_t threadsNeeded = state_size + control_size;
		const T Q_cost = static_cast<T>(.1);
		const T R_cost = static_cast<T>(0.0001);

		uint32_t offset;
		T err;

		for (int i = g.thread_rank(); i < threadsNeeded; i += g.size()){

			// EMRE->ORDER
			
			if(i < state_size){
				err = s_xu[i] - s_xu_traj[i];
				//gradient
				s_qk[i] = 2 * Q_cost * err;
				//hessian
				for(int j = 0; j < state_size; j++){
					s_Qk[i*state_size+j] = (i == j) ? Q_cost : static_cast<T>(0);
				}
			}
			else{
				err = s_xu[i];
				offset = i - state_size;
				//gradient
				s_rk[offset] = 2 * R_cost * err;
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
							    				  T *s_xux_traj, 
							    				  T *s_Qk, 
							    				  T *s_qk, 
							    				  T *s_Rk, 
							    				  T *s_rk, 
							    				  T *s_Qkp1, 
							    				  T *s_qkp1,
							    				  uint32_t block_id, 
							    				  cooperative_groups::thread_group g)
	{
		unsigned threadsNeeded = 2*state_size + control_size;
		const T Q_cost = static_cast<T>(.1);
		const T R_cost = static_cast<T>(0.0001);

		T err;
		uint32_t offset;

		for (int i = g.thread_rank(); i < threadsNeeded; i += g.size()){

			// EMRE->ORDER


			if (i < state_size){
				err = s_xux[i] - s_xux_traj[i];
				s_qk[i] = 2 * Q_cost * err;
				
				for(int j = 0; j < state_size; j++){
					s_Qk[i*state_size + j] = (i == j) ? Q_cost : static_cast<T>(0);
				}
			}
			else if(i < state_size + control_size){
				err = s_xux[i];
				offset = i - state_size;
				s_rk[offset] = 2 * R_cost * err;

				for(int j = 0; j < control_size; j++){
					s_Rk[offset*control_size + j] = (offset == j) ? R_cost : static_cast<T>(0);
				}
			}
			else{
				offset = i - state_size - control_size;
				err = s_xux[i] - s_xux_traj[i];
				s_qkp1[offset] = 2 * Q_cost * err;

				for(int j = 0; j < state_size; j++){
					s_Qkp1[offset*state_size+j] = (offset == j) ? Q_cost : static_cast<T>(0);
				}

			}
		}
	}

	__host__ __device__
	constexpr unsigned costGradientAndHessian_TempMemSize_Shared(){return 0;}
}

