#pragma once

#include <cstdint>
#include <cooperative_groups.h>

#include "iiwa_plant.cuh"
#include "integrator.cuh"

template <typename T>
size_t get_merit_smem_size(uint32_t state_size, uint32_t control_size)
{
    return sizeof(T) * ((4 * state_size + 2 * control_size ) + max((2 * state_size + control_size), state_size + gato_plant::forwardDynamics_TempMemSize_Shared()));
}

// cost compute for line search
template <typename T>
__global__
void ls_gato_compute_merit(uint32_t state_size,
                           uint32_t control_size,
                           uint32_t knot_points,
                           T *d_xs,
                           T *d_xu, 
                           T *d_xu_traj, 
                           T mu, 
                           T dt, 
                           void *d_dynMem_const, 
                           T *d_dz,
                           uint32_t alpha_multiplier, 
                           T *d_merits_out, 
                           T *d_merit_temp)
{
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;

    T alpha = -1.0 / (1 << alpha_multiplier);   // alpha sign
    T *s_xux_k_traj = s_xux_k + 2*state_size+control_size;
    T *s_temp = s_xux_k_traj + 2*state_size+control_size;


    for(unsigned knot = block_id; knot < knot_points; knot += num_blocks){

        for(int i = thread_id; i < state_size+(knot < knot_points-1)*(states_s_controls); i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i] + alpha * d_dz[knot*states_s_controls+i];  
            s_xux_k_traj[i] = d_xu_traj[knot*states_s_controls+i];                            
        }
        block.sync();
        
        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points, s_xux_k, s_xux_k_traj, s_temp);
        
        block.sync();
        if(knot < knot_points-1){
            ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_dynMem_const, dt, block);
        }
        else{
            // diff xs vs xs_traj
            for(int i = threadIdx.x; i < state_size; i++){
                s_temp[i] = abs((d_xu[i] + alpha *d_dz[i]) - d_xs[i]);
            }
            block.sync();
            glass::reduce<T>(state_size, s_temp, block);
            block.sync();
            ck = s_temp[0];
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            d_merit_temp[alpha_multiplier*knot_points+knot] = pointmerit;
            // printf("alpha: %f knot: %d reporting merit: %f\n", alpha, knot, pointmerit);
        }
    }
    cooperative_groups::this_grid().sync();
    if(block_id == 0){
        glass::reduce<T>(knot_points, &d_merit_temp[alpha_multiplier*knot_points], block);
    
        if(thread_id == 0){
            d_merits_out[alpha_multiplier] = d_merit_temp[alpha_multiplier*knot_points];
        }
    }
}

// zero merit out
// shared mem size get_merit_smem_size()
// cost compute for non line search
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void compute_merit(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu, T *d_xu_traj, T mu, float dt, void *d_dynMem_const, T *d_merit_out)
{
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t states_s_controls = state_size + control_size;
    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;
    T *s_xux_k_traj = s_xux_k + 2 * state_size + control_size;
    T *s_temp = s_xux_k_traj + 2 * state_size + control_size;

    for(unsigned knot = block_id; knot < knot_points; knot += gridDim.x){

        for(int i = thread_id; i < state_size+(knot < knot_points-1)*(states_s_controls); i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i];  
            s_xux_k_traj[i] = d_xu_traj[knot*states_s_controls+i];                            
        }
        block.sync();
        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points, s_xux_k, s_xux_k_traj, s_temp);


        block.sync();
        if(knot < knot_points-1){
            ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_dynMem_const, dt, block);
        }
        else{
            ck = 0;
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            atomicAdd(d_merit_out, pointmerit);
        }
    }
}
