#pragma once
#include <cstdint>
#include "important_numbers.cuh"
#include "iiwa_plant.cuh"

template <typename T>
uint32_t get_merit_smem_size()
{
    const uint32_t state_size = important_numbers::state_size;
    const uint32_t control_size = important_numbers::control_size;
    return sizeof(T) * (4 * state_size + 2 * control_size + max((2 * state_size + control_size), (state_size + gato_plant::forwardDynamics_TempMemSize_Shared())));
}

// cost compute for line search
template <typename T>
__global__
void ls_gato_compute_merit(T *d_xu, 
                           T *d_xu_traj, 
                           T mu, 
                           T dt, 
                           void *d_dynMem_const, 
                           T *d_dz,
                           uint32_t alpha_multiplier, 
                           T *d_merits_out, 
                           T *d_merit_temp)
{
    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t state_size = important_numbers::state_size;
    const uint32_t control_size = important_numbers::control_size;
    const uint32_t knot_points = important_numbers::knot_points;

    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;

    T alpha = -1.0 / (1 << alpha_multiplier);   // alpha sign


    for(unsigned knot = block_id; knot < knot_points; knot += num_blocks){

        T *s_xux_k_traj = s_xux_k + 2*state_size+control_size;
        T *s_temp = s_xux_k_traj + 2*state_size+control_size;
        

        for(int i = thread_id; i < 2*state_size+control_size; i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i] + alpha * d_dz[knot*states_s_controls+i];  
            s_xux_k_traj[i] = d_xu_traj[knot*states_s_controls+i];                            
        }
        block.sync();
        Jk = gato_plant::trackingcost<T>(s_xux_k, s_xux_k_traj, s_temp);
        block.sync();
        if(knot < knot_points-1){
            ck = integratorError<T, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, &s_xux_k[states_s_controls], s_temp, d_dynMem_const, dt, block);
        }
        else{
            ck = 0;
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            d_merit_temp[alpha_multiplier*knot_points+knot] = pointmerit;
        }
    }
    cgrps::this_grid().sync();
    if(block_id == 0){
        glass::reduce<T>(knot_points, &d_merit_temp[alpha_multiplier*knot_points], block);
    
        if(thread_id == 0){
            d_merits_out[alpha_multiplier] = d_merit_temp[alpha_multiplier*knot_points];
        }
    }
}


// shared mem size get_merit_smem_size()
// cost compute for non line search
template <unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void compute_merit(T *d_xu, T *d_xu_traj, T mu, T dt, void *d_dynMem_const, T *d_merit_out)
{
    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t state_size = important_numbers::state_size;
    const uint32_t control_size = important_numbers::control_size;
    const uint32_t knot_points = important_numbers::knot_points;


    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;

    if(block_id == 0 && thread_id == 0){ d_merit_out[0] = static_cast<T>(0); }

    for(unsigned knot = blockIdx.x; knot < knot_points; knot += gridDim.x){
        T *s_xux_k_traj = s_xux_k + 2 * state_size + control_size;
        T *s_temp = s_xux_k_traj + 2 * state_size + control_size;

        for(int i = thread_id; i < 2*state_size+control_size; i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i];  
            s_xux_k_traj[i] = d_xu_traj[knot*states_s_controls+i];                            
        }
        block.sync();
        Jk = gato_plant::trackingcost<T>(s_xux_k, s_xux_k_traj, s_temp);
        block.sync();
        if (knot < knot_points - 1){
            ck = integratorError<T, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, &s_xux_k[states_s_controls], s_temp, d_dynMem_const, dt, block);
        }
        else{
            ck = 0;
        }
        block.sync();
        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            atonicAdd(d_merit_out, pointmerit);
        }
    }
}
