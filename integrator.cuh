#pragma once
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
namespace cgrps = cooperative_groups;
#include "iiwa_plant.cuh"

#include "glass.cuh"


template<typename T>
__host__ __device__ 
T angleWrap(T input){
    const T pi = static_cast<T>(3.14159);
    if(input > pi){input = -(input - pi);}
    if(input < -pi){input = -(input + pi);}
    return input;
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(uint32_t state_size, T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, float dt, cgrps::thread_block block, bool absval = false){
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        }
        else {printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
        // block.sync();
        // if(blockIdx.x==0 && threadIdx.x==0){
        //     printf("position: estimated vs. integrated\n");
        //     for(int i = 0; i < state_size/2; i++){
        //         printf("%f %f\n", s_qkp1[i], s_q[i] + dt*s_qd[i]);
        //     }
        //     printf("\nvelocity: estimated vs. integrated\n");
        //     for(int i = 0; i < state_size/2; i++){
        //         printf("%f %f\n",s_qdkp1[i], s_qd[i] + dt*s_qdd[i]);
        //     }
        //     printf("\nqdd\n");
        //     for(int i = 0; i < state_size / 2; i++){
        //         printf("%f ", s_qdd[i]);
        //     }
        //     printf("\n");
        //     float sum = 0;
        //     for(int i = 0; i < state_size/2; i++){
        //         sum += abs(s_qkp1[i] - (s_q[i] + dt*s_qd[i]));
        //         sum += abs(s_qdkp1[i] - (s_qd[i] + dt*s_qdd[i]));
        //     }
        //     printf("first block constriant violation: %f\n", sum);
        // }
        // block.sync();

        // wrap angles if needed
        if(ANGLE_WRAP){ printf("ANGLE_WRAP!\n");
            new_qkp1 = angleWrap(new_qkp1);
        }

        // then computre error
        if(absval){
            s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
            s_err[ind + state_size/2] = abs(s_qdkp1[ind] - new_qdkp1);    
        }
        else{
            s_err[ind] = s_qkp1[ind] - new_qkp1;
            s_err[ind + state_size/2] = s_qdkp1[ind] - new_qdkp1;
        }
        // printf("err[%f] with new qkp1[%f] vs orig[%f] and new qdkp1[%f] vs orig[%f] with qk[%f] qdk[%f] qddk[%f] and dt[%f]\n",s_err[ind],new_qkp1,s_qkp1[ind],new_qdkp1,s_qdkp1[ind],s_q[ind],s_qd[ind],s_qdd[ind],dt);
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 0>
__device__
void exec_integrator_gradient(uint32_t state_size, uint32_t control_size, T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cgrps::thread_block block){
        
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    // and finally A and B
    if (INTEGRATOR_TYPE == 0){
        // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
        // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*(state_size + control_size); ind += block_dim){
            int c = ind / state_size; int r = ind % state_size;
            T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
            if(r >= state_size/2) { val += dt * s_dqdd[c*state_size/2 + r - state_size/2]; }
            ///TODO: EMRE why didn't this error before?
            // val += (r >= state_size/2) * dt * s_dqdd[c*state_size/2 + r - state_size/2]; // second dxd term (non-branching)
            *dst = val;
        }
    }
    else if (INTEGRATOR_TYPE == 1){
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1 = qk + dt*qdk + dt^2*qddk
        // dxkp1 = [Ix | 0u ] + dt*[[0q, Iqd, 0u] + dt*dqdd
        //                                             dqdd]
        // Ak = I + dt * [[0,I] + dt*dqdd/dx; dqdd/dx]
        // Bk = [dt*dqdd/du; dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
            int c = ind / state_size; int r = ind % state_size; int rdqdd = r % (state_size/2);
            T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
            s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - state_size/2)) +
                        dt * s_dqdd[c*state_size/2 + rdqdd] * dtVal;
            if(c < control_size){
                s_Bk[ind] = dt * s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd] * dtVal;
            }
        }
    }
    else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(uint32_t state_size, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    for (unsigned ind = thread_id; ind < state_size/2; ind += block_dim){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){
            s_qkp1[ind] = angleWrap(s_qkp1[ind]);
        }
    }
}

// s_temp of size state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){

    
    // first compute qdd and dqdd
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_qdd + state_size/2;	
    T *s_extra_temp = s_dqdd + state_size/2*(state_size+control_size);
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    gato_plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    // first compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    else{
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    
    // then compute gradient
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, block);
}


// s_temp of size 3*state_size/2 + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, float dt, cgrps::thread_block block){

    // first compute qdd
    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    // if(blockIdx.x == 0 && threadIdx.x==0){
    //     printf("\n");
    //     for(int i = 0; i < state_size/2; i++){
    //         printf("%f ", s_qdd[i]);
    //     }
    //     printf("\n");
    // }
    // block.sync();
    // then apply the integrator and compute error
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block, true);
    block.sync();

    // finish off forming the error
    glass::reduce<T>(state_size, s_err);
    block.sync();
    // if(GATO_LEAD_THREAD){printf("in integratorError with reduced error of [%f]\n",s_err[0]);}
    return s_err[0];
}



template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void integrator(uint32_t state_size, T *s_xkp1, T *s_xuk, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    // first compute qdd
    T *s_q = s_xuk; 					T *s_qd = s_q + state_size/2; 				T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					T *s_extra_temp = s_qdd + state_size/2;
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block);
}




template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_xkp1, T *d_xuk, void *d_dynMem_const, T dt){
    extern __shared__ T s_smem[];
    T *s_xkp1 = s_smem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    cgrps::grid_group grid = cgrps::this_grid();
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        s_xuk[ind] = d_xuk[ind];
    }

    block.sync();
    integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_xkp1[ind] = s_xkp1[ind];
    }
}

// We take start state from h_xs, and control input from h_xu, and update h_xs
template <typename T>
void integrator_host(uint32_t state_size, uint32_t control_size, T *d_xs, T *d_xu, void *d_dynMem_const, T dt){
    // T *d_xu;
    // T *d_xs_new;
    // gpuErrchk(cudaMalloc(&d_xu, xu_size));
    // gpuErrchk(cudaMalloc(&d_xs_new, xs_size));

    // gpuErrchk(cudaMemcpy(d_xu, h_xs, state_size*sizeof(T), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_xu + state_size, h_xu + state_size, control_size*sizeof(T), cudaMemcpyHostToDevice));
    //TODO: needs sync?

    const size_t integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
    //TODO: one block one thread? Why?
    integrator_kernel<T><<<1,1, integrator_kernel_smem_size>>>(state_size, control_size, d_xs, d_xu, d_dynMem_const, dt);

    //TODO: needs sync?
    // gpuErrchk(cudaMemcpy(h_xs, d_xs_new, xs_size, cudaMemcpyDeviceToHost));

    // gpuErrchk(cudaFree(d_xu));
    // gpuErrchk(cudaFree(d_xs_new));
}
/*
//TODO - xs_new, xs_end can be done together concurrently
template <typename T>
void shift(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs, T *d_xu, void *d_dynMem_const, float traj_timestep){
    
    const uint32_t states_s_controls = state_size + control_size;
    uint32_t stepsize;

    ///TODO: this is stupid, don't use in timed code
    for (int knot = 0; knot < knot_points-1; knot++){
        stepsize = (state_size+(knot<knot_points-2)*control_size);
        gpuErrchk(cudaMemcpy(&d_xu[knot*states_s_controls], &d_xu[knot*states_s_controls+stepsize], stepsize*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));

    // Fill in the end
    unsigned last_step = (knot_points-1)*states_s_controls;
    unsigned integrator_step = (knot_points-2)*states_s_controls;
    integrator_host<T>(state_size, control_size, d_xu + last_step, d_xu + integrator_step, d_dynMem_const, traj_timestep);
    
}
*/

template <typename T>
void just_shift(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu){
    for (int knot = 0; knot < knot_points-1; knot++){
        uint32_t stepsize = (state_size+(knot<knot_points-2)*control_size);
        gpuErrchk(cudaMemcpy(&d_xu[knot*(state_size+control_size)], &d_xu[knot*(state_size+control_size)+stepsize], stepsize*sizeof(T), cudaMemcpyDeviceToDevice));
    }
}
/*
template <typename T>
void integrator_shift(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs, T *d_xu, void *d_dynMem_const, float traj_timestep, float time_offset, float simulation_time){

    float time_offset_left = time_offset;
    float simulation_time_left = simulation_time;
    float simulation_iter_time;

    if (time_offset != 0){
        // STARTS AT 1 KNOT IN
        integrator_host<T>(state_size, control_size, d_xs, &d_xu[state_size+control_size], d_dynMem_const, traj_timestep);
        shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);
        shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);
    }
    else{
        integrator_host<T>(state_size, control_size, d_xs, d_xu, d_dynMem_const, traj_timestep);
        shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);
    }


    // shifts over unused timesteps
    // while (time_offset_left > traj_timestep){
    //     shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);
    //     time_offset_left -= traj_timestep;
    // }
    
    // // simulate half-used timestep
    // integrator_host<T>(state_size, control_size, d_xs, d_xu, d_dynMem_const, traj_timestep-time_offset_left);
    // shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);

    // simulation_time_left -= traj_timestep-time_offset_left;


    // while(simulation_time_left > 0){
        
    //     simulation_iter_time = fminf(simulation_time_left, traj_timestep);

    //     integrator_host<T>(state_size, control_size, d_xs, d_xu, d_dynMem_const, simulation_iter_time);
        
        
    //     if (simulation_iter_time == traj_timestep){
    //         shift<T>(state_size, control_size, knot_points, d_xs, d_xu, d_dynMem_const, traj_timestep);
    //     }

    //     simulation_time_left -= simulation_iter_time;
    // }
}
*/

template <typename T>
__global__
void simple_integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_x, T *d_u, void *d_dynMem_const, float dt){

    extern __shared__ T s_mem[];
    T *s_xkp1 = s_mem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    cgrps::grid_group grid = cgrps::this_grid();
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        if(ind < state_size){
            s_xuk[ind] = d_x[ind];
        }
        else{
            s_xuk[ind] = d_u[ind-state_size];
        }
    }

    block.sync();
    integrator<T,0,0>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_x[ind] = s_xkp1[ind];
    }
}

///TODO: edge case where all knots are used, shift at some point
template <typename T>
void simple_integrator(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs, T *d_xu, void *d_dynMem_const, float timestep, float time_offset, float simulation_time){

    const float epsilon = .0001; // this is to prevent float error from messing up floor
    const size_t simple_integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
    const uint32_t states_s_controls = state_size + control_size;
    float time_since_traj_start = time_offset;
    float simulation_time_left = simulation_time;
    uint32_t prev_knot = static_cast<uint32_t>(floorf(time_since_traj_start / timestep + epsilon));
    float simulation_iter_time;

    simulation_iter_time = std::min(simulation_time, std::min(timestep, timestep - time_offset));  //EMRE verify this

    // std::cout << "simulation iter time: " << simulation_iter_time << " prev knot: " << prev_knot << std::endl;

    // float h_temp[21];
    // gpuErrchk(cudaMemcpy(h_temp, d_xs, 14*sizeof(float), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(&h_temp[14], &d_xu[prev_knot*states_s_controls + 14], 7*sizeof(float), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 14; i++){
    //     std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_temp[i] << " ";
    // }
    // std::cout <<"\n";
    // for (int i = 0; i < 7; i++){
    //     std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_temp[14 + i] << " ";
    // }
    // std::cout <<"\n\n";


    // simulate half-used timestep
    simple_integrator_kernel<T><<< 1, 1, simple_integrator_kernel_smem_size>>>(state_size, control_size, d_xs, &d_xu[prev_knot*states_s_controls + state_size], d_dynMem_const, simulation_iter_time);

    simulation_time_left -= simulation_iter_time;
    time_since_traj_start += simulation_iter_time;

    // simulate remaining full timesteps and partial timestep at end (if applicable)
    while(simulation_time_left > 0.001){
        
        simulation_iter_time = std::min(simulation_time_left, timestep);
        
        prev_knot = static_cast<uint32_t>(floorf(time_since_traj_start / timestep + epsilon));

        simple_integrator_kernel<T><<< 1, 1, simple_integrator_kernel_smem_size>>>(state_size, control_size, d_xs, &d_xu[prev_knot*states_s_controls + state_size], d_dynMem_const, simulation_iter_time);

        simulation_time_left -= simulation_iter_time;
        time_since_traj_start += simulation_iter_time;
    }
}


template <typename T>
void simple_simulate(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs, T *d_xu, void *d_dynMem_const, float timestep, float time_offset, float sim_time){


    const float sim_step_time = .002;
    const size_t simple_integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());
    const uint32_t states_s_controls = state_size + control_size;
    uint32_t control_offset;
    T *control;


    uint32_t sim_steps_needed = static_cast<uint32_t>(sim_time / sim_step_time);



    for(int step = 0; step < sim_steps_needed; step++){
        control_offset = static_cast<uint32_t>((time_offset + step * sim_step_time) / timestep);
        control = &d_xu[control_offset * states_s_controls + state_size];

        simple_integrator_kernel<T><<<1,32,simple_integrator_kernel_smem_size>>>(state_size, control_size, d_xs, control, d_dynMem_const, sim_step_time);
    }

}