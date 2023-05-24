#pragma once
#include "iiwa_plant.cuh"
#include "important_numbers.cuh"
#include "glass.cuh"

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, float dt, cgrps::thread_block block, bool absval = false){
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = GATO_THREAD_ID; ind < state_size/2; ind += GATO_THREADS_PER_BLOCK){
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
void exec_integrator_gradient(T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cgrps::thread_block block){
        
    // and finally A and B
    if (INTEGRATOR_TYPE == 0){
        // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
        // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
        for (unsigned ind = GATO_THREAD_ID; ind < state_size*(state_size + control_size); ind += GATO_THREADS_PER_BLOCK){
            int c = ind / state_size; int r = ind % state_size;
            T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
            val += (r >= state_size/2) * dt * s_dqdd[c*state_size/2 + r - state_size/2]; // second dxd term (non-branching)
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
        for (unsigned ind = GATO_THREAD_ID; ind < state_size*state_size; ind += GATO_THREADS_PER_BLOCK){
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
void exec_integrator(T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block){
    for (unsigned ind = GATO_THREAD_ID; ind < state_size/2; ind += GATO_THREADS_PER_BLOCK){
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
void integratorAndGradient(T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    // first compute qdd and dqdd
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_temp + state_size/2;	
    T *s_extra_temp = s_dqdd + state_size/2*(state_size+control_size);
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    gato_plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    // first compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, &s_xux[STATES_S_CONTROLS], &s_xux[STATES_S_CONTROLS+state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    else{
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    
    // then compute gradient
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(s_Ak, s_Bk, s_dqdd, dt, block);
}


// s_temp of size 3*state_size/2 + DYNAMICS_TEMP
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
T integratorError(T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, float dt, cgrps::thread_block block){

    uint32_t state_size = important_numbers::state_size;

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
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block, true);
    block.sync();

    // finish off forming the error
    glass::reduce<T>(state_size, s_err, block);
    block.sync();
    // if(GATO_LEAD_THREAD){printf("in integratorError with reduced error of [%f]\n",s_err[0]);}
    return s_err[0];
}