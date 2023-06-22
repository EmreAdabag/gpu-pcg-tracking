#pragma once
#include <cstdint>
#include "utils.cuh"
#include "gpuassert.cuh"
#include "glass.cuh"
#include "iiwa_plant.cuh"
#include "merit.cuh"
#include "oldschur.cuh"
#include "integrator.cuh"


template <typename T>
__global__
void gato_form_schur_jacobi(uint32_t state_size,
                            uint32_t control_size,
                            uint32_t knot_points,
                            T *d_G,
                            T *d_C,
                            T *d_g,
                            T *d_c,
                            T *d_S,
                            T *d_Pinv, 
                            T *d_gamma,
                            T rho,
                            uint32_t num_blocks)
{


    
    extern __shared__ T s_temp[ ];


    for(unsigned blockrow=blockIdx.x; blockrow<knot_points; blockrow+=num_blocks){


        oldschur::gato_form_schur_jacobi_inner(state_size, control_size, knot_points, d_G, d_C, d_g, d_c, d_S, d_Pinv, d_gamma, rho, s_temp, blockrow);
        // gato_form_schur_jacobi_inner(
        //     state_size,
        //     control_size,
        //     knot_points,
        //     d_G,
        //     d_C,
        //     d_g,
        //     d_c,
        //     d_S,
        //     d_Pinv,
        //     d_gamma,
        //     rho,
        //     s_temp,
        //     blockrow
        // );
    
    }
}



/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void gato_form_kkt(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                   T *d_G_dense, T *d_C_dense, T *d_g, T *d_c,
                   void *d_dynMem_const, float timestep,
                   T *d_xu_traj, T *d_xs, T *d_xu)
{

    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_xux_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_xux_traj + 2*state_size + control_size;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;

    
    for(unsigned k = block_id; k < knot_points-1; k += num_blocks){

        glass::copy<T>(2*state_size + control_size, &d_xu[k*states_s_controls], s_xux);
        glass::copy<T>(2*state_size + control_size, &d_xu_traj[k*states_s_controls], s_xux_traj);
        
        __syncthreads();    

        if(k==knot_points-2){          // last block

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_Qkp1 = s_Bk + states_p_controls;
            T *s_qkp1 = s_Qkp1 + states_sq;
            T *s_integrator_error = s_qkp1 + state_size;
            T *s_extra_temp = s_integrator_error + state_size;
            
            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            __syncthreads();
            
            gato_plant::trackingCostGradientAndHessian_lastblock<T>(
                state_size,
                control_size,
                s_xux,
                s_xux_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_Qkp1,
                s_qkp1,
                block_id,
                block
            );
            __syncthreads();

            for(int i = thread_id; i < state_size; i+=num_threads){
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense[(states_sq+controls_sq)*(k+1)]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g[states_s_controls*(k+1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);

        }
        else{                               // not last knot

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_integrator_error = s_Bk + states_p_controls;
            T *s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, 
                                  INTEGRATOR_TYPE, 
                                  ANGLE_WRAP, 
                                  true>
                                 (state_size, control_size,
                                  s_xux,
                                  s_Ak,
                                  s_Bk,
                                  s_integrator_error,
                                  s_extra_temp,
                                  d_dynMem_const,
                                  timestep,
                                  block);
            __syncthreads();
           
            gato_plant::trackingCostGradientAndHessian<T>(state_size,
                                                  control_size,
                                                  s_xux,
                                                  s_xux_traj,
                                                  s_Qk,
                                                  s_qk,
                                                  s_Rk,
                                                  s_rk,
                                                  block_id,
                                                  block);
            __syncthreads();
 
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);
        }
    }
}



template <typename T>
void form_schur(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                T *d_G_dense, T *d_C_dense, T *d_g, T *d_c, 
                T *d_S, T *d_Pinv, T *d_gamma, 
                T rho)
{
    const uint32_t s_temp_size =sizeof(T)*(8 * state_size*state_size+   
                                7 * state_size+ 
                                state_size * control_size+
                                3 * control_size + 2 * control_size * control_size + 3);

    // form Schur, Pinv
    gato_form_schur_jacobi<T><<<knot_points, SCHUR_THREADS, s_temp_size>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_S, d_Pinv, d_gamma, rho, knot_points);// hard coded
    
}


template <typename T>
void compute_dz(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){
    
    oldschur::compute_dz<<<knot_points, DZ_THREADS, sizeof(T)*(2*state_size*state_size+state_size)>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz);
}


void parallel_line_search(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float *d_xs, float *d_xu, float *d_xu_traj, void *d_dynMem_const, float *d_dz, float timestep, float *d_merits_out, float *d_merit_temp)
{

    int alphas = 8;
    
    cudaStream_t streams[alphas];
    for(int st = 0; st < alphas; st++){
        gpuErrchk(cudaStreamCreate(&streams[st]));
    }


    void *ls_merit_kernel = (void *) ls_gato_compute_merit<float>;
    
    float mu = 10.0f;

    for(int p = 0; p < alphas; p++){
        void *kernelArgs[] = {
            (void *)&state_size,
            (void *)&control_size,
            (void *)&knot_points,
            (void *)&d_xs,
            (void *)&d_xu,
            (void *)&d_xu_traj,
            (void *)&mu, 
            (void *)&timestep,
            (void *)&d_dynMem_const,
            (void *)&d_dz,
            (void *)&p,
            (void *)&d_merits_out,
            (void *)&d_merit_temp
        };
        gpuErrchk(cudaLaunchCooperativeKernel(ls_merit_kernel, knot_points, MERIT_THREADS, kernelArgs, get_merit_smem_size<float>(state_size, knot_points), streams[p]));
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    for(int st=0; st < alphas; st++){
        gpuErrchk(cudaStreamDestroy(streams[st]));
    }
}
