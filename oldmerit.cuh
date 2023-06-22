#pragma once

#error "old merit defined"

#include "iiwa_plant.cuh"

namespace oldmerit{
// cost compute for line search
template <unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void ls_gato_compute_merit(float *d_xu, float *d_xu_traj, float mu, float dt, void *d_dynMem_const, float *d_dz, int alpha_multiplier, int max_alphas, float *d_merits_out, float *d_merit_temp)
{
    const cgrps::thread_block block = cgrps::this_thread_block();

    extern __shared__ float s_xux_k[];

    float Jk, ck, pointmerit;


    float alpha = -1.0 / (1 << alpha_multiplier);   // alpha sign



    for(unsigned knot = GATO_BLOCK_ID; knot < KNOT_POINTS; knot += GATO_NUM_BLOCKS){
        if(knot==KNOT_POINTS-1){            // last block
            float *s_x0xN = s_xux_k;
            float *s_xs = s_x0xN + 2 *STATE_SIZE;
            float *s_xg = s_xs + STATE_SIZE;
            float *s_temp = s_xg + STATE_SIZE; 
            ///TODO: these can be concurrent
            gato_memcpy<float>(s_xg, d_xg, STATE_SIZE);
            for(int i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
                s_xs[i] = d_xu[i];
                s_x0xN[i] = d_xu[i] + alpha * d_dz[i];                                                                                  
                s_x0xN[STATE_SIZE+i] = d_xu[STATES_S_CONTROLS*(KNOT_POINTS-1)+i] + alpha * d_dz[STATES_S_CONTROLS*(KNOT_POINTS-1)+i];   // alpha sign
            }
            block.sync();

            Jk = gato_plant::cost<float>(STATE_SIZE, KNOT_POINTS, &s_x0xN[STATE_SIZE], s_xg, s_temp, d_dynMem_const, block, true);
            Jk = gato_plant::cost<float>(&s_x0xN[STATE_SIZE], s_xg, s_temp, d_dynMem_const, block, true);
            block.sync();
            if(GATO_LEAD_THREAD){
                float val = 0;
                for(int i = 0; i < STATE_SIZE; i++){
                    val += abs(s_xs[i] - s_x0xN[i]);
                } 
                s_temp[0] = val;
                ck = s_temp[0];
                pointmerit = Jk + mu*ck;
                d_merit_temp[alpha_multiplier*KNOT_POINTS+knot] = pointmerit;
            }
        }
        else{
            float *s_xg = s_xux_k + 2*STATE_SIZE+CONTROL_SIZE;
            float *s_temp = s_xg + STATE_SIZE;
            
            for(int i = GATO_THREAD_ID; i < 2*STATE_SIZE+CONTROL_SIZE; i+=GATO_THREADS_PER_BLOCK){
                s_xux_k[i] = d_xu[knot*STATES_S_CONTROLS+i] + alpha * d_dz[knot*STATES_S_CONTROLS+i];                              
            }
            gato_memcpy<float>(s_xg, d_xg, STATE_SIZE);
            block.sync();
            Jk = gato_plant::cost<float>(s_xux_k, s_xg, s_temp, d_dynMem_const, block, false);
            block.sync();
            ck = integratorError<float, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, &s_xux_k[STATES_S_CONTROLS], s_temp, d_dynMem_const, dt, block);
            block.sync();

            if(GATO_LEAD_THREAD){
                pointmerit = Jk + mu*ck;
                d_merit_temp[alpha_multiplier*KNOT_POINTS+knot] = pointmerit;
            }
        }
    }
    cgrps::this_grid().sync();
    if(GATO_LEAD_BLOCK){
        reducePlus<float>(&d_merit_temp[alpha_multiplier*KNOT_POINTS], KNOT_POINTS, cgrps::this_thread_block());
        if(GATO_LEAD_THREAD){
            d_merits_out[alpha_multiplier] = d_merit_temp[alpha_multiplier*KNOT_POINTS];
        }
    }
}

// shared mem size get_merit_smem_size()
// cost compute for non line search
template <unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void gato_compute_merit(float *d_xu, float *d_xg, float mu, float dt, void *d_dynMem_const, float *d_merit_out)
{
    const cgrps::thread_block block = cgrps::this_thread_block();

    extern __shared__ float s_xux_k[];

    float Jk, ck, pointmerit;

    // if(GATO_LEAD_THREAD && GATO_BLOCK_ID==0){
    //     for(int k=0;k<KNOT_POINTS;k++){
    //         for(int j=0;j<STATES_S_CONTROLS;j++){
    //             printf("%.4f ", d_xu[STATES_S_CONTROLS*k+j]);
    //         }
    //     printf("\n");
    //     }
    // }

    for(unsigned knot = GATO_BLOCK_ID; knot < KNOT_POINTS; knot += GATO_NUM_BLOCKS){
        if(knot < KNOT_POINTS-1){
            float *s_xg = s_xux_k + 2*STATE_SIZE+CONTROL_SIZE;
            float *s_temp = s_xg + STATE_SIZE;
            
            gato_memcpy<float>(s_xux_k, d_xu+knot*STATES_S_CONTROLS, 2*STATE_SIZE+CONTROL_SIZE);
            gato_memcpy<float>(s_xg, d_xg, STATE_SIZE);
            block.sync();

            // make these concurrent
            Jk = gato_plant::cost<float>(s_xux_k, s_xg, s_temp, d_dynMem_const, block, false);
            block.sync();
            ck = integratorError<float, INTEGRATOR_TYPE, ANGLE_WRAP>(s_xux_k, &s_xux_k[STATES_S_CONTROLS], s_temp, d_dynMem_const, dt, block);
            block.sync();

            if(GATO_LEAD_THREAD){
                pointmerit = Jk + mu*ck;
                atomicAdd(d_merit_out, pointmerit);
            }
        }
        else{
            float *s_x0xN = s_xux_k;
            float *s_xg = s_x0xN + 2*STATE_SIZE;
            float *s_temp = s_xg + STATE_SIZE;
            gato_memcpy<float>(s_x0xN, d_xu, STATE_SIZE);
            gato_memcpy<float>(s_x0xN+STATE_SIZE, d_xu+STATES_S_CONTROLS*(KNOT_POINTS-1), STATE_SIZE);
            gato_memcpy<float>(s_xg, d_xg, STATE_SIZE);
            block.sync();
            
            Jk = gato_plant::cost<float>(&s_x0xN[STATE_SIZE], s_xg, s_temp, d_dynMem_const, block, 1);
            block.sync(); 
            if (GATO_LEAD_THREAD){
                pointmerit = Jk;
                atomicAdd(d_merit_out, pointmerit); 
            }
        }
    }

}
}
