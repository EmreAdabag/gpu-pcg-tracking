#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"

/*******************************************************************************
 *                           Private Functions                                 *
 *******************************************************************************/

template <typename T>
__device__
void gato_form_schur_jacobi_inner(
    uint32_t state_size,
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
    T *s_temp, 
    unsigned blockrow)
{
    
    const uint32_t states_sq = states_sq;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    
    //  SPACE ALLOCATION IN SHARED MEM
    //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
    //     s^2      s^2         s^2         s
    T *s_phi_k = s_temp; 	                            	    // phi_k        states^2
    T *s_theta_k = s_phi_k + states_sq; 			            // theta_k      states^2
    T *s_thetaInv_k = s_theta_k + states_sq; 			        // thetaInv_k   states^2
    T *s_gamma_k = s_thetaInv_k + states_sq;                       // gamma_k      states
    T *s_end_main = s_gamma_k + state_size;                               

    if(blockrow==0){

        //  LEADING BLOCK GOAL SHARED MEMORY STATE
        //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
        //              s^2   s^2     s   s^2   s^2     s      ? 
    
        T *s_QN = s_end_main;
        T *s_QN_i = s_QN + states_sq;
        T *s_qN = s_QN_i + states_sq;
        T *s_Q0 = s_qN + state_size;
        T *s_Q0_i = s_Q0 + states_sq;
        T *s_q0 = s_Q0_i + states_sq;
        T *s_end = s_q0 + state_size;

        // scratch space
        T *s_R_not_needed = s_end;
        T *s_r_not_needed = s_R_not_needed + control_size * control_size;
        T *s_extra_temp = s_r_not_needed + control_size * control_size;

        __syncthreads();//----------------------------------------------------------------

        glass::copy<T>(states_sq, s_Q0, d_G);
        glass::copy<T>(states_sq, &d_G[(knot_points-1)*(states_sq+controls_sq)], s_QN);
        glass::copy<T>(state_size, s_q0, d_g);
        glass::copy<T>(state_size, &d_g[(knot_points-1)*(states_s_controls)], s_qN);

        __syncthreads();//----------------------------------------------------------------

        glass::addI(state_size, s_Q0, rho, block);
        glass::addI(state_size, s_QN, rho, block);

        __syncthreads();//----------------------------------------------------------------
        
        // SHARED MEMORY STATE
        // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
        

        // save -Q_0 in PhiInv spot 00
        store_block_bd<T, state_size, knot_points>(
            s_Q0,                       // src     
            d_Pinv,                   // dst         
            1,                          // col
            blockrow,                    // blockrow
            -1                          //  multiplier
        );
        __syncthreads();//----------------------------------------------------------------


        // invert Q_N, Q_0
        loadIdentity<T, state_size,state_size>(s_Q0_i, s_QN_i);
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<T, state_size,state_size,state_size>(s_Q0, s_QN, s_extra_temp);
        
        __syncthreads();//----------------------------------------------------------------


        // if(PRINT_THREAD){
        //     printf("Q0Inv\n");
        //     printMat<Tstate_size,state_size>(s_Q0_i,state_size);
        //     printf("QNInv\n");
        //     printMat<Tstate_size,state_size>(s_QN_i,state_size);
        //     printf("theta\n");
        //     printMat<Tstate_size,state_size>(s_theta_k,state_size);
        //     printf("thetaInv\n");
        //     printMat<Tstate_size,state_size>(s_thetaInv_k,state_size);
        //     printf("\n");
        // }
        __syncthreads();//----------------------------------------------------------------

        // SHARED MEMORY STATE
        // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
        

        // compute gamma
        mat_vec_prod<T, state_size, state_size>(
            s_Q0_i,                                    
            s_q0,                                       
            s_gamma_k 
        );
        __syncthreads();//----------------------------------------------------------------
        

        // save -Q0_i in spot 00 in S
        store_block_bd<T, state_size, knot_points>(
            s_Q0_i,                         // src             
            d_S,                            // dst              
            1,                              // col   
            blockrow,                        // blockrow         
            -1                              //  multiplier   
        );
        __syncthreads();//----------------------------------------------------------------


        // compute Q0^{-1}q0
        mat_vec_prod<T, state_size, state_size>(
            s_Q0_i,
            s_q0,
            s_Q0
        );
        __syncthreads();//----------------------------------------------------------------


        // SHARED MEMORY STATE
        // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


        // save -Q0^{-1}q0 in spot 0 in gamma
        for(unsigned ind = GATO_THREAD_ID; ind < state_size; ind += GATO_THREADS_PER_BLOCK){
            d_gamma[ind] = -s_Q0[ind];
        }
        __syncthreads();//----------------------------------------------------------------

    }
    else{                       // blockrow!=LEAD_BLOCK


        const unsigned C_set_size = states_sq+states_p_controls;
        const unsigned G_set_size = states_sq+controls_sq;

        //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
        //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
        //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

        T *s_Ak = s_end_main; 								
        T *s_Bk = s_Ak +        states_sq;
        T *s_Qk = s_Bk +        states_p_controls; 	
        T *s_Qk_i = s_Qk +      states_sq;	
        T *s_Qkp1 = s_Qk_i +    states_sq;
        T *s_Qkp1_i = s_Qkp1 +  states_sq;
        T *s_Rk = s_Qkp1_i +    states_sq;
        T *s_Rk_i = s_Rk +      controls_sq;
        T *s_qk = s_Rk_i +      controls_sq; 	
        T *s_qkp1 = s_qk +      state_size; 			
        T *s_rk = s_qkp1 +      state_size;
        T *s_end = s_rk +       control_size;
        
        // scratch
        T *s_extra_temp = s_end;
        

        // if(PRINT_THREAD){
        //     printf("xk\n");
        //     printMat<T1,state_size>(s_xux,1);
        //     printf("uk\n");
        //     printMat<T1,control_size>(&s_xux[state_size],1);
        //     printf("xkp1\n");
        //     printMat<T1,state_size>(&s_xux[states_s_controls],1);
        //     printf("\n");
        // }

        __syncthreads();//----------------------------------------------------------------

        gato_memcpy<T>(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        states_sq);
        gato_memcpy<T>(s_Bk,   d_C+      (blockrow-1)*C_set_size+states_sq,              states_p_controls);
        gato_memcpy<T>(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        states_sq);
        gato_memcpy<T>(s_Qkp1, d_G+    (blockrow*G_set_size),                          states_sq);
        gato_memcpy<T>(s_Rk,   d_G+      ((blockrow-1)*G_set_size+states_sq),            controls_sq);
        gato_memcpy<T>(s_qk,   d_g+      (blockrow-1)*(states_s_controls),               state_size);
        gato_memcpy<T>(s_qkp1, d_g+    (blockrow)*(states_s_controls),                 state_size);
        gato_memcpy<T>(s_rk,   d_g+      ((blockrow-1)*(states_s_controls)+state_size),  control_size);

        __syncthreads();//----------------------------------------------------------------

        add_identity(s_Qk, state_size, rho);
        add_identity(s_Qkp1, state_size, rho);
        add_identity(s_Rk, control_size, rho);

#if DEBUG_MODE    
        if(thread_id==1 && GATO_THREAD_ID==0){
            printf("Ak\n");
            printMat<T,state_size,state_size>(s_Ak,state_size);
            printf("Bk\n");
            printMat<T,state_size,control_size>(s_Bk,state_size);
            printf("Qk\n");
            printMat<T,state_size,state_size>(s_Qk,state_size);
            printf("Rk\n");
            printMat<T,control_size,control_size>(s_Rk,control_size);
            printf("qk\n");
            printMat<T,state_size, 1>(s_qk,1);
            printf("rk\n");
            printMat<T,control_size, 1>(s_rk,1);
            printf("Qkp1\n");
            printMat<T,state_size,state_size>(s_Qkp1,state_size);
            printf("qkp1\n");
            printMat<T,state_size, 1>(s_qkp1,1);
            printf("integrator error\n");
        }
        __syncthreads();//----------------------------------------------------------------
#endif /* #if DEBUG_MODE */
        
        // Invert Q, Qp1, R 
        loadIdentity<T, state_size,state_size,control_size>(
            s_Qk_i, 
            s_Qkp1_i, 
            s_Rk_i
        );
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<T, state_size,state_size,control_size,state_size>(
            s_Qk, 
            s_Qkp1, 
            s_Rk, 
            s_extra_temp
        );
        __syncthreads();//----------------------------------------------------------------

        // save Qk_i into G (now Ginv) for calculating dz
        gato_memcpy<T>(
            d_G+(blockrow-1)*G_set_size,
            s_Qk_i,
            states_sq
        );

        // save Rk_i into G (now Ginv) for calculating dz
        gato_memcpy<T>( 
            d_G+(blockrow-1)*G_set_size+states_sq,
            s_Rk_i,
            controls_sq
        );

        if(blockrow==knot_points-1){
            // save Qkp1_i into G (now Ginv) for calculating dz
            gato_memcpy<T>(
                d_G+(blockrow)*G_set_size,
                s_Qkp1_i,
                states_sq
            );
        }
        __syncthreads();//----------------------------------------------------------------

#if DEBUG_MODE
        if(blockrow==1&&GATO_THREAD_ID==0){
            printf("Qk\n");
            printMat<T, state_size,state_size>(s_Qk_i,state_size);
            printf("RkInv\n");
            printMat<T,control_size,control_size>(s_Rk_i,control_size);
            printf("Qkp1Inv\n");
            printMat<T, state_size,state_size>(s_Qkp1_i,state_size);
            printf("\n");
        }
        __syncthreads();//----------------------------------------------------------------
#endif /* #if DEBUG_MODE */


        // Compute -AQ^{-1} in phi
        mat_mat_prod(
            s_phi_k,
            s_Ak,
            s_Qk_i,
            state_size, 
            state_size, 
            state_size, 
            state_size
        );
        // for(int i = GATO_THREAD_ID; i < states_sq; i++){
        //     s_phi_k[i] *= -1;
        // }

        __syncthreads();//----------------------------------------------------------------

        // Compute -BR^{-1} in Qkp1
        mat_mat_prod(
            s_Qkp1,
            s_Bk,
            s_Rk_i,
            state_size,
            control_size,
            control_size,
            control_size
        );

        __syncthreads();//----------------------------------------------------------------

        // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
        mat_vec_prod<T, state_size, state_size>(
            s_Qkp1_i,
            s_qkp1,
            s_gamma_k
        );
        for(unsigned i = GATO_THREAD_ID; i < state_size; i += GATO_THREADS_PER_BLOCK){
            s_gamma_k[i] -= d_c[(blockrow*state_size)+i];
        }
        __syncthreads();//----------------------------------------------------------------

        // compute -AQ^{-1}q for gamma         temp storage in extra temp
        mat_vec_prod<T, state_size, state_size>(
            s_phi_k,
            s_qk,
            s_extra_temp
        );
        

        __syncthreads();//----------------------------------------------------------------
        
        // compute -BR^{-1}r for gamma           temp storage in extra temp + states
        mat_vec_prod<T, state_size, control_size>(
            s_Qkp1,
            s_rk,
            s_extra_temp + state_size
        );

        __syncthreads();//----------------------------------------------------------------
        
        // gamma = yeah...
        for(unsigned i = GATO_THREAD_ID; i < state_size; i += GATO_THREADS_PER_BLOCK){
            s_gamma_k[i] += s_extra_temp[state_size + i] + s_extra_temp[i]; 
        }
        __syncthreads();//----------------------------------------------------------------

        // compute AQ^{-1}AT   -   Qkp1^{-1} for theta
        mat_mat_prod(
            s_theta_k,
            s_phi_k,
            s_Ak,
            state_size,
            state_size,
            state_size,
            state_size,
            true
        );

        __syncthreads();//----------------------------------------------------------------

#if DEBUG_MODE
        if(blockrow==1&&GATO_THREAD_ID==0){
            printf("this is the A thing\n");
            printMat<T, state_size, state_size>(s_theta_k, 234);
        }
#endif /* #if DEBUG_MODE */

        for(unsigned i = GATO_THREAD_ID; i < states_sq; i += GATO_THREADS_PER_BLOCK){
            s_theta_k[i] += s_Qkp1_i[i];
        }
        
        __syncthreads();//----------------------------------------------------------------

        // compute BR^{-1}BT for theta            temp storage in QKp1{-1}
        mat_mat_prod(
            s_Qkp1_i,
            s_Qkp1,
            s_Bk,
            state_size,
            control_size,
            state_size,
            control_size,
            true
        );

        __syncthreads();//----------------------------------------------------------------

        for(unsigned i = GATO_THREAD_ID; i < states_sq; i += GATO_THREADS_PER_BLOCK){
            s_theta_k[i] += s_Qkp1_i[i];
        }
        __syncthreads();//----------------------------------------------------------------

        // save phi_k into left off-diagonal of S, 
        store_block_bd<T, state_size, knot_points>(
            s_phi_k,                        // src             
            d_S,                            // dst             
            0,                              // col
            blockrow,                        // blockrow    
            -1
        );
        __syncthreads();//----------------------------------------------------------------

        // save -s_theta_k main diagonal S
        store_block_bd<T, state_size, knot_points>(
            s_theta_k,                                               
            d_S,                                                 
            1,                                               
            blockrow,
            -1                                             
        );          
        __syncthreads();//----------------------------------------------------------------

#if BLOCK_J_PRECON || SS_PRECON
    // invert theta
    loadIdentity<T,state_size>(s_thetaInv_k);
    __syncthreads();//----------------------------------------------------------------
    invertMatrix<T,state_size>(s_theta_k, s_extra_temp);
    __syncthreads();//----------------------------------------------------------------


    // save thetaInv_k main diagonal PhiInv
    store_block_bd<T, state_size, knot_points>(
        s_thetaInv_k, 
        d_Pinv,
        1,
        blockrow,
        -1
    );
#else /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */

    // save 1 / diagonal to PhiInv
    for(int i = GATO_THREAD_ID; i < state_size; i+=GATO_THREADS_PER_BLOCK){
        d_Pinv[blockrow*(3*states_sq)+states_sq+i*state_size+i]= 1 / d_S[blockrow*(3*states_sq)+states_sq+i*state_size+i]; 
    }
#endif /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */
    

    __syncthreads();//----------------------------------------------------------------

    // save gamma_k in gamma
    for(unsigned ind = GATO_THREAD_ID; ind < state_size; ind += GATO_THREADS_PER_BLOCK){
        unsigned offset = (blockrow)*state_size + ind;
        d_gamma[offset] = s_gamma_k[ind]*-1;
    }

    __syncthreads();//----------------------------------------------------------------

    //transpose phi_k
    loadIdentity<T,state_size>(s_Ak);
    __syncthreads();//----------------------------------------------------------------
    mat_mat_prod(s_Qkp1,s_Ak,s_phi_k,state_size,state_size,state_size,state_size,true);
    __syncthreads();//----------------------------------------------------------------

    // save phi_k_T into right off-diagonal of S,
    store_block_bd<T, state_size, knot_points>(
        s_Qkp1,                        // src             
        d_S,                            // dst             
        2,                              // col
        blockrow-1,                      // blockrow    
        -1
    );

    __syncthreads();//----------------------------------------------------------------
    }

}

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

    const uint32_t s_temp_size =    8 * state_size*state_size+   
                                    7 * state_size+ 
                                    state_size * control_size+
                                     3 * control_size + 2 * control_size * control_size + 3;
    
    __shared__ T s_temp[ s_temp_size ];


    for(unsigned blockrow=thread_id; blockrow<knot_points; blockrow+=num_blocks){

        gato_form_schur_jacobi_inner(
            state_size,
            control_size,
            knot_points,
            d_G,
            d_C,
            d_g,
            d_c,
            d_S,
            d_Pinv,
            d_gamma,
            rho,
            s_temp,
            blockrow
        );
    
    }
}


__device__
void gato_form_ss_inner(T *d_S, T *d_Pinv, T *d_gamma, T *s_temp, unsigned blockrow){
    
    //  STATE OF DEVICE MEM
    //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal
    //  Phi:    -Q0 in spot 00, theta_invk main diagonal
    //  gamma:  -Q0_i*q0 spot 0, gammak


    // GOAL SPACE ALLOCATION IN SHARED MEM
    // s_temp  = | phi_k_T | phi_k | phi_kp1 | thetaInv_k | thetaInv_kp1 | thetaInv_km1 | PhiInv_R | PhiInv_L | scratch
    T *s_phi_k = s_temp;
    T *s_phi_kp1_T = s_phi_k + states_sq;
    T *s_thetaInv_k = s_phi_kp1_T + states_sq;
    T *s_thetaInv_km1 = s_thetaInv_k + states_sq;
    T *s_thetaInv_kp1 = s_thetaInv_km1 + states_sq;
    T *s_PhiInv_k_R = s_thetaInv_kp1 + states_sq;
    T *s_PhiInv_k_L = s_PhiInv_k_R + states_sq;
    T *s_scratch = s_PhiInv_k_L + states_sq;

    const unsigned lastrow = knot_points - 1;

    // load phi_kp1_T
    if(blockrow!=lastrow){
        load_block_bd<T, state_size, knot_points>(
            d_S,                // src
            s_phi_kp1_T,        // dst
            0,                  // block column (0, 1, or 2)
            blockrow+1,          // block row
            true                // transpose
        );
    }
    
    __syncthreads();//----------------------------------------------------------------

    // load phi_k
    if(blockrow!=0){
        load_block_bd<T, state_size, knot_points>(
            d_S,
            s_phi_k,
            0,
            blockrow
        );
    }
    
    __syncthreads();//----------------------------------------------------------------


    // load thetaInv_k
    load_block_bd<T, state_size, knot_points>(
        d_Pinv,
        s_thetaInv_k,
        1,
        blockrow
    );

    __syncthreads();//----------------------------------------------------------------

    // load thetaInv_km1
    if(blockrow!=0){
        load_block_bd<T, state_size, knot_points>(
            d_Pinv,
            s_thetaInv_km1,
            1,
            blockrow-1
        );
    }

    __syncthreads();//----------------------------------------------------------------

    // load thetaInv_kp1
    if(blockrow!=lastrow){
        load_block_bd<T, state_size, knot_points>(
            d_Pinv,
            s_thetaInv_kp1,
            1,
            blockrow+1
        );
    }
    

    __syncthreads();//----------------------------------------------------------------

    if(blockrow!=0){

        // compute left off diag    
        mat_mat_prod(
            s_scratch,
            s_thetaInv_k,
            s_phi_k,
            state_size,
            state_size,
            state_size,
            state_size                           
        );
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod(
            s_PhiInv_k_L,
            s_scratch,
            s_thetaInv_km1,
            state_size,
            state_size,
            state_size,
            state_size
        );
        __syncthreads();//----------------------------------------------------------------

        // store left diagonal in Phi
        store_block_bd<T, state_size, knot_points>(
            s_PhiInv_k_L, 
            d_Pinv,
            0,
            blockrow,
            -1
        );
        __syncthreads();//----------------------------------------------------------------
    }


    if(blockrow!=lastrow){

        // calculate Phi right diag
        mat_mat_prod(
            s_scratch,
            s_thetaInv_k,
            s_phi_kp1_T,
            state_size,                           
            state_size,                           
            state_size,                           
            state_size                           
        );
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod(
            s_PhiInv_k_R,
            s_scratch,
            s_thetaInv_kp1,
            state_size,
            state_size,
            state_size,
            state_size
        );
        __syncthreads();//----------------------------------------------------------------

        // store Phi right diag
        store_block_bd<T, state_size, knot_points>(
            s_PhiInv_k_R, 
            d_Pinv,
            2,
            blockrow,
            -1
        );

    }
}


__global__
void gato_form_ss(T *d_S, T *d_Pinv, T *d_gamma){
    
    const unsigned s_temp_size = 9 * states_sq;
    // 8 * states^2
    // scratch space = states^2

    __shared__ T s_temp[ s_temp_size ];

    for(unsigned ind=thread_id; ind<knot_points; ind+=GATO_NUM_BLOCKS){
        gato_form_ss_inner(
            d_S,
            d_Pinv,
            d_gamma,
            s_temp,
            ind
        );
    }
}

__device__
void gato_compute_dz_inner(T *d_Ginv_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz, T *s_mem, int blockrow){

    const unsigned set = blockrow/2;
    
    if(blockrow%2){ // control row
        // shared mem config
        //    Rkinv |   BkT
        //      C^2  |  S*C

        T *s_Rk_i = s_mem;
        T *s_BkT = s_Rk_i + controls_sq;
        T *s_scratch = s_BkT + states_p_controls;

        // load Rkinv from G
        gato_memcpy(s_Rk_i, 
                    d_Ginv_dense+set*(states_sq+controls_sq)+states_sq, 
                    controls_sq);

        // load Bk from C
        gato_memcpy(s_BkT,
                    d_C_dense+set*(states_sq+states_p_controls)+states_sq,
                    states_p_controls);

        __syncthreads();

        // // compute BkT*lkp1
        gato_ATx(s_scratch,
                 s_BkT,
                 d_lambda+(set+1)*state_size,
                 state_size,
                 control_size);
        __syncthreads();

        // subtract from rk
        gato_vec_dif(s_scratch,
                     d_g_val+set*(states_s_controls)+state_size,
                     s_scratch,
                     control_size);
        __syncthreads();

        // multiply Rk_i*scratch in scratch + C
        mat_vec_prod<T, control_size, control_size>(s_Rk_i,
                                                        s_scratch,
                                                        s_scratch+control_size);
        __syncthreads();
        
        // store in d_dz
        gato_memcpy<T>(d_dz+set*(states_s_controls)+state_size,
                           s_scratch+control_size,
                           control_size);

    }
    else{   // state row

        T *s_Qk_i = s_mem;
        T *s_AkT = s_Qk_i + states_sq;
        T *s_scratch = s_AkT + states_sq;
        
        // shared mem config
        //    Qkinv |  AkT | scratch
        //      S^2     S^2

        /// TODO: error check
        // load Qkinv from G
        gato_memcpy(s_Qk_i, 
                    d_Ginv_dense+set*(states_sq+controls_sq), 
                    states_sq);

                    ///TODO: linsys solver hasn't been checked with this change
        if(set != knot_points-1){
            // load Ak from C
            gato_memcpy(s_AkT,
                d_C_dense+set*(states_sq+states_p_controls),
                states_sq);
            __syncthreads();
                        
            // // compute AkT*lkp1 in scratch
            gato_ATx(s_scratch,
                    s_AkT,
                    d_lambda+(set+1)*state_size,
                    state_size,
                    state_size);
            __syncthreads();
        }
        else{
            // cudaMemsetAsync(s_scratch, 0, state_size);       
            // need to compile with -dc flag to use deivce functions like that but having issues TODO: 
            for(int i = threadIdx.x; i < state_size; i+=GATO_THREADS_PER_BLOCK){
                s_scratch[i] = 0;
            }
        }
        

        // add lk to scratch
        gato_vec_sum(s_scratch,     // out
                     d_lambda+set*state_size,
                     s_scratch,
                     state_size);
        __syncthreads();

        // subtract from qk in scratch
        gato_vec_dif(s_scratch,
                     d_g_val+set*(states_s_controls),
                     s_scratch,
                     state_size);
        __syncthreads();
        
        
        // multiply Qk_i(scratch) in Akt
        mat_vec_prod<T, state_size, state_size>(s_Qk_i,
                                                    s_scratch,
                                                    s_AkT);
        __syncthreads();

        // store in dz
        gato_memcpy<T>(d_dz+set*(states_s_controls),
                           s_AkT,
                           state_size);
    }
}

__global__
void gato_compute_dz(T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){
    
    // const unsigned s_mem_size = max(2*control_size, state_size);

    __shared__ T s_mem[2*states_sq+state_size]; 

    for(int ind = thread_id; ind < 2*knot_points-1; ind+=GATO_NUM_BLOCKS){
        gato_compute_dz_inner(d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz, s_mem, ind);
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
                   T *d_xu_traj, T *d_xu)
{

    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const unit32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_sq = states_sq;
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

        glass::copy<T>(2*states_s_controls, &d_xu[k*states_s_controls], s_xux);
        glass::copy<T>(2*states_s_controls, &d_xu_traj[k*states_s_controls], s_xux_traj);
        
        __syncthreads();    

        if(k==knot_points-2){          // last block

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_Qkp1 = s_Bk + states_p_controls;
            T *s_qkp1 = s_Qkp1 + states_sq;
            T *s_integrator_error = s_qkp1 + state_size;
            T *s_extra_temp = s_integrator_error + state_size;
            
            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
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
            
            gato_plant::trackingCostGradientAndHessian<T>(
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
                ///TODO: EMRE what to do here
                d_c[i] = 0;
            }
            glass::copy<T>(states_sq, &d_G_dense[(states_sq+controls_sq)*k], s_Qk);
            glass::copy<T>(controls_sq, &d_G_dense[(states_sq+controls_sq)*k+states_sq], s_Rk);
            glass::copy<T>(states_sq, &d_G_dense[(states_sq+controls_sq)*(k+1)], s_Qkp1);
            glass::copy<T>(state_size, &d_g[states_s_controls*k], s_qk);
            glass::copy<T>(control_size, &d_g[states_s_controls*k+state_size], s_rk);
            glass::copy<T>(state_size, &d_g[states_s_controls*(k+1)], s_qkp1);
            glass::copy<T>(states_sq, static_cast<T>(-1), &d_C_dense[(states_sq+states_p_controls)*k], s_Ak);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), &d_C_dense[(states_sq+states_p_controls)*k+states_sq],   s_Bk);
            glass::copy<T>(state_size, &d_c[state_size*(k+1)], s_integrator_error);

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
                                 (s_xux,
                                  s_Ak,
                                  s_Bk,
                                  s_integrator_error,
                                  s_extra_temp,
                                  d_dynMem_const,
                                  dt,
                                  block);
            __syncthreads();
           
            gato_plant::costGradientAndHessian<T>(state_size,
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
 
            glass::copy<T>(states_sq, &d_G_dense[(states_sq+controls_sq)*k], s_Qk);
            glass::copy<T>(controls_sq, &d_G_dense[(states_sq+controls_sq)*k+states_sq], s_Rk);
            glass::copy<T>(state_size, &d_g[states_s_controls*k], s_qk);
            glass::copy<T>(control_size, &d_g[states_s_controls*k+state_size], s_rk);
            glass::copy<T>(states_sq, static_cast<T>(-1), &d_C_dense[(states_sq+states_p_controls)*k], s_Ak);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), &d_C_dense[(states_sq+states_p_controls)*k+states_sq], s_Bk);
            glass::copy<T>(state_size, &d_c[state_size*(k+1)], s_integrator_error);
        }
    }
}



template <typename T>
void form_schur(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                T *d_G_dense, T *d_C_dense, T *d_g, T *d_c, 
                T *d_S, T *d_Pinv, T *d_gamma, 
                T rho)
{
    // form Schur, Pinv
    gato_form_schur_jacobi<T><<<knot_points, 64>>>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_S, d_Pinv, d_gamma, rho, gridDim.x);// hard coded

    gato_form_ss<T><<<knot_points, 64>>>(d_S, d_Pinv, d_gamma);// hard coded

}



void compute_dz(T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz, gato_config *config){
    
    gato_compute_dz<<<config->dz_grid, config->dz_block>>>(d_G_dense, d_C_dense, d_g_val, d_lambda, d_dz);
}

