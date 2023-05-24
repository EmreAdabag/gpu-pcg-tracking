#pragma once
#include <cstdint>


#define GATO_BLOCK_ID (blockIdx.x)
#define GATO_THREAD_ID (threadIdx.x)
#define GATO_THREADS_PER_BLOCK (blockDim.x)
#define GATO_NUM_BLOCKS   (gridDim.x)
#define GATO_LEAD_THREAD (GATO_THREAD_ID == 0)
#define GATO_LEAD_BLOCK (GATO_BLOCK_ID == 0)
#define GATO_LAST_BLOCK (GATO_BLOCK_ID == GATO_NUM_BLOCKS - 1)


#define STATES_SQ       (STATE_SIZE*STATE_SIZE)
#define CONTROLS_SQ     (CONTROL_SIZE*CONTROL_SIZE)
#define STATES_S_CONTROLS (STATE_SIZE+CONTROL_SIZE)
#define STATES_P_CONTROLS (STATE_SIZE*CONTROL_SIZE)

#define KKT_G_DENSE_SIZE_BYTES   (((STATES_SQ+CONTROLS_SQ)*KNOT_POINTS-CONTROLS_SQ)*sizeof(float))
#define KKT_C_DENSE_SIZE_BYTES   (((STATES_SQ+STATES_P_CONTROLS)*(KNOT_POINTS-1))*sizeof(float))
#define KKT_g_SIZE_BYTES         ((STATES_S_CONTROLS*KNOT_POINTS-CONTROL_SIZE)*sizeof(float))
#define KKT_c_SIZE_BYTES         ((STATE_SIZE*KNOT_POINTS)*sizeof(float))     

#define LAMBDA_SIZE_BYTES        ((STATE_SIZE*KNOT_POINTS*sizeof(float)))
#define DZ_SIZE_BYTES            ((STATES_S_CONTROLS*KNOT_POINTS-CONTROL_SIZE)*sizeof(float))


namespace oldschur{
    template <typename T>
    __device__
    void gato_memcpy(T *dst, T *src, unsigned size_Ts){
        unsigned ind;
        for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
            dst[ind] = src[ind];
        }
    }

    // just negates the val lol
    template <typename T>
    __device__
    void gato_nmemcpy(T *dst, T *src, unsigned size_Ts){
        unsigned ind;
        for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
            dst[ind] = -src[ind];
        }
    }

    template <typename T>
    __device__
    void gato_memcpy_regularized(T *dst, T *src, unsigned size_Ts, int col, float rho){
        unsigned ind;
        for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
            dst[ind] = src[ind] + (ind==col)*rho;
        }
    }


    template <typename T, unsigned MAT_ROWS, unsigned MAT_COLS>
    __device__
    void mat_vec_prod(T *mat, T *vec, T *out){
        
        for(unsigned row=GATO_THREAD_ID; row<MAT_ROWS; row+=GATO_THREADS_PER_BLOCK){
            T res = static_cast<T>(0);
            for (unsigned col = 0; col < MAT_COLS; col++){
                res += mat[row + col*MAT_ROWS] * vec[col];
            }
            out[row] = res;
        }
    }

    template <typename T, unsigned B_DIM, unsigned M_DIM>
    __device__
    void store_block_bd(T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1){
        
        unsigned block_row_offset, block_col_offset, ind;

        assert(col<3);


        block_row_offset = BLOCKNO * (3 * B_DIM * B_DIM);
        block_col_offset = col*B_DIM*B_DIM;


        if(multiplier==1){

            gato_memcpy<T>(
                dst+block_row_offset+block_col_offset,
                src,
                B_DIM*B_DIM
            );

        }
        else{
            
            for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
                dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
            }

        }
    }

    template <typename T, unsigned m, unsigned n>
    __device__
    void printMat(float *mat, int asdf){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                printf("%f ", mat[j*m+i]);
            }
            printf("\n");
        }
    }

    template <typename T, unsigned B_DIM, unsigned M_DIM>
    __device__
    void load_block_bd(T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false){
        
        // EMRE assert this
        if(bcol > 2 || brow > M_DIM-1)
            return;
        

        unsigned block_row_offset, block_col_offset;

        block_row_offset = brow * (3 * B_DIM * B_DIM);
        block_col_offset = bcol*B_DIM*B_DIM;

        if(!transpose){

            gato_memcpy<T>(
                dst,
                src+block_row_offset+block_col_offset,
                B_DIM*B_DIM
            );

        }
        else{

            unsigned ind, transpose_col, transpose_row;

            for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
                transpose_col = ind%B_DIM * B_DIM;
                transpose_row = ind/B_DIM;
                dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
            }
        }
    }

    ///TODO: this could be more better
    __device__
    void mat_mat_prod(float *out, float *mat_A, float *mat_B, int A_rows, int A_cols, int B_rows, int B_cols, bool transposeB = false){

        if(!transposeB){

            unsigned ind, thing;
            unsigned maxind = A_rows*B_cols;
            float res;
            int row, col;

            for(ind=threadIdx.x; ind<maxind; ind+=blockDim.x){
                // ind x takes row x/A_cols and col x%b_rows
                res = 0;
                row = ind % A_rows;
                col = ind / A_rows;

                for(thing=0; thing<A_cols; thing++){
                    res += mat_A[thing*A_rows+row] * mat_B[col*B_rows+thing];
                }

                out[col*A_rows+row] = res;

            } 
        }
        else{                       // transpose matrix B


            unsigned ind, thing;
            unsigned maxind = A_rows*B_rows;
            float res;
            int row, col;

            for(ind=threadIdx.x; ind<maxind; ind+=blockDim.x){
                // ind x takes row x/A_cols and col x%b_rows
                res = 0;
                row = ind % A_rows;
                col = ind / A_rows;

                for(thing=0; thing<A_cols; thing++){
                    res += mat_A[thing*A_rows+row] * mat_B[thing*B_rows+col];
                }

                out[col*A_rows+row] = res;

            } 

        }
    }

    __device__
    void add_identity(float *A, unsigned dim, float factor){
        for(unsigned i = GATO_THREAD_ID; i < dim*dim; i+=GATO_THREADS_PER_BLOCK){
            if(i/dim == i%dim){ A[i] += factor; }
        }
    }



    // load identity in so memory is [A | I]
    template <typename T, unsigned DIM>
    __device__ __forceinline__
    void loadIdentity(T *A){
        for (unsigned ind = GATO_THREAD_ID; ind < DIM*DIM; ind += GATO_THREADS_PER_BLOCK){
            unsigned r, c;
            r = ind % DIM; 
            c = ind / DIM;
            A[ind] = static_cast<T>(r == c);
        }
    }

    // load identity in so memory is [V | I]
    template <typename T, unsigned DIMA, unsigned DIMB>
    __device__ __forceinline__
    void loadIdentity(T *A, T *B){
        for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB; ind += GATO_THREADS_PER_BLOCK){
            unsigned r, c, indAdj; T *V;
            if (ind < DIMA*DIMA){
                indAdj = ind;
                r = indAdj % DIMA; c = indAdj/DIMA; V = A;
            }
            else {
                indAdj = ind - DIMA*DIMA;
                r = indAdj % DIMB; c = indAdj/DIMB; V = B;
            }
            V[indAdj] = static_cast<T>(r == c);
        }
    }


    // load identity in so memory is [V | I]
    template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC>
    __device__ __forceinline__
    void loadIdentity(T *A, T *B, T *C){
        for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB+DIMC*DIMC; ind += GATO_THREADS_PER_BLOCK){
            unsigned r, c, indAdj; T *V;
            if (ind < DIMA*DIMA){
                indAdj = ind;
                r = indAdj % DIMA; c = indAdj/DIMA; V = A;
            }
            else if (ind < DIMA*DIMA+DIMB*DIMB){
                indAdj = ind - DIMA*DIMA;
                r = indAdj % DIMB; c = indAdj/DIMB; V = B;
            }
            else{
                indAdj = ind - DIMA*DIMA - DIMB*DIMB;
                r = indAdj % DIMC; c = indAdj/DIMC; V = C;
            }
            V[indAdj] = static_cast<T>(r == c);
        }
    }


    template <typename T, unsigned DIM>
    __device__
    void invertMatrix(T *A, T *s_temp){ 
    // we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        for (unsigned pivRC = 0; pivRC < DIM; pivRC++){
            unsigned pivColOffset = pivRC*DIM;
            // save the pivot and pivot column and row
            T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
            for (unsigned ind = GATO_THREAD_ID; ind < 2*DIM+1; ind++){
                unsigned AInd;
                if (ind < DIM){AInd = ind + pivColOffset;}
                else{AInd = pivRC + pivColOffset + (ind-DIM)*DIM;}
                s_temp[ind] = A[AInd];
            }
            __syncthreads(); //----------------------
            // make the pivot update
            for (unsigned ind = GATO_THREAD_ID; ind < DIM*(DIM+1); ind += GATO_THREADS_PER_BLOCK){
                unsigned row = ind % DIM; unsigned col = ind / DIM; unsigned colOffset = ind - row;
                // s_temp = orpcvs|prvOld
                if (row == pivRC){A[row + pivColOffset + colOffset] *= pvInv;}
                else{A[row + pivColOffset + colOffset] -= s_temp[row]*pvInv*s_temp[DIM+col];}
            }
        __syncthreads(); //----------------------
        }
    }


    template <typename T, unsigned DIMA, unsigned DIMB, unsigned MAX_DIM>
    __device__
    void invertMatrix(T *A, T *B, T *s_temp){

        // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
        // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
        // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
        // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
        // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1];
        for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
            bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB;
            unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB;
            // save the pivot column and row
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
                if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
                if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
            }
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
                if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
                if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
            }
            __syncthreads(); //----------------------
            // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
                if (AActive && ind < DIMA*(DIMA+1)){
                    unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                    if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                    else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
                }
                if (BActive && ind < DIMB*(DIMB+1)){
                    unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                    if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                    else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
                }
            }
            __syncthreads(); //----------------------
        }
    }

    // invert A,B,C assume memory for all is [V | VInv] where both are DIMxDIM and continguous
    // relies on s_temp of size [2*DIMA + 2*DIMB + 2*DIMC + 3]
    template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC, unsigned MAX_DIM>
    __device__
    void invertMatrix(T *A, T *B, T *C, T *s_temp){

        // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
        // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
        // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
        // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
        // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
        T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1]; T *s_memC = &s_memB[2*DIMB+1];
        for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
            bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB; bool CActive = pivRC < DIMC;
            unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB; unsigned pivColOffsetC = pivRC*DIMC;
            // save the pivot column and row
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
                if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
                if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
                if (CActive && ind < DIMC){s_memC[ind] = C[ind + pivColOffsetC];}
            }
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
                if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
                if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
                if (CActive && ind < DIMC+1){s_memC[ind + DIMC] = C[ind*DIMC + pivRC + pivColOffsetC];}
            }
            __syncthreads(); //----------------------
            // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
            for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
                if (AActive && ind < DIMA*(DIMA+1)){
                    unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                    if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                    else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
                }
                if (BActive && ind < DIMB*(DIMB+1)){
                    unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                    if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                    else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
                }
                if (CActive && ind < DIMC*(DIMC+1)){
                    unsigned row = ind % DIMC; unsigned col = ind / DIMC;
                    if (row == pivRC){C[pivColOffsetC + ind] /= s_memC[pivRC];}
                    else{C[pivColOffsetC + ind] -= s_memC[row]/s_memC[pivRC]*s_memC[DIMC+col];}
                }
            }
            __syncthreads(); //----------------------
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
    void gato_form_schur_jacobi_inner(float *d_G, float *d_C, float *d_g, float *d_c, float *d_S, float *d_Pinv, float *d_gamma, float rho, float *s_temp, unsigned blockrow){
        const uint32_t STATE_SIZE = 14;
        const uint32_t CONTROL_SIZE = 7;
        const uint32_t KNOT_POINTS = 10;    
        
        //  SPACE ALLOCATION IN SHARED MEM
        //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
        //     s^2      s^2         s^2         s
        float *s_phi_k = s_temp; 	                            	    // phi_k        states^2
        float *s_theta_k = s_phi_k + STATES_SQ; 			            // theta_k      states^2
        float *s_thetaInv_k = s_theta_k + STATES_SQ; 			        // thetaInv_k   states^2
        float *s_gamma_k = s_thetaInv_k + STATES_SQ;                       // gamma_k      states
        float *s_end_main = s_gamma_k + STATE_SIZE;                               

        if(blockrow==0){

            //  LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
            //              s^2   s^2     s   s^2   s^2     s      ? 
        
            float *s_QN = s_end_main;
            float *s_QN_i = s_QN + STATE_SIZE * STATE_SIZE;
            float *s_qN = s_QN_i + STATE_SIZE * STATE_SIZE;
            float *s_Q0 = s_qN + STATE_SIZE;
            float *s_Q0_i = s_Q0 + STATE_SIZE * STATE_SIZE;
            float *s_q0 = s_Q0_i + STATE_SIZE * STATE_SIZE;
            float *s_end = s_q0 + STATE_SIZE;

            // scratch space
            float *s_R_not_needed = s_end;
            float *s_r_not_needed = s_R_not_needed + CONTROL_SIZE * CONTROL_SIZE;
            float *s_extra_temp = s_r_not_needed + CONTROL_SIZE * CONTROL_SIZE;

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy<float>(s_Q0, d_G, STATES_SQ);
            gato_memcpy<float>(s_QN, d_G+(KNOT_POINTS-1)*(STATES_SQ+CONTROLS_SQ), STATES_SQ);
            gato_memcpy<float>(s_q0, d_g, STATE_SIZE);
            gato_memcpy<float>(s_qN, d_g+(KNOT_POINTS-1)*(STATE_SIZE+CONTROL_SIZE), STATE_SIZE);

            __syncthreads();//----------------------------------------------------------------

            add_identity(s_Q0, STATE_SIZE, rho);
            add_identity(s_QN, STATE_SIZE, rho);
            // if(PRINT_THREAD){
            //     printf("Q0\n");
            //     printMat<float,STATE_SIZE,STATE_SIZE>(s_Q0,STATE_SIZE);
            //     printf("q0\n");
            //     printMat<float,1,STATE_SIZE>(s_q0,1);
            //     printf("QN\n");
            //     printMat<float,STATE_SIZE,STATE_SIZE>(s_QN,STATE_SIZE);
            //     printf("qN\n");
            //     printMat<float,1,STATE_SIZE>(s_qN,1);
            //     printf("start error\n");
            //     printMat<float,1,STATE_SIZE>(s_integrator_error,1);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------
            
            // SHARED MEMORY STATE
            // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
            

            // save -Q_0 in PhiInv spot 00
            store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
                s_Q0,                       // src     
                d_Pinv,                   // dst         
                1,                          // col
                blockrow,                    // blockrow
                -1                          //  multiplier
            );
            __syncthreads();//----------------------------------------------------------------


            // invert Q_N, Q_0
            loadIdentity<float, STATE_SIZE,STATE_SIZE>(s_Q0_i, s_QN_i);
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<float, STATE_SIZE,STATE_SIZE,STATE_SIZE>(s_Q0, s_QN, s_extra_temp);
            
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("Q0Inv\n");
            //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_Q0_i,STATE_SIZE);
            //     printf("QNInv\n");
            //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_QN_i,STATE_SIZE);
            //     printf("theta\n");
            //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_theta_k,STATE_SIZE);
            //     printf("thetaInv\n");
            //     printMat<floatSTATE_SIZE,STATE_SIZE>(s_thetaInv_k,STATE_SIZE);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
            

            // compute gamma
            mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
                s_Q0_i,                                    
                s_q0,                                       
                s_gamma_k 
            );
            __syncthreads();//----------------------------------------------------------------
            

            // save -Q0_i in spot 00 in S
            store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
                s_Q0_i,                         // src             
                d_S,                            // dst              
                1,                              // col   
                blockrow,                        // blockrow         
                -1                              //  multiplier   
            );
            __syncthreads();//----------------------------------------------------------------


            // compute Q0^{-1}q0
            mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
                s_Q0_i,
                s_q0,
                s_Q0
            );
            __syncthreads();//----------------------------------------------------------------


            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


            // save -Q0^{-1}q0 in spot 0 in gamma
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                d_gamma[ind] = -s_Q0[ind];
            }
            __syncthreads();//----------------------------------------------------------------

        }
        else{                       // blockrow!=LEAD_BLOCK


            const unsigned C_set_size = STATES_SQ+STATES_P_CONTROLS;
            const unsigned G_set_size = STATES_SQ+CONTROLS_SQ;

            //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
            //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

            float *s_Ak = s_end_main; 								
            float *s_Bk = s_Ak +        STATES_SQ;
            float *s_Qk = s_Bk +        STATES_P_CONTROLS; 	
            float *s_Qk_i = s_Qk +      STATES_SQ;	
            float *s_Qkp1 = s_Qk_i +    STATES_SQ;
            float *s_Qkp1_i = s_Qkp1 +  STATES_SQ;
            float *s_Rk = s_Qkp1_i +    STATES_SQ;
            float *s_Rk_i = s_Rk +      CONTROLS_SQ;
            float *s_qk = s_Rk_i +      CONTROLS_SQ; 	
            float *s_qkp1 = s_qk +      STATE_SIZE; 			
            float *s_rk = s_qkp1 +      STATE_SIZE;
            float *s_end = s_rk +       CONTROL_SIZE;
            
            // scratch
            float *s_extra_temp = s_end;
            

            // if(PRINT_THREAD){
            //     printf("xk\n");
            //     printMat<float1,STATE_SIZE>(s_xux,1);
            //     printf("uk\n");
            //     printMat<float1,CONTROL_SIZE>(&s_xux[STATE_SIZE],1);
            //     printf("xkp1\n");
            //     printMat<float1,STATE_SIZE>(&s_xux[STATE_SIZE+CONTROL_SIZE],1);
            //     printf("\n");
            // }

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy<float>(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        STATES_SQ);
            gato_memcpy<float>(s_Bk,   d_C+      (blockrow-1)*C_set_size+STATES_SQ,              STATES_P_CONTROLS);
            gato_memcpy<float>(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        STATES_SQ);
            gato_memcpy<float>(s_Qkp1, d_G+    (blockrow*G_set_size),                          STATES_SQ);
            gato_memcpy<float>(s_Rk,   d_G+      ((blockrow-1)*G_set_size+STATES_SQ),            CONTROLS_SQ);
            gato_memcpy<float>(s_qk,   d_g+      (blockrow-1)*(STATES_S_CONTROLS),               STATE_SIZE);
            gato_memcpy<float>(s_qkp1, d_g+    (blockrow)*(STATES_S_CONTROLS),                 STATE_SIZE);
            gato_memcpy<float>(s_rk,   d_g+      ((blockrow-1)*(STATES_S_CONTROLS)+STATE_SIZE),  CONTROL_SIZE);

            __syncthreads();//----------------------------------------------------------------

            add_identity(s_Qk, STATE_SIZE, rho);
            add_identity(s_Qkp1, STATE_SIZE, rho);
            add_identity(s_Rk, CONTROL_SIZE, rho);

    #if DEBUG_MODE    
            if(GATO_BLOCK_ID==1 && GATO_THREAD_ID==0){
                printf("Ak\n");
                printMat<float,STATE_SIZE,STATE_SIZE>(s_Ak,STATE_SIZE);
                printf("Bk\n");
                printMat<float,STATE_SIZE,CONTROL_SIZE>(s_Bk,STATE_SIZE);
                printf("Qk\n");
                printMat<float,STATE_SIZE,STATE_SIZE>(s_Qk,STATE_SIZE);
                printf("Rk\n");
                printMat<float,CONTROL_SIZE,CONTROL_SIZE>(s_Rk,CONTROL_SIZE);
                printf("qk\n");
                printMat<float,STATE_SIZE, 1>(s_qk,1);
                printf("rk\n");
                printMat<float,CONTROL_SIZE, 1>(s_rk,1);
                printf("Qkp1\n");
                printMat<float,STATE_SIZE,STATE_SIZE>(s_Qkp1,STATE_SIZE);
                printf("qkp1\n");
                printMat<float,STATE_SIZE, 1>(s_qkp1,1);
                printf("integrator error\n");
            }
            __syncthreads();//----------------------------------------------------------------
    #endif /* #if DEBUG_MODE */
            
            // Invert Q, Qp1, R 
            loadIdentity<float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE>(
                s_Qk_i, 
                s_Qkp1_i, 
                s_Rk_i
            );
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE,STATE_SIZE>(
                s_Qk, 
                s_Qkp1, 
                s_Rk, 
                s_extra_temp
            );
            __syncthreads();//----------------------------------------------------------------

            // save Qk_i into G (now Ginv) for calculating dz
            gato_memcpy<float>(
                d_G+(blockrow-1)*G_set_size,
                s_Qk_i,
                STATES_SQ
            );

            // save Rk_i into G (now Ginv) for calculating dz
            gato_memcpy<float>( 
                d_G+(blockrow-1)*G_set_size+STATES_SQ,
                s_Rk_i,
                CONTROLS_SQ
            );

            if(blockrow==KNOT_POINTS-1){
                // save Qkp1_i into G (now Ginv) for calculating dz
                gato_memcpy<float>(
                    d_G+(blockrow)*G_set_size,
                    s_Qkp1_i,
                    STATES_SQ
                );
            }
            __syncthreads();//----------------------------------------------------------------

    #if DEBUG_MODE
            if(blockrow==1&&GATO_THREAD_ID==0){
                printf("Qk\n");
                printMat<float, STATE_SIZE,STATE_SIZE>(s_Qk_i,STATE_SIZE);
                printf("RkInv\n");
                printMat<float,CONTROL_SIZE,CONTROL_SIZE>(s_Rk_i,CONTROL_SIZE);
                printf("Qkp1Inv\n");
                printMat<float, STATE_SIZE,STATE_SIZE>(s_Qkp1_i,STATE_SIZE);
                printf("\n");
            }
            __syncthreads();//----------------------------------------------------------------
    #endif /* #if DEBUG_MODE */


            // Compute -AQ^{-1} in phi
            mat_mat_prod(
                s_phi_k,
                s_Ak,
                s_Qk_i,
                STATE_SIZE, 
                STATE_SIZE, 
                STATE_SIZE, 
                STATE_SIZE
            );
            // for(int i = GATO_THREAD_ID; i < STATES_SQ; i++){
            //     s_phi_k[i] *= -1;
            // }

            __syncthreads();//----------------------------------------------------------------

            // Compute -BR^{-1} in Qkp1
            mat_mat_prod(
                s_Qkp1,
                s_Bk,
                s_Rk_i,
                STATE_SIZE,
                CONTROL_SIZE,
                CONTROL_SIZE,
                CONTROL_SIZE
            );

            __syncthreads();//----------------------------------------------------------------

            // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
            mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
                s_Qkp1_i,
                s_qkp1,
                s_gamma_k
            );
            for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
                s_gamma_k[i] -= d_c[(blockrow*STATE_SIZE)+i];
            }
            __syncthreads();//----------------------------------------------------------------

            // compute -AQ^{-1}q for gamma         temp storage in extra temp
            mat_vec_prod<float, STATE_SIZE, STATE_SIZE>(
                s_phi_k,
                s_qk,
                s_extra_temp
            );
            

            __syncthreads();//----------------------------------------------------------------
            
            // compute -BR^{-1}r for gamma           temp storage in extra temp + states
            mat_vec_prod<float, STATE_SIZE, CONTROL_SIZE>(
                s_Qkp1,
                s_rk,
                s_extra_temp + STATE_SIZE
            );

            __syncthreads();//----------------------------------------------------------------
            
            // gamma = yeah...
            for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
                s_gamma_k[i] += s_extra_temp[STATE_SIZE + i] + s_extra_temp[i]; 
            }
            __syncthreads();//----------------------------------------------------------------

            // compute AQ^{-1}AT   -   Qkp1^{-1} for theta
            mat_mat_prod(
                s_theta_k,
                s_phi_k,
                s_Ak,
                STATE_SIZE,
                STATE_SIZE,
                STATE_SIZE,
                STATE_SIZE,
                true
            );

            __syncthreads();//----------------------------------------------------------------

    #if DEBUG_MODE
            if(blockrow==1&&GATO_THREAD_ID==0){
                printf("this is the A thing\n");
                printMat<float, STATE_SIZE, STATE_SIZE>(s_theta_k, 234);
            }
    #endif /* #if DEBUG_MODE */

            for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            
            __syncthreads();//----------------------------------------------------------------

            // compute BR^{-1}BT for theta            temp storage in QKp1{-1}
            mat_mat_prod(
                s_Qkp1_i,
                s_Qkp1,
                s_Bk,
                STATE_SIZE,
                CONTROL_SIZE,
                STATE_SIZE,
                CONTROL_SIZE,
                true
            );

            __syncthreads();//----------------------------------------------------------------

            for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            __syncthreads();//----------------------------------------------------------------

            // save phi_k into left off-diagonal of S, 
            store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
                s_phi_k,                        // src             
                d_S,                            // dst             
                0,                              // col
                blockrow,                        // blockrow    
                -1
            );
            __syncthreads();//----------------------------------------------------------------

            // save -s_theta_k main diagonal S
            store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
                s_theta_k,                                               
                d_S,                                                 
                1,                                               
                blockrow,
                -1                                             
            );          
            __syncthreads();//----------------------------------------------------------------

    #if BLOCK_J_PRECON || SS_PRECON
        // invert theta
        loadIdentity<float,STATE_SIZE>(s_thetaInv_k);
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<float,STATE_SIZE>(s_theta_k, s_extra_temp);
        __syncthreads();//----------------------------------------------------------------


        // save thetaInv_k main diagonal PhiInv
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_thetaInv_k, 
            d_Pinv,
            1,
            blockrow,
            -1
        );
    #else /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */

        // save 1 / diagonal to PhiInv
        for(int i = GATO_THREAD_ID; i < STATE_SIZE; i+=GATO_THREADS_PER_BLOCK){
            d_Pinv[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]= 1 / d_S[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]; 
        }
    #endif /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */
        

        __syncthreads();//----------------------------------------------------------------

        // save gamma_k in gamma
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            unsigned offset = (blockrow)*STATE_SIZE + ind;
            d_gamma[offset] = s_gamma_k[ind]*-1;
        }

        __syncthreads();//----------------------------------------------------------------

        //transpose phi_k
        loadIdentity<float,STATE_SIZE>(s_Ak);
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod(s_Qkp1,s_Ak,s_phi_k,STATE_SIZE,STATE_SIZE,STATE_SIZE,STATE_SIZE,true);
        __syncthreads();//----------------------------------------------------------------

        // save phi_k_T into right off-diagonal of S,
        store_block_bd<float, STATE_SIZE, KNOT_POINTS>(
            s_Qkp1,                        // src             
            d_S,                            // dst             
            2,                              // col
            blockrow-1,                      // blockrow    
            -1
        );

        __syncthreads();//----------------------------------------------------------------
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

}




















/*******************************************************************************
 *                           Private Functions                                 *
 *******************************************************************************/

// template <typename T>
// __device__
// void gato_form_schur_jacobi_inner(
//     uint32_t state_size,
//     uint32_t control_size,
//     uint32_t knot_points,
//     T *d_G, 
//     T *d_C, 
//     T *d_g, 
//     T *d_c, 
//     T *d_S, 
//     T *d_Pinv, 
//     T *d_gamma, 
//     T rho, 
//     T *s_temp, 
//     unsigned blockrow)
// {
//     const cgrps::thread_block = block;
//     const uint32_t states_sq = states_sq;
//     const uint32_t states_p_controls = state_size * control_size;
//     const uint32_t controls_sq = control_size * control_size;
//     const uint32_t states_s_controls = state_size + control_size;
//     const uint32_t thread_id = threadIdx.x;
//     const unit32_t num_threads = blockDim.x;
//     const uint32_t block_id = blockIdx.x;
//     const uint32_t num_blocks = gridDim.x;

    
//     //  SPACE ALLOCATION IN SHARED MEM
//     //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
//     //     s^2      s^2         s^2         s
//     T *s_phi_k = s_temp; 	                            	    // phi_k        states^2
//     T *s_theta_k = s_phi_k + states_sq; 			            // theta_k      states^2
//     T *s_thetaInv_k = s_theta_k + states_sq; 			        // thetaInv_k   states^2
//     T *s_gamma_k = s_thetaInv_k + states_sq;                       // gamma_k      states
//     T *s_end_main = s_gamma_k + state_size;                               

//     if(blockrow==0){

//         //  LEADING BLOCK GOAL SHARED MEMORY STATE
//         //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
//         //              s^2   s^2     s   s^2   s^2     s      ? 
    
//         T *s_QN = s_end_main;
//         T *s_QN_i = s_QN + states_sq;
//         T *s_qN = s_QN_i + states_sq;
//         T *s_Q0 = s_qN + state_size;
//         T *s_Q0_i = s_Q0 + states_sq;
//         T *s_q0 = s_Q0_i + states_sq;
//         T *s_end = s_q0 + state_size;

//         // scratch space
//         T *s_R_not_needed = s_end;
//         T *s_r_not_needed = s_R_not_needed + control_size * control_size;
//         T *s_extra_temp = s_r_not_needed + control_size * control_size;

//         __syncthreads();//----------------------------------------------------------------

//         glass::copy<T>(states_sq, d_G, s_Q0);
//         glass::copy<T>(states_sq, s_QN, &d_G[(knot_points-1)*(states_sq+controls_sq)]);
//         glass::copy<T>(state_size, d_g, s_q0);
//         glass::copy<T>(state_size, s_qN, &d_g[(knot_points-1)*(states_s_controls)]);

//         __syncthreads();//----------------------------------------------------------------

//         glass::addI(state_size, s_Q0, rho, block);
//         glass::addI(state_size, s_QN, rho, block);

//         __syncthreads();//----------------------------------------------------------------
        
//         // SHARED MEMORY STATE
//         // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
        

//         // save -Q_0 in PhiInv spot 00
//         store_block_bd<T>(
//             state_size, 
//             knot_points,
//             s_Q0,                       // src     
//             d_Pinv,                   // dst         
//             1,                          // col
//             blockrow,                    // blockrow
//             -1,                          //  multiplier
//             block
//         );
//         __syncthreads();//----------------------------------------------------------------


//         // invert Q_N, Q_0
//         loadIdentity<T>(state_size, s_Q0_i, state_size, s_QN_i);
//         __syncthreads();//----------------------------------------------------------------
//         invertMatrix<T>(state_size, s_Q0, state_size, s_QN, s_extra_temp);
        
//         __syncthreads();//----------------------------------------------------------------

//         // SHARED MEMORY STATE
//         // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
        

//         // compute gamma
//         glass::gemv<T>(state_size, state_size, static_cast<T>(1), S_Q0_i, s_q0, s_gamma_k);
//         // mat_vec_prod<T, state_size, state_size>(
//         //     s_Q0_i,                                    
//         //     s_q0,                                       
//         //     s_gamma_k 
//         // );
//         __syncthreads();//----------------------------------------------------------------
        

//         // save -Q0_i in spot 00 in S
//         store_block_bd<T>(
//             state_size,
//             knot_points,
//             s_Q0_i,                         // src             
//             d_S,                            // dst              
//             1,                              // col   
//             blockrow,                        // blockrow         
//             -1                              //  multiplier   
//         );
//         __syncthreads();//----------------------------------------------------------------


//         // compute Q0^{-1}q0
//         glass::gemv<T>(state_size, state_size, static_cast<T>(1), s_Q0_i, s_q0, s_Q0);
//         // mat_vec_prod<T, state_size, state_size>(
//         //     s_Q0_i,
//         //     s_q0,
//         //     s_Q0
//         // );
//         __syncthreads();//----------------------------------------------------------------


//         // SHARED MEMORY STATE
//         // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


//         // save -Q0^{-1}q0 in spot 0 in gamma
//         for(unsigned ind = thread_id; ind < state_size; ind += num_threads){
//             d_gamma[ind] = -s_Q0[ind];
//         }
//         __syncthreads();//----------------------------------------------------------------

//     }
//     else{                       // blockrow!=LEAD_BLOCK


//         const unsigned C_set_size = states_sq+states_p_controls;
//         const unsigned G_set_size = states_sq+controls_sq;

//         //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
//         //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
//         //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

//         T *s_Ak = s_end_main; 								
//         T *s_Bk = s_Ak +        states_sq;
//         T *s_Qk = s_Bk +        states_p_controls; 	
//         T *s_Qk_i = s_Qk +      states_sq;	
//         T *s_Qkp1 = s_Qk_i +    states_sq;
//         T *s_Qkp1_i = s_Qkp1 +  states_sq;
//         T *s_Rk = s_Qkp1_i +    states_sq;
//         T *s_Rk_i = s_Rk +      controls_sq;
//         T *s_qk = s_Rk_i +      controls_sq; 	
//         T *s_qkp1 = s_qk +      state_size; 			
//         T *s_rk = s_qkp1 +      state_size;
//         T *s_end = s_rk +       control_size;
        
//         // scratch
//         T *s_extra_temp = s_end;
        

//         __syncthreads();//----------------------------------------------------------------

//         glass::copy<T>(states_sq,  d_C+      (blockrow-1)*C_set_size,  s_Ak);
//         glass::copy<T>(states_p_controls,  d_C+      (blockrow-1)*C_set_size+states_sq,  s_Bk);
//         glass::copy<T>(states_sq,  d_G+      (blockrow-1)*G_set_size,  s_Qk);
//         glass::copy<T>(states_sq,  d_G+    (blockrow*G_set_size),  s_Qkp1);
//         glass::copy<T>(controls_sq,  d_G+      ((blockrow-1)*G_set_size+states_sq),  s_Rk);
//         glass::copy<T>(state_size,  d_g+      (blockrow-1)*(states_s_controls),  s_qk);
//         glass::copy<T>(state_size,  d_g+    (blockrow)*(states_s_controls),  s_qkp1);
//         glass::copy<T>(control_size,  d_g+      ((blockrow-1)*(states_s_controls)+state_size),  s_rk);

//         __syncthreads();//----------------------------------------------------------------

//         addI<T>(state_size, s_Qk, rho);
//         addI<T>(state_size, s_Qkp1, rho);
//         addI<T>(control_size, s_Rk, rho);
//         // add_identity(s_Qk, state_size, rho);
//         // add_identity(s_Qkp1, state_size, rho);
//         // add_identity(s_Rk, control_size, rho);


        
//         // Invert Q, Qp1, R 
//         loadIdentity<T>(
//             state_size,
//             s_Qk_i, 
//             state_size,
//             s_Qkp1_i, 
//             control_size,
//             s_Rk_i
//         );
//         __syncthreads();//----------------------------------------------------------------
//         invertMatrix<T>(
//             state_size,
//             s_Qk, 
//             state_size,
//             s_Qkp1, 
//             control_size
//             s_Rk, 
//             s_extra_temp
//         );
//         __syncthreads();//----------------------------------------------------------------

//         // save Qk_i into G (now Ginv) for calculating dz
//         gato_memcpy<T>(
//             d_G+(blockrow-1)*G_set_size,
//             s_Qk_i,
//             states_sq
//         );


//         // save Rk_i into G (now Ginv) for calculating dz
//         glass::copy<T>(controls_sq, s_Rk_i, d_G+(blockrow-1)*G_set_size+states_sq);
//         // gato_memcpy<T>( 
//         //     d_G+(blockrow-1)*G_set_size+states_sq,
//         //     s_Rk_i,
//         //     controls_sq
//         // );

//         if(blockrow==knot_points-1){
//             // save Qkp1_i into G (now Ginv) for calculating dz
//             glass::copy<T>(states_sq, s_Qkp1_i, d_G+(blockrow)*G_set_size);
//             // gato_memcpy<T>(
//             //     d_G+(blockrow)*G_set_size,
//             //     s_Qkp1_i,
//             //     states_sq
//             // );
//         }
//         __syncthreads();//----------------------------------------------------------------


//         glass::gemm<T>(state_size,
//                        state_size,
//                        state_size,
//                        static_cast<T>(1),
//                        s_Ak,
//                        s_Qk_i,
//                        s_phi_k);

//         // Compute -AQ^{-1} in phi
//         // mat_mat_prod(
//         //     s_phi_k,
//         //     s_Ak,
//         //     s_Qk_i,
//         //     state_size, 
//         //     state_size, 
//         //     state_size, 
//         //     state_size
//         // );

//         __syncthreads();//----------------------------------------------------------------

//         // Compute -BR^{-1} in Qkp1
//         glass::gemm<T>(state_size, control_size, control_size, static_cast<T>(1), s_Bk, s_Rk_i, s_Qkp1);
//         // mat_mat_prod(
//         //     s_Qkp1,
//         //     s_Bk,
//         //     s_Rk_i,
//         //     state_size,
//         //     control_size,
//         //     control_size,
//         //     control_size
//         // );

//         __syncthreads();//----------------------------------------------------------------

//         // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
//         glass::gemv<T>(state_size, state_size, static_cast<T>(1), s_Qkp1_i, s_qkp1, s_gamma_k);
//         // mat_vec_prod<T, state_size, state_size>(
//         //     s_Qkp1_i,
//         //     s_qkp1,
//         //     s_gamma_k
//         // );
//         for(unsigned i = thread_id; i < state_size; i += num_threads){
//             s_gamma_k[i] -= d_c[(blockrow*state_size)+i];
//         }
//         __syncthreads();//----------------------------------------------------------------

//         // compute -AQ^{-1}q for gamma         temp storage in extra temp
//         glass::gemv<T>(state_size, state_size, static_cast<T>(1), s_phi_k, s_qk, s_extra_temp);
//         // mat_vec_prod<T, state_size, state_size>(
//         //     s_phi_k,
//         //     s_qk,
//         //     s_extra_temp
//         // );
        

//         __syncthreads();//----------------------------------------------------------------
        
//         // compute -BR^{-1}r for gamma           temp storage in extra temp + states
//         glass::gemv<T>(state_size, control_size, static_cast<T>(1), s_Qkp1, s_rk, &s_extra_temp[state_size]);
//         // mat_vec_prod<T, state_size, control_size>(
//         //     s_Qkp1,
//         //     s_rk,
//         //     s_extra_temp + state_size
//         // );

//         __syncthreads();//----------------------------------------------------------------
        
//         // gamma = yeah...
//         for(unsigned i = thread_id; i < state_size; i += num_threads){
//             s_gamma_k[i] += s_extra_temp[state_size + i] + s_extra_temp[i]; 
//         }
//         __syncthreads();//----------------------------------------------------------------


//         // compute AQ^{-1}AT   -   Qkp1^{-1} for theta
//         glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1), s_phi_k, s_Ak, s_theta_k);
//         // mat_mat_prod(
//         //     s_theta_k,
//         //     s_phi_k,
//         //     s_Ak,
//         //     state_size,
//         //     state_size,
//         //     state_size,
//         //     state_size,
//         //     true
//         // );

//         __syncthreads();//----------------------------------------------------------------

//         for(unsigned i = thread_id; i < states_sq; i += num_threads){
//             s_theta_k[i] += s_Qkp1_i[i];
//         }
        
//         __syncthreads();//----------------------------------------------------------------

//         // compute BR^{-1}BT for theta            temp storage in QKp1{-1}
//         glass::gemm<T, true>(state_size, control_size, state_size, static_cast<T>(1), s_Qkp1_i, s_Qkp1, s_Bk);
//         //         mat_mat_prod(
//         //     s_Qkp1_i,
//         //     s_Qkp1,
//         //     s_Bk,
//         //     state_size,
//         //     control_size,
//         //     state_size,
//         //     control_size,
//         //     true
//         // );

//         __syncthreads();//----------------------------------------------------------------

//         for(unsigned i = thread_id; i < states_sq; i += GATO_THREADS_PER_BLOCK){
//             s_theta_k[i] += s_Qkp1_i[i];
//         }
//         __syncthreads();//----------------------------------------------------------------

//         // save phi_k into left off-diagonal of S, 
//         store_block_bd<T, state_size, knot_points>(
//             s_phi_k,                        // src             
//             d_S,                            // dst             
//             0,                              // col
//             blockrow,                        // blockrow    
//             -1
//         );
//         __syncthreads();//----------------------------------------------------------------

//         // save -s_theta_k main diagonal S
//         store_block_bd<T, state_size, knot_points>(
//             s_theta_k,                                               
//             d_S,                                                 
//             1,                                               
//             blockrow,
//             -1                                             
//         );          
//         __syncthreads();//----------------------------------------------------------------

//         // invert theta
//         loadIdentity<T,state_size>(s_thetaInv_k);
//         __syncthreads();//----------------------------------------------------------------
//         invertMatrix<T,state_size>(s_theta_k, s_extra_temp);
//         __syncthreads();//----------------------------------------------------------------


//         // save thetaInv_k main diagonal PhiInv
//         store_block_bd<T, state_size, knot_points>(
//             s_thetaInv_k, 
//             d_Pinv,
//             1,
//             blockrow,
//             -1
//         );
        

//         __syncthreads();//----------------------------------------------------------------

//         // save gamma_k in gamma
//         for(unsigned ind = GATO_THREAD_ID; ind < state_size; ind += GATO_THREADS_PER_BLOCK){
//             unsigned offset = (blockrow)*state_size + ind;
//             d_gamma[offset] = s_gamma_k[ind]*-1;
//         }

//         __syncthreads();//----------------------------------------------------------------

//         //transpose phi_k
//         loadIdentity<T,state_size>(s_Ak);
//         __syncthreads();//----------------------------------------------------------------
//         mat_mat_prod(s_Qkp1,s_Ak,s_phi_k,state_size,state_size,state_size,state_size,true);
//         __syncthreads();//----------------------------------------------------------------

//         // save phi_k_T into right off-diagonal of S,
//         store_block_bd<T, state_size, knot_points>(
//             s_Qkp1,                        // src             
//             d_S,                            // dst             
//             2,                              // col
//             blockrow-1,                      // blockrow    
//             -1
//         );

//         __syncthreads();//----------------------------------------------------------------
//     }

// }