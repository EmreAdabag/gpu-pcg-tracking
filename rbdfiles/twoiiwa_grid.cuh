#pragma once
/**
 * This instance of grid.cuh is optimized for the urdf: iiwa14_twoArm
 *
 * Notes:
 *   Interface is:
 *       __host__   robotModel<T> *d_robotModel = init_robotModel<T>()
 *       __host__   cudaStream_t streams = init_grid<T>()
 *       __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)
 *   
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *   
 *       __device__ direct_minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)
 *       __global__ direct_minv_Kernel<T>(T *d_Minv, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)
 *       __host__   direct_minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_device<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_kernel<T>(T *d_qdd, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)
 *       __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)
 *       __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)
 *       __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)
 *   
 *   Suggested Type T is float
 *   
 *   Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file
 *   
 *   By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [ID, MINV, FD, ID_DU, FD_DU]
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
// single kernel timing helper code
#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

/**
 * Check for runtime errors using the CUDA API
 *
 * Notes:
 *   Adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 *
 */


template <typename T, int M, int N>
__host__ __device__
void printMat(T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

template <typename T, int M, int N>
__host__ __device__
void printMat(const T *A, int lda){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){printf("%.4f ",A[i + lda*j]);}
        printf("\n");
    }
}

/**
 * All functions are kept in this namespace
 *
 */
namespace grid {
    const int NUM_JOINTS = 14;
    const int ID_DYNAMIC_SHARED_MEM_COUNT = 1092;
    const int MINV_DYNAMIC_SHARED_MEM_COUNT = 2930;
    const int FD_DYNAMIC_SHARED_MEM_COUNT = 3126;
    const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = 4452;
    const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = 4452;
    const int ID_DU_MAX_SHARED_MEM_COUNT = 5138;
    const int FD_DU_MAX_SHARED_MEM_COUNT = 5740;
    const int SUGGESTED_THREADS = 512;
    // Define custom structs
    template <typename T>
    struct robotModel {
        T *d_XImats;
        int *d_topology_helpers;
    };
    template <typename T>
    struct gridData {
        // GPU INPUTS
        T *d_q_qd_u;
        T *d_q_qd;
        T *d_q;
        // CPU INPUTS
        T *h_q_qd_u;
        T *h_q_qd;
        T *h_q;
        // GPU OUTPUTS
        T *d_c;
        T *d_Minv;
        T *d_qdd;
        T *d_dc_du;
        T *d_df_du;
        // CPU OUTPUTS
        T *h_c;
        T *h_Minv;
        T *h_qdd;
        T *h_dc_du;
        T *h_df_du;
    };
    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, const T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(const T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Compute the dot product between two vectors
     *
     * Notes:
     *   Assumes computed by a single thread
     *
     * @param vec1 is the first vector of length N with stride S1
     * @param vec2 is the second vector of length N with stride S2
     * @return the resulting final value
     */
    template <typename T, int N, int S1, int S2>
    __device__
    T dot_prod(T *vec1, T *vec2) {
        T result = 0;
        for(int i = 0; i < N; i++) {
            result += vec1[i*S1] * vec2[i*S2];
        }
        return result;
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2];
        s_vecX[2] = -s_vec[1];
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5];
        s_vecX[5] = -s_vec[4];
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx0_peq(T *s_vecX, const T *s_vec) {
        s_vecX[1] += s_vec[2];
        s_vecX[2] += -s_vec[1];
        s_vecX[4] += s_vec[5];
        s_vecX[5] += -s_vec[4];
    }

    /**
     * Generates the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = s_vec[2]*alpha;
        s_vecX[2] = -s_vec[1]*alpha;
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[5]*alpha;
        s_vecX[5] = -s_vec[4]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 0
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx0_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[1] += s_vec[2]*alpha;
        s_vecX[2] += -s_vec[1]*alpha;
        s_vecX[4] += s_vec[5]*alpha;
        s_vecX[5] += -s_vec[4]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1(T *s_vecX, const T *s_vec) {
        s_vecX[0] = -s_vec[2];
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0];
        s_vecX[3] = -s_vec[5];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3];
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx1_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += -s_vec[2];
        s_vecX[2] += s_vec[0];
        s_vecX[3] += -s_vec[5];
        s_vecX[5] += s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = -s_vec[2]*alpha;
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = s_vec[0]*alpha;
        s_vecX[3] = -s_vec[5]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[3]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 1
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx1_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += -s_vec[2]*alpha;
        s_vecX[2] += s_vec[0]*alpha;
        s_vecX[3] += -s_vec[5]*alpha;
        s_vecX[5] += s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2(T *s_vecX, const T *s_vec) {
        s_vecX[0] = s_vec[1];
        s_vecX[1] = -s_vec[0];
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4];
        s_vecX[4] = -s_vec[3];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx2_peq(T *s_vecX, const T *s_vec) {
        s_vecX[0] += s_vec[1];
        s_vecX[1] += -s_vec[0];
        s_vecX[3] += s_vec[4];
        s_vecX[4] += -s_vec[3];
    }

    /**
     * Generates the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = s_vec[1]*alpha;
        s_vecX[1] = -s_vec[0]*alpha;
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[4]*alpha;
        s_vecX[4] = -s_vec[3]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 2
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx2_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] += s_vec[1]*alpha;
        s_vecX[1] += -s_vec[0]*alpha;
        s_vecX[3] += s_vec[4]*alpha;
        s_vecX[4] += -s_vec[3]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2];
        s_vecX[5] = -s_vec[1];
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx3_peq(T *s_vecX, const T *s_vec) {
        s_vecX[4] += s_vec[2];
        s_vecX[5] += -s_vec[1];
    }

    /**
     * Generates the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = static_cast<T>(0);
        s_vecX[4] = s_vec[2]*alpha;
        s_vecX[5] = -s_vec[1]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 3
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx3_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[4] += s_vec[2]*alpha;
        s_vecX[5] += -s_vec[1]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2];
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0];
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx4_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += -s_vec[2];
        s_vecX[5] += s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = -s_vec[2]*alpha;
        s_vecX[4] = static_cast<T>(0);
        s_vecX[5] = s_vec[0]*alpha;
    }

    /**
     * Adds the motion vector cross product matrix column 4
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx4_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += -s_vec[2]*alpha;
        s_vecX[5] += s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5(T *s_vecX, const T *s_vec) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1];
        s_vecX[4] = -s_vec[0];
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mx5_peq(T *s_vecX, const T *s_vec) {
        s_vecX[3] += s_vec[1];
        s_vecX[4] += -s_vec[0];
    }

    /**
     * Generates the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[0] = static_cast<T>(0);
        s_vecX[1] = static_cast<T>(0);
        s_vecX[2] = static_cast<T>(0);
        s_vecX[3] = s_vec[1]*alpha;
        s_vecX[4] = -s_vec[0]*alpha;
        s_vecX[5] = static_cast<T>(0);
    }

    /**
     * Adds the motion vector cross product matrix column 5
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mx5_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {
        s_vecX[3] += s_vec[1]*alpha;
        s_vecX[4] += -s_vec[0]*alpha;
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0<T>(s_vecX, s_vec); break;
            case 1: mx1<T>(s_vecX, s_vec); break;
            case 2: mx2<T>(s_vecX, s_vec); break;
            case 3: mx3<T>(s_vecX, s_vec); break;
            case 4: mx4<T>(s_vecX, s_vec); break;
            case 5: mx5<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     */
    template <typename T>
    __device__
    void mxX_peq(T *s_vecX, const T *s_vec, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq<T>(s_vecX, s_vec); break;
            case 1: mx1_peq<T>(s_vecX, s_vec); break;
            case 2: mx2_peq<T>(s_vecX, s_vec); break;
            case 3: mx3_peq<T>(s_vecX, s_vec); break;
            case 4: mx4_peq<T>(s_vecX, s_vec); break;
            case 5: mx5_peq<T>(s_vecX, s_vec); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix for a runtime selected column
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_vecX is the destination vector
     * @param s_vec is the source vector
     * @param alpha is the scaling factor
     */
    template <typename T>
    __device__
    void mxX_peq_scaled(T *s_vecX, const T *s_vec, const T alpha, const int S_ind) {
        switch(S_ind){
            case 0: mx0_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 1: mx1_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 2: mx2_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 3: mx3_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 4: mx4_peq_scaled<T>(s_vecX, s_vec, alpha); break;
            case 5: mx5_peq_scaled<T>(s_vecX, s_vec, alpha); break;
        }
    }

    /**
     * Generates the motion vector cross product matrix
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 0] = static_cast<T>(0);
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*0 + 3] = static_cast<T>(0);
        s_matX[6*0 + 4] = static_cast<T>(0);
        s_matX[6*0 + 5] = static_cast<T>(0);
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 1] = static_cast<T>(0);
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*1 + 3] = static_cast<T>(0);
        s_matX[6*1 + 4] = static_cast<T>(0);
        s_matX[6*1 + 5] = static_cast<T>(0);
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*2 + 2] = static_cast<T>(0);
        s_matX[6*2 + 3] = static_cast<T>(0);
        s_matX[6*2 + 4] = static_cast<T>(0);
        s_matX[6*2 + 5] = static_cast<T>(0);
        s_matX[6*3 + 0] = static_cast<T>(0);
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 3] = static_cast<T>(0);
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 1] = static_cast<T>(0);
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 4] = static_cast<T>(0);
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 2] = static_cast<T>(0);
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
        s_matX[6*5 + 5] = static_cast<T>(0);
    }

    /**
     * Generates the motion vector cross product matrix for a pre-zeroed destination
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *   Assumes destination is zeroed
     *
     * @param s_matX is the destination matrix
     * @param s_vecX is the source vector
     */
    template <typename T>
    __device__
    void fx_zeroed(T *s_matX, const T *s_vecX) {
        s_matX[6*0 + 1] = s_vecX[2];
        s_matX[6*0 + 2] = -s_vecX[1];
        s_matX[6*1 + 0] = -s_vecX[2];
        s_matX[6*1 + 2] = s_vecX[0];
        s_matX[6*2 + 0] = s_vecX[1];
        s_matX[6*2 + 1] = -s_vecX[0];
        s_matX[6*3 + 1] = s_vecX[5];
        s_matX[6*3 + 2] = -s_vecX[4];
        s_matX[6*3 + 4] = s_vecX[2];
        s_matX[6*3 + 5] = -s_vecX[1];
        s_matX[6*4 + 0] = -s_vecX[5];
        s_matX[6*4 + 2] = s_vecX[3];
        s_matX[6*4 + 3] = -s_vecX[2];
        s_matX[6*4 + 5] = s_vecX[0];
        s_matX[6*5 + 0] = s_vecX[4];
        s_matX[6*5 + 1] = -s_vecX[3];
        s_matX[6*5 + 3] = s_vecX[1];
        s_matX[6*5 + 4] = -s_vecX[0];
    }

    /**
     * Generates the motion vector cross product matrix and multiples by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] = -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] =  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] = -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] =                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] =                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] =                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }

    /**
     * Adds the motion vector cross product matrix multiplied by the input vector
     *
     * Notes:
     *   Assumes only one thread is running each function call
     *
     * @param s_result is the result vector
     * @param s_fxVec is the fx vector
     * @param s_timesVec is the multipled vector
     */
    template <typename T>
    __device__
    void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {
        s_result[0] += -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];
        s_result[1] +=  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];
        s_result[2] += -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];
        s_result[3] +=                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];
        s_result[4] +=                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];
        s_result[5] +=                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];
    }

    /**
     * Initializes the topology_helpers in GPU memory
     *
     * @return A pointer to the topology_helpers memory in the GPU
     */
    template <typename T>
    __host__
    int *init_topology_helpers() {
        int h_topology_helpers[] = {-1,0,1,2,3,4,5,-1,7,8,9,10,11,12, // parent_inds
                                    0,1,2,3,4,5,6,0,1,2,3,4,5,6, // num_ancestors
                                    7,6,5,4,3,2,1,7,6,5,4,3,2,1, // num_subtree
                                    0,0,1,3,6,10,15,21,21,22,24,27,31,36,42, // running_sum_num_ancestors
                                    0,7,13,18,22,25,27,28,35,41,46,50,53,55}; // running_sum_num_subtree
        int *d_topology_helpers; gpuErrchk(cudaMalloc((void**)&d_topology_helpers,71*sizeof(int)));
        gpuErrchk(cudaMemcpy(d_topology_helpers,h_topology_helpers,71*sizeof(int),cudaMemcpyHostToDevice));
        return d_topology_helpers;
    }

    /**
     * Initializes the Xmats and Imats in GPU memory
     *
     * Notes:
     *   Memory order is X[0...N], I[0...N]
     *
     * @return A pointer to the XI memory in the GPU
     */
    template <typename T>
    __host__
    T* init_XImats() {
        T *h_XImats = (T *)malloc(1008*sizeof(T));
        // X[0]
        h_XImats[0] = static_cast<T>(0);
        h_XImats[1] = static_cast<T>(0);
        h_XImats[2] = static_cast<T>(0);
        h_XImats[3] = static_cast<T>(0);
        h_XImats[4] = static_cast<T>(0);
        h_XImats[5] = static_cast<T>(0);
        h_XImats[6] = static_cast<T>(0);
        h_XImats[7] = static_cast<T>(0);
        h_XImats[8] = static_cast<T>(0);
        h_XImats[9] = static_cast<T>(0);
        h_XImats[10] = static_cast<T>(0);
        h_XImats[11] = static_cast<T>(0);
        h_XImats[12] = static_cast<T>(0);
        h_XImats[13] = static_cast<T>(0);
        h_XImats[14] = static_cast<T>(1.00000000000000);
        h_XImats[15] = static_cast<T>(0);
        h_XImats[16] = static_cast<T>(0);
        h_XImats[17] = static_cast<T>(0);
        h_XImats[18] = static_cast<T>(0);
        h_XImats[19] = static_cast<T>(0);
        h_XImats[20] = static_cast<T>(0);
        h_XImats[21] = static_cast<T>(0);
        h_XImats[22] = static_cast<T>(0);
        h_XImats[23] = static_cast<T>(0);
        h_XImats[24] = static_cast<T>(0);
        h_XImats[25] = static_cast<T>(0);
        h_XImats[26] = static_cast<T>(0);
        h_XImats[27] = static_cast<T>(0);
        h_XImats[28] = static_cast<T>(0);
        h_XImats[29] = static_cast<T>(0);
        h_XImats[30] = static_cast<T>(0);
        h_XImats[31] = static_cast<T>(0);
        h_XImats[32] = static_cast<T>(0);
        h_XImats[33] = static_cast<T>(0);
        h_XImats[34] = static_cast<T>(0);
        h_XImats[35] = static_cast<T>(1.00000000000000);
        // X[1]
        h_XImats[36] = static_cast<T>(0);
        h_XImats[37] = static_cast<T>(0);
        h_XImats[38] = static_cast<T>(0);
        h_XImats[39] = static_cast<T>(0);
        h_XImats[40] = static_cast<T>(0);
        h_XImats[41] = static_cast<T>(-0.202500000000000);
        h_XImats[42] = static_cast<T>(0);
        h_XImats[43] = static_cast<T>(0);
        h_XImats[44] = static_cast<T>(1.00000000000000);
        h_XImats[45] = static_cast<T>(0);
        h_XImats[46] = static_cast<T>(0);
        h_XImats[47] = static_cast<T>(0);
        h_XImats[48] = static_cast<T>(0);
        h_XImats[49] = static_cast<T>(0);
        h_XImats[50] = static_cast<T>(0);
        h_XImats[51] = static_cast<T>(0);
        h_XImats[52] = static_cast<T>(0);
        h_XImats[53] = static_cast<T>(0);
        h_XImats[54] = static_cast<T>(0);
        h_XImats[55] = static_cast<T>(0);
        h_XImats[56] = static_cast<T>(0);
        h_XImats[57] = static_cast<T>(0);
        h_XImats[58] = static_cast<T>(0);
        h_XImats[59] = static_cast<T>(0);
        h_XImats[60] = static_cast<T>(0);
        h_XImats[61] = static_cast<T>(0);
        h_XImats[62] = static_cast<T>(0);
        h_XImats[63] = static_cast<T>(0);
        h_XImats[64] = static_cast<T>(0);
        h_XImats[65] = static_cast<T>(1.00000000000000);
        h_XImats[66] = static_cast<T>(0);
        h_XImats[67] = static_cast<T>(0);
        h_XImats[68] = static_cast<T>(0);
        h_XImats[69] = static_cast<T>(0);
        h_XImats[70] = static_cast<T>(0);
        h_XImats[71] = static_cast<T>(0);
        // X[2]
        h_XImats[72] = static_cast<T>(0);
        h_XImats[73] = static_cast<T>(0);
        h_XImats[74] = static_cast<T>(0);
        h_XImats[75] = static_cast<T>(0);
        h_XImats[76] = static_cast<T>(0);
        h_XImats[77] = static_cast<T>(0);
        h_XImats[78] = static_cast<T>(0);
        h_XImats[79] = static_cast<T>(0);
        h_XImats[80] = static_cast<T>(1.00000000000000);
        h_XImats[81] = static_cast<T>(0);
        h_XImats[82] = static_cast<T>(0);
        h_XImats[83] = static_cast<T>(0);
        h_XImats[84] = static_cast<T>(0);
        h_XImats[85] = static_cast<T>(0);
        h_XImats[86] = static_cast<T>(0);
        h_XImats[87] = static_cast<T>(0);
        h_XImats[88] = static_cast<T>(0);
        h_XImats[89] = static_cast<T>(0);
        h_XImats[90] = static_cast<T>(0);
        h_XImats[91] = static_cast<T>(0);
        h_XImats[92] = static_cast<T>(0);
        h_XImats[93] = static_cast<T>(0);
        h_XImats[94] = static_cast<T>(0);
        h_XImats[95] = static_cast<T>(0);
        h_XImats[96] = static_cast<T>(0);
        h_XImats[97] = static_cast<T>(0);
        h_XImats[98] = static_cast<T>(0);
        h_XImats[99] = static_cast<T>(0);
        h_XImats[100] = static_cast<T>(0);
        h_XImats[101] = static_cast<T>(1.00000000000000);
        h_XImats[102] = static_cast<T>(0);
        h_XImats[103] = static_cast<T>(0);
        h_XImats[104] = static_cast<T>(0);
        h_XImats[105] = static_cast<T>(0);
        h_XImats[106] = static_cast<T>(0);
        h_XImats[107] = static_cast<T>(0);
        // X[3]
        h_XImats[108] = static_cast<T>(0);
        h_XImats[109] = static_cast<T>(0);
        h_XImats[110] = static_cast<T>(0);
        h_XImats[111] = static_cast<T>(0);
        h_XImats[112] = static_cast<T>(0);
        h_XImats[113] = static_cast<T>(0.215500000000000);
        h_XImats[114] = static_cast<T>(0);
        h_XImats[115] = static_cast<T>(0);
        h_XImats[116] = static_cast<T>(-1.00000000000000);
        h_XImats[117] = static_cast<T>(0);
        h_XImats[118] = static_cast<T>(0);
        h_XImats[119] = static_cast<T>(0);
        h_XImats[120] = static_cast<T>(0);
        h_XImats[121] = static_cast<T>(0);
        h_XImats[122] = static_cast<T>(0);
        h_XImats[123] = static_cast<T>(0);
        h_XImats[124] = static_cast<T>(0);
        h_XImats[125] = static_cast<T>(0);
        h_XImats[126] = static_cast<T>(0);
        h_XImats[127] = static_cast<T>(0);
        h_XImats[128] = static_cast<T>(0);
        h_XImats[129] = static_cast<T>(0);
        h_XImats[130] = static_cast<T>(0);
        h_XImats[131] = static_cast<T>(0);
        h_XImats[132] = static_cast<T>(0);
        h_XImats[133] = static_cast<T>(0);
        h_XImats[134] = static_cast<T>(0);
        h_XImats[135] = static_cast<T>(0);
        h_XImats[136] = static_cast<T>(0);
        h_XImats[137] = static_cast<T>(-1.00000000000000);
        h_XImats[138] = static_cast<T>(0);
        h_XImats[139] = static_cast<T>(0);
        h_XImats[140] = static_cast<T>(0);
        h_XImats[141] = static_cast<T>(0);
        h_XImats[142] = static_cast<T>(0);
        h_XImats[143] = static_cast<T>(0);
        // X[4]
        h_XImats[144] = static_cast<T>(0);
        h_XImats[145] = static_cast<T>(0);
        h_XImats[146] = static_cast<T>(0);
        h_XImats[147] = static_cast<T>(0);
        h_XImats[148] = static_cast<T>(0);
        h_XImats[149] = static_cast<T>(0);
        h_XImats[150] = static_cast<T>(0);
        h_XImats[151] = static_cast<T>(0);
        h_XImats[152] = static_cast<T>(1.00000000000000);
        h_XImats[153] = static_cast<T>(0);
        h_XImats[154] = static_cast<T>(0);
        h_XImats[155] = static_cast<T>(0);
        h_XImats[156] = static_cast<T>(0);
        h_XImats[157] = static_cast<T>(0);
        h_XImats[158] = static_cast<T>(0);
        h_XImats[159] = static_cast<T>(0);
        h_XImats[160] = static_cast<T>(0);
        h_XImats[161] = static_cast<T>(0);
        h_XImats[162] = static_cast<T>(0);
        h_XImats[163] = static_cast<T>(0);
        h_XImats[164] = static_cast<T>(0);
        h_XImats[165] = static_cast<T>(0);
        h_XImats[166] = static_cast<T>(0);
        h_XImats[167] = static_cast<T>(0);
        h_XImats[168] = static_cast<T>(0);
        h_XImats[169] = static_cast<T>(0);
        h_XImats[170] = static_cast<T>(0);
        h_XImats[171] = static_cast<T>(0);
        h_XImats[172] = static_cast<T>(0);
        h_XImats[173] = static_cast<T>(1.00000000000000);
        h_XImats[174] = static_cast<T>(0);
        h_XImats[175] = static_cast<T>(0);
        h_XImats[176] = static_cast<T>(0);
        h_XImats[177] = static_cast<T>(0);
        h_XImats[178] = static_cast<T>(0);
        h_XImats[179] = static_cast<T>(0);
        // X[5]
        h_XImats[180] = static_cast<T>(0);
        h_XImats[181] = static_cast<T>(0);
        h_XImats[182] = static_cast<T>(0);
        h_XImats[183] = static_cast<T>(0);
        h_XImats[184] = static_cast<T>(0);
        h_XImats[185] = static_cast<T>(0.215500000000000);
        h_XImats[186] = static_cast<T>(0);
        h_XImats[187] = static_cast<T>(0);
        h_XImats[188] = static_cast<T>(-1.00000000000000);
        h_XImats[189] = static_cast<T>(0);
        h_XImats[190] = static_cast<T>(0);
        h_XImats[191] = static_cast<T>(0);
        h_XImats[192] = static_cast<T>(0);
        h_XImats[193] = static_cast<T>(0);
        h_XImats[194] = static_cast<T>(0);
        h_XImats[195] = static_cast<T>(0);
        h_XImats[196] = static_cast<T>(0);
        h_XImats[197] = static_cast<T>(0);
        h_XImats[198] = static_cast<T>(0);
        h_XImats[199] = static_cast<T>(0);
        h_XImats[200] = static_cast<T>(0);
        h_XImats[201] = static_cast<T>(0);
        h_XImats[202] = static_cast<T>(0);
        h_XImats[203] = static_cast<T>(0);
        h_XImats[204] = static_cast<T>(0);
        h_XImats[205] = static_cast<T>(0);
        h_XImats[206] = static_cast<T>(0);
        h_XImats[207] = static_cast<T>(0);
        h_XImats[208] = static_cast<T>(0);
        h_XImats[209] = static_cast<T>(-1.00000000000000);
        h_XImats[210] = static_cast<T>(0);
        h_XImats[211] = static_cast<T>(0);
        h_XImats[212] = static_cast<T>(0);
        h_XImats[213] = static_cast<T>(0);
        h_XImats[214] = static_cast<T>(0);
        h_XImats[215] = static_cast<T>(0);
        // X[6]
        h_XImats[216] = static_cast<T>(0);
        h_XImats[217] = static_cast<T>(0);
        h_XImats[218] = static_cast<T>(0);
        h_XImats[219] = static_cast<T>(0);
        h_XImats[220] = static_cast<T>(0);
        h_XImats[221] = static_cast<T>(0);
        h_XImats[222] = static_cast<T>(0);
        h_XImats[223] = static_cast<T>(0);
        h_XImats[224] = static_cast<T>(1.00000000000000);
        h_XImats[225] = static_cast<T>(0);
        h_XImats[226] = static_cast<T>(0);
        h_XImats[227] = static_cast<T>(0);
        h_XImats[228] = static_cast<T>(0);
        h_XImats[229] = static_cast<T>(0);
        h_XImats[230] = static_cast<T>(0);
        h_XImats[231] = static_cast<T>(0);
        h_XImats[232] = static_cast<T>(0);
        h_XImats[233] = static_cast<T>(0);
        h_XImats[234] = static_cast<T>(0);
        h_XImats[235] = static_cast<T>(0);
        h_XImats[236] = static_cast<T>(0);
        h_XImats[237] = static_cast<T>(0);
        h_XImats[238] = static_cast<T>(0);
        h_XImats[239] = static_cast<T>(0);
        h_XImats[240] = static_cast<T>(0);
        h_XImats[241] = static_cast<T>(0);
        h_XImats[242] = static_cast<T>(0);
        h_XImats[243] = static_cast<T>(0);
        h_XImats[244] = static_cast<T>(0);
        h_XImats[245] = static_cast<T>(1.00000000000000);
        h_XImats[246] = static_cast<T>(0);
        h_XImats[247] = static_cast<T>(0);
        h_XImats[248] = static_cast<T>(0);
        h_XImats[249] = static_cast<T>(0);
        h_XImats[250] = static_cast<T>(0);
        h_XImats[251] = static_cast<T>(0);
        // X[7]
        h_XImats[252] = static_cast<T>(0);
        h_XImats[253] = static_cast<T>(0);
        h_XImats[254] = static_cast<T>(0);
        h_XImats[255] = static_cast<T>(0);
        h_XImats[256] = static_cast<T>(0);
        h_XImats[257] = static_cast<T>(0);
        h_XImats[258] = static_cast<T>(0);
        h_XImats[259] = static_cast<T>(0);
        h_XImats[260] = static_cast<T>(0);
        h_XImats[261] = static_cast<T>(0);
        h_XImats[262] = static_cast<T>(0);
        h_XImats[263] = static_cast<T>(0);
        h_XImats[264] = static_cast<T>(0);
        h_XImats[265] = static_cast<T>(0);
        h_XImats[266] = static_cast<T>(1.00000000000000);
        h_XImats[267] = static_cast<T>(0);
        h_XImats[268] = static_cast<T>(0);
        h_XImats[269] = static_cast<T>(0);
        h_XImats[270] = static_cast<T>(0);
        h_XImats[271] = static_cast<T>(0);
        h_XImats[272] = static_cast<T>(0);
        h_XImats[273] = static_cast<T>(0);
        h_XImats[274] = static_cast<T>(0);
        h_XImats[275] = static_cast<T>(0);
        h_XImats[276] = static_cast<T>(0);
        h_XImats[277] = static_cast<T>(0);
        h_XImats[278] = static_cast<T>(0);
        h_XImats[279] = static_cast<T>(0);
        h_XImats[280] = static_cast<T>(0);
        h_XImats[281] = static_cast<T>(0);
        h_XImats[282] = static_cast<T>(0);
        h_XImats[283] = static_cast<T>(0);
        h_XImats[284] = static_cast<T>(0);
        h_XImats[285] = static_cast<T>(0);
        h_XImats[286] = static_cast<T>(0);
        h_XImats[287] = static_cast<T>(1.00000000000000);
        // X[8]
        h_XImats[288] = static_cast<T>(0);
        h_XImats[289] = static_cast<T>(0);
        h_XImats[290] = static_cast<T>(0);
        h_XImats[291] = static_cast<T>(0);
        h_XImats[292] = static_cast<T>(0);
        h_XImats[293] = static_cast<T>(-0.202500000000000);
        h_XImats[294] = static_cast<T>(0);
        h_XImats[295] = static_cast<T>(0);
        h_XImats[296] = static_cast<T>(1.00000000000000);
        h_XImats[297] = static_cast<T>(0);
        h_XImats[298] = static_cast<T>(0);
        h_XImats[299] = static_cast<T>(0);
        h_XImats[300] = static_cast<T>(0);
        h_XImats[301] = static_cast<T>(0);
        h_XImats[302] = static_cast<T>(0);
        h_XImats[303] = static_cast<T>(0);
        h_XImats[304] = static_cast<T>(0);
        h_XImats[305] = static_cast<T>(0);
        h_XImats[306] = static_cast<T>(0);
        h_XImats[307] = static_cast<T>(0);
        h_XImats[308] = static_cast<T>(0);
        h_XImats[309] = static_cast<T>(0);
        h_XImats[310] = static_cast<T>(0);
        h_XImats[311] = static_cast<T>(0);
        h_XImats[312] = static_cast<T>(0);
        h_XImats[313] = static_cast<T>(0);
        h_XImats[314] = static_cast<T>(0);
        h_XImats[315] = static_cast<T>(0);
        h_XImats[316] = static_cast<T>(0);
        h_XImats[317] = static_cast<T>(1.00000000000000);
        h_XImats[318] = static_cast<T>(0);
        h_XImats[319] = static_cast<T>(0);
        h_XImats[320] = static_cast<T>(0);
        h_XImats[321] = static_cast<T>(0);
        h_XImats[322] = static_cast<T>(0);
        h_XImats[323] = static_cast<T>(0);
        // X[9]
        h_XImats[324] = static_cast<T>(0);
        h_XImats[325] = static_cast<T>(0);
        h_XImats[326] = static_cast<T>(0);
        h_XImats[327] = static_cast<T>(0);
        h_XImats[328] = static_cast<T>(0);
        h_XImats[329] = static_cast<T>(0);
        h_XImats[330] = static_cast<T>(0);
        h_XImats[331] = static_cast<T>(0);
        h_XImats[332] = static_cast<T>(1.00000000000000);
        h_XImats[333] = static_cast<T>(0);
        h_XImats[334] = static_cast<T>(0);
        h_XImats[335] = static_cast<T>(0);
        h_XImats[336] = static_cast<T>(0);
        h_XImats[337] = static_cast<T>(0);
        h_XImats[338] = static_cast<T>(0);
        h_XImats[339] = static_cast<T>(0);
        h_XImats[340] = static_cast<T>(0);
        h_XImats[341] = static_cast<T>(0);
        h_XImats[342] = static_cast<T>(0);
        h_XImats[343] = static_cast<T>(0);
        h_XImats[344] = static_cast<T>(0);
        h_XImats[345] = static_cast<T>(0);
        h_XImats[346] = static_cast<T>(0);
        h_XImats[347] = static_cast<T>(0);
        h_XImats[348] = static_cast<T>(0);
        h_XImats[349] = static_cast<T>(0);
        h_XImats[350] = static_cast<T>(0);
        h_XImats[351] = static_cast<T>(0);
        h_XImats[352] = static_cast<T>(0);
        h_XImats[353] = static_cast<T>(1.00000000000000);
        h_XImats[354] = static_cast<T>(0);
        h_XImats[355] = static_cast<T>(0);
        h_XImats[356] = static_cast<T>(0);
        h_XImats[357] = static_cast<T>(0);
        h_XImats[358] = static_cast<T>(0);
        h_XImats[359] = static_cast<T>(0);
        // X[10]
        h_XImats[360] = static_cast<T>(0);
        h_XImats[361] = static_cast<T>(0);
        h_XImats[362] = static_cast<T>(0);
        h_XImats[363] = static_cast<T>(0);
        h_XImats[364] = static_cast<T>(0);
        h_XImats[365] = static_cast<T>(0.215500000000000);
        h_XImats[366] = static_cast<T>(0);
        h_XImats[367] = static_cast<T>(0);
        h_XImats[368] = static_cast<T>(-1.00000000000000);
        h_XImats[369] = static_cast<T>(0);
        h_XImats[370] = static_cast<T>(0);
        h_XImats[371] = static_cast<T>(0);
        h_XImats[372] = static_cast<T>(0);
        h_XImats[373] = static_cast<T>(0);
        h_XImats[374] = static_cast<T>(0);
        h_XImats[375] = static_cast<T>(0);
        h_XImats[376] = static_cast<T>(0);
        h_XImats[377] = static_cast<T>(0);
        h_XImats[378] = static_cast<T>(0);
        h_XImats[379] = static_cast<T>(0);
        h_XImats[380] = static_cast<T>(0);
        h_XImats[381] = static_cast<T>(0);
        h_XImats[382] = static_cast<T>(0);
        h_XImats[383] = static_cast<T>(0);
        h_XImats[384] = static_cast<T>(0);
        h_XImats[385] = static_cast<T>(0);
        h_XImats[386] = static_cast<T>(0);
        h_XImats[387] = static_cast<T>(0);
        h_XImats[388] = static_cast<T>(0);
        h_XImats[389] = static_cast<T>(-1.00000000000000);
        h_XImats[390] = static_cast<T>(0);
        h_XImats[391] = static_cast<T>(0);
        h_XImats[392] = static_cast<T>(0);
        h_XImats[393] = static_cast<T>(0);
        h_XImats[394] = static_cast<T>(0);
        h_XImats[395] = static_cast<T>(0);
        // X[11]
        h_XImats[396] = static_cast<T>(0);
        h_XImats[397] = static_cast<T>(0);
        h_XImats[398] = static_cast<T>(0);
        h_XImats[399] = static_cast<T>(0);
        h_XImats[400] = static_cast<T>(0);
        h_XImats[401] = static_cast<T>(0);
        h_XImats[402] = static_cast<T>(0);
        h_XImats[403] = static_cast<T>(0);
        h_XImats[404] = static_cast<T>(1.00000000000000);
        h_XImats[405] = static_cast<T>(0);
        h_XImats[406] = static_cast<T>(0);
        h_XImats[407] = static_cast<T>(0);
        h_XImats[408] = static_cast<T>(0);
        h_XImats[409] = static_cast<T>(0);
        h_XImats[410] = static_cast<T>(0);
        h_XImats[411] = static_cast<T>(0);
        h_XImats[412] = static_cast<T>(0);
        h_XImats[413] = static_cast<T>(0);
        h_XImats[414] = static_cast<T>(0);
        h_XImats[415] = static_cast<T>(0);
        h_XImats[416] = static_cast<T>(0);
        h_XImats[417] = static_cast<T>(0);
        h_XImats[418] = static_cast<T>(0);
        h_XImats[419] = static_cast<T>(0);
        h_XImats[420] = static_cast<T>(0);
        h_XImats[421] = static_cast<T>(0);
        h_XImats[422] = static_cast<T>(0);
        h_XImats[423] = static_cast<T>(0);
        h_XImats[424] = static_cast<T>(0);
        h_XImats[425] = static_cast<T>(1.00000000000000);
        h_XImats[426] = static_cast<T>(0);
        h_XImats[427] = static_cast<T>(0);
        h_XImats[428] = static_cast<T>(0);
        h_XImats[429] = static_cast<T>(0);
        h_XImats[430] = static_cast<T>(0);
        h_XImats[431] = static_cast<T>(0);
        // X[12]
        h_XImats[432] = static_cast<T>(0);
        h_XImats[433] = static_cast<T>(0);
        h_XImats[434] = static_cast<T>(0);
        h_XImats[435] = static_cast<T>(0);
        h_XImats[436] = static_cast<T>(0);
        h_XImats[437] = static_cast<T>(0.215500000000000);
        h_XImats[438] = static_cast<T>(0);
        h_XImats[439] = static_cast<T>(0);
        h_XImats[440] = static_cast<T>(-1.00000000000000);
        h_XImats[441] = static_cast<T>(0);
        h_XImats[442] = static_cast<T>(0);
        h_XImats[443] = static_cast<T>(0);
        h_XImats[444] = static_cast<T>(0);
        h_XImats[445] = static_cast<T>(0);
        h_XImats[446] = static_cast<T>(0);
        h_XImats[447] = static_cast<T>(0);
        h_XImats[448] = static_cast<T>(0);
        h_XImats[449] = static_cast<T>(0);
        h_XImats[450] = static_cast<T>(0);
        h_XImats[451] = static_cast<T>(0);
        h_XImats[452] = static_cast<T>(0);
        h_XImats[453] = static_cast<T>(0);
        h_XImats[454] = static_cast<T>(0);
        h_XImats[455] = static_cast<T>(0);
        h_XImats[456] = static_cast<T>(0);
        h_XImats[457] = static_cast<T>(0);
        h_XImats[458] = static_cast<T>(0);
        h_XImats[459] = static_cast<T>(0);
        h_XImats[460] = static_cast<T>(0);
        h_XImats[461] = static_cast<T>(-1.00000000000000);
        h_XImats[462] = static_cast<T>(0);
        h_XImats[463] = static_cast<T>(0);
        h_XImats[464] = static_cast<T>(0);
        h_XImats[465] = static_cast<T>(0);
        h_XImats[466] = static_cast<T>(0);
        h_XImats[467] = static_cast<T>(0);
        // X[13]
        h_XImats[468] = static_cast<T>(0);
        h_XImats[469] = static_cast<T>(0);
        h_XImats[470] = static_cast<T>(0);
        h_XImats[471] = static_cast<T>(0);
        h_XImats[472] = static_cast<T>(0);
        h_XImats[473] = static_cast<T>(0);
        h_XImats[474] = static_cast<T>(0);
        h_XImats[475] = static_cast<T>(0);
        h_XImats[476] = static_cast<T>(1.00000000000000);
        h_XImats[477] = static_cast<T>(0);
        h_XImats[478] = static_cast<T>(0);
        h_XImats[479] = static_cast<T>(0);
        h_XImats[480] = static_cast<T>(0);
        h_XImats[481] = static_cast<T>(0);
        h_XImats[482] = static_cast<T>(0);
        h_XImats[483] = static_cast<T>(0);
        h_XImats[484] = static_cast<T>(0);
        h_XImats[485] = static_cast<T>(0);
        h_XImats[486] = static_cast<T>(0);
        h_XImats[487] = static_cast<T>(0);
        h_XImats[488] = static_cast<T>(0);
        h_XImats[489] = static_cast<T>(0);
        h_XImats[490] = static_cast<T>(0);
        h_XImats[491] = static_cast<T>(0);
        h_XImats[492] = static_cast<T>(0);
        h_XImats[493] = static_cast<T>(0);
        h_XImats[494] = static_cast<T>(0);
        h_XImats[495] = static_cast<T>(0);
        h_XImats[496] = static_cast<T>(0);
        h_XImats[497] = static_cast<T>(1.00000000000000);
        h_XImats[498] = static_cast<T>(0);
        h_XImats[499] = static_cast<T>(0);
        h_XImats[500] = static_cast<T>(0);
        h_XImats[501] = static_cast<T>(0);
        h_XImats[502] = static_cast<T>(0);
        h_XImats[503] = static_cast<T>(0);
        // I[0]
        h_XImats[504] = static_cast<T>(0.20912799999999998);
        h_XImats[505] = static_cast<T>(0.0);
        h_XImats[506] = static_cast<T>(0.0);
        h_XImats[507] = static_cast<T>(0.0);
        h_XImats[508] = static_cast<T>(-0.6911999999999999);
        h_XImats[509] = static_cast<T>(-0.17279999999999998);
        h_XImats[510] = static_cast<T>(0.0);
        h_XImats[511] = static_cast<T>(0.198944);
        h_XImats[512] = static_cast<T>(-0.0002640000000000038);
        h_XImats[513] = static_cast<T>(0.6911999999999999);
        h_XImats[514] = static_cast<T>(0.0);
        h_XImats[515] = static_cast<T>(0.0);
        h_XImats[516] = static_cast<T>(0.0);
        h_XImats[517] = static_cast<T>(-0.0002640000000000038);
        h_XImats[518] = static_cast<T>(0.022684000000000003);
        h_XImats[519] = static_cast<T>(0.17279999999999998);
        h_XImats[520] = static_cast<T>(0.0);
        h_XImats[521] = static_cast<T>(0.0);
        h_XImats[522] = static_cast<T>(0.0);
        h_XImats[523] = static_cast<T>(0.6911999999999999);
        h_XImats[524] = static_cast<T>(0.17279999999999998);
        h_XImats[525] = static_cast<T>(5.76);
        h_XImats[526] = static_cast<T>(0.0);
        h_XImats[527] = static_cast<T>(0.0);
        h_XImats[528] = static_cast<T>(-0.6911999999999999);
        h_XImats[529] = static_cast<T>(0.0);
        h_XImats[530] = static_cast<T>(0.0);
        h_XImats[531] = static_cast<T>(0.0);
        h_XImats[532] = static_cast<T>(5.76);
        h_XImats[533] = static_cast<T>(0.0);
        h_XImats[534] = static_cast<T>(-0.17279999999999998);
        h_XImats[535] = static_cast<T>(0.0);
        h_XImats[536] = static_cast<T>(0.0);
        h_XImats[537] = static_cast<T>(0.0);
        h_XImats[538] = static_cast<T>(0.0);
        h_XImats[539] = static_cast<T>(5.76);
        // I[1]
        h_XImats[540] = static_cast<T>(0.09710574999999999);
        h_XImats[541] = static_cast<T>(-1.2394999999999979e-05);
        h_XImats[542] = static_cast<T>(-9.999999999994822e-09);
        h_XImats[543] = static_cast<T>(0.0);
        h_XImats[544] = static_cast<T>(-0.2667);
        h_XImats[545] = static_cast<T>(0.37465);
        h_XImats[546] = static_cast<T>(-1.2394999999999979e-05);
        h_XImats[547] = static_cast<T>(0.052801971499999996);
        h_XImats[548] = static_cast<T>(-3.5300000000001996e-05);
        h_XImats[549] = static_cast<T>(0.2667);
        h_XImats[550] = static_cast<T>(0.0);
        h_XImats[551] = static_cast<T>(-0.0019049999999999998);
        h_XImats[552] = static_cast<T>(-9.99999999998127e-09);
        h_XImats[553] = static_cast<T>(-3.529999999999853e-05);
        h_XImats[554] = static_cast<T>(0.0552049215);
        h_XImats[555] = static_cast<T>(-0.37465);
        h_XImats[556] = static_cast<T>(0.0019049999999999998);
        h_XImats[557] = static_cast<T>(0.0);
        h_XImats[558] = static_cast<T>(0.0);
        h_XImats[559] = static_cast<T>(0.2667);
        h_XImats[560] = static_cast<T>(-0.37465);
        h_XImats[561] = static_cast<T>(6.35);
        h_XImats[562] = static_cast<T>(0.0);
        h_XImats[563] = static_cast<T>(0.0);
        h_XImats[564] = static_cast<T>(-0.2667);
        h_XImats[565] = static_cast<T>(0.0);
        h_XImats[566] = static_cast<T>(0.0019049999999999998);
        h_XImats[567] = static_cast<T>(0.0);
        h_XImats[568] = static_cast<T>(6.35);
        h_XImats[569] = static_cast<T>(0.0);
        h_XImats[570] = static_cast<T>(0.37465);
        h_XImats[571] = static_cast<T>(-0.0019049999999999998);
        h_XImats[572] = static_cast<T>(0.0);
        h_XImats[573] = static_cast<T>(0.0);
        h_XImats[574] = static_cast<T>(0.0);
        h_XImats[575] = static_cast<T>(6.35);
        // I[2]
        h_XImats[576] = static_cast<T>(0.1496);
        h_XImats[577] = static_cast<T>(0.0);
        h_XImats[578] = static_cast<T>(0.0);
        h_XImats[579] = static_cast<T>(0.0);
        h_XImats[580] = static_cast<T>(-0.455);
        h_XImats[581] = static_cast<T>(0.105);
        h_XImats[582] = static_cast<T>(0.0);
        h_XImats[583] = static_cast<T>(0.14215);
        h_XImats[584] = static_cast<T>(0.0003499999999999996);
        h_XImats[585] = static_cast<T>(0.455);
        h_XImats[586] = static_cast<T>(0.0);
        h_XImats[587] = static_cast<T>(0.0);
        h_XImats[588] = static_cast<T>(0.0);
        h_XImats[589] = static_cast<T>(0.0003499999999999996);
        h_XImats[590] = static_cast<T>(0.01395);
        h_XImats[591] = static_cast<T>(-0.105);
        h_XImats[592] = static_cast<T>(0.0);
        h_XImats[593] = static_cast<T>(0.0);
        h_XImats[594] = static_cast<T>(0.0);
        h_XImats[595] = static_cast<T>(0.455);
        h_XImats[596] = static_cast<T>(-0.105);
        h_XImats[597] = static_cast<T>(3.5);
        h_XImats[598] = static_cast<T>(0.0);
        h_XImats[599] = static_cast<T>(0.0);
        h_XImats[600] = static_cast<T>(-0.455);
        h_XImats[601] = static_cast<T>(0.0);
        h_XImats[602] = static_cast<T>(0.0);
        h_XImats[603] = static_cast<T>(0.0);
        h_XImats[604] = static_cast<T>(3.5);
        h_XImats[605] = static_cast<T>(0.0);
        h_XImats[606] = static_cast<T>(0.105);
        h_XImats[607] = static_cast<T>(0.0);
        h_XImats[608] = static_cast<T>(0.0);
        h_XImats[609] = static_cast<T>(0.0);
        h_XImats[610] = static_cast<T>(0.0);
        h_XImats[611] = static_cast<T>(3.5);
        // I[3]
        h_XImats[612] = static_cast<T>(0.056557500000000004);
        h_XImats[613] = static_cast<T>(0.0);
        h_XImats[614] = static_cast<T>(0.0);
        h_XImats[615] = static_cast<T>(0.0);
        h_XImats[616] = static_cast<T>(-0.11900000000000001);
        h_XImats[617] = static_cast<T>(0.23450000000000001);
        h_XImats[618] = static_cast<T>(0.0);
        h_XImats[619] = static_cast<T>(0.024496);
        h_XImats[620] = static_cast<T>(2.6999999999999247e-05);
        h_XImats[621] = static_cast<T>(0.11900000000000001);
        h_XImats[622] = static_cast<T>(0.0);
        h_XImats[623] = static_cast<T>(0.0);
        h_XImats[624] = static_cast<T>(0.0);
        h_XImats[625] = static_cast<T>(2.6999999999999247e-05);
        h_XImats[626] = static_cast<T>(0.0374215);
        h_XImats[627] = static_cast<T>(-0.23450000000000001);
        h_XImats[628] = static_cast<T>(0.0);
        h_XImats[629] = static_cast<T>(0.0);
        h_XImats[630] = static_cast<T>(0.0);
        h_XImats[631] = static_cast<T>(0.11900000000000001);
        h_XImats[632] = static_cast<T>(-0.23450000000000001);
        h_XImats[633] = static_cast<T>(3.5);
        h_XImats[634] = static_cast<T>(0.0);
        h_XImats[635] = static_cast<T>(0.0);
        h_XImats[636] = static_cast<T>(-0.11900000000000001);
        h_XImats[637] = static_cast<T>(0.0);
        h_XImats[638] = static_cast<T>(0.0);
        h_XImats[639] = static_cast<T>(0.0);
        h_XImats[640] = static_cast<T>(3.5);
        h_XImats[641] = static_cast<T>(0.0);
        h_XImats[642] = static_cast<T>(0.23450000000000001);
        h_XImats[643] = static_cast<T>(0.0);
        h_XImats[644] = static_cast<T>(0.0);
        h_XImats[645] = static_cast<T>(0.0);
        h_XImats[646] = static_cast<T>(0.0);
        h_XImats[647] = static_cast<T>(3.5);
        // I[4]
        h_XImats[648] = static_cast<T>(0.0535595);
        h_XImats[649] = static_cast<T>(-3.500000000000009e-07);
        h_XImats[650] = static_cast<T>(3.9999999999999956e-07);
        h_XImats[651] = static_cast<T>(0.0);
        h_XImats[652] = static_cast<T>(-0.266);
        h_XImats[653] = static_cast<T>(0.07350000000000001);
        h_XImats[654] = static_cast<T>(-3.5000000000000173e-07);
        h_XImats[655] = static_cast<T>(0.049132035000000004);
        h_XImats[656] = static_cast<T>(0.0);
        h_XImats[657] = static_cast<T>(0.266);
        h_XImats[658] = static_cast<T>(0.0);
        h_XImats[659] = static_cast<T>(-0.00035);
        h_XImats[660] = static_cast<T>(3.9999999999999617e-07);
        h_XImats[661] = static_cast<T>(0.0);
        h_XImats[662] = static_cast<T>(0.0075435350000000005);
        h_XImats[663] = static_cast<T>(-0.07350000000000001);
        h_XImats[664] = static_cast<T>(0.00035);
        h_XImats[665] = static_cast<T>(0.0);
        h_XImats[666] = static_cast<T>(0.0);
        h_XImats[667] = static_cast<T>(0.266);
        h_XImats[668] = static_cast<T>(-0.07350000000000001);
        h_XImats[669] = static_cast<T>(3.5);
        h_XImats[670] = static_cast<T>(0.0);
        h_XImats[671] = static_cast<T>(0.0);
        h_XImats[672] = static_cast<T>(-0.266);
        h_XImats[673] = static_cast<T>(0.0);
        h_XImats[674] = static_cast<T>(0.00035);
        h_XImats[675] = static_cast<T>(0.0);
        h_XImats[676] = static_cast<T>(3.5);
        h_XImats[677] = static_cast<T>(0.0);
        h_XImats[678] = static_cast<T>(0.07350000000000001);
        h_XImats[679] = static_cast<T>(-0.00035);
        h_XImats[680] = static_cast<T>(0.0);
        h_XImats[681] = static_cast<T>(0.0);
        h_XImats[682] = static_cast<T>(0.0);
        h_XImats[683] = static_cast<T>(3.5);
        // I[5]
        h_XImats[684] = static_cast<T>(0.004900936);
        h_XImats[685] = static_cast<T>(0.0);
        h_XImats[686] = static_cast<T>(0.0);
        h_XImats[687] = static_cast<T>(0.0);
        h_XImats[688] = static_cast<T>(-0.00072);
        h_XImats[689] = static_cast<T>(0.00108);
        h_XImats[690] = static_cast<T>(0.0);
        h_XImats[691] = static_cast<T>(0.004700288);
        h_XImats[692] = static_cast<T>(-4.32e-07);
        h_XImats[693] = static_cast<T>(0.00072);
        h_XImats[694] = static_cast<T>(0.0);
        h_XImats[695] = static_cast<T>(0.0);
        h_XImats[696] = static_cast<T>(0.0);
        h_XImats[697] = static_cast<T>(-4.32e-07);
        h_XImats[698] = static_cast<T>(0.0036006479999999997);
        h_XImats[699] = static_cast<T>(-0.00108);
        h_XImats[700] = static_cast<T>(0.0);
        h_XImats[701] = static_cast<T>(0.0);
        h_XImats[702] = static_cast<T>(0.0);
        h_XImats[703] = static_cast<T>(0.00072);
        h_XImats[704] = static_cast<T>(-0.00108);
        h_XImats[705] = static_cast<T>(1.8);
        h_XImats[706] = static_cast<T>(0.0);
        h_XImats[707] = static_cast<T>(0.0);
        h_XImats[708] = static_cast<T>(-0.00072);
        h_XImats[709] = static_cast<T>(0.0);
        h_XImats[710] = static_cast<T>(0.0);
        h_XImats[711] = static_cast<T>(0.0);
        h_XImats[712] = static_cast<T>(1.8);
        h_XImats[713] = static_cast<T>(0.0);
        h_XImats[714] = static_cast<T>(0.00108);
        h_XImats[715] = static_cast<T>(0.0);
        h_XImats[716] = static_cast<T>(0.0);
        h_XImats[717] = static_cast<T>(0.0);
        h_XImats[718] = static_cast<T>(0.0);
        h_XImats[719] = static_cast<T>(1.8);
        // I[6]
        h_XImats[720] = static_cast<T>(0.00598);
        h_XImats[721] = static_cast<T>(0.0);
        h_XImats[722] = static_cast<T>(0.0);
        h_XImats[723] = static_cast<T>(0.0);
        h_XImats[724] = static_cast<T>(-0.024);
        h_XImats[725] = static_cast<T>(0.0);
        h_XImats[726] = static_cast<T>(0.0);
        h_XImats[727] = static_cast<T>(0.00598);
        h_XImats[728] = static_cast<T>(0.0);
        h_XImats[729] = static_cast<T>(0.024);
        h_XImats[730] = static_cast<T>(0.0);
        h_XImats[731] = static_cast<T>(0.0);
        h_XImats[732] = static_cast<T>(0.0);
        h_XImats[733] = static_cast<T>(0.0);
        h_XImats[734] = static_cast<T>(0.005);
        h_XImats[735] = static_cast<T>(0.0);
        h_XImats[736] = static_cast<T>(0.0);
        h_XImats[737] = static_cast<T>(0.0);
        h_XImats[738] = static_cast<T>(0.0);
        h_XImats[739] = static_cast<T>(0.024);
        h_XImats[740] = static_cast<T>(0.0);
        h_XImats[741] = static_cast<T>(1.2);
        h_XImats[742] = static_cast<T>(0.0);
        h_XImats[743] = static_cast<T>(0.0);
        h_XImats[744] = static_cast<T>(-0.024);
        h_XImats[745] = static_cast<T>(0.0);
        h_XImats[746] = static_cast<T>(0.0);
        h_XImats[747] = static_cast<T>(0.0);
        h_XImats[748] = static_cast<T>(1.2);
        h_XImats[749] = static_cast<T>(0.0);
        h_XImats[750] = static_cast<T>(0.0);
        h_XImats[751] = static_cast<T>(0.0);
        h_XImats[752] = static_cast<T>(0.0);
        h_XImats[753] = static_cast<T>(0.0);
        h_XImats[754] = static_cast<T>(0.0);
        h_XImats[755] = static_cast<T>(1.2);
        // I[7]
        h_XImats[756] = static_cast<T>(0.20912799999999998);
        h_XImats[757] = static_cast<T>(0.0);
        h_XImats[758] = static_cast<T>(0.0);
        h_XImats[759] = static_cast<T>(0.0);
        h_XImats[760] = static_cast<T>(-0.6911999999999999);
        h_XImats[761] = static_cast<T>(-0.17279999999999998);
        h_XImats[762] = static_cast<T>(0.0);
        h_XImats[763] = static_cast<T>(0.198944);
        h_XImats[764] = static_cast<T>(-0.0002640000000000038);
        h_XImats[765] = static_cast<T>(0.6911999999999999);
        h_XImats[766] = static_cast<T>(0.0);
        h_XImats[767] = static_cast<T>(0.0);
        h_XImats[768] = static_cast<T>(0.0);
        h_XImats[769] = static_cast<T>(-0.0002640000000000038);
        h_XImats[770] = static_cast<T>(0.022684000000000003);
        h_XImats[771] = static_cast<T>(0.17279999999999998);
        h_XImats[772] = static_cast<T>(0.0);
        h_XImats[773] = static_cast<T>(0.0);
        h_XImats[774] = static_cast<T>(0.0);
        h_XImats[775] = static_cast<T>(0.6911999999999999);
        h_XImats[776] = static_cast<T>(0.17279999999999998);
        h_XImats[777] = static_cast<T>(5.76);
        h_XImats[778] = static_cast<T>(0.0);
        h_XImats[779] = static_cast<T>(0.0);
        h_XImats[780] = static_cast<T>(-0.6911999999999999);
        h_XImats[781] = static_cast<T>(0.0);
        h_XImats[782] = static_cast<T>(0.0);
        h_XImats[783] = static_cast<T>(0.0);
        h_XImats[784] = static_cast<T>(5.76);
        h_XImats[785] = static_cast<T>(0.0);
        h_XImats[786] = static_cast<T>(-0.17279999999999998);
        h_XImats[787] = static_cast<T>(0.0);
        h_XImats[788] = static_cast<T>(0.0);
        h_XImats[789] = static_cast<T>(0.0);
        h_XImats[790] = static_cast<T>(0.0);
        h_XImats[791] = static_cast<T>(5.76);
        // I[8]
        h_XImats[792] = static_cast<T>(0.09710574999999999);
        h_XImats[793] = static_cast<T>(-1.2394999999999979e-05);
        h_XImats[794] = static_cast<T>(-9.999999999994822e-09);
        h_XImats[795] = static_cast<T>(0.0);
        h_XImats[796] = static_cast<T>(-0.2667);
        h_XImats[797] = static_cast<T>(0.37465);
        h_XImats[798] = static_cast<T>(-1.2394999999999979e-05);
        h_XImats[799] = static_cast<T>(0.052801971499999996);
        h_XImats[800] = static_cast<T>(-3.5300000000001996e-05);
        h_XImats[801] = static_cast<T>(0.2667);
        h_XImats[802] = static_cast<T>(0.0);
        h_XImats[803] = static_cast<T>(-0.0019049999999999998);
        h_XImats[804] = static_cast<T>(-9.99999999998127e-09);
        h_XImats[805] = static_cast<T>(-3.529999999999853e-05);
        h_XImats[806] = static_cast<T>(0.0552049215);
        h_XImats[807] = static_cast<T>(-0.37465);
        h_XImats[808] = static_cast<T>(0.0019049999999999998);
        h_XImats[809] = static_cast<T>(0.0);
        h_XImats[810] = static_cast<T>(0.0);
        h_XImats[811] = static_cast<T>(0.2667);
        h_XImats[812] = static_cast<T>(-0.37465);
        h_XImats[813] = static_cast<T>(6.35);
        h_XImats[814] = static_cast<T>(0.0);
        h_XImats[815] = static_cast<T>(0.0);
        h_XImats[816] = static_cast<T>(-0.2667);
        h_XImats[817] = static_cast<T>(0.0);
        h_XImats[818] = static_cast<T>(0.0019049999999999998);
        h_XImats[819] = static_cast<T>(0.0);
        h_XImats[820] = static_cast<T>(6.35);
        h_XImats[821] = static_cast<T>(0.0);
        h_XImats[822] = static_cast<T>(0.37465);
        h_XImats[823] = static_cast<T>(-0.0019049999999999998);
        h_XImats[824] = static_cast<T>(0.0);
        h_XImats[825] = static_cast<T>(0.0);
        h_XImats[826] = static_cast<T>(0.0);
        h_XImats[827] = static_cast<T>(6.35);
        // I[9]
        h_XImats[828] = static_cast<T>(0.1496);
        h_XImats[829] = static_cast<T>(0.0);
        h_XImats[830] = static_cast<T>(0.0);
        h_XImats[831] = static_cast<T>(0.0);
        h_XImats[832] = static_cast<T>(-0.455);
        h_XImats[833] = static_cast<T>(0.105);
        h_XImats[834] = static_cast<T>(0.0);
        h_XImats[835] = static_cast<T>(0.14215);
        h_XImats[836] = static_cast<T>(0.0003499999999999996);
        h_XImats[837] = static_cast<T>(0.455);
        h_XImats[838] = static_cast<T>(0.0);
        h_XImats[839] = static_cast<T>(0.0);
        h_XImats[840] = static_cast<T>(0.0);
        h_XImats[841] = static_cast<T>(0.0003499999999999996);
        h_XImats[842] = static_cast<T>(0.01395);
        h_XImats[843] = static_cast<T>(-0.105);
        h_XImats[844] = static_cast<T>(0.0);
        h_XImats[845] = static_cast<T>(0.0);
        h_XImats[846] = static_cast<T>(0.0);
        h_XImats[847] = static_cast<T>(0.455);
        h_XImats[848] = static_cast<T>(-0.105);
        h_XImats[849] = static_cast<T>(3.5);
        h_XImats[850] = static_cast<T>(0.0);
        h_XImats[851] = static_cast<T>(0.0);
        h_XImats[852] = static_cast<T>(-0.455);
        h_XImats[853] = static_cast<T>(0.0);
        h_XImats[854] = static_cast<T>(0.0);
        h_XImats[855] = static_cast<T>(0.0);
        h_XImats[856] = static_cast<T>(3.5);
        h_XImats[857] = static_cast<T>(0.0);
        h_XImats[858] = static_cast<T>(0.105);
        h_XImats[859] = static_cast<T>(0.0);
        h_XImats[860] = static_cast<T>(0.0);
        h_XImats[861] = static_cast<T>(0.0);
        h_XImats[862] = static_cast<T>(0.0);
        h_XImats[863] = static_cast<T>(3.5);
        // I[10]
        h_XImats[864] = static_cast<T>(0.056557500000000004);
        h_XImats[865] = static_cast<T>(0.0);
        h_XImats[866] = static_cast<T>(0.0);
        h_XImats[867] = static_cast<T>(0.0);
        h_XImats[868] = static_cast<T>(-0.11900000000000001);
        h_XImats[869] = static_cast<T>(0.23450000000000001);
        h_XImats[870] = static_cast<T>(0.0);
        h_XImats[871] = static_cast<T>(0.024496);
        h_XImats[872] = static_cast<T>(2.6999999999999247e-05);
        h_XImats[873] = static_cast<T>(0.11900000000000001);
        h_XImats[874] = static_cast<T>(0.0);
        h_XImats[875] = static_cast<T>(0.0);
        h_XImats[876] = static_cast<T>(0.0);
        h_XImats[877] = static_cast<T>(2.6999999999999247e-05);
        h_XImats[878] = static_cast<T>(0.0374215);
        h_XImats[879] = static_cast<T>(-0.23450000000000001);
        h_XImats[880] = static_cast<T>(0.0);
        h_XImats[881] = static_cast<T>(0.0);
        h_XImats[882] = static_cast<T>(0.0);
        h_XImats[883] = static_cast<T>(0.11900000000000001);
        h_XImats[884] = static_cast<T>(-0.23450000000000001);
        h_XImats[885] = static_cast<T>(3.5);
        h_XImats[886] = static_cast<T>(0.0);
        h_XImats[887] = static_cast<T>(0.0);
        h_XImats[888] = static_cast<T>(-0.11900000000000001);
        h_XImats[889] = static_cast<T>(0.0);
        h_XImats[890] = static_cast<T>(0.0);
        h_XImats[891] = static_cast<T>(0.0);
        h_XImats[892] = static_cast<T>(3.5);
        h_XImats[893] = static_cast<T>(0.0);
        h_XImats[894] = static_cast<T>(0.23450000000000001);
        h_XImats[895] = static_cast<T>(0.0);
        h_XImats[896] = static_cast<T>(0.0);
        h_XImats[897] = static_cast<T>(0.0);
        h_XImats[898] = static_cast<T>(0.0);
        h_XImats[899] = static_cast<T>(3.5);
        // I[11]
        h_XImats[900] = static_cast<T>(0.0535595);
        h_XImats[901] = static_cast<T>(-3.500000000000009e-07);
        h_XImats[902] = static_cast<T>(3.9999999999999956e-07);
        h_XImats[903] = static_cast<T>(0.0);
        h_XImats[904] = static_cast<T>(-0.266);
        h_XImats[905] = static_cast<T>(0.07350000000000001);
        h_XImats[906] = static_cast<T>(-3.5000000000000173e-07);
        h_XImats[907] = static_cast<T>(0.049132035000000004);
        h_XImats[908] = static_cast<T>(0.0);
        h_XImats[909] = static_cast<T>(0.266);
        h_XImats[910] = static_cast<T>(0.0);
        h_XImats[911] = static_cast<T>(-0.00035);
        h_XImats[912] = static_cast<T>(3.9999999999999617e-07);
        h_XImats[913] = static_cast<T>(0.0);
        h_XImats[914] = static_cast<T>(0.0075435350000000005);
        h_XImats[915] = static_cast<T>(-0.07350000000000001);
        h_XImats[916] = static_cast<T>(0.00035);
        h_XImats[917] = static_cast<T>(0.0);
        h_XImats[918] = static_cast<T>(0.0);
        h_XImats[919] = static_cast<T>(0.266);
        h_XImats[920] = static_cast<T>(-0.07350000000000001);
        h_XImats[921] = static_cast<T>(3.5);
        h_XImats[922] = static_cast<T>(0.0);
        h_XImats[923] = static_cast<T>(0.0);
        h_XImats[924] = static_cast<T>(-0.266);
        h_XImats[925] = static_cast<T>(0.0);
        h_XImats[926] = static_cast<T>(0.00035);
        h_XImats[927] = static_cast<T>(0.0);
        h_XImats[928] = static_cast<T>(3.5);
        h_XImats[929] = static_cast<T>(0.0);
        h_XImats[930] = static_cast<T>(0.07350000000000001);
        h_XImats[931] = static_cast<T>(-0.00035);
        h_XImats[932] = static_cast<T>(0.0);
        h_XImats[933] = static_cast<T>(0.0);
        h_XImats[934] = static_cast<T>(0.0);
        h_XImats[935] = static_cast<T>(3.5);
        // I[12]
        h_XImats[936] = static_cast<T>(0.004900936);
        h_XImats[937] = static_cast<T>(0.0);
        h_XImats[938] = static_cast<T>(0.0);
        h_XImats[939] = static_cast<T>(0.0);
        h_XImats[940] = static_cast<T>(-0.00072);
        h_XImats[941] = static_cast<T>(0.00108);
        h_XImats[942] = static_cast<T>(0.0);
        h_XImats[943] = static_cast<T>(0.004700288);
        h_XImats[944] = static_cast<T>(-4.32e-07);
        h_XImats[945] = static_cast<T>(0.00072);
        h_XImats[946] = static_cast<T>(0.0);
        h_XImats[947] = static_cast<T>(0.0);
        h_XImats[948] = static_cast<T>(0.0);
        h_XImats[949] = static_cast<T>(-4.32e-07);
        h_XImats[950] = static_cast<T>(0.0036006479999999997);
        h_XImats[951] = static_cast<T>(-0.00108);
        h_XImats[952] = static_cast<T>(0.0);
        h_XImats[953] = static_cast<T>(0.0);
        h_XImats[954] = static_cast<T>(0.0);
        h_XImats[955] = static_cast<T>(0.00072);
        h_XImats[956] = static_cast<T>(-0.00108);
        h_XImats[957] = static_cast<T>(1.8);
        h_XImats[958] = static_cast<T>(0.0);
        h_XImats[959] = static_cast<T>(0.0);
        h_XImats[960] = static_cast<T>(-0.00072);
        h_XImats[961] = static_cast<T>(0.0);
        h_XImats[962] = static_cast<T>(0.0);
        h_XImats[963] = static_cast<T>(0.0);
        h_XImats[964] = static_cast<T>(1.8);
        h_XImats[965] = static_cast<T>(0.0);
        h_XImats[966] = static_cast<T>(0.00108);
        h_XImats[967] = static_cast<T>(0.0);
        h_XImats[968] = static_cast<T>(0.0);
        h_XImats[969] = static_cast<T>(0.0);
        h_XImats[970] = static_cast<T>(0.0);
        h_XImats[971] = static_cast<T>(1.8);
        // I[13]
        h_XImats[972] = static_cast<T>(0.00598);
        h_XImats[973] = static_cast<T>(0.0);
        h_XImats[974] = static_cast<T>(0.0);
        h_XImats[975] = static_cast<T>(0.0);
        h_XImats[976] = static_cast<T>(-0.024);
        h_XImats[977] = static_cast<T>(0.0);
        h_XImats[978] = static_cast<T>(0.0);
        h_XImats[979] = static_cast<T>(0.00598);
        h_XImats[980] = static_cast<T>(0.0);
        h_XImats[981] = static_cast<T>(0.024);
        h_XImats[982] = static_cast<T>(0.0);
        h_XImats[983] = static_cast<T>(0.0);
        h_XImats[984] = static_cast<T>(0.0);
        h_XImats[985] = static_cast<T>(0.0);
        h_XImats[986] = static_cast<T>(0.005);
        h_XImats[987] = static_cast<T>(0.0);
        h_XImats[988] = static_cast<T>(0.0);
        h_XImats[989] = static_cast<T>(0.0);
        h_XImats[990] = static_cast<T>(0.0);
        h_XImats[991] = static_cast<T>(0.024);
        h_XImats[992] = static_cast<T>(0.0);
        h_XImats[993] = static_cast<T>(1.2);
        h_XImats[994] = static_cast<T>(0.0);
        h_XImats[995] = static_cast<T>(0.0);
        h_XImats[996] = static_cast<T>(-0.024);
        h_XImats[997] = static_cast<T>(0.0);
        h_XImats[998] = static_cast<T>(0.0);
        h_XImats[999] = static_cast<T>(0.0);
        h_XImats[1000] = static_cast<T>(1.2);
        h_XImats[1001] = static_cast<T>(0.0);
        h_XImats[1002] = static_cast<T>(0.0);
        h_XImats[1003] = static_cast<T>(0.0);
        h_XImats[1004] = static_cast<T>(0.0);
        h_XImats[1005] = static_cast<T>(0.0);
        h_XImats[1006] = static_cast<T>(0.0);
        h_XImats[1007] = static_cast<T>(1.2);
        T *d_XImats; gpuErrchk(cudaMalloc((void**)&d_XImats,1008*sizeof(T)));
        gpuErrchk(cudaMemcpy(d_XImats,h_XImats,1008*sizeof(T),cudaMemcpyHostToDevice));
        free(h_XImats);
        return d_XImats;
    }

    /**
     * Initializes the robotModel helpers in GPU memory
     *
     * @return A pointer to the robotModel struct
     */
    template <typename T>
    __host__
    robotModel<T>* init_robotModel() {
        robotModel<T> h_robotModel;
        h_robotModel.d_XImats = init_XImats<T>();
        h_robotModel.d_topology_helpers = init_topology_helpers<T>();
        robotModel<T> *d_robotModel; gpuErrchk(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));
        gpuErrchk(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));
        return d_robotModel;
    }

    /**
     * Allocated device and host memory for all computations
     *
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T, int NUM_TIMESTEPS>
    __host__
    gridData<T> *init_gridData(){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Allocated device and host memory for all computations
     *
     * @param Max number of timesteps in the trajectory
     * @return A pointer to the gridData struct of pointers
     */
    template <typename T>
    __host__
    gridData<T> *init_gridData(int NUM_TIMESTEPS){
        gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));// first the input variables on the GPU
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        // then the GPU outputs
        gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));
        // and the CPU
        hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));
        return hd_data;
    }

    /**
     * Updates the Xmats in (shared) GPU memory acording to the configuration
     *
     * @param s_XImats is the (shared) memory destination location for the XImats
     * @param s_q is the (shared) memory location of the current configuration
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)
     * @param s_temp is temporary (shared) memory used to compute sin and cos if needed of size: 28
     */
    template <typename T>
    __device__
    void load_update_XImats_helpers(T *s_XImats, const T *s_q, int *s_topology_helpers, const robotModel<T> *d_robotModel, T *s_temp) {
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1008; ind += blockDim.x*blockDim.y){
            s_XImats[ind] = d_robotModel->d_XImats[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 71; ind += blockDim.x*blockDim.y){
            s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];
        }
        for(int k = threadIdx.x + threadIdx.y*blockDim.x; k < 14; k += blockDim.x*blockDim.y){
            s_temp[k] = static_cast<T>(sin(s_q[k]));
            s_temp[k+14] = static_cast<T>(cos(s_q[k]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0){
            // X[0]
            s_XImats[0] = static_cast<T>(1.0*s_temp[14]);
            s_XImats[1] = static_cast<T>(-1.0*s_temp[0]);
            s_XImats[3] = static_cast<T>(-0.1575*s_temp[0]);
            s_XImats[4] = static_cast<T>(-0.1575*s_temp[14]);
            s_XImats[6] = static_cast<T>(1.0*s_temp[0]);
            s_XImats[7] = static_cast<T>(1.0*s_temp[14]);
            s_XImats[9] = static_cast<T>(0.1575*s_temp[14]);
            s_XImats[10] = static_cast<T>(-0.1575*s_temp[0]);
            // X[1]
            s_XImats[36] = static_cast<T>(-s_temp[15]);
            s_XImats[37] = static_cast<T>(s_temp[1]);
            s_XImats[45] = static_cast<T>(-0.2025*s_temp[15]);
            s_XImats[46] = static_cast<T>(0.2025*s_temp[1]);
            s_XImats[48] = static_cast<T>(s_temp[1]);
            s_XImats[49] = static_cast<T>(s_temp[15]);
            // X[2]
            s_XImats[72] = static_cast<T>(-s_temp[16]);
            s_XImats[73] = static_cast<T>(s_temp[2]);
            s_XImats[75] = static_cast<T>(0.2045*s_temp[2]);
            s_XImats[76] = static_cast<T>(0.2045*s_temp[16]);
            s_XImats[84] = static_cast<T>(s_temp[2]);
            s_XImats[85] = static_cast<T>(s_temp[16]);
            s_XImats[87] = static_cast<T>(0.2045*s_temp[16]);
            s_XImats[88] = static_cast<T>(-0.2045*s_temp[2]);
            // X[3]
            s_XImats[108] = static_cast<T>(s_temp[17]);
            s_XImats[109] = static_cast<T>(-s_temp[3]);
            s_XImats[117] = static_cast<T>(0.2155*s_temp[17]);
            s_XImats[118] = static_cast<T>(-0.2155*s_temp[3]);
            s_XImats[120] = static_cast<T>(s_temp[3]);
            s_XImats[121] = static_cast<T>(s_temp[17]);
            // X[4]
            s_XImats[144] = static_cast<T>(-s_temp[18]);
            s_XImats[145] = static_cast<T>(s_temp[4]);
            s_XImats[147] = static_cast<T>(0.1845*s_temp[4]);
            s_XImats[148] = static_cast<T>(0.1845*s_temp[18]);
            s_XImats[156] = static_cast<T>(s_temp[4]);
            s_XImats[157] = static_cast<T>(s_temp[18]);
            s_XImats[159] = static_cast<T>(0.1845*s_temp[18]);
            s_XImats[160] = static_cast<T>(-0.1845*s_temp[4]);
            // X[5]
            s_XImats[180] = static_cast<T>(s_temp[19]);
            s_XImats[181] = static_cast<T>(-s_temp[5]);
            s_XImats[189] = static_cast<T>(0.2155*s_temp[19]);
            s_XImats[190] = static_cast<T>(-0.2155*s_temp[5]);
            s_XImats[192] = static_cast<T>(s_temp[5]);
            s_XImats[193] = static_cast<T>(s_temp[19]);
            // X[6]
            s_XImats[216] = static_cast<T>(-s_temp[20]);
            s_XImats[217] = static_cast<T>(s_temp[6]);
            s_XImats[219] = static_cast<T>(0.081*s_temp[6]);
            s_XImats[220] = static_cast<T>(0.081*s_temp[20]);
            s_XImats[228] = static_cast<T>(s_temp[6]);
            s_XImats[229] = static_cast<T>(s_temp[20]);
            s_XImats[231] = static_cast<T>(0.081*s_temp[20]);
            s_XImats[232] = static_cast<T>(-0.081*s_temp[6]);
            // X[7]
            s_XImats[252] = static_cast<T>(1.0*s_temp[21]);
            s_XImats[253] = static_cast<T>(-1.0*s_temp[7]);
            s_XImats[255] = static_cast<T>(-0.1575*s_temp[7]);
            s_XImats[256] = static_cast<T>(-0.1575*s_temp[21]);
            s_XImats[258] = static_cast<T>(1.0*s_temp[7]);
            s_XImats[259] = static_cast<T>(1.0*s_temp[21]);
            s_XImats[261] = static_cast<T>(0.1575*s_temp[21]);
            s_XImats[262] = static_cast<T>(-0.1575*s_temp[7]);
            // X[8]
            s_XImats[288] = static_cast<T>(-s_temp[22]);
            s_XImats[289] = static_cast<T>(s_temp[8]);
            s_XImats[297] = static_cast<T>(-0.2025*s_temp[22]);
            s_XImats[298] = static_cast<T>(0.2025*s_temp[8]);
            s_XImats[300] = static_cast<T>(s_temp[8]);
            s_XImats[301] = static_cast<T>(s_temp[22]);
            // X[9]
            s_XImats[324] = static_cast<T>(-s_temp[23]);
            s_XImats[325] = static_cast<T>(s_temp[9]);
            s_XImats[327] = static_cast<T>(0.2045*s_temp[9]);
            s_XImats[328] = static_cast<T>(0.2045*s_temp[23]);
            s_XImats[336] = static_cast<T>(s_temp[9]);
            s_XImats[337] = static_cast<T>(s_temp[23]);
            s_XImats[339] = static_cast<T>(0.2045*s_temp[23]);
            s_XImats[340] = static_cast<T>(-0.2045*s_temp[9]);
            // X[10]
            s_XImats[360] = static_cast<T>(s_temp[24]);
            s_XImats[361] = static_cast<T>(-s_temp[10]);
            s_XImats[369] = static_cast<T>(0.2155*s_temp[24]);
            s_XImats[370] = static_cast<T>(-0.2155*s_temp[10]);
            s_XImats[372] = static_cast<T>(s_temp[10]);
            s_XImats[373] = static_cast<T>(s_temp[24]);
            // X[11]
            s_XImats[396] = static_cast<T>(-s_temp[25]);
            s_XImats[397] = static_cast<T>(s_temp[11]);
            s_XImats[399] = static_cast<T>(0.1845*s_temp[11]);
            s_XImats[400] = static_cast<T>(0.1845*s_temp[25]);
            s_XImats[408] = static_cast<T>(s_temp[11]);
            s_XImats[409] = static_cast<T>(s_temp[25]);
            s_XImats[411] = static_cast<T>(0.1845*s_temp[25]);
            s_XImats[412] = static_cast<T>(-0.1845*s_temp[11]);
            // X[12]
            s_XImats[432] = static_cast<T>(s_temp[26]);
            s_XImats[433] = static_cast<T>(-s_temp[12]);
            s_XImats[441] = static_cast<T>(0.2155*s_temp[26]);
            s_XImats[442] = static_cast<T>(-0.2155*s_temp[12]);
            s_XImats[444] = static_cast<T>(s_temp[12]);
            s_XImats[445] = static_cast<T>(s_temp[26]);
            // X[13]
            s_XImats[468] = static_cast<T>(-s_temp[27]);
            s_XImats[469] = static_cast<T>(s_temp[13]);
            s_XImats[471] = static_cast<T>(0.081*s_temp[13]);
            s_XImats[472] = static_cast<T>(0.081*s_temp[27]);
            s_XImats[480] = static_cast<T>(s_temp[13]);
            s_XImats[481] = static_cast<T>(s_temp[27]);
            s_XImats[483] = static_cast<T>(0.081*s_temp[27]);
            s_XImats[484] = static_cast<T>(-0.081*s_temp[13]);
        }
        __syncthreads();
        for(int kcr = threadIdx.x + threadIdx.y*blockDim.x; kcr < 126; kcr += blockDim.x*blockDim.y){
            int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;
            int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols
            s_XImats[dstInd] = s_XImats[srcInd];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 252
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 84
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 0 + (ind >= 6) * 7;
            int jid6 = 6*jid;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[84 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[jid]; s_vaf[84 + jid6 + 2] += s_qdd[jid];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 1 + (comp_mod == 1) * 8;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 1 + (ind == 1) * 8;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 2 + (comp_mod == 1) * 9;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 2 + (ind == 1) * 9;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 3 + (comp_mod == 1) * 10;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 3 + (ind == 1) * 10;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 4 + (comp_mod == 1) * 11;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 4 + (ind == 1) * 11;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 5 + (comp_mod == 1) * 12;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 5 + (ind == 1) * 12;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 6 + (comp_mod == 1) * 13;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 6 + (ind == 1) * 13;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 14;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 84 + jid6;
            T *dst = IaFlag ? &s_vaf[168] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[504 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[168 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 6 + (ind >= 6) * 13;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 5 + (ind >= 6) * 12;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 4 + (ind >= 6) * 11;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 3 + (ind >= 6) * 10;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 2 + (ind >= 6) * 9;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[168 + 6*jid + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 252
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 84
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner(T *s_c,  T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 0 + (ind >= 6) * 7;
            int jid6 = 6*jid;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[84 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[jid];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 1 + (comp_mod == 1) * 8;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 1 + (ind == 1) * 8;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 2 + (comp_mod == 1) * 9;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 2 + (ind == 1) * 9;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 3 + (comp_mod == 1) * 10;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 3 + (ind == 1) * 10;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 4 + (comp_mod == 1) * 11;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 4 + (ind == 1) * 11;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 5 + (comp_mod == 1) * 12;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 5 + (ind == 1) * 12;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 6 + (comp_mod == 1) * 13;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 6 + (ind == 1) * 13;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 14;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 84 + jid6;
            T *dst = IaFlag ? &s_vaf[168] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[504 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[168 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 6 + (ind >= 6) * 13;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 5 + (ind >= 6) * 12;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 4 + (ind >= 6) * 11;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 3 + (ind >= 6) * 10;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 2 + (ind >= 6) * 9;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        //
        // s_c extracted in parallel (S*f)
        //
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            s_c[jid] = s_vaf[168 + 6*jid + 2];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 252
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is (optional vector of joint accelerations
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 84
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravityS[k]*qdd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 0 + (ind >= 6) * 7;
            int jid6 = 6*jid;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[84 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[jid]; s_vaf[84 + jid6 + 2] += s_qdd[jid];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 1 + (comp_mod == 1) * 8;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 1 + (ind == 1) * 8;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 2 + (comp_mod == 1) * 9;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 2 + (ind == 1) * 9;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 3 + (comp_mod == 1) * 10;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 3 + (ind == 1) * 10;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 4 + (comp_mod == 1) * 11;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 4 + (ind == 1) * 11;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 5 + (comp_mod == 1) * 12;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 5 + (ind == 1) * 12;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + S[k]*qdd[k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 6 + (comp_mod == 1) * 13;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid] + !vFlag * s_qdd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 6 + (ind == 1) * 13;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 14;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 84 + jid6;
            T *dst = IaFlag ? &s_vaf[168] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[504 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[168 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 6 + (ind >= 6) * 13;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 5 + (ind >= 6) * 12;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 4 + (ind >= 6) * 11;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 3 + (ind >= 6) * 10;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 2 + (ind >= 6) * 9;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = 252
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_XI is the pointer to the transformation and inertia matricies 
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = 84
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_inner_vaf(T *s_vaf, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // Forward Pass
        //
        // s_v, s_a where parent is base
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 0 + (ind >= 6) * 7;
            int jid6 = 6*jid;
            s_vaf[jid6 + row] = static_cast<T>(0);
            s_vaf[84 + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;
            if (row == 2){s_vaf[jid6 + 2] += s_qd[jid];}
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 1 + (comp_mod == 1) * 8;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 1 + (ind == 1) * 8;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 2 + (comp_mod == 1) * 9;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 2 + (ind == 1) * 9;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 3 + (comp_mod == 1) * 10;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 3 + (ind == 1) * 10;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 4 + (comp_mod == 1) * 11;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 4 + (ind == 1) * 11;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 5 + (comp_mod == 1) * 12;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 5 + (ind == 1) * 12;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        // s_v and s_a where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k] + mxS[k](v[k])*qd[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int comp_mod = comp % 2; int vFlag = comp == comp_mod;
            // non-branching pointer selector
            int jid = (comp_mod == 0) * 6 + (comp_mod == 1) * 13;
            int vaOffset = !vFlag * 84; int jid6 = 6 * jid;
            T qd_qdd_val = (row == 2) * (vFlag * s_qd[jid]);
            // compute based on the branch and use bool multiply for no branch
            s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*s_topology_helpers[jid]]) + qd_qdd_val;
        }
        // sync before a += MxS(v)*qd[S] 
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind == 0) * 6 + (ind == 1) * 13;
            mx2_peq_scaled<T>(&s_vaf[84 + 6*jid], &s_vaf[6*jid], s_qd[jid]);
        }
        __syncthreads();
        //
        // s_f in parallel given all v, a
        //
        // s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]
        // start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int comp = ind / 6; int jid = comp % 14;
            bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * 84 + jid6;
            T *dst = IaFlag ? &s_vaf[168] : s_temp;
            // compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync
            dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[504 + 6*jid6 + row], &s_vaf[vaOffset]);
        }
        __syncthreads();
        // finish with s_f[k] += fx(v[k])*Iv[k]
        for(int jid = threadIdx.x + threadIdx.y*blockDim.x; jid < 14; jid += blockDim.x*blockDim.y){
            int jid6 = 6*jid;
            fx_times_v_peq<T>(&s_vaf[168 + jid6], &s_vaf[jid6], &s_temp[jid6]);
        }
        __syncthreads();
        //
        // Backward Pass
        //
        // s_f update where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 6 + (ind >= 6) * 13;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 5 + (ind >= 6) * 12;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 4 + (ind >= 6) * 11;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 3 + (ind >= 6) * 10;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 2 + (ind >= 6) * 9;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
        // s_f update where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // s_f[parent_k] += X[k]^T*f[k]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            T val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row], &s_vaf[168 + 6*jid]);
            int dstOffset = 168 + 6*s_topology_helpers[jid] + row;
            s_vaf[dstOffset] += val;
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_c is the vector of output torques
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_device(T *s_c,  const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   used to compute vaf as helper values
     *   optimized for qdd = 0
     *
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_vaf_device(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_qdd[14]; 
        __shared__ T s_c[14];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_qdd[14]; 
        __shared__ T s_c[14];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel_single_timing(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_c[14];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            d_c[ind] = s_c[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_c[14];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Assumes the XI matricies have already been updated for the given q
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 1922
     */
    template <typename T>
    __device__
    void direct_minv_inner(T *s_Minv, const T *s_q, T *s_XImats, int *s_topology_helpers, T *s_temp) {
        // T *s_F = &s_temp[0]; T *s_IA = &s_temp[1176]; T *s_U = &s_temp[1680]; T *s_Dinv = &s_temp[1764]; T *s_Ia = &s_temp[1778]; T *s_IaTemp = &s_temp[1850];
        // Initialize IA = I
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 504; ind += blockDim.x*blockDim.y){
            s_temp[1176 + ind] = s_XImats[504 + ind];
        }
        // Zero Minv and F
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
            if(ind < 1176){s_temp[0 + ind] = static_cast<T>(0);}
            else{s_Minv[ind - 1176] = static_cast<T>(0);}
        }
        //
        // Backward Pass
        //
        // backward pass updates where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 6 + (ind >= 6) * 13;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 1) * 6 + (ind >= 1) * 13;
            int subTreeAdj = (ind < 1) * 0 + (ind >= 1) * 1;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 6 + (ind >= 36) * 13;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 1) * 6 + (col >= 1) * 13;
            int subTreeAdj = (col < 1) * 0 + (col >= 1) * 1;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 2) {
                col -= 2; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 6 + (jid_selector == 1) * 13;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 6 + (jid_ind == 1) * 13;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 5 + (ind >= 6) * 12;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 2) * 5 + (ind >= 2) * 12;
            int subTreeAdj = (ind < 2) * 0 + (ind >= 2) * 2;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 5 + (ind >= 36) * 12;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 96; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 2) * 5 + (col >= 2) * 12;
            int subTreeAdj = (col < 2) * 0 + (col >= 2) * 2;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 4) {
                col -= 4; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 5 + (jid_selector == 1) * 12;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 5 + (jid_ind == 1) * 12;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 4 + (ind >= 6) * 11;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 3) * 4 + (ind >= 3) * 11;
            int subTreeAdj = (ind < 3) * 0 + (ind >= 3) * 3;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 4 + (ind >= 36) * 11;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 108; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 3) * 4 + (col >= 3) * 11;
            int subTreeAdj = (col < 3) * 0 + (col >= 3) * 3;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 6) {
                col -= 6; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 4 + (jid_selector == 1) * 11;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 4 + (jid_ind == 1) * 11;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 3 + (ind >= 6) * 10;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 8; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 4) * 3 + (ind >= 4) * 10;
            int subTreeAdj = (ind < 4) * 0 + (ind >= 4) * 4;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 3 + (ind >= 36) * 10;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 120; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 4) * 3 + (col >= 4) * 10;
            int subTreeAdj = (col < 4) * 0 + (col >= 4) * 4;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 8) {
                col -= 8; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 3 + (jid_selector == 1) * 10;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 3 + (jid_ind == 1) * 10;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 2 + (ind >= 6) * 9;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 10; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 5) * 2 + (ind >= 5) * 9;
            int subTreeAdj = (ind < 5) * 0 + (ind >= 5) * 5;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 2 + (ind >= 36) * 9;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 132; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 5) * 2 + (col >= 5) * 9;
            int subTreeAdj = (col < 5) * 0 + (col >= 5) * 5;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 10) {
                col -= 10; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 2 + (jid_selector == 1) * 9;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 2 + (jid_ind == 1) * 9;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        // Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 6) * 1 + (ind >= 6) * 8;
            int subTreeAdj = (ind < 6) * 0 + (ind >= 6) * 6;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
            for(int row = 0; row < 6; row++) {
                s_temp[0 + 84*jid + jid_subtree6 + row] += s_temp[1680 + 6*jid + row] * s_Minv[jid_subtreeN + jid];
            }
        }
        // Ia = IA - U^T Dinv U | to start IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 36) * 1 + (ind >= 36) * 8;
            int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;
            s_temp[1778 + ind] = s_temp[1176 + 6*jid6 + ind36] - (s_temp[1680 + jid6 + row] * s_temp[1764 + jid] * s_temp[1680 + jid6 + col]);
        }
        __syncthreads();
        // F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]
        // IA_Update_Temp = Xmat^T * Ia | for IAparent Update
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 144; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            // non-branching pointer selector
            int jid = (col < 6) * 1 + (col >= 6) * 8;
            int subTreeAdj = (col < 6) * 0 + (col >= 6) * 6;
            int jid_subtree = jid + (col - subTreeAdj);
            T *src = &s_temp[0 + 84*jid + 6*jid_subtree]; T *dst = &s_temp[0 + 84*s_topology_helpers[jid] + 6*jid_subtree];
            // adjust for temp comps
            if (col >= 12) {
                col -= 12; src = &s_temp[1778 + 6*col]; dst = &s_temp[1850 + 6*col];
                int jid_selector = col / 6;
                // non-branching pointer selector
                jid = (jid_selector == 0) * 1 + (jid_selector == 1) * 8;
            }
            dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],src);
        }
        __syncthreads();
        // IA[parent_ind] += IA_Update_Temp * Xmat
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int col = ind / 6; int row = ind % 6;
            int col_max6 = col % 6; int jid_ind = col / 6;
            // non-branching pointer selector
            int jid = (jid_ind == 0) * 1 + (jid_ind == 1) * 8;
            T * src = &s_temp[1850 + 36*jid_ind + row]; T * dst = &s_temp[1176 + 36*s_topology_helpers[jid] + 6*col_max6 + row];
            *dst += dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6]);
        }
        __syncthreads();
        // backward pass updates where bfs_level is 0
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6;
            // non-branching pointer selector
            int jid = (ind < 6) * 0 + (ind >= 6) * 7;
            int jid6 = 6*jid;
            s_temp[1680 + jid6 + row] = s_temp[1176 + 6*jid6 + 6*2 + row];
            if(row == 2){
                s_temp[1764 + jid] = static_cast<T>(1)/s_temp[1680 + jid6 + 2];
                s_Minv[15 * jid] = s_temp[1764 + jid];
            }
        }
        __syncthreads();
        // Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            // non-branching pointer selector
            int jid = (ind < 7) * 0 + (ind >= 7) * 7;
            int subTreeAdj = (ind < 7) * 0 + (ind >= 7) * 7;
            int jid_subtree = jid + (ind - subTreeAdj); int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = 14*jid_subtree;
            s_Minv[jid_subtreeN + jid] -= s_temp[1764 + jid] * s_temp[0 + 84*jid + jid_subtree6 + 2];
        }
        __syncthreads();
        //
        // Forward Pass
        //   Note that due to the i: operation we need to go serially over all n
        //
        // forward pass for jid: 0
        // F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[0 + ind] = (row == 2) * s_Minv[0 + 14 * col];
        }
        __syncthreads();
        // forward pass for jid: 1
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 78; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 6;
            s_temp[84 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[36 + row], &s_temp[0 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 13; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 1;
            T *s_Fcol = &s_temp[84 + 6*col_ind];
            s_Minv[14 * col_ind + 1] -= s_temp[1765] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1686]);
            s_Fcol[2] += s_Minv[14 * col_ind + 1];
        }
        __syncthreads();
        // forward pass for jid: 2
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 12;
            s_temp[168 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[72 + row], &s_temp[84 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 2;
            T *s_Fcol = &s_temp[168 + 6*col_ind];
            s_Minv[14 * col_ind + 2] -= s_temp[1766] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1692]);
            s_Fcol[2] += s_Minv[14 * col_ind + 2];
        }
        __syncthreads();
        // forward pass for jid: 3
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 66; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 18;
            s_temp[252 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[108 + row], &s_temp[168 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 11; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 3;
            T *s_Fcol = &s_temp[252 + 6*col_ind];
            s_Minv[14 * col_ind + 3] -= s_temp[1767] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1698]);
            s_Fcol[2] += s_Minv[14 * col_ind + 3];
        }
        __syncthreads();
        // forward pass for jid: 4
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 60; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 24;
            s_temp[336 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[144 + row], &s_temp[252 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 10; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 4;
            T *s_Fcol = &s_temp[336 + 6*col_ind];
            s_Minv[14 * col_ind + 4] -= s_temp[1768] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1704]);
            s_Fcol[2] += s_Minv[14 * col_ind + 4];
        }
        __syncthreads();
        // forward pass for jid: 5
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 54; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 30;
            s_temp[420 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[180 + row], &s_temp[336 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 9; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 5;
            T *s_Fcol = &s_temp[420 + 6*col_ind];
            s_Minv[14 * col_ind + 5] -= s_temp[1769] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1710]);
            s_Fcol[2] += s_Minv[14 * col_ind + 5];
        }
        __syncthreads();
        // forward pass for jid: 6
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 36;
            s_temp[504 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[216 + row], &s_temp[420 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 8; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 6;
            T *s_Fcol = &s_temp[504 + 6*col_ind];
            s_Minv[14 * col_ind + 6] -= s_temp[1770] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1716]);
            s_Fcol[2] += s_Minv[14 * col_ind + 6];
        }
        __syncthreads();
        // forward pass for jid: 7
        // F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6;
            s_temp[630 + ind] = (row == 2) * s_Minv[105 + 14 * col];
        }
        __syncthreads();
        // forward pass for jid: 8
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 36; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 48;
            s_temp[672 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[288 + row], &s_temp[588 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 8;
            T *s_Fcol = &s_temp[672 + 6*col_ind];
            s_Minv[14 * col_ind + 8] -= s_temp[1772] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1728]);
            s_Fcol[2] += s_Minv[14 * col_ind + 8];
        }
        __syncthreads();
        // forward pass for jid: 9
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 30; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 54;
            s_temp[756 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[324 + row], &s_temp[672 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 5; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 9;
            T *s_Fcol = &s_temp[756 + 6*col_ind];
            s_Minv[14 * col_ind + 9] -= s_temp[1773] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1734]);
            s_Fcol[2] += s_Minv[14 * col_ind + 9];
        }
        __syncthreads();
        // forward pass for jid: 10
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 60;
            s_temp[840 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[360 + row], &s_temp[756 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 4; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 10;
            T *s_Fcol = &s_temp[840 + 6*col_ind];
            s_Minv[14 * col_ind + 10] -= s_temp[1774] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1740]);
            s_Fcol[2] += s_Minv[14 * col_ind + 10];
        }
        __syncthreads();
        // forward pass for jid: 11
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 18; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 66;
            s_temp[924 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[396 + row], &s_temp[840 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 11;
            T *s_Fcol = &s_temp[924 + 6*col_ind];
            s_Minv[14 * col_ind + 11] -= s_temp[1775] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1746]);
            s_Fcol[2] += s_Minv[14 * col_ind + 11];
        }
        __syncthreads();
        // forward pass for jid: 12
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 12; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 72;
            s_temp[1008 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[432 + row], &s_temp[924 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 2; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 12;
            T *s_Fcol = &s_temp[1008 + 6*col_ind];
            s_Minv[14 * col_ind + 12] -= s_temp[1776] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1752]);
            s_Fcol[2] += s_Minv[14 * col_ind + 12];
        }
        __syncthreads();
        // forward pass for jid: 13
        // Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        // F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        //   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col_ind = ind - row + 78;
            s_temp[1092 + col_ind + row] = dot_prod<T,6,6,1>(&s_XImats[468 + row], &s_temp[1008 + col_ind]);
        }
        __syncthreads();
        //   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]
        //     and then update F[i,:,i:] += S*Minv[i,i:]
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1; ind += blockDim.x*blockDim.y){
            int col_ind = ind + 13;
            T *s_Fcol = &s_temp[1092 + 6*col_ind];
            s_Minv[14 * col_ind + 13] -= s_temp[1777] * dot_prod<T,6,1,1>(s_Fcol,&s_temp[1758]);
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param s_Minv is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     */
    template <typename T>
    __device__
    void direct_minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel_single_timing(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_q[ind] = d_q[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
            d_Minv[ind] = s_Minv[ind];
        }
        __syncthreads();
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     *   Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            __syncthreads();
            // save down to global
            T *d_Minv_k = &d_Minv[k*196];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
                d_Minv_k[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                     const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q;
        if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_COMPRESSED_MEM) {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel_single_timing<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call Minv %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the inverse of the mass matrix
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_COMPRESSED_MEM = false>
    __host__
    void direct_minv_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,
                                  const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_COMPRESSED_MEM) {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,d_robotModel,num_timesteps);}
        else                    {direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q_qd_u,stride_q,d_robotModel,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Finish the forward dynamics computation with qdd = Minv*(u-c)
     *
     * Notes:
     *   Assumes s_Minv and s_c are already computed
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_u is the vector of joint input torques
     * @param s_c is the bias vector
     * @param s_Minv is the inverse mass matrix
     */
    template <typename T>
    __device__
    void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {
        for(int row = threadIdx.x + threadIdx.y*blockDim.x; row < 14; row += blockDim.x*blockDim.y){
            T val = static_cast<T>(0);
            for(int col = 0; col < 14; col++) {
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                val += s_Minv[index] * (s_u[col] - s_c[col]);
            }
            s_qdd[row] = val;
        }
    }

    /**
     * Computes forward dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is the pointer to the shared memory needed of size: 2118
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        direct_minv_inner<T>(s_temp, s_q, s_XImats, s_topology_helpers, &s_temp[196]);
        inverse_dynamics_inner<T>(&s_temp[196], &s_temp[210], s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[462], gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, &s_temp[196], s_temp);
    }

    /**
     * Computes forward dynamics
     *
     * @param s_qdd is a pointer to memory for the final result
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of joint input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        __syncthreads();
        forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel_single_timing(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[42]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[14]; T *s_u = &s_q_qd_u[28];
        __shared__ T s_qdd[14];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            d_qdd[ind] = s_qdd[ind];
        }
        __syncthreads();
    }

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[42]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[14]; T *s_u = &s_q_qd_u[28];
        __shared__ T s_qdd[14];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_qdd_k = &d_qdd[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                d_qdd_k[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        forward_dynamics_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T>
    __host__
    void forward_dynamics_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd_u = 3*NUM_JOINTS;
        // then call the kernel
        forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,d_robotModel,gravity,num_timesteps);
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   Assumes s_XImats is updated already for the current s_q
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_vaf are the helper intermediate variables computed by inverse_dynamics
     * @param s_XImats is the (shared) memory holding the updated XI matricies for the given s_q
     * @param s_topology_helpers is the (shared) memory destination location for the topology_helpers
     * @param s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = 3444
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, int *s_topology_helpers, T *s_temp, const T gravity) {
        //
        // dv and da need 56 cols per dq,dqd
        // df needs 98 cols per dq,dqd
        //    out of a possible 196 cols per dq,dqd
        // Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc
        //    where a and b are the needed number of columns
        //
        // Temp memory offsets are as follows:
        // T *s_dv_dq = &s_temp[0]; T *s_dv_dqd = &s_temp[336]; T *s_da_dq = &s_temp[672];
        // T *s_da_dqd = &s_temp[1008]; T *s_df_dq = &s_temp[1344]; T *s_df_dqd = &s_temp[1932];
        // T *s_FxvI = &s_temp[2520]; T *s_MxXv = &s_temp[3024]; T *s_MxXa = &s_temp[3108];
        // T *s_Mxv = &s_temp[3192]; T *s_Mxf = &s_temp[3276]; T *s_Iv = &s_temp[3360];
        //
        // Initial Temp Comps
        //
        // First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)
        // Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 252; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int jid = col % 14; int jid6 = 6*jid;
            bool parentIsBase = s_topology_helpers[jid] == -1;
            bool comp1 = col < 14; bool comp3 = col >= 28;
            int XIOffset  =  comp1 * 504 + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)
            int vaOffset  = comp1 * jid6 + !comp1 * 6*s_topology_helpers[jid] + comp3 * 84; // v_i (comp1) or va_parent (comp 2 and 3)
            int dstOffset = comp1 * 3360 + !comp1 * 2520 + comp3 * 84 + jid6 + row; // rowCol of dst
            s_temp[dstOffset] = (parentIsBase && !comp1) ? comp3 * s_XImats[XIOffset + 30] * gravity : 
                                                           dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);
        }
        __syncthreads();
        // Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 56; col += blockDim.x*blockDim.y){
            int jid = col / 4; int selector = col % 4; int jid6 = 6*jid;
            // branch to get pointer locations
            int dstOffset; const T * src;
                 if (selector == 0){ dstOffset = 3024; src = &s_temp[2520]; }
            else if (selector == 1){ dstOffset = 3108; src = &s_temp[2604]; }
            else if (selector == 2){ dstOffset = 3192; src = &s_vaf[0]; }
            else              { dstOffset = 3276; src = &s_vaf[168]; }
            mx2<T>(&s_temp[dstOffset + jid6], &src[jid6]);
        }
        __syncthreads();
        //
        // Forward Pass
        //
        // We start with dv/du noting that we only have values
        //    for ancestors and for the current index else 0
        // dv/du where bfs_level is 0
        //     joints are: iiwa_joint_1, iiwa2_joint_1
        //     links are: iiwa_link_1, iiwa2_link_1
        // when parent is base dv_dq = 0, dv_dqd = S
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2; bool dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du >= 1) * 7;
            int du_offset = dq_flag ? 0 : 336;
            s_temp[du_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2; int col_jid = col_du % 1;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 1) * 1 + (col_du >= 1) * 8;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
        }
        __syncthreads();
        // dv/du where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4; int col_jid = col_du % 2;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 2) * 2 + (col_du >= 2) * 9;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 1) {
                s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                    dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6; int col_jid = col_du % 3;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 3) * 3 + (col_du >= 3) * 10;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 2) {
                s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                    dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 96; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 8; int col_jid = col_du % 4;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 4) * 4 + (col_du >= 4) * 11;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 3) {
                s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                    dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 120; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 10; int col_jid = col_du % 5;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 5) * 5 + (col_du >= 5) * 12;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 4) {
                s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                    dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // dv/du where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}
        // first compute dv/du = Xmat*dv_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 144; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 12; int col_jid = col_du % 6;
            int dq_flag = col == col_du;
            // non-branching pointer selector
            int jid = (col_du < 6) * 6 + (col_du >= 6) * 13;
            int du_col_offset = dq_flag * 0 + !dq_flag * 336 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] = 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
            // then add {Mx(Xv) or S for col ind}
            if (col_jid == 5) {
                s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + 6 + row] = 
                    dq_flag * s_temp[3024 + 6*jid + row] + (!dq_flag && row == 2) * static_cast<T>(1);
            }
        }
        __syncthreads();
        // start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel
        // start with da/du = MxS(dv/du)*qd
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 112; col += blockDim.x*blockDim.y){
            int col_du = col % 56;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du < 28 && col_du >= 21) * 6 + (col_du < 29 && col_du >= 28) * 7 + (col_du < 31 && col_du >= 29) * 8 + (col_du < 34 && col_du >= 31) * 9 + (col_du < 38 && col_du >= 34) * 10 + (col_du < 43 && col_du >= 38) * 11 + (col_du < 49 && col_du >= 43) * 12 + (col_du >= 49) * 13;
            mx2_scaled<T>(&s_temp[672 + 6*col], &s_temp[0 + 6*col], s_qd[jid]);
            // then add {MxXa, Mxv} to the appropriate column
            int dq_flag = col == col_du; int src_offset = dq_flag * 3108 + !dq_flag * 3192 + 6*jid;
            if(col_du == ((s_topology_helpers[42 + jid + 1] + jid + 1) - 1)){
                for(int row = 0; row < 6; row++){
                    s_temp[672 + 6*col + row] += s_temp[src_offset + row];
                }
            }
        }
        __syncthreads();
        // Finish da/du with parent updates noting that we only have values
        //    for ancestors and for the current index and nothing for bfs 0
        // da/du where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 24; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 2;
            int dq_flag = col == col_du; int col_jid = col_du % 1;
            // non-branching pointer selector
            int jid = (col_du < 1) * 1 + (col_du >= 1) * 8;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // da/du where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 48; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 4;
            int dq_flag = col == col_du; int col_jid = col_du % 2;
            // non-branching pointer selector
            int jid = (col_du < 2) * 2 + (col_du >= 2) * 9;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // da/du where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 72; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 6;
            int dq_flag = col == col_du; int col_jid = col_du % 3;
            // non-branching pointer selector
            int jid = (col_du < 3) * 3 + (col_du >= 3) * 10;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // da/du where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 96; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 8;
            int dq_flag = col == col_du; int col_jid = col_du % 4;
            // non-branching pointer selector
            int jid = (col_du < 4) * 4 + (col_du >= 4) * 11;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // da/du where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 120; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 10;
            int dq_flag = col == col_du; int col_jid = col_du % 5;
            // non-branching pointer selector
            int jid = (col_du < 5) * 5 + (col_du >= 5) * 12;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // da/du where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // da/du += Xmat*da_parent/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 144; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 12;
            int dq_flag = col == col_du; int col_jid = col_du % 6;
            // non-branching pointer selector
            int jid = (col_du < 6) * 6 + (col_du >= 6) * 13;
            int du_col_offset = dq_flag * 672 + !dq_flag * 1008 + 6 * col_jid;
            s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + jid) + row] += 
                dot_prod<T,6,6,1>(&s_XImats[36*jid + row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[jid])]);
        }
        __syncthreads();
        // Init df/du to 0
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1176; ind += blockDim.x*blockDim.y){
            s_temp[1344 + ind] = static_cast<T>(0);
        }
        __syncthreads();
        // Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
        //    aka do all of the Fx comps in parallel
        // note that while df has more cols than dva the dva cols are the first few df cols
        for(int col = threadIdx.x + threadIdx.y*blockDim.x; col < 196; col += blockDim.x*blockDim.y){
            int col_du = col % 56;
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du < 28 && col_du >= 21) * 6 + (col_du < 29 && col_du >= 28) * 7 + (col_du < 31 && col_du >= 29) * 8 + (col_du < 34 && col_du >= 31) * 9 + (col_du < 38 && col_du >= 34) * 10 + (col_du < 43 && col_du >= 38) * 11 + (col_du < 49 && col_du >= 43) * 12 + (col_du >= 49) * 13;
            // Compute Offsets and Pointers
            int dq_flag = col == col_du; int dva_to_df_adjust = (s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid]) - (s_topology_helpers[42 + jid] + jid);
            int Offset_col_du_src = dq_flag * 0 + !dq_flag * 336 + 6*col_du;
            int Offset_col_du_dst = dq_flag * 1344 + !dq_flag * 1932 + 6*(col_du + dva_to_df_adjust);
            T *dst = &s_temp[Offset_col_du_dst]; const T *fx_src = &s_temp[Offset_col_du_src]; const T *mult_src = &s_temp[3360 + 6*jid];
            // Adjust pointers for temp comps (if applicable)
            if (col >= 112) {
                int comp = col - 112; int comp_col = comp % 6; // int jid = comp / 6;
                int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;
                dst = &s_temp[2520 + jid36_col6]; fx_src = &s_vaf[jid6]; mult_src = &s_XImats[504 + jid36_col6];
            }
            fx_times_v<T>(dst, fx_src, mult_src);
        }
        __syncthreads();
        // Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 672; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % 56);
            // non-branching pointer selector
            int jid = (col_du < 1) * 0 + (col_du < 3 && col_du >= 1) * 1 + (col_du < 6 && col_du >= 3) * 2 + (col_du < 10 && col_du >= 6) * 3 + (col_du < 15 && col_du >= 10) * 4 + (col_du < 21 && col_du >= 15) * 5 + (col_du < 28 && col_du >= 21) * 6 + (col_du < 29 && col_du >= 28) * 7 + (col_du < 31 && col_du >= 29) * 8 + (col_du < 34 && col_du >= 31) * 9 + (col_du < 38 && col_du >= 34) * 10 + (col_du < 43 && col_du >= 38) * 11 + (col_du < 49 && col_du >= 43) * 12 + (col_du >= 49) * 13;
            // Compute Offsets and Pointers
            int dva_to_df_adjust = (s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid]) - (s_topology_helpers[42 + jid] + jid);
            if (col >= 56){dva_to_df_adjust += 42;}
            T *df_row_col = &s_temp[1344 + 6*dva_to_df_adjust + ind];
            const T *dv_col = &s_temp[0 + col6]; const T *da_col = &s_temp[672 + col6];
            int jid36 = 36*jid; const T *I_row = &s_XImats[504 + jid36 + row]; const T *FxvI_row = &s_temp[2520 + jid36 + row];
            // Compute the values
            *df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);
        }
        // At the same time compute the last temp var: -X^T * mx(f)
        // use Mx(Xv) temp memory as those values are no longer needed
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 84; ind += blockDim.x*blockDim.y){
            int XTcol = ind % 6; int jid6 = ind - XTcol;
            s_temp[3024 + ind] = -dot_prod<T,6,1,1>(&s_XImats[6*(jid6 + XTcol)], &s_temp[3276 + jid6]);
        }
        __syncthreads();
        //
        // BACKWARD Pass
        //
        // df/du update where bfs_level is 6
        //     joints are: iiwa_joint_7, iiwa2_joint_7
        //     links are: iiwa_link_7, iiwa2_link_7
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 6 + (col_du >= 7) * 13;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 5
        //     joints are: iiwa_joint_6, iiwa2_joint_6
        //     links are: iiwa_link_6, iiwa2_link_6
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 5 + (col_du >= 7) * 12;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 4
        //     joints are: iiwa_joint_5, iiwa2_joint_5
        //     links are: iiwa_link_5, iiwa2_link_5
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 4 + (col_du >= 7) * 11;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 3
        //     joints are: iiwa_joint_4, iiwa2_joint_4
        //     links are: iiwa_link_4, iiwa2_link_4
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 3 + (col_du >= 7) * 10;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 2
        //     joints are: iiwa_joint_3, iiwa2_joint_3
        //     links are: iiwa_link_3, iiwa2_link_3
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 2 + (col_du >= 7) * 9;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // df/du update where bfs_level is 1
        //     joints are: iiwa_joint_2, iiwa2_joint_2
        //     links are: iiwa_link_2, iiwa2_link_2
        // df_lambda/du += X^T * df/du + {Xmx(f), 0}
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 168; ind += blockDim.x*blockDim.y){
            int row = ind % 6; int col = ind / 6; int col_du = col % 14;
            // non-branching pointer selector
            int jid = (col_du < 7) * 1 + (col_du >= 7) * 8;
            int col_adjust = (col_du < 7) * 0 + (col_du >= 7) * 7;
            int dq_flag = col == col_du;
            col_du -= col_adjust; // adjust for variable number of columns
            int dst_adjust = (col_du >= s_topology_helpers[14 + jid]) * 6 * 0; // adjust for sparsity compression offsets
            int du_col_offset = dq_flag * 1344 + !dq_flag * 1932 + 6*col_du;
            T *dst = &s_temp[du_col_offset + 6*(s_topology_helpers[42 + s_topology_helpers[jid]] + s_topology_helpers[57 + s_topology_helpers[jid]]) + dst_adjust + row];
            T update_val = dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid])])
                          + dq_flag * (col_du == s_topology_helpers[14 + jid]) * s_temp[3024 + 6*jid + row];
            *dst += update_val;
        }
        __syncthreads();
        // Finally dc[i]/du = S[i]^T*df[i]/du
        for(int jid_dq_qd = threadIdx.x + threadIdx.y*blockDim.x; jid_dq_qd < 28; jid_dq_qd += blockDim.x*blockDim.y){
            int jid = jid_dq_qd % 14; int dq_flag = jid == jid_dq_qd;
            // Note that this gets a tad complicated due to memory compression and variable column length
            //    so we need to fully unroll the loop -- this will not be the most efficient for a serial
            //    chain manipulator but will generalize to branched robots
            int Offset_src = dq_flag * 1344 + !dq_flag * 1932 + 6*(s_topology_helpers[42 + jid] + s_topology_helpers[57 + jid]) + 2;
            int Offset_dst = !dq_flag * 196 + jid; bool flag = 0;
            // dc[jid]/du[0]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[1]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[2]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[3]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[4]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[5]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[6]
            flag = ((jid == 0) || (jid == 1) || (jid == 2) || (jid == 3) || (jid == 4) || (jid == 5) || (jid == 6));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[7]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[8]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[9]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[10]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[11]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[12]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
            // dc[jid]/du[13]
            flag = ((jid == 7) || (jid == 8) || (jid == 9) || (jid == 10) || (jid == 11) || (jid == 12) || (jid == 13));
            s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += 14;
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_qdd[14]; 
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_qdd[14]; 
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*392];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel_single_timing(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            d_dc_du[ind] = s_dc_du[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of inverse dynamics
     *
     * Notes:
     *   optimized for qdd = 0
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*392];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                   const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        // start code with memory transfer
        int stride_q_qd;
        if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));}
        if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));}
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call ID_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>
    __host__
    void inverse_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                                const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_FLAG) {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, d_robotModel,gravity,num_timesteps);}
        }
        else {
            if (USE_COMPRESSED_MEM) {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);}
            else                    {inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        }
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_u is the vector of input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ T s_dc_du[392];
        __shared__ T s_Minv[196];
        __shared__ T s_qdd[14];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
        inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[14], gravity);
        forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            int row = ind % 14; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 14; col++) {
                int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * Notes:
     *   Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'
     *
     * @param s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param s_q is the vector of joint positions
     * @param s_qd is the vector of joint velocities
     * @param s_qdd is the vector of joint accelerations
     * @param s_Minv is the mass matrix
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     */
    template <typename T>
    __device__
    void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity) {
        __shared__ T s_vaf[252];
        __shared__ T s_dc_du[392];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
        inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            int row = ind % 14; int dc_col_offset = ind - row;
            // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
            T val = static_cast<T>(0);
            for(int col = 0; col < 14; col++) {
                int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
            }
            s_df_du[ind] = -val;
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ T s_qdd[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
            s_q_qd[ind] = d_q_qd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
            s_qdd[ind] = d_qdd[ind];
        }
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
            s_Minv[ind] = d_Minv[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                int row = ind % 14; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 14; col++) {
                    int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*14]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[14];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ T s_qdd[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 28; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*14];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            const T *d_Minv_k = &d_Minv[k*196];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 196; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                int row = ind % 14; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 14; col++) {
                    int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*392];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel_single_timing(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[3*14]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[14]; T *s_u = &s_q_qd_u[28];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ T s_qdd[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        // load to shared mem
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
            s_q_qd_u[ind] = d_q_qd_u[ind];
        }
        __syncthreads();
        // compute with NUM_TIMESTEPS as NUM_REPS for timing
        for (int rep = 0; rep < NUM_TIMESTEPS; rep++){
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[14], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                int row = ind % 14; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 14; col++) {
                    int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
        }
        // save down to global
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
            d_df_du[ind] = s_temp[ind];
        }
        __syncthreads();
    }

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 392
     * @param d_q_dq is the vector of joint positions, velocities, and input torques
     * @param stride_q_qd_u is the stide between each q, qd, u
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[3*14]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[14]; T *s_u = &s_q_qd_u[28];
        __shared__ T s_dc_du[392];
        __shared__ T s_vaf[252];
        __shared__ T s_qdd[14];
        __shared__ T s_Minv[196];
        __shared__ int s_topology_helpers[71];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[1008];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp);
            //TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_topology_helpers, s_temp);
            inverse_dynamics_inner<T>(s_temp, s_vaf, s_q, s_qd, s_XImats, s_topology_helpers, &s_temp[14], gravity);
            forward_dynamics_finish<T>(s_qdd, s_u, s_temp, s_Minv);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_topology_helpers, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_topology_helpers, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                int row = ind % 14; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 14; col++) {
                    int index = (row <= col) * (col * 14 + row) + (row > col) * (row * 14 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*392];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 392; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                          const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*num_timesteps*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_single_timing(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                        const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {
        int stride_q_qd= 3*NUM_JOINTS;
        // start code with memory transfer
        gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));
        if (USE_QDD_MINV_FLAG) {
            gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[1]));
            gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice,streams[2]));
        }
        gpuErrchk(cudaDeviceSynchronize());
        // then call the kernel
        struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel_single_timing<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&end);
        // finally transfer the result back
        gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*sizeof(T),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        printf("Single Call FD_DU %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));
    }

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param hd_data is the packaged input and output pointers
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     * @param streams are pointers to CUDA streams for async memory transfers (if needed)
     */
    template <typename T, bool USE_QDD_MINV_FLAG = false>
    __host__
    void forward_dynamics_gradient_compute_only(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,
                                       const dim3 block_dimms, const dim3 thread_dimms) {
        int stride_q_qd= 3*NUM_JOINTS;
        // then call the kernel
        if (USE_QDD_MINV_FLAG) {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,hd_data->d_qdd, hd_data->d_Minv, d_robotModel,gravity,num_timesteps);}
        else {forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);}
        gpuErrchk(cudaDeviceSynchronize());
    }

    /**
     * Sets shared mem needed for gradient kernels and initializes streams for host functions
     *
     * @return A pointer to the array of streams
     */
    template <typename T>
    __host__
    cudaStream_t *init_grid(){
        // set the max temp memory for the gradient kernels to account for large robots
        auto id_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);
        auto id_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto id_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);
        auto fd_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        auto fd_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);
        cudaFuncSetAttribute(id_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(id_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        cudaFuncSetAttribute(fd_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));
        gpuErrchk(cudaDeviceSynchronize());
        // allocate streams
        cudaStream_t *streams = (cudaStream_t *)malloc(3*sizeof(cudaStream_t));
        int priority, minPriority, maxPriority;
        gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
        for(int i=0; i<3; i++){
            int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;
            gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));
        }
        return streams;
    }

    /**
     * Frees the memory used by grid
     *
     * @param streams allocated by init_grid
     * @param robotModel allocated by init_robotModel
     * @param data allocated by init_gridData
     */
    template <typename T>
    __host__
    void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data){
        gpuErrchk(cudaFree(d_robotModel));
        gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));
        gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd));
        gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));
        free(hd_data->h_q_qd_u); free(hd_data->h_q_qd); free(hd_data->h_q);
        free(hd_data->h_c); free(hd_data->h_Minv); free(hd_data->h_qdd);
        free(hd_data->h_dc_du); free(hd_data->h_df_du);
        for(int i=0; i<3; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);
    }

}
