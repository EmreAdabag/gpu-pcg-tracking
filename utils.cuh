#pragma once
#include <cstdint>
#include <cooperative_groups.h>
#include "glass.cuh"


template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
    unsigned ind;
    for(ind=threadIdx.x; ind < size_Ts; ind+=blockDim.x){
        dst[ind] = src[ind];
    }
}


template <typename T, bool store_ptr_ind=false>
__device__
void store_block_csr_lowertri(uint32_t bdim, uint32_t mdim, T *d_src, int32_t *d_row_ptr, int32_t *d_col_ind, float *d_val, bool col0, uint32_t bd_block_row, int32_t multiplier=1){

    const int brow_val_ct = bdim*bdim + ((bdim+1)*bdim)/2;
    int row, col, csr_row_offset, full_csr_offset, bd_row_len;
    int write_len;
    int cur_triangle_offset;

    for(row = threadIdx.x; row < bdim; row += blockDim.x){


        if(bd_block_row==0 && row==0){
            d_row_ptr[0] = 0;
        }
        
        cur_triangle_offset = ((row+1)*row)/2;
        csr_row_offset = (bd_block_row>0)*((bdim+1)*bdim)/2 +                   // add triangle if not first block row
                         (bd_block_row>0) * (bd_block_row-1)*brow_val_ct +      // add previous full block rows if not first block row
                         (bd_block_row>0)*row*bdim +                            // 
                         cur_triangle_offset;                                   // triangle offset


        if (store_ptr_ind){
            bd_row_len = (bd_block_row>0)*bdim + row+1;
            d_row_ptr[bd_block_row*bdim + row+1] = csr_row_offset+bd_row_len;
            
            for(col = 0; col < bd_row_len; col++){
                full_csr_offset = csr_row_offset + col;
                d_col_ind[full_csr_offset] = (bd_block_row>0)*(bd_block_row-1)*bdim + col;
            }
        }
        else{
            write_len = (bd_block_row>0)*((col0)*(bdim)+(!col0)*(row+1)) + (col0)*(bd_block_row==0)*(row+1);
            
            for(col = 0; col<write_len; col++){
                full_csr_offset = csr_row_offset + (bd_block_row>0)*(!col0)*bdim + col;
                d_val[full_csr_offset] = static_cast<float>(d_src[row + col*bdim]) * multiplier;
            }
        }

    }
}


__global__
void prep_csr(uint32_t state_size, uint32_t knot_points, QDLDL_int *d_col_ptr, QDLDL_int *d_row_ind){
    
    float *unusedpfloat;
    bool unusedbool;

    for (uint32_t blockrow = blockIdx.x; blockrow < knot_points; blockrow+=gridDim.x)
    {
        const int brow_val_ct = state_size*state_size + ((state_size+1)*state_size)/2;
        int row, col, csr_row_offset, full_csr_offset, bd_row_len;
        int write_len;
        int cur_triangle_offset;

        for(row = threadIdx.x; row < state_size; row += blockDim.x){


            if(blockrow==0 && row==0){
                d_col_ptr[0] = 0;
            }
            
            cur_triangle_offset = ((row+1)*row)/2;
            csr_row_offset = (blockrow>0)*((state_size+1)*state_size)/2 +                   // add triangle if not first block row
                            (blockrow>0) * (blockrow-1)*brow_val_ct +      // add previous full block rows if not first block row
                            (blockrow>0)*row*state_size +                            // 
                            cur_triangle_offset;                                   // triangle offset


            bd_row_len = (blockrow>0)*state_size + row+1;
            d_col_ptr[blockrow*state_size + row+1] = csr_row_offset+bd_row_len;
            
            for(col = 0; col < bd_row_len; col++){
                full_csr_offset = csr_row_offset + col;
                d_row_ind[full_csr_offset] = (blockrow>0)*(blockrow-1)*state_size + col;
            }

        }
    }
    
}


template <typename T>
__device__
void store_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    unsigned block_row_offset, block_col_offset, ind;


    block_row_offset = BLOCKNO * (3 * b_dim * b_dim);
    block_col_offset = col*b_dim*b_dim;


    if(multiplier==1){

        glass::copy<T>(b_dim*b_dim, src, &dst[block_row_offset+block_col_offset]);

        gato_memcpy<T>(
            dst+block_row_offset+block_col_offset,
            src,
            b_dim*b_dim
        );

    }
    else{
        
        for(ind=g.thread_rank(); ind<b_dim*b_dim; ind+=g.size()){
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}


template <typename T>
__device__
void load_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    if(bcol > 2 || brow > m_dim-1){
        printf("doing somehting wrong in load_block_bd\n");
        return;
    }
    

    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * b_dim * b_dim);
    block_col_offset = bcol*b_dim*b_dim;

    if(!transpose){

        gato_memcpy<T>(
            dst,
            src+block_row_offset+block_col_offset,
            b_dim*b_dim
        );

    }
    else{

        unsigned ind, transpose_col, transpose_row;

        for(ind=threadIdx.x; ind<b_dim*b_dim; ind+=blockDim.x){
            transpose_col = ind%b_dim * b_dim;
            transpose_row = ind/b_dim;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
        }
    }
}