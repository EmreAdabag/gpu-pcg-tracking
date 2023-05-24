#pragma once

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