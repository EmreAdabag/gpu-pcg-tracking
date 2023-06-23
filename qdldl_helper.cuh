#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "qdldl.h"


namespace qdl{

template <typename ind_T, typename val_T> 
__global__
void bd_to_csr_lowertri(
                unsigned               n,           ///< number of rows
                ind_T               *row_ptr,    ///< row pointers (size m+1)
                ind_T               *col_ind,    ///< column indices (size nnz)
                val_T                  *val,        ///< numerical values (size nnz)
                float  *bdmat,
                unsigned bdim,
                unsigned mdim)
{

    const int brow_val_ct = bdim*bdim + ((bdim+1)*bdim)/2;
    int row, col, csr_row_offset, basic_col_offset, bd_block_row, bd_block_col, bd_col, bd_row, bd_row_len;
    int iter, bd_offset, row_adj;


    for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < n; row += GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){

        bd_block_row = row/bdim;

        bd_row_len = (bd_block_row>0)*bdim + row%bdim+1;
        
        if(row==0){
            row_ptr[row] = 0;
        }
        
        row_adj = (row%bdim);    
        // int thisthing = ((row_adj+1)*(2*(bdim-row_adj)+row_adj))/2;
        int thisthing = ((row_adj+1)*row_adj)/2;
        csr_row_offset = (bd_block_row>0)*((bdim+1)*bdim)/2 + (bd_block_row>0) * (bd_block_row-1)*brow_val_ct + (bd_block_row>0)*(row%bdim)*bdim + thisthing;

        basic_col_offset = (bd_block_row>0)*(bd_block_row-1)*bdim;
        row_ptr[row+1] = csr_row_offset+bd_row_len;

        for(iter=0; iter<bd_row_len; iter++){

            col = basic_col_offset+iter;
            bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
            bd_col = col % bdim;
            bd_row = row % bdim;

            bd_offset = bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row;
            
            col_ind[csr_row_offset+iter] = col;
            val[csr_row_offset+iter] = bdmat[bd_offset];
        }

    }
}

__host__
double qdldl_solve_schur(uint32_t state_size, uint32_t knot_points, float *d_S, float *d_gamma, float *d_lambda){

    const uint32_t states_sq = state_size * state_size;

	float h_gamma_f[state_size*knot_points];
	double h_gamma[state_size*knot_points];
	gpuErrchk(cudaMemcpy(h_gamma_f, d_gamma, state_size*knot_points*sizeof(float), cudaMemcpyDeviceToHost));
	for(int j = 0; j < state_size*knot_points; j++){
		h_gamma[j] = h_gamma_f[j];
    }

	const int nnz = (knot_points-1)*states_sq + knot_points*((state_size+1)*state_size/2);

	long long *d_col_ptr, *d_row_ind;
	double *d_val;

	gpuErrchk(cudaMalloc(&d_col_ptr, (state_size*knot_points+1)*sizeof(long long)));
	gpuErrchk(cudaMalloc(&d_row_ind, nnz*sizeof(long long)));
	gpuErrchk(cudaMalloc(&d_val, nnz*sizeof(double)));

	bd_to_csr_lowertri<long long, double><<<1,32*((knot_points/32)+1)>>>(state_size*knot_points, d_col_ptr, d_row_ind, d_val, d_S, state_size, knot_points);
	

	long long h_col_ptr[state_size*knot_points+1];
	long long h_row_ind[nnz];
	double h_val[nnz];


	gpuErrchk(cudaMemcpy(h_col_ptr, d_col_ptr, (state_size*knot_points+1)*sizeof(long long), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_row_ind, d_row_ind, (nnz)*sizeof(long long), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_val, d_val, (nnz)*sizeof(double), cudaMemcpyDeviceToHost));
	

	

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC,&start);



    const QDLDL_int An = state_size*knot_points;
    const QDLDL_int *Ai = (QDLDL_int *) h_row_ind; 
    const QDLDL_int *Ap = (QDLDL_int *) h_col_ptr;
    const QDLDL_float *Ax = (QDLDL_float *)  h_val;
    const QDLDL_float *b =  (QDLDL_float *) h_gamma;



    QDLDL_int i;

    //data for L and D factors
	QDLDL_int Ln = An;
	QDLDL_int *Lp;
	QDLDL_int *Li;
	QDLDL_float *Lx;
	QDLDL_float *D;
	QDLDL_float *Dinv;

	//data for elim tree calculation
	QDLDL_int *etree;
	QDLDL_int *Lnz;
	QDLDL_int  sumLnz;

	//working data for factorisation
	QDLDL_int   *iwork;
	QDLDL_bool  *bwork;
	QDLDL_float *fwork;
	//Data for results of A\b
	QDLDL_float *x;


	/*--------------------------------
	* pre-factorisation memory allocations
	*---------------------------------*/

	//These can happen *before* the etree is calculated
	//since the sizes are not sparsity pattern specific

	//For the elimination tree
	etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
	Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);

	//For the L factors.   Li and Lx are sparsity dependent
	//so must be done after the etree is constructed
	Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
	D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
	Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

	//Working memory.  Note that both the etree and factor
	//calls requires a working vector of QDLDL_int, with
	//the factor function requiring 3*An elements and the
	//etree only An elements.   Just allocate the larger
	//amount here and use it in both places
	iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
	bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
	fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
	/*--------------------------------
	* elimination tree calculation
	*---------------------------------*/

	sumLnz = QDLDL_etree(An,Ap,Ai,iwork,Lnz,etree);

	/*--------------------------------
	* LDL factorisation
	*---------------------------------*/

	//First allocate memory for Li and Lx
	Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
	Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

	//now factor
	QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork);
	/*--------------------------------
	* solve
	*---------------------------------*/
	x = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

	//when solving A\b, start with x = b
	for(i=0;i < Ln; i++) x[i] = b[i];

	QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);

	/*--------------------------------
	* print factors and solution
	*---------------------------------*/
/*	printf("\n");
	printf("A (CSC format):\n");
	print_line();
	print_arrayi(Ap, An + 1, "A.p");
	print_arrayi(Ai, Ap[An], "A.i");
	print_arrayf(Ax, Ap[An], "A.x");
	printf("\n\n");

	printf("elimination tree:\n");
	print_line();
	print_arrayi(etree, Ln, "etree");
	print_arrayi(Lnz, Ln, "Lnz");
	printf("\n\n");

	printf("L (CSC format):\n");
	print_line();
	print_arrayi(Lp, Ln + 1, "L.p");
	print_arrayi(Li, Lp[Ln], "L.i");
	print_arrayf(Lx, Lp[Ln], "L.x");
	printf("\n\n");

	printf("D:\n");
	print_line();
	print_arrayf(D, An,    "diag(D)     ");
	print_arrayf(Dinv, An, "diag(D^{-1})");
	printf("\n\n");

	printf("solve results:\n");
	print_line();
	print_arrayf(b, An, "b");
	print_arrayf(x, An, "A\\b");
	printf("\n\n");
*/

	// TIMER ENDS HERE
	clock_gettime(CLOCK_MONOTONIC,&end);
	double qdldl_time_us = time_delta_us_timespec(start,end);


	for(int j = 0; j < state_size*knot_points; j++){
		h_gamma_f[j] = (float) x[j];
    }

	/*--------------------------------
	* clean up
	*---------------------------------*/
	free(Lp);
	free(Li);
	free(Lx);
	free(D);
	free(Dinv);
	free(etree);
	free(Lnz);
	free(iwork);
	free(bwork);
	free(fwork);
	free(x);





    gpuErrchk(cudaMemcpy(d_lambda, h_gamma_f, state_size*knot_points*sizeof(float), cudaMemcpyHostToDevice));

	gpuErrchk(cudaFree(d_col_ptr));
	gpuErrchk(cudaFree(d_row_ind));
	gpuErrchk(cudaFree(d_val));


    return qdldl_time_us;

}



}