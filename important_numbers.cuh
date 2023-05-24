#pragma once
#include <cstdint>
#include "gpuassert.cuh"

namespace important_numbers {
    __constant__ uint32_t state_size;
    __constant__ uint32_t control_size;
    __constant__ uint32_t knot_points;
}


void set_important_numbers(uint32_t state_size, uint32_t control_size, uint32_t knot_points){
    gpuErrchk(cudaMemcpyToSymbol(important_numbers::state_size, &state_size, sizeof(uint32_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(important_numbers::control_size, &control_size, sizeof(uint32_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(important_numbers::knot_points, &knot_points, sizeof(uint32_t), cudaMemcpyHostToDevice));
}