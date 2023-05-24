#include "toplevel.cuh"
#include "important_numbers.cuh"










int main(){
    const uint32_t state_size = 14;
    const uint32_t control_size = 7;
    const uint32_t knot_points = 10;

    set_important_numbers(state_size, control_size, knot_points);

    float *d_traj, *d_lambda, *d_xu;
    void *d_dynmem;

    sqpSolve<float>(state_size, control_size, knot_points, .1, d_traj, d_lambda, d_xu, d_dynmem);    

    return 0;
}