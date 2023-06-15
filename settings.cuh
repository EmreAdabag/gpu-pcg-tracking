#pragma once

#define ADD_NOISE  0


#define CONSTANT_SOLVE_TIME 1
#define SOLVE_TIME  .005


#define PRINT_LINE_SEARCH   0
#define PRINT_PCG_ITERS     0

// this sets lambda to zeros each time pcg is called
// should be set to 1 if not using 50 knot points because the lambdas used to warm start are taken from a traj planned with 50 knots
#define ZERO_LAMBDA     0