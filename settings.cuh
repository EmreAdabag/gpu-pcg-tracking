#pragma once

#define ADD_NOISE  0


#define CONSTANT_SOLVE_TIME 0
#if CONSTANT_SOLVE_TIME
#define SOLVE_TIME  .005
#endif

#define PRINT_LINE_SEARCH   0
#define PRINT_PCG_ITERS     0

// this sets lambda to zeros each time pcg is called
// should be set to 1 if not using 50 knot points because the lambdas used to warm start are taken from a traj planned with 50 knots
#define ZERO_LAMBDA     0


// cost things
// when enabled this penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#define ABSOLUTE_QD_PENALTY 0
#define Q_COST          (1.0)
#define QD_COST         (0.010)
#define R_COST          (0.0001)


// this constant controls when xu will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (.8 * timestep)


#define MAX_SQP_ITERS   5