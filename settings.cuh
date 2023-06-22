#pragma once

// if ADD_NOISE is enabled a value from a rand_normal(mean=0, std_dev=1)*NOISE_MULTIPLIER*joint_velocity will be added to a current joint state every NOISE_FREQUENCY control updates
#define ADD_NOISE  1
#define NOISE_FREQUENCY .8
#define NOISE_MULTIPLIER .0001


// turn off warm start
#define ZERO_LAMBDA     0


// when enabled ABSOLUTE_QD_PENALTY penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#define ABSOLUTE_QD_PENALTY 0
#define Q_COST          (1.0)
#define QD_COST         (0.010)
#define R_COST          (0.0001)


// this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (1 * timestep)


#define MAX_SQP_ITERS   5

#define PCG_NUM_THREADS     64
#define PCG_EXIT_TOL        1e-6
#define PCG_MAX_ITER        25

#define TRACKING_EXIT_TOL   .1

// where to store test results — manually create this directory
#define RESULTS_DIRECTORY   "./testresults_pcg25/"


// prints state while tracking
#define LIVE_PRINT_PATH 0