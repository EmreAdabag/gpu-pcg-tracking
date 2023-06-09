#pragma once

#define KNOT_POINTS 256 
#define STATE_SIZE  14


#define ADD_NOISE  0


// qdldl if 0
#define PCG_SOLVE       0

// doubles if 1, floats if 0
#define USE_DOUBLES 0

#if USE_DOUBLES
typedef double pcg_t;
#else
typedef float pcg_t;
#endif

// when enabled ABSOLUTE_QD_PENALTY penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#define ABSOLUTE_QD_PENALTY 0
#define Q_COST          (1.0)
#define QD_COST         (0.100)
#define R_COST          (0.0001)


// this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (1 * timestep)


// if 1 sqp exits on time or rho, else sqp exits on max iters or rho
#define CONST_UPDATE_FREQ   1

#if CONST_UPDATE_FREQ 
#define SQP_MAX_TIME_US 3600        // should be some buffer between sqp max time and simulation period there's probably a correct way to implement this
#define SIMULATION_PERIOD 4000
#define SQP_MAX_ITER    100
#else
#define SQP_MAX_ITER    1
#endif



#define TIME_LINSYS     1


#define PCG_NUM_THREADS     128
#define PCG_EXIT_TOL        1e-5
#define PCG_MAX_ITER        500

#define MERIT_THREADS       128
#define SCHUR_THREADS       128
#define DZ_THREADS          128
#define KKT_THREADS         128




// prints state while tracking
#define LIVE_PRINT_PATH 0
#define LIVE_PRINT_STATS 0

// runs sqp a bunch of times before starting to track
#define REMOVE_JITTERS  1

// where to store test results — manually create this directory
#define SAVE_DATA   0
#define DATA_DIRECTORY   "./testresults_qdl/"
