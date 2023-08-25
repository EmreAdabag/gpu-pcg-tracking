#!/bin/bash

# Define a function to run when the script receives a SIGINT signal
function handle_sigint {
    echo "Received SIGINT, exiting"
    exit 1
}

# Tell the script to use the handle_sigint function to handle SIGINT signals
trap handle_sigint SIGINT


base_compile_command="nvcc --compiler-options -Wall -arch=sm_86 -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl "
end_compile_command="-o runme.exe runme.cu"



rho_maxs=("1e1", "1e2")
rho_factors=("1.2", "2", "4", "6")

for rm in "${rho_maxs[@]}"; do
    for rf in "${rho_factors[@]}"; do
        
        # # test qdl
        # compile_command=$base_compile_command
        # compile_command+="-DPCG_SOLVE=0 "
        # compile_command+="-DKNOT_POINTS=63 "
        # compile_command+="-DPCG_EXIT_TOL=-1 "
        # compile_command+="-DPCG_MAX_ITER=-1 "
        # compile_command+="-DRHO_FACTOR=$rf "
        # compile_command+="-DRHO_MAX=$rm "
        # compile_command+="-diag-suppress 68 "
        # compile_command+=$end_compile_command

        # eval $compile_command
        # ./runme.exe
        # echo "-----------------------------------------------------------"
        # rm runme.exe
        
        
        compile_command=$base_compile_command
        compile_command+="-DPCG_SOLVE=1 "
        compile_command+="-DKNOT_POINTS=128 "
        compile_command+="-DPCG_MAX_ITER=90 "
        compile_command+="-DRHO_FACTOR=$rf "
        compile_command+="-DRHO_MAX=$rm "
        compile_command+=$end_compile_command

        eval $compile_command
        ./runme.exe
        echo "-----------------------------------------------------------"
        rm runme.exe

        done
    done

