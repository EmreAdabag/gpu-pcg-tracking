#!/bin/bash

# Define a function to run when the script receives a SIGINT signal
function handle_sigint {
    echo "Received SIGINT, exiting"
    exit 1
}

# Tell the script to use the handle_sigint function to handle SIGINT signals
trap handle_sigint SIGINT


base_compile_command="nvcc --compiler-options -Wall -arch=sm_89  -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl "
end_compile_command="-o runme.exe runme.cu"

declare -A pcg_max_iters
pcg_max_iters["32"]="173"
pcg_max_iters["64"]="167"
pcg_max_iters["128"]="167"
pcg_max_iters["256"]="118"
pcg_max_iters["512"]="67"


knot_points=("32" "64" "128" "256" "512")

for knot in "${knot_points[@]}"; do
    
    # test qdl
    compile_command=$base_compile_command
    compile_command+="-DPCG_SOLVE=0 "
    compile_command+="-DKNOT_POINTS=$knot "
    compile_command+="-DPCG_EXIT_TOL=-1 "
    compile_command+="-DPCG_MAX_ITER=-1 "
    if [ $knot = "64" ]; then
        compile_command+="-DR_COST=.001 "
    else
        compile_command+="-DR_COST=.0001 "
    fi
    compile_command+="-diag-suppress 68 "
    compile_command+=$end_compile_command

    eval $compile_command
    ./runme.exe
    echo "-----------------------------------------------------------"
    
    
    compile_command=$base_compile_command
    compile_command+="-DPCG_SOLVE=1 "
    compile_command+="-DKNOT_POINTS=$knot "
    compile_command+="-DPCG_MAX_ITER=${pcg_max_iters[$knot]} "
    if [ $knot = "64" ]
    then
        compile_command+="-DR_COST=.001 "
    else
        compile_command+="-DR_COST=.0001 "
    fi
    compile_command+=$end_compile_command

    eval $compile_command
    ./runme.exe
    echo "-----------------------------------------------------------"

    done


echo "\n\n\n\n\n\n\n\nSQP DATA\n\n"
cd ../gpu-pcg-tracking-1
./final_sqp_test.sh