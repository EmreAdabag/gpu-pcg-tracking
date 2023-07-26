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

declare -A pcg_exit_tolerances
pcg_exit_tolerances["32"]="173"
pcg_exit_tolerances["64"]="173"
pcg_exit_tolerances["128"]="167"
pcg_exit_tolerances["256"]="118"
pcg_exit_tolerances["512"]="67"


knot_points=("32" "64" "128" "256" "512")
exit_tols=("1e-9", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3")

for knot in "${knot_points[@]}"; do
    # test qdl
    compile_command=$base_compile_command
    compile_command+="-DPCG_SOLVE=0 "
    compile_command+="-DKNOT_POINTS=$knot "
    compile_command+="-DPCG_EXIT_TOL=-1 "
    compile_command+="-DPCG_MAX_ITER=-1 "
    compile_command+="-diag-suppress 68 "
    compile_command+=$end_compile_command

    eval $compile_command
    ./runme.exe
    echo "-----------------------------------------------------------"
    
    for exit_tol in "${exit_tols[@]}"; do
        
        compile_command=$base_compile_command
        compile_command+="-DPCG_SOLVE=1 "
        compile_command+="-DKNOT_POINTS=$knot "
        compile_command+="-DPCG_EXIT_TOL=$exit_tol "
        compile_command+="-DPCG_MAX_ITER=${pcg_exit_tolerances[$knot]} "
        compile_command+=$end_compile_command

        eval $compile_command
        ./runme.exe
        echo "-----------------------------------------------------------"
        done
    done

