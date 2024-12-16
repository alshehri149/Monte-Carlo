
# Accelerated Monte Carlo Simulation

## Brief:
    This project focuses on accelerating a scalar implementation of Monte Carlo simulation to price financial derivatives using CUDA and OpenACC.

## Project Structure:

    C++_Serial:     Contains the scalar implementation in C++.
    Common:         Contains shared headers.
    Cuda:           Contains CUDA implementation in C.
    Dataset:        Contains a dataset file. Only a single data point is currently used for testing.
    Numba:          Contains CUDA implementation in Python.
    OpenACC:        Contains OpenACC implementation.
    Output:         Contains C/C++ binaries generated using "Make".
    Python_Serial:  Contains the scalar implementation in Python.


## Prerequisites:
    NVIDIA HPC SDK 24.9
    Numba (To run CUDA Python implementation)
    NVIDIA GPU

## Build:
    - Open a terminal.
    - Navigate to the project's root.
    - Run "make all".
    - If you encounter any errors, make sure the prerequisites are met and installed correctly, including any additions to PATH.
    - You might need to add NVIDIA HPC SDK bin folder to the path in .bashrc, e.g. "export PATH="$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/bin/""
    - If errors are encountered while running the applications, you might need to add NVIDIA math lib and NVTX lib, e.g. "export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/lib;/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/lib""

## Usage:
    - Either navigate to Output directory or just execute the binaries directly, e.g. "Output/monteCarloSerial"
    - An argument can specify the number of simulations, e.g. "Output/monteCarloCuda 50000" will run monteCarloCuda with 50000 simulations. Otherwise, if no argument is passed, a default value of 100000000 simulations is used instead.
