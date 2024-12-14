#Compiler definitions
CC := nvc++
NVCC := nvcc
CC_INC := -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/include
CC_LNK := -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/lib -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/lib -lnvToolsExt -lcurand
NVCC_LNK := -lcurand
CC_ACC_FLAGS := -acc -gpu=cc86
CC_CUDA_FLAGS := -cuda

#Output meta-data
EXEC_NAME := monteCarlo
OUTPUT_DIR := Output

#Sorce code paths
SERIAL_SRC := C++_Serial/*.cpp
OPENACC_SRC := OpenACC/*.cpp
CUDA_SRC := Cuda/*.cu

all:
	@make --no-print-directory serial
	@make --no-print-directory openacc
	@make --no-print-directory cuda

serial:
	@echo "Building monteCarloSerial..."
	${CC} ${SERIAL_SRC} ${CC_INC} ${CC_LNK} -o ${OUTPUT_DIR}/${EXEC_NAME}Serial
	@echo "Done."
	@echo ""
	
openacc:
	@echo "Building monteCarloOpenacc..."
	${CC} ${OPENACC_SRC} ${CC_ACC_FLAGS} ${CC_INC} ${CC_LNK} -o ${OUTPUT_DIR}/${EXEC_NAME}Openacc
	@echo "Done."
	@echo ""
	
cuda:
	@echo "Building monteCarloCuda..."
	${NVCC} ${CUDA_SRC} ${NVCC_LNK} -o ${OUTPUT_DIR}/${EXEC_NAME}Cuda
	@echo "Done."
	@echo ""

clean:
	@echo "Cleaning up..."
	@rm -f ${OUTPUT_DIR}/*
	@echo "Done."
	@echo ""
