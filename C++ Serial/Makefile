# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

CC := nvc++
#ACCFLAGS := -ta=tesla:managed,lineinfo -Minfo=accel

ACCFLAGS := 

rdf: monteCarlo.cpp
	${CC} ${CFLAGS} ${ACCFLAGS} -o monteCarlo *.cpp -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/lib64 -lnvToolsExt

clean:
	rm -f *.o monteCarlo
