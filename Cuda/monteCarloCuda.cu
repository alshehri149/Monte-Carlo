#include <math.h>
#include <nvtx3/nvToolsExt.h>
#include <curand.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../Common/option.h"

__global__ void reducef(float *array_in, double *reduct, size_t array_len)
{
	extern volatile __shared__ double sdata[];
	unsigned int blockSize = 128;
	size_t tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while(i < array_len)
	{
		sdata[tid] += array_in[i];
		i += gridSize;
	}
	__syncthreads();
	if(blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
	if(blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
	if(blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
	
	if (tid < 32)
	{
		if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if(tid == 0) reduct[blockIdx.x] = sdata[0];
}

__global__ void reduced(double *array_in, double *reduct, size_t array_len)
{
	extern volatile __shared__ double sdata[];
	unsigned int blockSize = 128;
	size_t tid = threadIdx.x, gridSize = blockSize * gridDim.x, i = blockIdx.x * blockSize + tid;
	sdata[tid] = 0;
	while(i < array_len)
	{
		sdata[tid] += array_in[i];
		i += gridSize;
	}
	__syncthreads();
	if(blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
	if(blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
	if(blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
	
	if (tid < 32)
	{
		if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if(tid == 0) reduct[blockIdx.x] = sdata[0];
}

void mcs_build_rando_list(uint32_t num_sims, float* rando_list)
{
	curandGenerator_t randGenerator = {0};
	curandCreateGenerator(&randGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randGenerator, 0xdead5eed);
	curandGenerateNormal(randGenerator, rando_list, num_sims, 0, 1);
	curandDestroyGenerator(randGenerator);
}

//Calculate call and put prices with Monte Carlo method
__global__ void mcs_calc_price(uint32_t num_sims, option_t *opt, float *rando_list, float *call_vec, float *put_vec)
{
	/*Repeated exprerssions***************************************************/
	float S_adjust = opt->s * exp(opt->t * (opt->r - 0.5 * opt->v * opt->v));
	float xpr = sqrt(opt->v * opt->v * opt->t);
	/*************************************************************************/

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < num_sims)
	{
		float S_cur = S_adjust * exp(xpr * rando_list[index]);
		float call_max = S_cur - opt->k;
		float put_max = opt->k - S_cur;
		call_vec[index] = (call_max > 0) ? call_max : 0;
		put_vec[index] = (put_max > 0) ? put_max : 0;
	}
}

int main(int argc, char **argv)
{
	uint32_t SIM_COUNT = (argc > 1) ? atol(argv[1]) : 100000000;
	float *rando_list = NULL;
	float *call_vec = NULL;
	float *put_vec = NULL;
	double *call_red = NULL;
	double *put_red = NULL;
	option_t *option_d = NULL;
	option_t option = 
	{
		.s = 42,
		.k = 40,
		.r = 0.1,
		.v = 0.2,
		.t = 0.5,
		.call = 0,
		.put = 0
	};
	
	nvtxRangePush("TOTAL");
	//First we populate rando_list with randomly generated numbers
	nvtxRangePush("Rando_gen");
	cudaMalloc((void**)&rando_list, sizeof(float) * SIM_COUNT);
	mcs_build_rando_list(SIM_COUNT, rando_list);
	//cudaDeviceSynchronize();
	nvtxRangePop();//Rando_gen - pop
	
	//Then we calculate the call and put values
	nvtxRangePush("Price_calc");
	uint32_t threads = 128;
	uint32_t blocks = (SIM_COUNT + threads - 1) / threads;
	cudaMalloc((void**)&call_vec, sizeof(float) * SIM_COUNT);
	cudaMalloc((void**)&put_vec, sizeof(float) * SIM_COUNT);
	cudaMalloc((void**)&option_d, sizeof(option_t));
	cudaMemcpy(option_d, &option, sizeof(option_t), cudaMemcpyHostToDevice);
	mcs_calc_price<<<blocks, threads>>>(SIM_COUNT, option_d, rando_list, call_vec, put_vec);
	
	//Reduce call_vec and put_vec and place result in call_red and put_red respectively
	cudaMalloc((void**)&call_red, blocks * sizeof(double));
	cudaMalloc((void**)&put_red, blocks * sizeof(double));
	//Reduce is called twice, the first time is to reduce each block of call_vec and put_vec in to a single block call_red and put_red respectively.
	//	The second is reduce the call_red and put_red to single values stored in (repurposed) call_vec and put_vec respectively.
	reducef<<< blocks, threads, threads * sizeof(double) >>> (call_vec, call_red, SIM_COUNT);
	reduced<<< 1, threads, threads * sizeof(double) >>> (call_red, (double*)call_vec, blocks);
	reducef<<< blocks, threads, threads * sizeof(double) >>> (put_vec, put_red, SIM_COUNT);
	reduced<<< 1, threads, threads * sizeof(double) >>> (put_red, (double*)put_vec, blocks);
	
	//Copy results back to host
	double temp1 = 0;
	double temp2 = 0;
	cudaMemcpy(&option, option_d, sizeof(option_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&temp1, call_vec, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&temp2, put_vec, sizeof(double), cudaMemcpyDeviceToHost);
	//Average the results
	option.call = (temp1 / (double)SIM_COUNT) * exp(-option.r * option.t);
	option.put = (temp2 / (double)SIM_COUNT) * exp(-option.r * option.t);
	nvtxRangePop();//Price_calc - pop
	nvtxRangePop();//TOTAL - pop

	//Finally we output the parameters and prices
	printf("Simulations count: %u\r\n", SIM_COUNT);
	printf("Spot price:        %f\r\n", option.s);
	printf("Strike price:      %f\r\n", option.k);
	printf("Risk-free rate:    %f\r\n", option.r);
	printf("Volatility:        %f\r\n", option.v);
	printf("Time to maturity:  %f\r\n", option.t);
	printf("\r\n");
	printf("\033[93;40mCall Price:        \033[92;40m%f\r\n", option.call);
	printf("\033[93;40mPut Price:         \033[91;40m%f\033[0m\r\n", option.put);
	
	cudaFree(rando_list);
	cudaFree(call_vec);
	cudaFree(put_vec);
	cudaFree(call_red);
	cudaFree(put_red);
	cudaFree(option_d);
	return 0;
}
