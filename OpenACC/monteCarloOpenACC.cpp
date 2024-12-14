#include <algorithm>
#include <cmath>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <openacc.h>
#include <curand.h>
#include "../Common/option.h"

void mcs_build_rando_list(const uint32_t& num_sims, float* rando_list)
{
	curandGenerator_t randGenerator = {0};
	curandCreateGeneratorHost(&randGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randGenerator, 0xdead5eed);
	curandGenerateNormal(randGenerator, rando_list, num_sims, 0, 1);
	curandDestroyGenerator(randGenerator);
}

//Calculate call and put prices with Monte Carlo method
void mcs_calc_price(const uint32_t& num_sims, option_t& opt, float *rando_list)
{
	/*Repeated exprerssions***************************************************/
	float S_adjust = opt.s * exp(opt.t * (opt.r - 0.5 * opt.v * opt.v));
	float xpr = sqrt(opt.v * opt.v * opt.t);
	/*************************************************************************/
	
	/*Reduction variables*****************************************************/
	double call_sum = 0;
	double put_sum = 0;
	/*************************************************************************/
	#pragma acc data copyin(rando_list[0 : num_sims])
	{
		#pragma acc parallel loop reduction(+:call_sum) reduction(+:put_sum)
		for (uint32_t i = 0; i < num_sims; i++)
		{
			float S_cur = S_adjust * exp(xpr * rando_list[i]);
			float call_max = std::max((S_cur - opt.k), static_cast<float>(0.0));
			float put_max = std::max((opt.k - S_cur), static_cast<float>(0.0));
			call_sum += call_max;
			put_sum += put_max;
		}
	}
	opt.call = (call_sum / static_cast<double>(num_sims)) * exp(-opt.r * opt.t);
	opt.put = (put_sum / static_cast<double>(num_sims)) * exp(-opt.r * opt.t);
}

int main(int argc, char **argv)
{
	uint32_t SIM_COUNT = (argc > 1) ? atol(argv[1]) : 100000000;
	float *rando_list = (float*)malloc(sizeof(float) * SIM_COUNT);
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
	mcs_build_rando_list(SIM_COUNT, rando_list);
	nvtxRangePop();//Rando_gen - pop

	//Then we calculate the call and put values
	nvtxRangePush("Price_calc");
	mcs_calc_price(SIM_COUNT, option, rando_list);
	nvtxRangePop();//Price_calc - pop
	nvtxRangePop();//TOTAL - pop

	//Finally we output the parameters and prices
	std::cout << "Simulations count: " << SIM_COUNT << std::endl;
	std::cout << "Spot price:        " << option.s << std::endl;
	std::cout << "Strike price:      " << option.k << std::endl;
	std::cout << "Risk-free rate:    " << option.r << std::endl;
	std::cout << "Volatility:        " << option.v << std::endl;
	std::cout << "Time to maturity:  " << option.t << std::endl;
	std::cout << std::endl;
	std::cout << "\033[93;40m" << "Call Price:        " << "\033[92;40m" << option.call << std::endl;
	std::cout << "\033[93;40m" << "Put Price:         " << "\033[91;40m" << option.put << "\033[0m" << std::endl;

	free(rando_list);
	return 0;
}
