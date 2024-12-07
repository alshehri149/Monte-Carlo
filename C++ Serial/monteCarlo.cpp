/*
Source code below obtained from:
https://www.quantstart.com/articles/European-vanilla-option-pricing-with-C-via-Monte-Carlo-methods/
*/
#include <algorithm>    // Needed for the "max" function
#include <cmath>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <ctime>

#define SIM_COUNT								1000000
float rando_list[SIM_COUNT] = {0};
int max_trials = 0;

// A simple implementation of the Box-Muller algorithm, used to generate
// gaussian random numbers - necessary for the Monte Carlo method below
// Note that C++11 actually provides std::normal_distribution<> in 
// the <random> library, which can be used instead of this function
float gaussian_box_muller()
{
	float x = 0.0;
	float y = 0.0;
	float euclid_sq = 0.0;
	float return_value = 0.0;
	int trials = 0;

	// Continue generating two uniform random variables
	// until the square of their "euclidean distance" 
	// is less than unity
	do
	{
		x = 2.0 * rand() / static_cast<float>(RAND_MAX)-1;
		y = 2.0 * rand() / static_cast<float>(RAND_MAX)-1;
		euclid_sq = x*x + y*y;
		trials++;
	} while (euclid_sq >= 1.0);
	return_value = x*sqrt(-2*log(euclid_sq)/euclid_sq);
	max_trials = max_trials > trials ? max_trials : trials;
	
	return return_value;
}

void build_rando_list(const int& num_sims, float* rando_list)
{
	for (int i = 0; i < num_sims; i++)
	{
		rando_list[i] = gaussian_box_muller();
	}
}

// Pricing a European vanilla call option with a Monte Carlo method
float monte_carlo_price(const int& num_sims, const float& S, const float& K,
						const float& r, const float& v, const float& T, 
						const bool& isCall)
{
	float S_adjust = S * exp(T*(r-0.5*v*v));
	float S_cur = 0.0;
	float payoff_sum = 0.0;
	float return_value = 0.0;
	float max_holder = 0.0;

	for (int i = 0; i < num_sims; i++)
	{
		S_cur = S_adjust * exp(sqrt(v*v*T)*rando_list[i]);
		max_holder = isCall ? (S_cur - K) : (K - S_cur);
		payoff_sum += std::max(max_holder, static_cast<float>(0.0));
	}
	return_value = (payoff_sum / static_cast<float>(num_sims)) * exp(-r*T);

	return return_value;
}

int main(int argc, char **argv)
{
	// First we create the parameter list
	float S = 42.0;					// Option price
	float K = 40.0;					// Strike price
	float r = 0.1;					// Risk-free rate (10%)
	float v = 0.2;					// Volatility of the underlying (20%)
	float T = 0.5;					// One year until expiry
	
	srand(time(NULL));
	
	nvtxRangePush("Rando_gen");
	build_rando_list(SIM_COUNT, rando_list);
	nvtxRangePop();

	// Then we calculate the call/put values via Monte Carlo
	nvtxRangePush("Price_calc");
	float call = monte_carlo_price(SIM_COUNT, S, K, r, v, T, true);
	nvtxRangePop();

	// Finally we output the parameters and prices
	std::cout << "Number of Paths: " << SIM_COUNT << std::endl;
	std::cout << "Underlying:      " << S << std::endl;
	std::cout << "Strike:          " << K << std::endl;
	std::cout << "Risk-Free Rate:  " << r << std::endl;
	std::cout << "Volatility:      " << v << std::endl;
	std::cout << "Maturity:        " << T << std::endl;

	std::cout << "Call Price:      " << call << std::endl;
	std::cout << "Max Rand Trials: " << max_trials << std::endl;

	return 0;
}
