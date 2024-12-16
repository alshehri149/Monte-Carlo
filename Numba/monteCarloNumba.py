from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, create_xoroshiro128p_states
import numpy as np
import math
import sys

@cuda.jit(device=True) 
def generate_random_number(rng_states, idx):
    return cuda.random.xoroshiro128p_normal_float32(rng_states, idx)


@cuda.jit
def monte_carlo_kernel(S, K, T, r, sigma, num_simulations, call_prices, put_prices, rng_states):
    # Get the thread index
    idx = cuda.grid(1)

    # Only run if within the number of simulations
    if idx < num_simulations:
        # Generate a random number using the dedicated function
        z = generate_random_number(rng_states, idx)

        # Calculate the stock price at maturity
        ST = S * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)

        # Calculate the payoff for call and put options
        call_payoff = max(ST - K, 0)
        put_payoff = max(K - ST, 0)

        # Store the payoffs in the output arrays
        call_prices[idx] = call_payoff
        put_prices[idx] = put_payoff

def monte_carlo_european_option_cuda(S, K, T, r, sigma, num_simulations):

    # Allocate memory on the device
    call_prices = cuda.device_array(num_simulations, dtype=np.float32)
    put_prices = cuda.device_array(num_simulations, dtype=np.float32)

    # Define the grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (num_simulations + threads_per_block - 1) // threads_per_block

    # Initialize RNG states
    rng_states = cuda.random.create_xoroshiro128p_states(num_simulations, seed=0xdead5eed)

    # Launch the kernel
    monte_carlo_kernel[blocks_per_grid, threads_per_block](S, K, T, r, sigma, num_simulations, call_prices, put_prices, rng_states)
    #synchronize
    cuda.synchronize()

    # Copy the results back to the host
    call_payoffs = call_prices.copy_to_host()
    put_payoffs = put_prices.copy_to_host()

    # Calculate the option prices by discounting the payoffs
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)

    return call_price, put_price
	
if __name__=="__main__":
	# Set the parameters
	S = 42      # Current stock price
	K = 40      # Strike price
	T = 0.5        # Time to maturity (in years)
	r = 0.1     # Risk-free interest rate
	sigma = 0.2  # Volatility
	if len(sys.argv) > 1:
		num_simulations = int(sys.argv[1])  # Number of simulations
	else:
		num_simulations = 100000000  # Number of simulations

	# Run the simulation
	call_price, put_price = monte_carlo_european_option_cuda(S, K, T, r, sigma, num_simulations)

	# Output the results
	print("Number of simulations:\t", num_simulations)
	print("Spot price:\t\t", S)
	print("Strike price:\t\t", K)
	print("Risk-free rate:\t\t", r)
	print("Volatility:\t\t", sigma)
	print("Time to maturity:\t", T)
	print("Call Price:\t\t", call_price)
	print("Put Price:\t\t", put_price)