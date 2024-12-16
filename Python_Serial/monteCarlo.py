import sys
import math
import random


# Function to generate normally distributed random numbers
def generateGaussianNoise(mean, stddev):
    return random.gauss(mean, stddev)


# Function to calculate the payoff of a European call option
def callOptionPayoff(S, K):
    return max(S - K, 0.0)


# Function to calculate the payoff of a European put option
def putOptionPayoff(S, K):
    return max(K - S, 0.0)


# Monte Carlo Simulation for European option pricing
def monteCarloOptionPricing(S0, K, r, sigma, T, numSimulations, isCallOption):
    payoffSum = 0.0

    for _ in range(numSimulations):
        # Generate a random price path
        ST = S0 * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * generateGaussianNoise(0.0, 1.0))

        # Calculate the payoff for this path
        payoff = callOptionPayoff(ST, K) if isCallOption else putOptionPayoff(ST, K)

        # Accumulate the payoff
        payoffSum += payoff

    # Calculate the average payoff and discount it to present value
    averagePayoff = payoffSum / numSimulations
    return math.exp(-r * T) * averagePayoff


# Main function
if __name__ == "__main__":
    # Option parameters
    S0 = 42.0  # Initial stock price
    K = 40.0   # Strike price
    r = 0.1    # Risk-free rate
    sigma = 0.2  # Volatility
    T = 0.5       # Time to maturity (1 year)
    if len(sys.argv) > 1:
        numSimulations = int(sys.argv[1])  # Number of simulations
    else:
        numSimulations = 100000000  # Number of simulations

    # Calculate option prices
    callPrice = monteCarloOptionPricing(S0, K, r, sigma, T, numSimulations, True)
    putPrice = monteCarloOptionPricing(S0, K, r, sigma, T, numSimulations, False)

    # Output the results
    print(f"European Call Option Price: {callPrice}")
    print(f"European Put Option Price: {putPrice}")
