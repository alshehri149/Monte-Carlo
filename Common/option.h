#ifndef __OPTION_H_
#define __OPTION_H_

typedef struct
{
	float s;		//Spot price
	float k;		//Strike price
	float r;		//Risk-free rate
	float v;		//Volatility
	float t;		//Time to maturity
	float call;		//Call price
	float put;		//Put price
} option_t;

#endif
