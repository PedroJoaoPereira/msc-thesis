#ifndef MATH_METHODS_H
#define MATH_METHODS_H

// int num1 - integer value
// int num2 - integer value
// Return least common multiple of two integers
int lcm(int num1, int num2);

// int num1 - integer value
// int num2 - integer value
// Return the nearest value of num2 that is a divisor of num1
int greatestDivisor(int num1, int num2);

// double value - value to be rounded
// Fast round a value
int roundFast(double value);

// TEMPLATES

// PrecisionType num1   - first value
// PrecisionType num2   - second value
// Return the minimum number of two values
template <class PrecisionType>
PrecisionType min(PrecisionType num1, PrecisionType num2);

// PrecisionType num1   - first value
// PrecisionType num2   - second value
// Return the maximum number of two values
template <class PrecisionType>
PrecisionType max(PrecisionType num1, PrecisionType num2);

// Include template methods implementations
#include "MathMethods.hpp"

#endif