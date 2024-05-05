#ifndef MATH_METHODS_H
#define MATH_METHODS_H

// int num1 - integer value
// int num2 - integer value
// Return least common multiple of two integers
int lcm(int num1, int num2);

// int num1 - first value
// int num2 - second value
// Return the minimum number of two values
int min(int num1, int num2);

// int num  - integer value
// Return the biggest binary divisor
int greatestDivisor(int num);

// double value - value to be rounded
// Fast round a value
int roundFast(double value);

// double* val  - value to be clamped
// double min   - minimum limit of clamping
// double max   - maximum limit of clamping
// Clamp a value to a defined interval
void clamp(double &val, double min, double max);

#endif