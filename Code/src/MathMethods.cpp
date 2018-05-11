#include "MathMethods.h"

// Return the minimum number of two values
int min(int num1, int num2){
    return (num1 > num2) ? num2 : num1;
}

// Return the biggest binary divisor
int greatestDivisor(int num){
    if(num == 0)
        return 0;
    int holder = 1;
    // Find the greatest divisor
    while((num & holder) == 0)
        holder <<= 1;

    return holder;
}

// Fast round a value
int roundFast(double value){
    return static_cast<int>(value + .5 - (value < 0.));
}
