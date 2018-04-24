#include "MathMethods.h"

// Return the minimum number of two values
template <class PrecisionType>
PrecisionType min(PrecisionType num1, PrecisionType num2){
    return (num1 > num2) ? num2 : num1;
}

// Return the maximum number of two values
template <class PrecisionType>
PrecisionType max(PrecisionType num1, PrecisionType num2){
    return (num1 < num2) ? num2 : num1;
}
