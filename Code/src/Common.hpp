#include "Common.h"

// Limit a value to a defined interval
template <class DataType>
void clamp(DataType &val, DataType min, DataType max){
    if(val < min)
        val = min;
    else if(val > max)
        val = max;
}

// Convert a floating point value to fixed point
template <class DataType, class PrecisionType>
DataType roundTo(PrecisionType value){
    return static_cast<DataType>(value + static_cast<PrecisionType>(0.5) - (value < static_cast<PrecisionType>(0.)));
}

// Get a valid pixel from the image
template <class DataType>
void getPixel(int lin, int col, int height, int width, DataType* data, DataType* pixelVal){
    // Clamp coords
    clamp<int>(lin, 0, height - 1);
    clamp<int>(col, 0, width - 1);

    // Assigns correct value to return
    *pixelVal = data[lin * width + col];
}

// Calculate nearest neighbor interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType NearestNeighborCoefficient(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Calculate coefficient
    if(valAbs < static_cast<PrecisionType>(0.5))
        return static_cast<PrecisionType>(1.);
    else
        return static_cast<PrecisionType>(0.);
}

// Calculate bilinear interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType BilinearCoefficient(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Calculate coefficient
    if(valAbs < static_cast<PrecisionType>(1.))
        return static_cast<PrecisionType>(1.) - valAbs;
    else
        return static_cast<PrecisionType>(0.);
}

// Calculate Mitchell interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType MitchellCoefficient(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Configurable parameters
    PrecisionType B = static_cast<PrecisionType>(0.);
    PrecisionType C = static_cast<PrecisionType>(0.6);

    // Calculate once
    PrecisionType valAbs2 = valAbs * valAbs;
    PrecisionType valAbs3 = valAbs2 * valAbs;
    PrecisionType val1div6 = static_cast<PrecisionType>(1.) / static_cast<PrecisionType>(6.);

    // Calculate coefficient
    if(valAbs < static_cast<PrecisionType>(1.))
        return val1div6 *
        ((static_cast<PrecisionType>(12.) - static_cast<PrecisionType>(9.) * B - static_cast<PrecisionType>(6.) * C) * valAbs3 +
        (static_cast<PrecisionType>(12.) * B + static_cast<PrecisionType>(6.) * C - static_cast<PrecisionType>(18.)) * valAbs2 +
         (static_cast<PrecisionType>(6.) - static_cast<PrecisionType>(2.) * B));
    else if(valAbs < static_cast<PrecisionType>(2.))
        return val1div6 *
        ((-B - static_cast<PrecisionType>(6.) * C) * valAbs3 +
        (static_cast<PrecisionType>(6.) * B + static_cast<PrecisionType>(30.) * C) * valAbs2 +
         (-static_cast<PrecisionType>(12.) * B - static_cast<PrecisionType>(48.) * C) * valAbs +
         (static_cast<PrecisionType>(8.) * B + static_cast<PrecisionType>(24.) * C));
    else
        return static_cast<PrecisionType>(0.);
}