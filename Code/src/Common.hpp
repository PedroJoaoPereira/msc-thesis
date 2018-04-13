#include "Common.h"

// Return coefficient function calculator
template <class PrecisionType>
PrecisionType(*getCoefMethod(int operation))(PrecisionType){
    // Resize operation with different kernels
    switch(operation){
        case SWS_POINT:
            return &NearestNeighborCoefficient<PrecisionType>;
        case SWS_BILINEAR:
            return &BilinearCoefficient<PrecisionType>;
        case SWS_BICUBIC:
            return &MitchellCoefficient<PrecisionType>;
        case SWS_LANCZOS:
            return &LanczosCoefficient<PrecisionType>;
    }

    // Insuccess
    return nullptr;
}

// Limit a value to a defined interval
template <class PrecisionType>
void clamp(PrecisionType &val, PrecisionType min, PrecisionType max){
    if(val < min)
        val = min;
    else if(val > max)
        val = max;
}

// Convert a floating point value to fixed point
template <class PrecisionType>
uint8_t roundTo(PrecisionType value){
    return static_cast<uint8_t>(value + static_cast<PrecisionType>(0.5) - (value < static_cast<PrecisionType>(0.)));
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

// Calculate bicubic interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType BicubicCoefficient(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Configurable parameters
    PrecisionType A = static_cast<PrecisionType>(-0.6);

    // Calculate once
    PrecisionType valAbs2 = valAbs * valAbs;
    PrecisionType valAbs3 = valAbs2 * valAbs;

    // Calculate coefficient
    if(valAbs < static_cast<PrecisionType>(1.))
        return (A + static_cast<PrecisionType>(2.)) * valAbs3 -
        (A + static_cast<PrecisionType>(3.)) * valAbs2 +
        static_cast<PrecisionType>(1.);
    else if(valAbs < static_cast<PrecisionType>(2.))
        return A * valAbs3 -
        static_cast<PrecisionType>(5.) * A * valAbs2 +
        static_cast<PrecisionType>(8.) * A * valAbs -
        static_cast<PrecisionType>(4.) * A;
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

// Calculate Lanczos interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType LanczosCoefficient(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Configurable parameters
    PrecisionType A = static_cast<PrecisionType>(3.);

    // Calculate coefficient
    if(valAbs < A){
        // Calculate once
        PrecisionType xpi = val * M_PI;
        PrecisionType xapi = val / A * M_PI;

        return sin(val * M_PI) * sin(val * M_PI / A) / (val * val * M_PI * M_PI / A);
    } else
        return static_cast<PrecisionType>(0.);
}