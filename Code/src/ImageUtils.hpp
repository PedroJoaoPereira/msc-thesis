#include "ImageUtils.h"

// Free the 2d buffer resources
template <class DataType>
void free2dBufferAuxDEPRECATED(DataType** &buffer, int bufferSize){
    // Free nested buffers first
    for(int i = 0; i < bufferSize; i++)
        free(buffer[i]);

    // Free main buffer
    free(buffer);
}

// Limit a value to a defined interval
template <class PrecisionType>
void clampDEPRECATED(PrecisionType &val, PrecisionType min, PrecisionType max){
    if(val < min)
        val = min;
    else if(val > max)
        val = max;
}

// Convert a floating point value to fixed point
template <class DataType, class PrecisionType>
DataType roundToDEPRECATED(PrecisionType value){
    return static_cast<DataType>(value + static_cast<PrecisionType>(0.5) - (value < static_cast<PrecisionType>(0.)));
}

// Return coefficient function calculator
template <class PrecisionType>
PrecisionType(*getCoefMethodAuxDEPRECATED(int operation))(PrecisionType){
    // Resize operation with different kernels
    switch(operation){
    case SWS_POINT:
        return &NearestNeighborCoefficientDEPRECATED<PrecisionType>;
    case SWS_BILINEAR:
        return &BilinearCoefficientDEPRECATED<PrecisionType>;
    case SWS_BICUBIC:
        return &MitchellCoefficientDEPRECATED<PrecisionType>;
    case SWS_LANCZOS:
        return &LanczosCoefficientDEPRECATED<PrecisionType>;
    }

    // Insuccess
    return nullptr;
}

// Calculate nearest neighbor interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType NearestNeighborCoefficientDEPRECATED(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Calculate coefficient
    if(valAbs <= static_cast<PrecisionType>(0.499999))
        return static_cast<PrecisionType>(1.);
    else
        return static_cast<PrecisionType>(0.);
}

// Calculate bilinear interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType BilinearCoefficientDEPRECATED(PrecisionType val){
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
PrecisionType MitchellCoefficientDEPRECATED(PrecisionType val){
    // Calculate absolute value to zero
    PrecisionType valAbs = abs(val);

    // Configurable parameters
    PrecisionType B = static_cast<PrecisionType>(0.);
    PrecisionType C = static_cast<PrecisionType>(.6);

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
PrecisionType LanczosCoefficientDEPRECATED(PrecisionType val){
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
