#ifndef OMP_UTILS_H
#define OMP_UTILS_H

#include <omp.h>

#include "ImageUtils.h"
#include "MathMethods.h"

extern "C" {
    #define __STDC_CONSTANT_MACROS
    #include <libswscale/swscale.h>
}

using namespace std;

// Image format conversion with openmp
void omp_formatConversion(int width, int height, int srcPixelFormat, uint8_t* srcSlice[], int dstPixelFormat, uint8_t* dstSlice[]);

#endif