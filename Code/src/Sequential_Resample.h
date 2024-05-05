#ifndef SEQUENTIAL_RESAMPLE_H
#define SEQUENTIAL_RESAMPLE_H

#include <iostream>
#include <chrono>

#include "ImageUtils.h"
#include "MathMethods.h"

extern "C" {
    #define __STDC_CONSTANT_MACROS
    #include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the openmp scale operation method
int sequential_resample(AVFrame* src, AVFrame* dst, int operation);

#endif