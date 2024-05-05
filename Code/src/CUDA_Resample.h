#ifndef CUDA_RESAMPLE_H
#define CUDA_RESAMPLE_H

#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "ImageUtils.h"
#include "MathMethods.h"
#include "OpenMP_Utils.h"

extern "C" {
    #define __STDC_CONSTANT_MACROS
    #include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Initializes data
void cuda_init(AVFrame* src, AVFrame* dst, int operation);

// Wrapper for the cuda resample operation method
int cuda_resample(AVFrame* src, AVFrame* dst, int operation, double* &times);

// Free resources
void cuda_finish();

#endif