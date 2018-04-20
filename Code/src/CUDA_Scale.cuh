#ifndef CUDA_SCALE_H
#define CUDA_SCALE_H

#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ImageInfo.h"
#include "Common.h"

extern "C" {
    #define __STDC_CONSTANT_MACROS
    #include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the cuda scale operation method
int cuda_scale(AVFrame* src, AVFrame* dst, int operation);

#endif