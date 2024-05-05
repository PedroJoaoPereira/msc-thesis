#ifndef OMP_SCALE_H
#define OMP_SCALE_H

#include <iostream>
#include <chrono>

#include "ImageInfo.h"
#include "Common.h"

extern "C" {
#define __STDC_CONSTANT_MACROS
#include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the openmp scale operation method
int omp_scale(AVFrame* src, AVFrame* dst, int operation);

#endif