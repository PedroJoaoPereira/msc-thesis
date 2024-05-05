#ifndef SEQUENTIAL_SCALE_H
#define SEQUENTIAL_SCALE_H

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

// Wrapper for the sequential scale operation method
int sequential_scale(AVFrame* src, AVFrame* dst, int operation);

#endif