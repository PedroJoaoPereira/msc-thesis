#ifndef SIMULATOR_SCALE_H
#define SIMULATOR_SCALE_H

#include <iostream>
#include <chrono>

#include "ImageInfo.h"
#include "Common.h"

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the ffmpeg simulator scale operation method
int simulator_scale(AVFrame* src, AVFrame* dst, int operation);

#endif