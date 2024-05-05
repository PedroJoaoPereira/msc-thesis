#ifndef FFMPEG_SCALE_H
#define FFMPEG_RESAMPLE_H

#include <iostream>
#include <chrono>

#include "ImageClass.h"
#include "ImageUtils.h"

extern "C"{
    #define __STDC_CONSTANT_MACROS
    #include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the ffmpeg resample operation method
int ffmpeg_resample(AVFrame* src, AVFrame* dst, int operation);

#endif