#ifndef OPENMP_SCALE_H
#define OPENMP_SCALE_H

#include <iostream>
#include <string>
#include <math.h>
#include <chrono>

#include "ImageInfo.h"
#include "Common.h"

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the ffmpeg scale operation method
int openmp_scale(ImageInfo src, ImageInfo dst, int operation);

#endif