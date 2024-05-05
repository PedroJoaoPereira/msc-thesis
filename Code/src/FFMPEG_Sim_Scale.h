#ifndef FFMPEG_SIM_SCALE_H
#define FFMPEG_SIM_SCALE_H

#include <iostream>
#include <algorithm>
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

// Wrapper for the ffmpeg simulator scale operation method
int ffmpeg_sim_scale(ImageInfo src, ImageInfo dst, int operation);

#endif