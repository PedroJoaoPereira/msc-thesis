#ifndef SEQUENTIAL_SCALE_H
#define SEQUENTIAL_SCALE_H

#include <iostream>
#include <string>
#include <chrono>

#include "ImageInfo.h"

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

// Wrapper for the ffmpeg scale operation method
int sequential_scale(ImageInfo src, ImageInfo dst, int operation);

#endif