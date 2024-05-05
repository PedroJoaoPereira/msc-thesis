#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <string>

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
}

using namespace std;

class ImageInfo{
    public:
    ImageInfo(string fileName, int width, int height, AVPixelFormat pixelFormat);
    string fileName;
    int width;
    int height;
    AVPixelFormat pixelFormat;
};

#endif