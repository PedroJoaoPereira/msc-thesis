#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <string>

#include "Common.h"

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
}

using namespace std;

class ImageInfo{
    public:
    // Variables
    string fileName;
    int width;
    int height;
    AVPixelFormat pixelFormat;
    uint8_t* frameBuffer;
    AVFrame* frame;

    // Constructor
    ImageInfo(string fileName, int width, int height, AVPixelFormat pixelFormat);

    // Load image into avframe
    void loadImage();
    // Create frame
    void initFrame();
    // Write image into a file
    void writeImage();
};

#endif