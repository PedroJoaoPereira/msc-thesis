#ifndef IMAGE_CLASS_H
#define IMAGE_CLASS_H

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "ImageUtils.h"

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
}

using namespace std;

class ImageClass{
public:
    bool isInitialized;
    // Variables
    string fileName;
    int width;
    int height;
    int pixelFormat;
    uint8_t* frameBuffer;
    AVFrame* frame;

    // Constructor
    ImageClass(string fileName, int width, int height, int pixelFormat);

    // Load image into avframe
    void loadImage();
    // Create frame
    void initFrame();
    // Write image into a file
    void writeImage();
    // Free image resources
    void freeResources();
};

#endif