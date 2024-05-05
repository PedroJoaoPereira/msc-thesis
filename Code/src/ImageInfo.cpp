
#include "ImageInfo.h"

// Constructor
ImageInfo::ImageInfo(string fileName, int width, int height, AVPixelFormat pixelFormat){
    this->fileName = fileName;
    this->width = width;
    this->height = height;
    this->pixelFormat = pixelFormat;
}

// Destructor
ImageInfo::~ImageInfo(){
    // Free used resources
    if(frame == NULL)
        av_frame_free(&frame);
    if(frameBuffer == NULL)
        free(frameBuffer);
}

// Load image into avframe
int ImageInfo::loadImage(){
    // Temporary variables
    int retVal = -1;

    // Read image from a file
    retVal = readImageFromFile(fileName, &frameBuffer);
    if(retVal < 0)
        return retVal;

    // Initialize frame
    retVal = initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame);
    if(retVal < 0){
        free(frameBuffer);
        return retVal;
    }

    // Success
    return 0;
}

// Create frame
int ImageInfo::initFrame(){
    // Temporary variables
    int retVal = -1;

    // Prepare to initialize frame
    retVal = createImageDataBuffer(width, height, pixelFormat, &frameBuffer);
    if(retVal < 0)
        return retVal;

    // Initialize frame
    retVal = initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame);
    if(retVal < 0){
        free(frameBuffer);
        return retVal;
    }

    // Success
    return 0;
}

// Write image into a file
int ImageInfo::writeImage(){
    // Temporary variables
    int retVal = -1;

    // Write image to a file
    retVal = writeImageToFile(fileName, &frame);
    if(retVal < 0){
        av_frame_free(&frame);
        free(frameBuffer);
        return retVal;
    }

    // Success
    return 0;
}