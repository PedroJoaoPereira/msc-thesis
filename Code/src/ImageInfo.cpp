#include "ImageInfo.h"

// Constructor
ImageInfo::ImageInfo(string fileName, int width, int height, int pixelFormat){
    this->fileName = fileName;
    this->width = width;
    this->height = height;
    this->pixelFormat = pixelFormat;
}

// Load image into avframe
void ImageInfo::loadImage(){
    // Read image from a file
    if(readImageFromFile(fileName, &frameBuffer) < 0)
        return;

    // Initialize frame
    if(initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame) < 0){
        free(frameBuffer);
    }
}

// Create frame
void ImageInfo::initFrame(){
    // Prepare to initialize frame
    if(createImageDataBuffer(width, height, pixelFormat, &frameBuffer) < 0)
        return;

    // Initialize frame
    if(initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame) < 0){
        free(frameBuffer);
    }
}

// Write image into a file
void ImageInfo::writeImage(){
    // Write image to file
    if(writeImageToFile(fileName, &frame) < 0){
        av_frame_free(&frame);
        free(frameBuffer);
    }
}