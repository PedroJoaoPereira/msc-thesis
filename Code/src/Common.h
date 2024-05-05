#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
}

using namespace std;

// string fileName      - path of the image file
// uint8_t* dataBuffer  - buffer that will contain the data
// Read image from a file
int readImageFromFile(string fileName, uint8_t** dataBuffer);

// string fileName  - path of the image file
// AVFrame* frame   - frame to be written in the file
// Write image to a file
int writeImageToFile(string fileName, AVFrame** frame);

// int width                    - width of the image
// int height                   - height of the image
// AVPixelFormat pixelFormat    - pixel format of the image
// uint8_t* dataBuffer          - buffer that will contain the data
// Create data buffer to hold image
int createImageDataBuffer(int width, int height, AVPixelFormat pixelFormat, uint8_t** dataBuffer);

// uint8_t* dataBuffer          - image data to transfer to the AVFrame
// int width                    - width of the image
// int height                   - height of the image
// AVPixelFormat pixelFormat    - pixel format of the image
// AVFrame* frame               - resulting AVframe properly initialized
// Initialize and transfer data to AVFrame
int initializeAVFrame(uint8_t** dataBuffer, int width, int height, AVPixelFormat pixelFormat, AVFrame** frame);

#endif