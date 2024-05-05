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

// int& index   - pixel index to be clamped
// int min      - minimum limit of the pixel index
// int max      - maximum limit of the pixel index
// Limit a pixel index value to a defined interval
void clampPixel(int &index, int min, int max);

// float val    - value to be clamped
// float min    - minimum limit of clamping
// float max    - maximum limit of clamping
// Limit a value to a defined interval
void clamp(float &val, float min, float max);

// float value  - value to be converted
// Convert a float to an uint8_t
uint8_t float2uint8_t(float value);

// float value  - value to be converted
// Convert a float to an int
int float2int(float value);

// uint8_t* data        - image data to retrieve color from
// int width            - width of the image
// int height           - height of the image
// int lin              - line coordinate of pixel to retrieve
// int col              - column coordinate of pixel to retrieve
// uint8_t* pixelVal    - value that will be filled with the retrieved color
// Get a valid pixel from the image
void getPixel(uint8_t* data, int width, int height, int lin, int col, uint8_t* pixelVal);

// float x  - distance to the neighboring pixel
// Get the bicubic coefficients
float getBicubicCoef(float x);

#endif