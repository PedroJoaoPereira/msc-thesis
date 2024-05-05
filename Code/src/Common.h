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

// AVPixelFormat format	- pixel format of the data
// Return if format is supported
bool isSupportedFormat(AVPixelFormat format);

// AVPixelFormat inFormat	- pixel format of the source data
// Return the temporary scale pixel format
AVPixelFormat getTempScaleFormat(AVPixelFormat inFormat);









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

float getBilinearCoef(float x);










// DataType val - value to be clamped
// DataType min - minimum limit of clamping
// DataType max - maximum limit of clamping
// Limit a value to a defined interval
template <class DataType>
void clamp(DataType &val, DataType min, DataType max);

// PrecisionType value  - value to be converted
// Convert a floating point value to fixed point
template <class DataType, class PrecisionType>
DataType roundTo(PrecisionType value);

// int lin              - line coordinate of pixel to retrieve
// int col              - column coordinate of pixel to retrieve
// int width            - width of the image
// int height           - height of the image
// DataType* data       - image data to retrieve color from
// DataType* pixelVal   - value that will be filled with the retrieved color
// Get a valid pixel from the image
template <class DataType>
void getPixel(int lin, int col, int width, int height, DataType* data, DataType* pixelVal);

// PrecisionType val    - distance value to calculate coefficient from
// Calculate nearest neighbor interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType NearestNeighborCoefficient(PrecisionType val);

// PrecisionType val    - distance value to calculate coefficient from
// Calculate bilinear interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType BilinearCoefficient(PrecisionType val);

// PrecisionType val    - distance value to calculate coefficient from
// Calculate Mitchell interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType MitchellCoefficient(PrecisionType val);

// Include template methods implementations
#include "Common.hpp"

#endif