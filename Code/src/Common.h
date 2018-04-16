#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

// int operation    - resizing operation
// Return if operation is supported
bool isSupportedOperation(int operation);

// AVPixelFormat format	- pixel format of the data
// Return if format is supported
bool isSupportedFormat(AVPixelFormat format);

// int num1 - integer value
// int num2 - integer value
// Return least common multiple of two integers
int lcm(int num1, int num2);

// int operation    - value to be clamped
// int scaleRatio   - the nearest integer value of the scaling ratio
// Return the value of the pixel support depending of the operation
int getPixelSupport(int operation, int scaleRatio);

// AVPixelFormat inFormat	- pixel format of the source data
// Return the temporary scale pixel format
AVPixelFormat getTempScaleFormat(AVPixelFormat inFormat);

// string fileName          - path of the image file
// uint8_t** dataBuffer    - buffer that will contain the data
// Read image from a file
int readImageFromFile(string fileName, uint8_t** dataBuffer);

// string fileName  - path of the image file
// AVFrame* frame   - frame to be written in the file
// Write image to a file
int writeImageToFile(string fileName, AVFrame** frame);

// int width                    - width of the image
// int height                   - height of the image
// AVPixelFormat pixelFormat    - pixel format of the image
// uint8_t** dataBuffer         - buffer that will contain the data
// Create data buffer to hold image
int createImageDataBuffer(int width, int height, AVPixelFormat pixelFormat, uint8_t** dataBuffer);

// uint8_t** dataBuffer         - image data to transfer to the AVFrame
// int width                    - width of the image
// int height                   - height of the image
// AVPixelFormat pixelFormat    - pixel format of the image
// AVFrame* frame               - resulting AVframe properly initialized
// Initialize and transfer data to AVFrame
int initializeAVFrame(uint8_t** dataBuffer, int width, int height, AVPixelFormat pixelFormat, AVFrame** frame);

// int lin              - line coordinate of pixel to retrieve
// int col              - column coordinate of pixel to retrieve
// int width            - width of the image
// int height           - height of the image
// uint8_t* data        - image data to retrieve color from
// Get a valid pixel from the image
uint8_t getPixel(int lin, int col, int width, int height, uint8_t* data);

// TEMPLATES

// PrecisionType num1   - first value
// PrecisionType num2   - second value
// Return the minimum number of two values
template <class PrecisionType>
PrecisionType min(PrecisionType num1, PrecisionType num2);

// PrecisionType num1   - first value
// PrecisionType num2   - second value
// Return the maximum number of two values
template <class PrecisionType>
PrecisionType max(PrecisionType num1, PrecisionType num2);

// int operation    - value to be clamped
// Return coefficient function calculator
template <class PrecisionType>
PrecisionType(*getCoefMethod(int operation))(PrecisionType);

// PrecisionType* val   - value to be clamped
// PrecisionType min    - minimum limit of clamping
// PrecisionType max    - maximum limit of clamping
// PrecisionType a value to a defined interval
template <class PrecisionType>
void clamp(PrecisionType &val, PrecisionType min, PrecisionType max);

// PrecisionType value  - value to be converted
// Convert a floating point value to fixed point
template <class PrecisionType>
uint8_t roundTo(PrecisionType value);

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

// PrecisionType val    - distance value to calculate coefficient from
// Calculate Lanczos interpolation coefficient from a distance
template <class PrecisionType>
PrecisionType LanczosCoefficient(PrecisionType val);

// Include template methods implementations
#include "Common.hpp"

#endif