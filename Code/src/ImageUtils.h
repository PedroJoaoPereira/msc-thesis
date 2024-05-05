#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <iostream>

extern "C"{
    #define __STDC_CONSTANT_MACROS
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
}

// Define pixel formats
#define AV_PIX_FMT_V210 -2
#define AV_PIX_FMT_YUV422PNORM -3

// int operation    - resizing operation
// Return if operation is supported
bool isSupportedOperation(int operation);

// int format   - pixel format of the data
// Return if format is supported
bool isSupportedFormat(int format);

// int inFormat	    - pixel format of the source data
// int outFormat    - pixel format of the target data
// Return the temporary scale pixel format
int getTempScaleFormat(int inFormat, int outFormat);

// int operation    - value to be clamped
// Return the value of the pixel support depending of the operation
int getPixelSupport(int operation);

// uint8_t** &buffer    - buffer to be allocated
// int width            - width of the image
// int height           - height of the image
// int pixelFormat      - pixel format of the image to be allocated
// Allocate image channels data buffers depending of the pixel format
void allocBuffers(uint8_t** &buffer, int width, int height, int pixelFormat);

// int lin              - line coordinate of pixel to retrieve
// int col              - column coordinate of pixel to retrieve
// int width            - width of the image
// int height           - height of the image
// uint8_t* data        - image data to retrieve color from
// Get a valid pixel from the image
uint8_t getPixel(int lin, int col, int width, int height, uint8_t* data);

// TEMPLATES

// DataType** &buffer    - buffer to be freed
// int bufferSize       - size of the buffer to be freed
// Free the 2d buffer resources
template <class DataType>
void free2dBuffer(DataType** &buffer, int bufferSize);

// PrecisionType* val   - value to be clamped
// PrecisionType min    - minimum limit of clamping
// PrecisionType max    - maximum limit of clamping
// PrecisionType a value to a defined interval
template <class PrecisionType>
void clamp(PrecisionType &val, PrecisionType min, PrecisionType max);

// PrecisionType value  - value to be converted
// Convert a floating point value to fixed point
template <class DataType, class PrecisionType>
DataType roundTo(PrecisionType value);

// int operation    - value to be clamped
// Return coefficient function calculator
template <class PrecisionType>
PrecisionType(*getCoefMethod(int operation))(PrecisionType);

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
#include "ImageUtils.hpp"

#endif