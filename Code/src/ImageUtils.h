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

// int inFormat     - pixel format of the input data
// int outFormat    - pixel format of the input data
// Return if format has supported conversion
bool hasSupportedConversion(int inFormat, int outFormat);

// int operation    - resizing operation
// Return if operation is supported
bool isSupportedOperation(int operation);

// int inFormat	    - pixel format of the source data
// int outFormat    - pixel format of the target data
// Return the temporary scale pixel format
int getScaleFormat(int inFormat, int outFormat);

#endif