#include "ImageUtils.h"

// Return if operation is supported
bool isSupportedOperation(int operation){
    // Verify if supported operation
    switch(operation){
    case SWS_POINT:
    case SWS_BILINEAR:
    case SWS_BICUBIC:
    case SWS_LANCZOS:
        return true;
    }

    // Not a supported operation
    return false;
}

// Return if format is supported
bool isSupportedFormat(int format){
    // Verify if supported format
    switch(format){
    case AV_PIX_FMT_UYVY422:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_V210:
    case AV_PIX_FMT_YUV422PNORM:
        return true;
    }

    // Not a supported format
    return false;
}

// Return the temporary scale pixel format
int getTempScaleFormat(int inFormat, int outFormat){
    // Retrieve normalized 8 bits format if scale will be done in 10 bits
    if(inFormat == AV_PIX_FMT_V210 && outFormat == AV_PIX_FMT_V210)
        return AV_PIX_FMT_YUV422PNORM;

    // Retrieve the temporary scale format
    switch(inFormat){
    case AV_PIX_FMT_YUV422P:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUV420P:
        return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_UYVY422:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_NV12:
        return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_V210:
        return AV_PIX_FMT_YUV422P;
    }

    // If the source pixel format is not supported
    return AV_PIX_FMT_NONE;
}

// Return the value of the pixel support depending of the operation
int getPixelSupport(int operation){
    // Resize operation with different kernels
    switch(operation){
    case SWS_POINT:
        return 2;
    case SWS_BILINEAR:
        return 2;
    case SWS_BICUBIC:
        return 4;
    case SWS_LANCZOS:
        return 6;
    }

    // Insuccess
    return -1;
}

// Allocate image channels data buffers depending of the pixel format
void allocBuffers(uint8_t** &buffer, int width, int height, int pixelFormat){
    // Allocate channel buffer pointers
    buffer = static_cast<uint8_t**>(malloc(3 * sizeof(uint8_t*)));

    // If format is 10bit V210
    if(pixelFormat == AV_PIX_FMT_V210){
        buffer[0] = static_cast<uint8_t*>(malloc(height * 128 * ((width + 47) / 48)));
        return;
    }

    // Luma component size is equal to image dimension
    int lumaChannelSize = width * height;

    // If is packed yuv
    if(pixelFormat == AV_PIX_FMT_UYVY422){
        buffer[0] = static_cast<uint8_t*>(malloc(lumaChannelSize * 2));
        return;
    }

    // All luma components have the same size
    buffer[0] = static_cast<uint8_t*>(malloc(lumaChannelSize));

    // Allocate memory for each format
    if(pixelFormat == AV_PIX_FMT_YUV444P || pixelFormat == AV_PIX_FMT_NV12){
        buffer[1] = static_cast<uint8_t*>(malloc(lumaChannelSize));
        if(pixelFormat == AV_PIX_FMT_YUV444P)
            buffer[2] = static_cast<uint8_t*>(malloc(lumaChannelSize));
        return;
    }

    // For subsampled chromas
    if(pixelFormat == AV_PIX_FMT_YUV422P || pixelFormat == AV_PIX_FMT_YUV422PNORM){
        int chromaSize = lumaChannelSize / 2;
        buffer[1] = static_cast<uint8_t*>(malloc(chromaSize));
        buffer[2] = static_cast<uint8_t*>(malloc(chromaSize));
        return;
    }
    if(pixelFormat == AV_PIX_FMT_YUV420P){
        int chromaSize = lumaChannelSize / 4;
        buffer[1] = static_cast<uint8_t*>(malloc(chromaSize));
        buffer[2] = static_cast<uint8_t*>(malloc(chromaSize));
        return;
    }
}

// Get a valid pixel from the image
uint8_t getPixel(int lin, int col, int width, int height, uint8_t* data){
    // Clamp coords
    clamp<int>(lin, 0, height - 1);
    clamp<int>(col, 0, width - 1);

    // Assigns correct value to return
    return data[lin * width + col];
}

