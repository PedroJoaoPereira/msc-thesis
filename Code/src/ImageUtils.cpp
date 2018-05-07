#include "ImageUtils.h"

// Return if format has supported conversion
bool hasSupportedConversion(int inFormat, int outFormat){
    // Used only in DEBUG
    if(inFormat == AV_PIX_FMT_V210 && outFormat == AV_PIX_FMT_YUV422PNORM)
        return true;
    if(inFormat == AV_PIX_FMT_YUV422PNORM && outFormat == AV_PIX_FMT_V210)
        return true;


    // Verify if supported input format
    switch(inFormat){
    case AV_PIX_FMT_UYVY422:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_V210:
        break;
    default:
        return false;
    }

    // Verify if supported output format
    switch(outFormat){
    case AV_PIX_FMT_UYVY422:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_V210:
        break;
    default:
        return false;
    }

    // If formats are the same
    if(inFormat == outFormat)
        return true;

    // If input format is uyvy422
    if(inFormat == AV_PIX_FMT_UYVY422){
        switch(outFormat){
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_V210:
            return true;
        }
    }

    // If input format is yuv422p
    if(inFormat == AV_PIX_FMT_YUV422P){
        switch(outFormat){
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_V210:
            return true;
        }
    }

    // If input format is yuv420p
    if(inFormat == AV_PIX_FMT_YUV420P){
        switch(outFormat){
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_V210:
            return true;
        }
    }

    // If input format is nv12
    if(inFormat == AV_PIX_FMT_NV12){
        switch(outFormat){
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_V210:
            return true;
        }
    }

    // If input format is v210
    if(inFormat == AV_PIX_FMT_V210){
        switch(outFormat){
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_NV12:
            return true;
        }
    }

    // Not a supported conversion
    return false;
}

// Return if operation is supported
bool isSupportedOperation(int operation){
    // Verify if supported operation
    switch(operation){
    case SWS_POINT:
    case SWS_BILINEAR:
    case SWS_BICUBIC:
        return true;
    }

    // Not a supported operation
    return false;
}

// Return if format is supported
bool isSupportedFormatDEPRECATED(int format){
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

// Free the 2d buffer resources
void free2dBuffer(uint8_t** &buffer, int bufferSize){
    // Free nested buffers first
    for(int i = 0; i < bufferSize; i++)
        free(buffer[i]);

    // Free main buffer
    free(buffer);
}

// Get a valid pixel from the image
uint8_t getPixel(int lin, int col, int width, int height, uint8_t* data){
    // Clamp coords
    if(lin < 0)
        lin = 0;
    else if(lin > height - 1)
        lin = height - 1;

    // Clamp coords
    if(col < 0)
        col = 0;
    else if(col > width - 1)
        col = width - 1;

    // Assigns correct value to return
    return data[lin * width + col];
}

// Return the temporary scale pixel format
int getScaleFormat(int inFormat, int outFormat){
    // Retrieve normalized 8 bits format if scale will be done in 10 bits
    if(inFormat == AV_PIX_FMT_V210 && outFormat == AV_PIX_FMT_V210)
        return AV_PIX_FMT_YUV422PNORM;

    // Retrieve the temporary scale format
    switch(inFormat){
    case AV_PIX_FMT_UYVY422:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUV422P:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUV420P:
        return AV_PIX_FMT_YUV420P;
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

// Return coefficient function calculator
double(*getCoefMethod(int operation))(double){
    // Resize operation with different kernels
    switch(operation){
    case SWS_POINT:
        return &NearestNeighborCoefficient;
    case SWS_BILINEAR:
        return &BilinearCoefficient;
    case SWS_BICUBIC:
        return &MitchellCoefficient;
    case SWS_LANCZOS:
        return &LanczosCoefficient;
    }

    // Insuccess
    return nullptr;
}

// Calculate nearest neighbor interpolation coefficient from a distance
double NearestNeighborCoefficient(double val){
    // Calculate absolute value to zero
    double valAbs = abs(val);

    // Calculate coefficient
    if(valAbs <= 0.499999)
        return 1.;
    else
        return 0.;
}

// Calculate bilinear interpolation coefficient from a distance
double BilinearCoefficient(double val){
    // Calculate absolute value to zero
    double valAbs = abs(val);

    // Calculate coefficient
    if(valAbs < 1.)
        return 1. - valAbs;
    else
        return 0.;
}

// Calculate Mitchell interpolation coefficient from a distance
double MitchellCoefficient(double val){
    // Calculate absolute value to zero
    double valAbs = abs(val);

    // Configurable parameters
    double B = 0.;
    double C = .6;

    // Calculate once
    double val1div6 = 1. / 6.;

    // Calculate coefficient
    if(valAbs < 1.)
        return val1div6 * ((6. - 2. * B) + valAbs * valAbs * ((12. * B + 6. * C - 18.) + valAbs * (12. - 9. * B - 6. * C)));
    else if(valAbs < 2.)
        return val1div6 * ((8. * B + 24. * C) + valAbs * ((-12. * B - 48. * C) + valAbs * ((6. * B + 30. * C) + valAbs * (-B - 6. * C))));
    else
        return 0.;
}

// Calculate Lanczos interpolation coefficient from a distance
double LanczosCoefficient(double val){
    // Calculate absolute value to zero
    double valAbs = abs(val);

    // Configurable parameters
    double A = 3.;

    // Calculate coefficient
    if(valAbs < A){
        // Calculate once
        double xpi = val * M_PI;
        double xapi = val / A * M_PI;

        return sin(val * M_PI) * sin(val * M_PI / A) / (val * val * M_PI * M_PI / A);
    } else
        return 0.;
}

// Type cast content of array from uint8_t to float
void arrayConvertToFloat(int size, uint8_t* src, float* dst){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < size; i++){
        union{ float f; uint32_t i; } u;
        u.f = 32768.0f;
        u.i |= src[i];
        dst[i] = u.f - 32768.0f;
    }
}