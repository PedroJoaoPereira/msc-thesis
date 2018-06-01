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

// Return the temporary scale pixel format
int getScaleFormat(int inFormat, int outFormat){
    // Retrieve normalized 8 bits format if scale will be done in 10 bits
    if(inFormat == AV_PIX_FMT_V210 && outFormat == AV_PIX_FMT_V210)
        return AV_PIX_FMT_YUV422PNORM;

    // Retrieve the temporary scale format
    switch(inFormat){
    case AV_PIX_FMT_UYVY422:
        if(outFormat == AV_PIX_FMT_YUV420P || outFormat == AV_PIX_FMT_NV12)
            return AV_PIX_FMT_YUV420P;
        else
            return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUV422P:
        if(outFormat == AV_PIX_FMT_YUV420P || outFormat == AV_PIX_FMT_NV12)
            return AV_PIX_FMT_YUV420P;
        else
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
