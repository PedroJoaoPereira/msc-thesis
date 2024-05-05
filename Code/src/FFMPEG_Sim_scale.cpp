#include "FFMPEG_Sim_Scale.h"

// Modify the color model of the image
template <class PrecisionType>
int ffmpeg_sim_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                         int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        // Copy data between buffers
        memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight);
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight);
        memcpy(dstSlice[3], srcSlice[3], srcStride[3] * srcHeight);

        // Success
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_GBRP){
        // Number of elements
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            // Retrieve values from source data
            PrecisionType y = static_cast<PrecisionType>(srcSlice[0][index]); // Y
            PrecisionType u = static_cast<PrecisionType>(srcSlice[1][index]); // U
            PrecisionType v = static_cast<PrecisionType>(srcSlice[2][index]); // V

            // BT 601
            y -= static_cast<PrecisionType>(16.);
            u -= static_cast<PrecisionType>(128.);
            v -= static_cast<PrecisionType>(128.);

            PrecisionType r = static_cast<PrecisionType>(1.164) * y + static_cast<PrecisionType>(1.596) * v;
            PrecisionType g = static_cast<PrecisionType>(1.164) * y - static_cast<PrecisionType>(0.392) * u - static_cast<PrecisionType>(0.813) * v;
            PrecisionType b = static_cast<PrecisionType>(1.164) * y + static_cast<PrecisionType>(2.017) * u;

            // Clamp values to avoid overshooting and undershooting
            clamp<PrecisionType>(r, static_cast<PrecisionType>(0.), static_cast<PrecisionType>(255.));
            clamp<PrecisionType>(g, static_cast<PrecisionType>(0.), static_cast<PrecisionType>(255.));
            clamp<PrecisionType>(b, static_cast<PrecisionType>(0.), static_cast<PrecisionType>(255.));

            dstSlice[0][index] = roundTo<PrecisionType>(g); // G
            dstSlice[1][index] = roundTo<PrecisionType>(b); // B
            dstSlice[2][index] = roundTo<PrecisionType>(r); // R
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_GBRP && dstPixelFormat == AV_PIX_FMT_YUV444P){
        // Number of elements
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            // Retrieve values from source data
            PrecisionType g = static_cast<PrecisionType>(srcSlice[0][index]);	// G
            PrecisionType b = static_cast<PrecisionType>(srcSlice[1][index]);	// B
            PrecisionType r = static_cast<PrecisionType>(srcSlice[2][index]);	// R

            // BT 601
            PrecisionType y = static_cast<PrecisionType>(0.257) * r + static_cast<PrecisionType>(0.504) * g + static_cast<PrecisionType>(0.098) * b + static_cast<PrecisionType>(16.);
            PrecisionType u = -static_cast<PrecisionType>(0.148) * r - static_cast<PrecisionType>(0.291) * g + static_cast<PrecisionType>(0.439) * b + static_cast<PrecisionType>(128.);
            PrecisionType v = static_cast<PrecisionType>(0.439) * r - static_cast<PrecisionType>(0.368) * g - static_cast<PrecisionType>(0.071) * b + static_cast<PrecisionType>(128.);

            // Clamp values to avoid overshooting and undershooting
            clamp<PrecisionType>(y, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(235.));
            clamp<PrecisionType>(u, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(240.));
            clamp<PrecisionType>(v, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(240.));

            dstSlice[0][index] = roundTo<PrecisionType>(y); // Y
            dstSlice[1][index] = roundTo<PrecisionType>(u); // U
            dstSlice[2][index] = roundTo<PrecisionType>(v); // V
        }

        // Success
        return 0;
    }

    // No conversion was supported
    return -1;
}

// Apply resizing operation
template <class PrecisionType>
void ffmpeg_sim_resize_operation(int srcWidth, int srcHeight, uint8_t* srcData,
                                 int dstWidth, int dstHeight, uint8_t* dstData,
                                 int pixelSupport, PrecisionType(*coefFunc)(PrecisionType)){

    // Get scale ratios
    PrecisionType scaleHeightRatio = static_cast<PrecisionType>(dstHeight) / static_cast<PrecisionType>(srcHeight);
    PrecisionType scaleWidthRatio = static_cast<PrecisionType>(dstWidth) / static_cast<PrecisionType>(srcWidth);

    // Calculate filter step in original data
    PrecisionType vFilterStep = scaleHeightRatio;
    PrecisionType hFilterStep = scaleWidthRatio;
    
    // Holds coefficients
    PrecisionType* vCoefficients = (PrecisionType*) malloc(pixelSupport * sizeof(PrecisionType));
    PrecisionType* hCoefficients = (PrecisionType*) malloc(pixelSupport * sizeof(PrecisionType));

    // Calculate once
    int pixelSupportDiv2 = pixelSupport / 2;

    // Iterate through each line of the scaled image
    for(int lin = 0; lin < dstHeight; lin++){
        // Calculate once the target line coordinate
        int targetLine = lin * dstWidth;

        // Original line index coordinate
        PrecisionType linOriginal = (static_cast<PrecisionType>(lin) + static_cast<PrecisionType>(0.5)) / scaleHeightRatio - static_cast<PrecisionType>(0.5);

        // Calculate nearest original position
        int linNearest = floor(linOriginal);
        // Calculate limit positions
        int linStart = linNearest - pixelSupportDiv2 + 1;
        int linStop = linStart + pixelSupport - 1;

        // Calculate distance to left nearest pixel
        PrecisionType vDist = linOriginal - static_cast<PrecisionType>(linNearest);
        // Calculate distance to original pixels
        PrecisionType vUpperCoef = vDist;
        PrecisionType vBottomtCoef = static_cast<PrecisionType>(1.) - vDist;

        // Calculate coefficients
        for(int index = 0; index < pixelSupportDiv2; index++){
            vCoefficients[pixelSupportDiv2 - index - 1] = coefFunc(vUpperCoef + index * vFilterStep);
            vCoefficients[index + pixelSupportDiv2] = coefFunc(vBottomtCoef + index * vFilterStep);
        }

        // Iterate through each column of the scaled image
        for(int col = 0; col < dstWidth; col++){
            // Original column index coordinate
            PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(0.5)) / scaleWidthRatio - static_cast<PrecisionType>(0.5);

            // Calculate nearest original position
            int colNearest = floor(colOriginal);
            // Calculate limit positions
            int colStart = colNearest - pixelSupportDiv2 + 1;
            int colStop = colStart + pixelSupport - 1;

            // Calculate distance to left nearest pixel
            PrecisionType hDist = colOriginal - static_cast<PrecisionType>(colNearest);
            // Calculate distance to original pixels
            PrecisionType hLeftCoef = hDist;
            PrecisionType hRightCoef = static_cast<PrecisionType>(1.) - hDist;

            // Calculate coefficients
            for(int index = 0; index < pixelSupportDiv2; index++){
                hCoefficients[pixelSupportDiv2 - index - 1] = coefFunc(hLeftCoef + index * hFilterStep);
                hCoefficients[index + pixelSupportDiv2] = coefFunc(hRightCoef + index * hFilterStep);
            }

            // Temporary variables used in the interpolation
            PrecisionType colorAcc = static_cast<PrecisionType>(0.);
            PrecisionType weightAcc = static_cast<PrecisionType>(0.);
            // Calculate resulting color from coefficients
            for(int linTemp = linStart; linTemp <= linStop; linTemp++){
                // Access once the memory
                PrecisionType vCoef = vCoefficients[linTemp - linStart];

                for(int colTemp = colStart; colTemp <= colStop; colTemp++){
                    // Access once the memory
                    PrecisionType hCoef = hCoefficients[colTemp - colStart];

                    // Get pixel from source data
                    uint8_t colorHolder = getPixel(linTemp, colTemp, srcWidth, srcHeight, srcData);

                    // Calculate pixel color weight
                    PrecisionType weight = vCoef * hCoef;

                    // Weights neighboring pixel and add it to the result
                    colorAcc += static_cast<PrecisionType>(colorHolder) * weight;
                    weightAcc += weight;
                }
            }

            // Calculate resulting color
            PrecisionType result = colorAcc / weightAcc;
            // Clamp value to avoid undershooting and overshooting
            clamp<PrecisionType>(result, static_cast<PrecisionType>(0.), static_cast<PrecisionType>(255.));
            // Assign calculated color to destiantion data
            dstData[targetLine + col] = roundTo<PrecisionType>(result);
        }
    }
}

// Prepare resizing operation
template <class PrecisionType>
void ffmpeg_sim_resize(int srcWidth, int srcHeight, uint8_t* srcData,
                       int dstWidth, int dstHeight, uint8_t* dstData, int operation){

    // Resize operation with different kernels
    if(operation == SWS_POINT)
        ffmpeg_sim_resize_operation<PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 2, NearestNeighborCoefficient<PrecisionType>);
    if(operation == SWS_BILINEAR)
        ffmpeg_sim_resize_operation<PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 2, BilinearCoefficient<PrecisionType>);
    if(operation == SWS_BICUBIC)
        ffmpeg_sim_resize_operation<PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 4, MitchellCoefficient<PrecisionType>);
}

// Prepares the scaling operation
template <class PrecisionType>
int ffmpeg_sim_scale_aux(AVFrame* src, AVFrame* dst, int operation){

    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    int dstWidth = dst->width, dstHeight = dst->height;
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Check if is only a resample operation
    bool isOnlyResample = false;
    if(srcWidth == dstWidth && srcHeight == dstHeight)
        isOnlyResample = true;

    // Initialize needed variables if it is a scaling operation
    AVPixelFormat scalingSupportedFormat = AV_PIX_FMT_GBRP;

#pragma region INITIALIZE TEMPORARY FRAMES
    // Temporary frames used in intermediate operations
    uint8_t* resampleBuffer,* scaleBuffer;
    AVFrame* resampleFrame,* scaleFrame;

    // Only initializes frames if is not only a resample opreration
    if(!isOnlyResample){
        // Initialize temporary frame buffers
        if(createImageDataBuffer(srcWidth, srcHeight, scalingSupportedFormat, &resampleBuffer) < 0)
            return -1;
        if(createImageDataBuffer(dstWidth, dstHeight, scalingSupportedFormat, &scaleBuffer) < 0){
            free(resampleBuffer);
            return -1;
        }

        // Initialize temporary frames
        if(initializeAVFrame(&resampleBuffer, srcWidth, srcHeight, scalingSupportedFormat, &resampleFrame) < 0){
            free(resampleBuffer);
            free(scaleBuffer);
            return -1;
        }
        if(initializeAVFrame(&scaleBuffer, dstWidth, dstHeight, scalingSupportedFormat, &scaleFrame) < 0){
            av_frame_free(&resampleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
            return -1;
        }
    }
#pragma endregion

    // Last resample frame
    AVFrame* lastResampleFrame = src;
    AVPixelFormat lastResamplePixelFormat = srcFormat;

#pragma region RESIZE OPERATION
    // Verify if is not only a resample operation
    if(!isOnlyResample){
        // Resamples image to a supported format
        if(ffmpeg_sim_resampler<PrecisionType>(srcWidth, srcHeight, srcFormat, src->data, src->linesize,
                                                         srcWidth, srcHeight, scalingSupportedFormat, resampleFrame->data, resampleFrame->linesize) < 0){
            av_frame_free(&resampleFrame);
            av_frame_free(&scaleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
            return -2;
        }

        // Apply the resizing operation to each color channel
        for(int colorChannel = 0; colorChannel < 3; colorChannel++){
            ffmpeg_sim_resize<PrecisionType>(srcWidth, srcHeight, resampleFrame->data[colorChannel],
                                             dstWidth, dstHeight, scaleFrame->data[colorChannel], operation);
        }

        // Assign correct values to apply last resample
        lastResampleFrame = scaleFrame;
        lastResamplePixelFormat = scalingSupportedFormat;
    }
#pragma endregion

#pragma region LAST RESAMPLE
    // Last resample to destination frame
    if(ffmpeg_sim_resampler<PrecisionType>(dstWidth, dstHeight, lastResamplePixelFormat, lastResampleFrame->data, lastResampleFrame->linesize,
                                                     dstWidth, dstHeight, dstFormat, dst->data, dst->linesize) < 0){
        if(!isOnlyResample){
            av_frame_free(&resampleFrame);
            av_frame_free(&scaleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
        }

        return -2;
    }
#pragma endregion

    // Free used resources
    if(!isOnlyResample){
        av_frame_free(&resampleFrame);
        av_frame_free(&scaleFrame);
        free(resampleBuffer);
        free(scaleBuffer);
    }

    //Success
    return 0;
}

// Wrapper for the ffmpeg simulator scale operation method
int ffmpeg_sim_scale(AVFrame* src, AVFrame* dst, int operation){
    // Variables used
    int retVal = -1, duration = -1;
    high_resolution_clock::time_point initTime, stopTime;
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid input dimensions
    if(src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0){
        cerr << "[SIMULATOR] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid input data
    if(!src->data || !src->linesize || !dst->data || !dst->linesize){
        cerr << "[SIMULATOR] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if(srcFormat != AV_PIX_FMT_YUV444P && srcFormat != AV_PIX_FMT_GBRP &&
       dstFormat != AV_PIX_FMT_YUV444P && dstFormat != AV_PIX_FMT_GBRP){
        cerr << "[SIMULATOR] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if(!isSupportedOperation(operation)){
        cerr << "[SIMULATOR] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(int retVal = ffmpeg_sim_scale_aux<double>(src, dst, operation) < 0){
        string error = "[SIMULATOR] Operation could not be done (";

        if(retVal == -1)
            error += "error initializing frames)!";

        if(retVal == -2)
            error += "resample - conversion not supported)!";

        // Display error
        cerr << error << endl;

        // Insuccess
        return retVal;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}