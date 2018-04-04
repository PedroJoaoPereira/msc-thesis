#include "FFMPEG_Sim_Scale.h"

// Modify the color model of the image
template <class DataType, class PrecisionType>
int ffmpeg_sim_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, DataType* srcSlice[], int srcStride[],
                         int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, DataType* dstSlice[], int dstStride[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        // Calculate the chroma size depending on the source data pixel format
        float heightPercentage = 1.f;
        if(srcPixelFormat == AV_PIX_FMT_YUV420P || srcPixelFormat == AV_PIX_FMT_NV12)
            heightPercentage = 0.5f;

        // Copy data between buffers
        memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight * sizeof(DataType));
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight * heightPercentage * sizeof(DataType));
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight * heightPercentage * sizeof(DataType));

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

            dstSlice[0][index] = roundTo<DataType, PrecisionType>(g); // G
            dstSlice[1][index] = roundTo<DataType, PrecisionType>(b); // B
            dstSlice[2][index] = roundTo<DataType, PrecisionType>(r); // R
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

            dstSlice[0][index] = roundTo<DataType, PrecisionType>(y); // Y
            dstSlice[1][index] = roundTo<DataType, PrecisionType>(u); // U
            dstSlice[2][index] = roundTo<DataType, PrecisionType>(v); // V
        }

        // Success
        return 0;
    }

    return -1;
}

// Apply resizing operation
template <class DataType, class PrecisionType>
void ffmpeg_sim_resize_operation(int srcWidth, int srcHeight, DataType* srcData, int dstWidth, int dstHeight, DataType* dstData, int pixelSupport, PrecisionType (*coefFunc)(PrecisionType)){

    // Get scale ratios
    PrecisionType scaleHeightRatio = static_cast<PrecisionType>(dstHeight) / static_cast<PrecisionType>(srcHeight);
    PrecisionType scaleWidthRatio = static_cast<PrecisionType>(dstWidth) / static_cast<PrecisionType>(srcWidth);

    // Calculate filter step in original data
    PrecisionType vFilterStep = scaleHeightRatio;// min(scaleHeightRatio, static_cast<PrecisionType>(1.));
    PrecisionType hFilterStep = scaleWidthRatio;// min(scaleWidthRatio, static_cast<PrecisionType>(1.));

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
            DataType colorHolder;
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
                    getPixel<DataType>(linTemp, colTemp, srcHeight, srcWidth, srcData, &colorHolder);

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
            dstData[targetLine + col] = roundTo<DataType, PrecisionType>(result);
        }
    }
}

// Prepare resizing operation
template <class DataType, class PrecisionType>
void ffmpeg_sim_resize(int srcWidth, int srcHeight, DataType* srcData, int dstWidth, int dstHeight, DataType* dstData, int operation){

    // Resize operation with different kernels
    if(operation == SWS_BILINEAR)
        ffmpeg_sim_resize_operation<DataType, PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 2, BilinearCoefficient<PrecisionType>);
    if(operation == SWS_BICUBIC)
        ffmpeg_sim_resize_operation<DataType, PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 4, MitchellCoefficient<PrecisionType>);

    // DEBUG - Nearest neighbor resize
    //ffmpeg_sim_resize_operation<DataType, PrecisionType>(srcWidth, srcHeight, srcData, dstWidth, dstHeight, dstData, 2, NearestNeighborCoefficient<PrecisionType>);
}

// Prepares the scaling operation
template <class DataType, class PrecisionType>
int ffmpeg_sim_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, DataType* srcSlice[], int srcStride[],
                         int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, DataType* dstSlice[], int dstStride[],
                         int operation){

    // Check if is only a resample operation
    bool isOnlyResample = false;
    if(srcWidth == dstWidth && srcHeight == dstHeight)
        isOnlyResample = true;

    // Initialize needed variables if it is a scaling operation
    AVPixelFormat scalingSupportedFormat;
    float widthPercentage = 1.f;
    float heightPercentage = 1.f;
    if(!isOnlyResample){
        // Retrieve intermediate supported pixel format for scaling
        scalingSupportedFormat = AV_PIX_FMT_GBRP;

        // Calculate the chroma size depending on the source data pixel format
        float heightPercentage = 1.f;
        if(srcPixelFormat == AV_PIX_FMT_YUV420P || srcPixelFormat == AV_PIX_FMT_NV12)
            heightPercentage = 0.5f;
    }

#pragma region INITIALIZE TEMPORARY FRAMES
    // Temporary frames used in intermediate operations
    DataType* resampleTempFrameBuffer, *scaleTempFrameBuffer;
    AVFrame* resampleTempFrame, *scaleTempFrame;

    // Only initializes frames if is not only a resample opreration
    if(!isOnlyResample){
        // Initialize temporary frame buffers
        if(createImageDataBuffer(srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrameBuffer) < 0)
            return -1;
        if(createImageDataBuffer(dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrameBuffer) < 0){
            free(resampleTempFrameBuffer);
            return -1;
        }

        // Initialize temporary frames
        if(initializeAVFrame(&resampleTempFrameBuffer, srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrame) < 0){
            free(resampleTempFrameBuffer);
            free(scaleTempFrameBuffer);
            return -1;
        }
        if(initializeAVFrame(&scaleTempFrameBuffer, dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrame) < 0){
            av_frame_free(&resampleTempFrame);
            free(resampleTempFrameBuffer);
            free(scaleTempFrameBuffer);
            return -1;
        }
    }
#pragma endregion

    // Last resample frame
    DataType** lastResampleData = srcSlice;
    int* lastResampleLinesize = srcStride;
    AVPixelFormat lastResamplePixelFormat = srcPixelFormat;

#pragma region RESIZE OPERATION
    // Verify if is not only a resample operation
    if(!isOnlyResample){
        // Resamples image to a supported format
        if(ffmpeg_sim_resampler<DataType, PrecisionType>(srcWidth, srcHeight, srcPixelFormat, srcSlice, srcStride,
                                          srcWidth, srcHeight, scalingSupportedFormat, resampleTempFrame->data, resampleTempFrame->linesize) < 0){
            av_frame_free(&resampleTempFrame);
            av_frame_free(&scaleTempFrame);
            free(resampleTempFrameBuffer);
            free(scaleTempFrameBuffer);
            return -2;
        }

        // Apply the resizing operation to each color channel
        for(int colorChannel = 0; colorChannel < 3; colorChannel++){
            // Avoid full chroma processing
            if(colorChannel == 0)
                ffmpeg_sim_resize<DataType, PrecisionType>(srcWidth, srcHeight, resampleTempFrame->data[colorChannel],
                                            dstWidth, dstHeight, scaleTempFrame->data[colorChannel], operation);
            else
                ffmpeg_sim_resize<DataType, PrecisionType>(static_cast<int>(srcWidth * widthPercentage), static_cast<int>(srcHeight * heightPercentage), resampleTempFrame->data[colorChannel],
                                            static_cast<int>(dstWidth * widthPercentage), static_cast<int>(dstHeight * heightPercentage), scaleTempFrame->data[colorChannel], operation);
        }

        // Assign correct values to apply last resample
        lastResampleData = scaleTempFrame->data;
        lastResampleLinesize = scaleTempFrame->linesize;
        lastResamplePixelFormat = scalingSupportedFormat;
    }
#pragma endregion

#pragma region LAST RESAMPLE
    // Last resample to destination frame
    if(ffmpeg_sim_resampler<DataType, PrecisionType>(dstWidth, dstHeight, lastResamplePixelFormat, lastResampleData, lastResampleLinesize,
                                      dstWidth, dstHeight, dstPixelFormat, dstSlice, dstStride) < 0){
        if(!isOnlyResample){
            av_frame_free(&resampleTempFrame);
            av_frame_free(&scaleTempFrame);
            free(resampleTempFrameBuffer);
            free(scaleTempFrameBuffer);
        }

        return -2;
    }
#pragma endregion
    
    // Free used resources
    if(!isOnlyResample){
        av_frame_free(&resampleTempFrame);
        av_frame_free(&scaleTempFrame);
        free(resampleTempFrameBuffer);
        free(scaleTempFrameBuffer);
    }

    //Success
    return 0;
}

// Wrapper for the ffmpeg simulator scale operation method
int ffmpeg_sim_scale(ImageInfo src, ImageInfo dst, int operation){
    // Variables used
    int retVal = -1, duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Verify valid input dimensions
    if(src.width < 0 || src.height < 0 || dst.width < 0 || dst.height < 0){
        cerr << "[SIMULATOR] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid input data
    if(!src.frame->data || !src.frame->linesize || !dst.frame->data || !dst.frame->linesize){
        cerr << "[SIMULATOR] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if(!isSupportedFormat(src.pixelFormat) || !isSupportedFormat(dst.pixelFormat)){
        cerr << "[SIMULATOR] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if(operation != SWS_BILINEAR && operation != SWS_BICUBIC){
        cerr << "[SIMULATOR] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(int retVal = ffmpeg_sim_scale_aux<uint8_t, double>(src.width, src.height, src.pixelFormat, src.frame->data, src.frame->linesize,
                                     dst.width, dst.height, dst.pixelFormat, dst.frame->data, dst.frame->linesize, operation) < 0){
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
    duration = duration_cast<milliseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}