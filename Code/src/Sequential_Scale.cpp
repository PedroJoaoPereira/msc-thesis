#include "Sequential_Scale.h"

#include "Common.h"

// Resampler sequential method
int seq_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]);

// Sequential scale method
int seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
                  int operation);

// Prepares the scaling operation
int seq_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation);

int sequential_scale(ImageInfo src, ImageInfo dst, int operation){
    // Variables used
    int retVal = -1, duration = -1;
    uint8_t* srcBuffer, *dstBuffer;
    AVFrame* srcFrame, *dstFrame;
    SwsContext* swsContext;
    high_resolution_clock::time_point initTime, stopTime;

    // Read image from a file
    retVal = readImageFromFile(src.fileName, &srcBuffer);
    if(retVal < 0)
        return retVal;

    // Initialize srcFrame
    retVal = initializeAVFrame(&srcBuffer, src.width, src.height, src.pixelFormat, &srcFrame);
    if(retVal < 0){
        free(srcBuffer);
        return retVal;
    }

    // Prepare to initialize dstFrame
    retVal = createImageDataBuffer(dst.width, dst.height, dst.pixelFormat, &dstBuffer);
    if(retVal < 0){
        av_frame_free(&srcFrame);
        free(srcBuffer);
        return retVal;
    }

    // Initialize dstFrame
    retVal = initializeAVFrame(&dstBuffer, dst.width, dst.height, dst.pixelFormat, &dstFrame);
    if(retVal < 0){
        av_frame_free(&srcFrame);
        free(srcBuffer);
        free(dstBuffer);
        return retVal;
    }

    // ------------------------------ SCALING ------------------------------
    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(seq_scale_aux(src.width, src.height, src.pixelFormat, srcFrame->data, srcFrame->linesize,
                 dst.width, dst.height, dst.pixelFormat, dstFrame->data, dstFrame->linesize,
                 operation) < 0) return -1;

    // DEBUG
    /*seq_resampler(src.width, src.height, src.pixelFormat, srcFrame->data, srcFrame->linesize,
                  dst.width, dst.height, dst.pixelFormat, dstFrame->data, dstFrame->linesize);*/

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<milliseconds>(stopTime - initTime).count();
    // ---------------------------------------------------------------------

    // Write image to a file
    retVal = writeImageToFile(dst.fileName, &dstFrame);
    if(retVal < 0){
        av_frame_free(&srcFrame);
        free(srcBuffer);
        av_frame_free(&dstFrame);
        free(dstBuffer);
        return retVal;
    }

    // Free used resources
    av_frame_free(&srcFrame);
    free(srcBuffer);
    av_frame_free(&dstFrame);
    free(dstBuffer);

    // Return execution time of the scaling operation
    return duration;
}

int seq_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight);
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight);
        memcpy(dstSlice[3], srcSlice[3], srcStride[3] * srcHeight);

        return 0;
    }

    // -----------------------------------------------
    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Calculate once
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index += 4){
            dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

            dstSlice[1][index / 4] = srcSlice[0][index];            // U
            dstSlice[2][index / 4] = srcSlice[0][index + 2];        // V
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Calculate once
        int stride = srcStride[0];
        long numElements = stride * srcHeight;
        int columnsByLine = stride / 2;

        // Loop through each pixel
        for(int index = 0; index < numElements; index += 4){
            dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

            int lineIndex = index / (stride * 2);
            if(lineIndex % 2 == 0){
                int columnIndex = (index / 4) % columnsByLine;
                int chromaIndex = lineIndex / 2 * columnsByLine + columnIndex;
                dstSlice[1][chromaIndex] = srcSlice[0][index];      // U
                dstSlice[2][chromaIndex] = srcSlice[0][index + 2];  // V
            }
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV444P){
        // Calculate once
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index += 4){
            dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

            dstSlice[1][index / 2] = srcSlice[0][index];            // U
            dstSlice[1][index / 2 + 1] = srcSlice[0][index];

            dstSlice[2][index / 2] = srcSlice[0][index + 2];        // V
            dstSlice[2][index / 2 + 1] = srcSlice[0][index + 2];
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV444P){
        // Calculate once
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            dstSlice[0][index] = srcSlice[0][index];        // Y

            dstSlice[1][index] = srcSlice[1][index / 2];    // U
            dstSlice[1][index] = srcSlice[1][index / 2];

            dstSlice[2][index] = srcSlice[2][index / 2];    // V
            dstSlice[2][index] = srcSlice[2][index / 2];
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Calculate once
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            dstSlice[0][index] = srcSlice[0][index];            // Y

            if(index % 2 == 0){
                dstSlice[1][index / 2] = srcSlice[1][index];    // U
                dstSlice[2][index / 2] = srcSlice[2][index];    // V
            }
        }

        // Success
        return 0;
    }

    cerr << "Conversion not supported" << endl;
    return -1;
}

int seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
              int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
              int operation){

    // Get scale ratios
    float scaleHeightRatio = (float) dstHeight / srcHeight;
    float scaleWidthRatio = (float) dstWidth / srcWidth;

    if(operation == SWS_BILINEAR){
        // Iterate through each line
        for(int lin = 0; lin < dstHeight; lin++){
            // Get line in original image
            float linOrigFloat = lin / scaleHeightRatio;
            float linOrigRemainder = fmod(lin, scaleHeightRatio);
            int linOrig = int(linOrigFloat);
            float linDist = linOrigFloat - floor(linOrigFloat);

            // Iterate through each column
            for(int col = 0; col < dstWidth; col++){
                // Get column in original image
                float colOrigFloat = col / scaleWidthRatio;
                float colOrigRemainder = fmod(col, scaleWidthRatio);
                int colOrig = int(colOrigFloat);
                float colDist = colOrigFloat - floor(colOrigFloat);

                // If same position as an original pixel
                if(linOrigRemainder == 0 && colOrigRemainder == 0){
                    // Original pixel to the result
                    dstSlice[0][lin * dstWidth + col] = srcSlice[0][linOrig * srcWidth + colOrig];
                    dstSlice[1][lin * dstWidth + col] = srcSlice[1][linOrig * srcWidth + colOrig];
                    dstSlice[2][lin * dstWidth + col] = srcSlice[2][linOrig * srcWidth + colOrig];

                    // Continue processing following pixels
                    continue;
                }

                // Get original pixels value - component-line-column
                uint8_t y00 = srcSlice[0][linOrig * srcWidth + colOrig];
                uint8_t y01 = srcSlice[0][linOrig * srcWidth + (colOrig + 1)];
                uint8_t y10 = srcSlice[0][(linOrig + 1) * srcWidth + colOrig];
                uint8_t y11 = srcSlice[0][(linOrig + 1) * srcWidth + (colOrig + 1)];

                uint8_t u00 = srcSlice[1][linOrig * srcWidth + colOrig];
                uint8_t u01 = srcSlice[1][linOrig * srcWidth + (colOrig + 1)];
                uint8_t u10 = srcSlice[1][(linOrig + 1) * srcWidth + colOrig];
                uint8_t u11 = srcSlice[1][(linOrig + 1) * srcWidth + (colOrig + 1)];

                uint8_t v00 = srcSlice[2][linOrig * srcWidth + colOrig];
                uint8_t v01 = srcSlice[2][linOrig * srcWidth + (colOrig + 1)];
                uint8_t v10 = srcSlice[2][(linOrig + 1) * srcWidth + colOrig];
                uint8_t v11 = srcSlice[2][(linOrig + 1) * srcWidth + (colOrig + 1)];

                // Horizontal linear interpolation
                float liy00y01 = lerp(y00, y01, colDist);
                float liy10y11 = lerp(y10, y11, colDist);
                float liYVertical = lerp(liy00y01, liy10y11, linDist);

                float liu00u01 = lerp(u00, u01, colDist);
                float liu10u11 = lerp(u10, u11, colDist);
                float liUVertical = lerp(liu00u01, liu10u11, linDist);

                float liv00v01 = lerp(v00, v01, colDist);
                float liv10v11 = lerp(v10, v11, colDist);
                float liVVertical = lerp(liv00v01, liv10v11, linDist);

                // Clamp result
                uint8_t newYValue = uint8_t(clamp(liYVertical, 0, 255));
                uint8_t newUValue = uint8_t(clamp(liUVertical, 0, 255));
                uint8_t newVValue = uint8_t(clamp(liVVertical, 0, 255));

                // Assign new value to the corresponding pixel
                dstSlice[0][lin * dstWidth + col] = newYValue;
                dstSlice[1][lin * dstWidth + col] = newUValue;
                dstSlice[2][lin * dstWidth + col] = newVValue;
            }
        }

        return 0;
    }

    if(operation == SWS_BICUBIC){

        return 0;
    }

    cerr << "Operation not supported" << endl;
    return -1;
}

int seq_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation){

    // Variables used
    int retVal = -1;
    AVPixelFormat scalingSupportedFormat = AV_PIX_FMT_YUV444P;
    uint8_t* resampleTempFrameBuffer,* scaleTempFrameBuffer;
    AVFrame* resampleTempFrame,* scaleTempFrame;

    // Verify input parameters
    if(srcWidth < 0 || srcHeight < 0 || dstWidth < 0 || dstHeight < 0){
        cerr << "One of input dimensions is negative!" << endl;
        return -1;
    }
    if(srcWidth % 2 != 0 || srcHeight % 2 != 0 || dstWidth % 2 != 0 || dstHeight % 2 != 0){
        cerr << "One of input dimensions is not divisible by 2!" << endl;
        return -1;
    }
    if(!srcSlice || !srcStride || !dstSlice || !dstStride){
        cerr << "One of input parameters is null!" << endl;
        return -1;
    }

    // Prepare to initialize resampleTempFrame
    retVal = createImageDataBuffer(srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrameBuffer);
    if(retVal < 0)
        return retVal;

    // Initialize resampleTempFrame
    retVal = initializeAVFrame(&resampleTempFrameBuffer, srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrame);
    if(retVal < 0){
        free(resampleTempFrameBuffer);
        return retVal;
    }

    // Verify if image is in right format to scale
    if(srcPixelFormat != scalingSupportedFormat){
        // Resamples image to a supported format
        retVal = seq_resampler(srcWidth, srcHeight, srcPixelFormat, srcSlice, srcStride,
                               srcWidth, srcHeight, scalingSupportedFormat, resampleTempFrame->data, resampleTempFrame->linesize);
        if(retVal < 0){
            av_frame_free(&resampleTempFrame);
            free(resampleTempFrameBuffer);
            return retVal;
        }
    } else{
        // Copy data from source frame to temp frame
        resampleTempFrame->data[0] = srcSlice[0];
        resampleTempFrame->data[1] = srcSlice[1];
        resampleTempFrame->data[2] = srcSlice[2];
        resampleTempFrame->data[3] = srcSlice[3];

        // Copy linesize from source frame to temp frame
        resampleTempFrame->linesize[0] = srcStride[0];
        resampleTempFrame->linesize[1] = srcStride[1];
        resampleTempFrame->linesize[2] = srcStride[2];
        resampleTempFrame->linesize[3] = srcStride[3];
    }

    // Prepare to initialize scaleTempFrame
    retVal = createImageDataBuffer(dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrameBuffer);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        return retVal;
    }

    // Initialize scaleTempFrame
    retVal = initializeAVFrame(&scaleTempFrameBuffer, dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrame);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        free(scaleTempFrameBuffer);
        return retVal;
    }

    // Apply the scaling operation
    retVal = seq_scale(srcWidth, srcHeight, scalingSupportedFormat, resampleTempFrame->data, resampleTempFrame->linesize,
                       dstWidth, dstHeight, scalingSupportedFormat, scaleTempFrame->data, scaleTempFrame->linesize,
                       operation);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        av_frame_free(&scaleTempFrame);
        free(scaleTempFrameBuffer);
        return retVal;
    }

    if(dstPixelFormat != scalingSupportedFormat){
        // Resamples results to the desired one
        retVal = seq_resampler(dstWidth, dstHeight, scalingSupportedFormat, scaleTempFrame->data, scaleTempFrame->linesize,
                               dstWidth, dstHeight, dstPixelFormat, dstSlice, dstStride);
        if(retVal < 0){
            av_frame_free(&resampleTempFrame);
            free(resampleTempFrameBuffer);
            av_frame_free(&scaleTempFrame);
            free(scaleTempFrameBuffer);
            return retVal;
        }
    } else{
        // Copy data from scaleTempFrame to result frame
        dstSlice = scaleTempFrame->data;
    }

    // Free used resources
    av_frame_free(&resampleTempFrame);
    free(resampleTempFrameBuffer);
    av_frame_free(&scaleTempFrame);
    free(scaleTempFrameBuffer);

    return 0;
}