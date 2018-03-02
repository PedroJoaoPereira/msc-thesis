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



float cubicInterpolate(uint8_t valA, uint8_t valB, uint8_t valC, uint8_t valD, float dist){
    return valB + 0.5f * dist * (valC - valA + dist * (2.0f * valA - 5.0f * valB + 4.0f * valC - valD + dist * (3.0f * (valB - valC) + valD - valA)));
}

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
    float scaleHeightRatio = static_cast<float>(dstHeight / srcHeight);
    float scaleWidthRatio = static_cast<float>(dstWidth / srcWidth);

    if(operation == SWS_BILINEAR){
        // Iterate through each line
        for(int lin = 0; lin < dstHeight; lin++){
            // Original coordinates
            float linInOriginal = (lin - 0.5) / scaleHeightRatio;

            // Calculate original pixels coordinates to interpolate
            int linTop = clamp(floor(linInOriginal), 0, srcHeight - 1);
            int linBottom = clamp(ceil(linInOriginal), 0, srcHeight - 1);

            // Calculate distance to the top left pixel
            float linDist = linInOriginal - linTop;

            // Iterate through each column
            for(int col = 0; col < dstWidth; col++){
                // Original coordinates
                float colInOriginal = (col - 0.5) / scaleWidthRatio;

                // Calculate original pixels coordinates to interpolate
                int colLeft = clamp(floor(colInOriginal), 0, srcWidth - 1);
                int colRight = clamp(ceil(colInOriginal), 0, srcWidth - 1);

                // Calculate distance to the top left pixel
                float colDist = colInOriginal - colLeft;

                // Calculate weight of neighboring pixels
                float leftRatio = 1 - colDist;
                float rightRatio = colDist;
                float topRatio = 1 - linDist;
                float bottomRatio = linDist;

                // Bilinear interpolation operation
                // Y
                dstSlice[0][lin * dstWidth + col] = double2uint8_t(
                    (srcSlice[0][linTop * srcWidth + colLeft] * leftRatio +
                     srcSlice[0][linTop * srcWidth + colRight] * rightRatio) *
                    topRatio +
                    (srcSlice[0][linBottom * srcWidth + colLeft] *
                     leftRatio + srcSlice[0][linBottom * srcWidth + colRight] *
                     rightRatio) *
                    bottomRatio);

                // U
                dstSlice[1][lin * dstWidth + col] = double2uint8_t(
                    (srcSlice[1][linTop * srcWidth + colLeft] * leftRatio +
                     srcSlice[1][linTop * srcWidth + colRight] * rightRatio) *
                    topRatio +
                    (srcSlice[1][linBottom * srcWidth + colLeft] *
                     leftRatio + srcSlice[1][linBottom * srcWidth + colRight] *
                     rightRatio) *
                    bottomRatio);

                // V
                dstSlice[2][lin * dstWidth + col] = double2uint8_t(
                    (srcSlice[2][linTop * srcWidth + colLeft] * leftRatio +
                     srcSlice[2][linTop * srcWidth + colRight] * rightRatio) *
                    topRatio +
                    (srcSlice[2][linBottom * srcWidth + colLeft] *
                     leftRatio + srcSlice[2][linBottom * srcWidth + colRight] *
                     rightRatio) *
                    bottomRatio);
            }
        }

        return 0;
    }

    if(operation == SWS_BICUBIC){
        // Iterate through each line
        for(int lin = 0; lin < dstHeight; lin++){
            // Original coordinates
            float linInOriginal = (lin - 0.5) / scaleHeightRatio;

            // Calculate original pixels coordinates to interpolate
            int linTopFurther = clamp(floor(linInOriginal - 1), 0, srcHeight - 1);
            int linTop = clamp(floor(linInOriginal), 0, srcHeight - 1);
            int linBottom = clamp(ceil(linInOriginal), 0, srcHeight - 1);
            int linBottomFurther = clamp(ceil(linInOriginal + 1), 0, srcHeight - 1);

            // Calculate distance to the top left pixel
            float linDist = linInOriginal - linTop;

            // Iterate through each column
            for(int col = 0; col < dstWidth; col++){
                // Original coordinates
                float colInOriginal = (col - 0.5) / scaleWidthRatio;

                // Calculate original pixels coordinates to interpolate
                int colLeftFurther = clamp(floor(colInOriginal - 1), 0, srcWidth - 1);
                int colLeft = clamp(floor(colInOriginal), 0, srcWidth - 1);
                int colRight = clamp(ceil(colInOriginal), 0, srcWidth - 1);
                int colRightFurther = clamp(ceil(colInOriginal + 1), 0, srcWidth - 1);

                // Calculate distance to the top left pixel
                float colDist = colInOriginal - colLeft;

                // Calculate weight of neighboring pixels
                float leftRatio = 1 - colDist;
                float rightRatio = colDist;
                float topRatio = 1 - linDist;
                float bottomRatio = linDist;

                // Gets the original pixels values
                // 1st row
                uint8_t p00 = srcSlice[0][linTopFurther * srcWidth + colLeftFurther];
                uint8_t p10 = srcSlice[0][linTopFurther * srcWidth + colLeft];
                uint8_t p20 = srcSlice[0][linTopFurther * srcWidth + colRight];
                uint8_t p30 = srcSlice[0][linTopFurther * srcWidth + colRightFurther];

                // 2nd row
                uint8_t p01 = srcSlice[0][linTop * srcWidth + colLeftFurther];
                uint8_t p11 = srcSlice[0][linTop * srcWidth + colLeft];
                uint8_t p21 = srcSlice[0][linTop * srcWidth + colRight];
                uint8_t p31 = srcSlice[0][linTop * srcWidth + colRightFurther];

                // 3rd row
                uint8_t p02 = srcSlice[0][linBottom * srcWidth + colLeftFurther];
                uint8_t p12 = srcSlice[0][linBottom * srcWidth + colLeft];
                uint8_t p22 = srcSlice[0][linBottom * srcWidth + colRight];
                uint8_t p32 = srcSlice[0][linBottom * srcWidth + colRightFurther];

                // 4th row
                uint8_t p03 = srcSlice[0][linBottomFurther * srcWidth + colLeftFurther];
                uint8_t p13 = srcSlice[0][linBottomFurther * srcWidth + colLeft];
                uint8_t p23 = srcSlice[0][linBottomFurther * srcWidth + colRight];
                uint8_t p33 = srcSlice[0][linBottomFurther * srcWidth + colRightFurther];

                // Bilinear interpolation operation
                // Y
                float col0 = cubicInterpolate(p00, p10, p20, p30, colDist);
                float col1 = cubicInterpolate(p01, p11, p21, p31, colDist);
                float col2 = cubicInterpolate(p02, p12, p22, p32, colDist);
                float col3 = cubicInterpolate(p03, p13, p23, p33, colDist);
                float value = cubicInterpolate(col0, col1, col2, col3, linDist);

                /*dstSlice[0][lin * dstWidth + col] = double2uint8_t(
                    (p00 * colDist * colDist * colDist +
                     p10 * colDist * colDist +
                     p20 * colDist +
                     p30) * linDist * linDist * linDist +
                     (p01 * colDist * colDist * colDist +
                      p11 * colDist * colDist +
                      p21 * colDist +
                      p31) * linDist * linDist +
                      (p02 * colDist * colDist * colDist +
                       p12 * colDist * colDist +
                       p22 * colDist +
                       p32) * linDist +
                       (p03 * colDist * colDist * colDist +
                        p13 * colDist * colDist +
                        p23 * colDist +
                        p33));*/

                dstSlice[0][lin * dstWidth + col] = value;
            }
        }

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
