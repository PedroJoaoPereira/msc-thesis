#include "Sequential_Scale.h"

#include "Common.h"

void getPixel(uint8_t** data, int channel, int width, int height, int lin, int col, uint8_t* pixelVal){
    // Clamp coords
    clampPixel(lin, 0, height - 1);
    clampPixel(col, 0, width - 1);

    // Assigns correct value to return
    *pixelVal = data[channel][lin * width + col];
}

double bcoef(double x){
    /*
    // For a = -0.5
    double xRounded = abs(x);
    if(xRounded <= 1.0){
        return xRounded * xRounded * (1.5 * xRounded - 2.5) + 1.0;
    } else if(xRounded < 2.0){
        return xRounded * (xRounded * (-0.5 * xRounded + 2.5) - 4.0) + 2.0;
    } else{
        return 0.0;
    }
    */

    // For a = -0.5
    double a = -0.57;
    //double a = -0.6;
    double xRounded = abs(x);
    if(xRounded <= 1.0){
        return (a + 2.0) * xRounded * xRounded * xRounded - (a + 3.0) * xRounded * xRounded + 1.0;
    } else if(xRounded < 2.0){
        return a * xRounded * xRounded * xRounded - 5.0 * a * xRounded * xRounded + 8.0 * a * xRounded - 4.0 * a;
    } else{
        return 0.0;
    }
}

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
    double scaleHeightRatio = static_cast<double>(dstHeight) / srcHeight;
    double scaleWidthRatio = static_cast<double>(dstWidth) / srcWidth;

    if(operation == SWS_BILINEAR){
        /*
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
        */

        return 0;
    }

    if(operation == SWS_BICUBIC){
        // Iterate through each line
        for(int lin = 0; lin < dstHeight; lin++){
            // Original coordinates
            double linInOriginal = (lin + 0.5) / scaleHeightRatio;
            // Original lin index
            double linIndexOriginalD = floor(linInOriginal);
            int linIndexOriginal = double2uint8_t(linIndexOriginalD);
            // Calculate distance to the original pixel
            double linDist = abs(linInOriginal - (linIndexOriginalD + 0.5));

            // Calculate neighboring pixels coords
            int linMin = linIndexOriginal - 1;
            int linMax = linIndexOriginal + 2;

            // Iterate through each column
            for(int col = 0; col < dstWidth; col++){
                // Original coordinates
                double colInOriginal = (col + 0.5) / scaleWidthRatio;
                // Original col index
                double colIndexOriginalD = floor(colInOriginal);
                int colIndexOriginal = double2uint8_t(colIndexOriginalD);
                // Calculate distance to the original pixel
                double colDist = abs(colInOriginal - (colIndexOriginalD + 0.5));

                // Calculate neighboring pixels coords
                int colMin = colIndexOriginal - 1;
                int colMax = colIndexOriginal + 2;

                double weightedValue = 0.0, sum = 0.0, wSum = 0.0;
                uint8_t* colorHolder = new uint8_t();
                double weight;
                // Iterate through each neighboring pixels
                for(int linTemp = linMin; linTemp <= linMax; linTemp++){
                    for(int colTemp = colMin; colTemp <= colMax; colTemp++){
                        // Retreive pixel from data buffer
                        getPixel(srcSlice, 0, srcWidth, srcHeight, linTemp, colTemp, colorHolder);
                        // Calculate weight of pixel in the bicubic interpolation
                        weight = bcoef(abs(linInOriginal - (static_cast<double>(linTemp) + 0.5)))
                            * bcoef(abs(colInOriginal - (static_cast<double>(colTemp) + 0.5)));
                        // Calculate weighted value
                        weightedValue = *colorHolder * weight;
                        // Sum weighted values
                        sum += weightedValue;
                        // Sum weights
                        wSum += weight;
                    }                    
                }

                // Calculate resulting color
                double result = sum / wSum;
                // Clamp value to avoid color undershooting and overshooting
                clamp(result, 0.0, 255.0);
                // Store the result value
                dstSlice[0][lin * dstWidth + col] = round(result);
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
