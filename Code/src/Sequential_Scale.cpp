#include "Sequential_Scale.h"

#include "Common.h"

// Resampler sequential method
int seq_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]);

// Sequential scale method
int seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
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
    // Create operation context
    swsContext = sws_getContext(src.width, src.height, src.pixelFormat,
                                dst.width, dst.height, dst.pixelFormat,
                                operation, NULL, NULL, NULL);

    // Verify if scaling context was created
    if(!swsContext){
        cerr << "Could not create SwsContext!" << endl;
        av_frame_free(&srcFrame);
        free(srcBuffer);
        free(dstBuffer);
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(seq_scale(src.width, src.height, src.pixelFormat, srcFrame->data, srcFrame->linesize,
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
        sws_freeContext(swsContext);
        av_frame_free(&srcFrame);
        free(srcBuffer);
        av_frame_free(&dstFrame);
        free(dstBuffer);
        return retVal;
    }

    // Free used resources
    sws_freeContext(swsContext);
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

    return -1;
}

int seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation){

    // Variables used
    int retVal = -1;
    AVPixelFormat tempFormat = AV_PIX_FMT_YUV444P;
    uint8_t* tempBuffer;
    AVFrame* tempFrame;

    // Verify input parameters
    if(!srcSlice || !srcStride || !dstSlice || !dstStride){
        cerr << "One of input parameters is null!" << endl;
        return -1;
    }

    // Verify if image is in right format to scale
    if(srcPixelFormat != tempFormat){
        // Prepare to initialize tempFrame
        retVal = createImageDataBuffer(srcWidth, srcHeight, tempFormat, &tempBuffer);
        if(retVal < 0)
            return retVal;

        // Initialize tempFrame
        retVal = initializeAVFrame(&tempBuffer, srcWidth, srcHeight, tempFormat, &tempFrame);
        if(retVal < 0){
            free(tempBuffer);
            return retVal;
        }

        if(seq_resampler(srcWidth, srcHeight, srcPixelFormat, srcSlice, srcStride,
                         srcWidth, srcHeight, tempFormat, tempFrame->data, tempFrame->linesize) < 0){

            // Free used resources
            av_frame_free(&tempFrame);
            free(tempBuffer);
            
            cerr << "Could not do conversion, not supported!" << endl;
            return -1;
        }
    } else{
        // Makes the tempFrame the srcFrame
        // TODO
    }

    // Apply the scaling operation
    // TODO

    if(dstPixelFormat != tempFormat){
        // Resamples results to the desired one
        if(seq_resampler(dstWidth, dstHeight, tempFormat, tempFrame->data, tempFrame->linesize,
                         dstWidth, dstHeight, dstPixelFormat, dstSlice, dstStride) < 0){

            // Free used resources
            av_frame_free(&tempFrame);
            free(tempBuffer);

            cerr << "Could not do conversion, not supported!" << endl;
            return -1;
        }
    } else{
        // Makes the dstFrame de tempFrame
        // TODO
    }

    // Free used resources
    // Free used resources
    av_frame_free(&tempFrame);
    free(tempBuffer);

    return 0;
}