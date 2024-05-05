#include "Sequential_Scale.h"

#include "Common.h"

// Resampler sequential method
int seq_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]);

// Sequential scale method
void seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
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

    // DEBUG
    seq_resampler(src.width, src.height, src.pixelFormat, srcFrame->data, srcFrame->linesize,
                  dst.width, dst.height, dst.pixelFormat, dstFrame->data, dstFrame->linesize);

    // Apply the scaling operation
    seq_scale(src.width, src.height, src.pixelFormat, srcFrame->data, srcFrame->linesize,
              dst.width, dst.height, dst.pixelFormat, dstFrame->data, dstFrame->linesize,
              operation);

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

    // Calculate once
    long numPixels = srcStride[0] * srcHeight;

    // -----------------------------------------------
    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Loop through each pixel
        for(int index = 0; index < numPixels; index += 4){
            dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

            dstSlice[1][index / 4] = srcSlice[0][index];            // U            
            dstSlice[2][index / 4] = srcSlice[0][index + 2];        // V
        }

        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Loop through each pixel
        for(int index = 0; index < numPixels; index += 4){
            dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

            if(((index / srcStride[0]) % 2) == 0){
                int tempIndex = srcStride[0] / (index / srcStride[0]) + index / 4;
                dstSlice[1][tempIndex] = srcSlice[0][index];            // U            
                dstSlice[2][tempIndex] = srcSlice[0][index + 2];        // V
            }

        }

        return 0;
    }

    cerr << "Could not do conversion, not supported!" << endl;
    return -1;
}

void seq_scale(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation){

    /*
    // Verify input parameters
    if(!srcSlice || !srcStride || !dstSlice || !dstStride){
        cerr << "One of input parameters is null!" << endl;
        return;
    }
    
    // Prepares frame slices
    if(isPackedFormat(srcPixelFormat)){
        srcSlice[0] = srcSlice[1] = srcSlice[2] = srcSlice[3] = srcSlice[0];
        srcStride[0] = srcStride[1] = srcStride[2] = srcStride[3] = srcStride[0];
    }

    // Separate image into components
    uint8_t* aBuffer,* bBuffer,* cBuffer;
    switch(srcPixelFormat){
        case AV_PIX_FMT_UYVY422:
            // Allocate memory for each component
            aBuffer = (uint8_t*) malloc(sizeof(uint8_t) * srcWidth * srcHeight); // Y
            bBuffer = (uint8_t*) malloc(sizeof(uint8_t) * srcWidth * srcHeight); // U
            cBuffer = (uint8_t*) malloc(sizeof(uint8_t) * srcWidth * srcHeight); // V

            // Iterate through each pixel and save it corresponding to its component
            for(size_t index = 0; index < (sizeof(uint8_t) * srcWidth * srcHeight); index += (sizeof(uint8_t) * 4)){
                bBuffer[index] = bBuffer[index + 1] = srcSlice[0][index];       // U
                aBuffer[index] = srcSlice[0][index + 1];                        // Ya
                cBuffer[index] = cBuffer[index + 1] = srcSlice[0][index + 2];   // V
                aBuffer[index + 1] = srcSlice[0][index + 3];                    // Yb
            }
            break;
        case AV_PIX_FMT_YUV422P:
            break;
        case AV_PIX_FMT_YUV420P:
            break;
        default:
            cerr << "Could not separate components!" << endl;
            return;
    }

    // Scale components
    // TODO

    // Organize components
    switch(dstPixelFormat){
        case AV_PIX_FMT_UYVY422:
            break;
        case AV_PIX_FMT_YUV422P:
            break;
        case AV_PIX_FMT_YUV420P:
            break;
        default:
            cerr << "Could not separate components!" << endl;
            return;
    }
    */
}