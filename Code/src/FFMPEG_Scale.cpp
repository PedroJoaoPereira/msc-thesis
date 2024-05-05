#include "FFMPEG_Scale.h"

#include "Common.h"

int ffmpeg_scale(ImageInfo src, ImageInfo dst, int operation){
    // Variables used
    int retVal = -1, duration = -1;
    uint8_t* srcBuffer,* dstBuffer;
    AVFrame* srcFrame,* dstFrame;
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
    sws_scale(swsContext, srcFrame->data, srcFrame->linesize, 0, src.height, dstFrame->data, dstFrame->linesize);

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