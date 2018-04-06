#include "FFMPEG_Scale.h"

int ffmpeg_scale(AVFrame* srcFrame, AVFrame* dstFrame, int operation){
    // Variables used
    int duration = -1;
    SwsContext* swsContext;
    high_resolution_clock::time_point initTime, stopTime;

    // Create operation context
    swsContext = sws_getContext(srcFrame->width, srcFrame->height, (AVPixelFormat) srcFrame->format,
                                dstFrame->width, dstFrame->height, (AVPixelFormat) dstFrame->format,
                                operation, NULL, NULL, NULL);

    // Verify if scaling context was created
    if(!swsContext){
        cerr << "[FFMPEG] Could not create SwsContext!" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(sws_scale(swsContext, srcFrame->data, srcFrame->linesize, 0, srcFrame->height, dstFrame->data, dstFrame->linesize) < 0){
        // Free used resources
        sws_freeContext(swsContext);

        cerr << "[FFMPEG] Could not apply sws_scale()!" << endl;
        return -1;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Free used resources
    sws_freeContext(swsContext);

    // Return execution time of the scaling operation
    return duration;
}