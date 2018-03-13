#include "FFMPEG_Scale.h"

int ffmpeg_scale(ImageInfo src, ImageInfo dst, int operation){
    // Variables used
    int duration = -1;
    SwsContext* swsContext;
    high_resolution_clock::time_point initTime, stopTime;

    // Create operation context
    swsContext = sws_getContext(src.width, src.height, src.pixelFormat,
                                dst.width, dst.height, dst.pixelFormat,
                                operation, NULL, NULL, NULL);

    // Verify if scaling context was created
    if(!swsContext){
        cerr << "[FFMPEG] Could not create SwsContext!" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(sws_scale(swsContext, src.frame->data, src.frame->linesize, 0, src.height, dst.frame->data, dst.frame->linesize) < 0){
        // Free used resources
        sws_freeContext(swsContext);

        cerr << "[FFMPEG] Could not apply sws_scale()!" << endl;
        return -1;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<milliseconds>(stopTime - initTime).count();

    // Free used resources
    sws_freeContext(swsContext);

    // Return execution time of the scaling operation
    return duration;
}