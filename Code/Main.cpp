#include <iostream>
#include <chrono>

extern "C"{
#define __STDC_CONSTANT_MACROS
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace std::chrono;

int main(){

    // DEBUG variables
    string inputFileName = "in-uyvy422.yuv"; // UYVY422 176x144
    int srcWidth = 176;
    int srcHeight = 144;
    AVPixelFormat srcPixelFormat = AV_PIX_FMT_UYVY422;
    int dstWidth = 176;
    int dstHeight = 144;
    AVPixelFormat dstPixelFormat = AV_PIX_FMT_UYVY422;
    int flagOperator = SWS_BILINEAR;

    // Initialize ffmpeg
    av_register_all();

    // READ FROM IMAGE FILE --------------------------
    // Open input file
    FILE* inputFile = fopen(inputFileName.c_str(), "rb");
    if(!inputFile){
        cerr << "Could not load image data into memory!" << endl;
        system("pause");
        return -1;
    }

    // Get input file size
    fseek(inputFile, 0, SEEK_END);
    long inputFileSize = ftell(inputFile);
    rewind(inputFile);

    // Allocate memory to contain the whole file:
    uint8_t* srcBuffer = (uint8_t*) malloc(sizeof(uint8_t) * inputFileSize);
    if(!srcBuffer){
        cerr << "Could not allocate source buffer in memory for image!" << endl;
        system("pause");
        return -1;
    }

    // Copy file into memory
    long readElements = fread(srcBuffer, sizeof(uint8_t), inputFileSize, inputFile);
    if(readElements != inputFileSize){
        cerr << "Function fread error!" << endl;
        system("pause");
        return -1;
    }

    // Close file
    fclose(inputFile);
    // -----------------------------------------------
    // INITIALIZES AVFRAME STRUCTURE -----------------
    // Allocates frame
    AVFrame* srcFrame = av_frame_alloc();
    if(!srcFrame){
        cerr << "Could not allocate source frame!" << endl;
        system("pause");
        return -1;
    }

    // Fill necessary frame fields
    srcFrame->width = srcWidth;
    srcFrame->height = srcHeight;
    srcFrame->format = srcPixelFormat;

    // Frame data fill
    // If contiguous data
    avpicture_fill((AVPicture*) srcFrame, srcBuffer, (AVPixelFormat) srcFrame->format, srcFrame->width, srcFrame->height);
    // If not
    //avpicture_fill((AVPicture*) frame, NULL, (AVPixelFormat) frame->format, frame->width, frame->height);
    //frame->data[0] = inputBufferY;
    //frame->data[1] = inputBufferU;
    //frame->data[2] = inputBufferV;
    // -----------------------------------------------
    // APPLY SWS_SCALE TO THE IMAGE ------------------
    // Allocate the result frame
    AVFrame* dstFrame = av_frame_alloc();
    if(!dstFrame){
        cerr << "Could not allocate destination frame!" << endl;
        system("pause");
        return -1;
    }

    // Allocate buffer of the result frame
    int numBytes = avpicture_get_size(dstPixelFormat, dstWidth, dstHeight);
    uint8_t* dstBuffer = (uint8_t*) malloc(sizeof(uint8_t) * numBytes);
    if(!dstBuffer){
        cerr << "Could not allocate destination buffer in memory for image!" << endl;
        system("pause");
        return -1;
    }

    // Fill frame->data and frame->linesize pointers
    avpicture_fill((AVPicture*) dstFrame, dstBuffer, dstPixelFormat, dstWidth, dstHeight);

    // Create operation context
    SwsContext* swsContext = sws_getContext(srcWidth, srcHeight, srcPixelFormat, dstWidth, dstHeight, dstPixelFormat, flagOperator, NULL, NULL, NULL);
    if(!swsContext){
        cerr << "Could not create SwsContext!" << endl;
        system("pause");
        return -1;
    }

    // Apply the operation and measure the its execution time
    high_resolution_clock::time_point initTime = high_resolution_clock::now();
    sws_scale(swsContext, srcFrame->data, srcFrame->linesize, 0, srcHeight, dstFrame->data, dstFrame->linesize);
    high_resolution_clock::time_point stopTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Displays the execution time
    cout << "[SWS_SCALE] took " << duration << " microseconds!" << endl;
    // -----------------------------------------------
    // APPLY SWS_SCALE TO THE IMAGE ------------------
    // Opens output file
    FILE* outputFile = fopen("o.yuv", "wb");
    if(!inputFile){
        cerr << "Could not write image data to a file!" << endl;
        system("pause");
        return -1;
    }

    // Write resulting frame to a file
    fwrite(dstFrame->data[0], sizeof(uint8_t), numBytes, outputFile);

    // Close file
    fclose(outputFile);
    // -----------------------------------------------

    // Free used resources
    av_frame_free(&srcFrame);
    av_frame_free(&dstFrame);
    free(srcBuffer);
    free(dstBuffer);
    sws_freeContext(swsContext);

    cerr << "Successful!" << endl;
    system("pause");
    return 0;
}