#include "Common.h"

// Read image from a file
int readImageFromFile(string fileName, uint8_t** dataBuffer){
    // Open input file
    FILE* inputFile = fopen(fileName.c_str(), "rb");
    if(!inputFile){
        cerr << "Could not open file!" << endl;
        return -1;
    }

    // Get input file size
    fseek(inputFile, 0, SEEK_END);
    long inputFileSize = ftell(inputFile);
    rewind(inputFile);

    // Allocate memory to contain the whole file:
    *dataBuffer = (uint8_t*) malloc(sizeof(uint8_t) * inputFileSize);
    if(!*dataBuffer){
        // Close file
        fclose(inputFile);

        cerr << "Could not allocate the buffer memory!" << endl;
        return -1;
    }

    // Copy file into memory
    long numElements = fread(*dataBuffer, sizeof(uint8_t), inputFileSize, inputFile);
    if(numElements != inputFileSize){
        // Close file
        fclose(inputFile);

        cerr << "Could not read whole file!" << endl;
        return -1;
    }

    // Close file
    fclose(inputFile);

    // Return number of elements read
    return numElements;
}

// Write image to a file
int writeImageToFile(string fileName, AVFrame** frame){
    // Opens output file
    FILE* outputFile = fopen(fileName.c_str(), "wb");
    if(!outputFile){
        cerr << "Could not open file!" << endl;
        return -1;
    }

    // Calculate the number of elements of the image
    int numElements = avpicture_get_size((AVPixelFormat) (*frame)->format, (*frame)->width, (*frame)->height);
    // Write resulting frame to a file
    fwrite((*frame)->data[0], sizeof(uint8_t), numElements, outputFile);

    // Close file
    fclose(outputFile);

    // Return number of elements written
    return numElements;
}

// Create data buffer to hold image
int createImageDataBuffer(int width, int height, AVPixelFormat pixelFormat, uint8_t** dataBuffer){
    // Calculate the number of elements of the image
    int numElements = avpicture_get_size(pixelFormat, width, height);

    // Allocate buffer of the frame
    *dataBuffer = (uint8_t*) malloc(sizeof(uint8_t) * numElements);
    if(!*dataBuffer){
        cerr << "Could not allocate the buffer memory!" << endl;
        return -1;
    }

    // Return the size of the allocated buffer
    return numElements;
}

// Initialize and transfer data to AVFrame
int initializeAVFrame(uint8_t** dataBuffer, int width, int height, AVPixelFormat pixelFormat, AVFrame** frame){
    // Allocate the frame
    *frame = av_frame_alloc();
    if(!*frame){
        cerr << "Could not allocate frame!" << endl;
        return -1;
    }

    // Fields frame information
    (*frame)->width = width;
    (*frame)->height = height;
    (*frame)->format = pixelFormat;

    // Fill frame->data and frame->linesize pointers
    if(avpicture_fill((AVPicture*) *frame, *dataBuffer, pixelFormat, width, height) < 0){
        cerr << "Could not initialize frame!" << endl;
        av_frame_free(&(*frame));
        return -1;
    }

    return 0;
}

// Limit a value to a defined interval
int clamp(int val, int min, int max){
    if(val < min)
        return min;
    else if(val > max)
        return max;
    else
        return val;
}

// Interpolate a value between two points
float lerp(float valA, float valB, float dist){
    return valA * (1.0f - dist) + valB * dist;
}