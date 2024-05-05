#include "Common.h"

// Return if operation is supported
bool isSupportedOperation(int operation){
    // Verify if supported operation
    switch(operation){
        case SWS_POINT:
        case SWS_BILINEAR:
        case SWS_BICUBIC:
        case SWS_LANCZOS:
            return true;
    }

    // Not a supported operation
    return false;
}

// Return if format is supported
bool isSupportedFormat(AVPixelFormat format){
    // Verify if supported format
    switch(format){
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_GBRP:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_NV12:
            return true;
    }

    // Not a supported format
    return false;
}

// Return least common multiple of two integers
int lcm(int num1, int num2){
    // Find the greater value of the two
    int max = (num1 > num2) ? num1 : num2;

    do{
        if(max % num1 == 0 && max % num2 == 0)
            return max;

        max++;
    } while(max < num1 * num2);

    // Insuccess
    return num1 * num2;
}

// Return minimum number of two integers
int min(int num1, int num2){
    return (num1 > num2) ? num2 : num1;
}

// Return the value of the pixel support depending of the operation
int getPixelSupport(int operation){
    // Resize operation with different kernels
    switch(operation){
        case SWS_POINT:
            return 2;
        case SWS_BILINEAR:
            return 2;
        case SWS_BICUBIC:
            return 4;
        case SWS_LANCZOS:
            return 4;
    }

    // Insuccess
    return -1;
}

// Return the temporary scale pixel format
AVPixelFormat getTempScaleFormat(AVPixelFormat inFormat){
    // Retrieve the temporary scale format
    switch(inFormat){
        case AV_PIX_FMT_YUV444P:
            return AV_PIX_FMT_YUV444P;
        case AV_PIX_FMT_YUV422P:
            return AV_PIX_FMT_YUV422P;
        case AV_PIX_FMT_YUV420P:
            return AV_PIX_FMT_YUV420P;
        case AV_PIX_FMT_RGB24:
            return AV_PIX_FMT_YUV444P;
        case AV_PIX_FMT_UYVY422:
            return AV_PIX_FMT_YUV422P;
        case AV_PIX_FMT_NV12:
            return AV_PIX_FMT_YUV420P;
    }

    // If the source pixel format is not supported
    return AV_PIX_FMT_NONE;
}

// Read image from a file
int readImageFromFile(string fileName, uint8_t** dataBuffer){
    // Open input file
    FILE* inputFile = fopen(fileName.c_str(), "rb");
    if(!inputFile){
        cout << "Could not open file!" << endl;
        return -1;
    }

    // Get input file size
    fseek(inputFile, 0, SEEK_END);
    long inputFileSize = ftell(inputFile);
    rewind(inputFile);

    // Allocate memory to contain the whole file:
    *dataBuffer = (uint8_t*) malloc(sizeof(uint8_t) * inputFileSize);
    if(!*dataBuffer){
        fclose(inputFile);
        cout << "Could not allocate the buffer memory!" << endl;
        return -1;
    }

    // Copy file into memory
    long numElements = fread(*dataBuffer, sizeof(uint8_t), inputFileSize, inputFile);
    if(numElements != inputFileSize){
        fclose(inputFile);
        cout << "Could not read whole file!" << endl;
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
        cout << "Could not open file!" << endl;
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
        cout << "Could not allocate the buffer memory!" << endl;
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
        cout << "Could not allocate frame!" << endl;
        return -1;
    }

    // Fields frame information
    (*frame)->width = width;
    (*frame)->height = height;
    (*frame)->format = pixelFormat;

    // Fill frame->data and frame->linesize pointers
    if(avpicture_fill((AVPicture*) *frame, *dataBuffer, pixelFormat, width, height) < 0){
        av_frame_free(&(*frame));
        cout << "Could not initialize frame!" << endl;
        return -1;
    }

    // Success
    return 0;
}

// Get a valid pixel from the image
uint8_t getPixel(int lin, int col, int width, int height, uint8_t* data){
    // Clamp coords
    clamp<int>(lin, 0, height - 1);
    clamp<int>(col, 0, width - 1);

    // Assigns correct value to return
    return data[lin * width + col];
}
