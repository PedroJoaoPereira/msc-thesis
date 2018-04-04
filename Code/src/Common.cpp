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
        av_frame_free(&(*frame));

        cerr << "Could not initialize frame!" << endl;
        return -1;
    }

    return 0;
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
        default:
            break;
    }

    // Not a supported format
    return false;
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
        default:
            break;
    }

    // If the source pixel format is not supported
    return AV_PIX_FMT_NONE;
}











// Get a valid pixel from the image
void getPixel(uint8_t* data, int width, int height, int lin, int col, uint8_t* pixelVal){
    // Clamp coords
    clamp<int>(lin, 0, height - 1);
    clamp<int>(col, 0, width - 1);

    // Assigns correct value to return
    *pixelVal = data[lin * width + col];
}

// Get the bicubic coefficients
float getBicubicCoef(float x){
    /*float a = -0.6f;
    float xRounded = abs(x);
    if(xRounded <= 1.0f){
    return (a + 2.0f) * xRounded * xRounded * xRounded - (a + 3.0f) * xRounded * xRounded + 1.0f;
    } else if(xRounded < 2.0f){
    return a * xRounded * xRounded * xRounded - 5.0f * a * xRounded * xRounded + 8.0f * a * xRounded - 4.0f * a;
    } else{
    return 0.0f;
    }*/

    float b = 0.f;
    float c = 0.6f;
    float xr = abs(x);
    float xr2 = xr * xr;
    float xr3 = xr2 * xr;

    if(xr <= 1.0f){
        return (1.f / 6.f) * ((12.f - 9.f * b - 6.f * c) * xr3 + (-18.f + 12.f * b + 6.f * c) * xr2 + (6.f - 2.f * b));
    } else if(xr < 2.0f){
        return (1.f / 6.f) * ((-b * -6.f*c) * xr3 + (6.f*b + 30.f*c) * xr2 + (-12.f*b - 48.f*c) * xr2 + (8.f*b + 24.f*c));
    } else{
        return 0.0f;
    }
}

float getBilinearCoef(float x){

}