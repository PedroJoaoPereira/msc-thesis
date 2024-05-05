#include "ImageClass.h"

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
    //*dataBuffer = (uint8_t*) malloc(sizeof(uint8_t) * inputFileSize);
    cudaMallocHost((void **) dataBuffer, inputFileSize);
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
    // Change pixel format if it is fundamentally the same
    int pixelFormat = (*frame)->format;
    if(pixelFormat == AV_PIX_FMT_YUV422PNORM)
        pixelFormat = AV_PIX_FMT_YUV422P;

    // Opens output file
    FILE* outputFile = fopen(fileName.c_str(), "wb");
    if(!outputFile){
        cout << "Could not open file!" << endl;
        return -1;
    }

    if(pixelFormat == AV_PIX_FMT_V210){
        // Calculate the number of elements of the image
        int numElements = (*frame)->width * (*frame)->height / 6 * 4;
        // Write resulting frame to a file
        fwrite((*frame)->data[0], sizeof(uint32_t), numElements, outputFile);
    } else{
        // Calculate the number of elements of the image
        int numElements = avpicture_get_size((AVPixelFormat) pixelFormat, (*frame)->width, (*frame)->height);
        // Write resulting frame to a file
        fwrite((*frame)->data[0], sizeof(uint8_t), numElements, outputFile);
    }

    // Close file
    fclose(outputFile);

    // Success
    return 1;
}

// Create data buffer to hold image
int createImageDataBuffer(int width, int height, int pixelFormat, uint8_t** dataBuffer){
    // Change pixel format if it is fundamentally the same
    if(pixelFormat == AV_PIX_FMT_YUV422PNORM)
        pixelFormat = AV_PIX_FMT_YUV422P;

    // Calculate the number of elements of the image
    int numElements;
    if(pixelFormat != AV_PIX_FMT_V210)
        numElements = avpicture_get_size((AVPixelFormat) pixelFormat, width, height);

    // Allocate buffer of the frame
    if(pixelFormat == AV_PIX_FMT_V210)
        cudaMallocHost((void **) dataBuffer, ((width + 47) / 48) * 128 * height * sizeof(uint8_t));
    else
        cudaMallocHost((void **) dataBuffer, numElements * sizeof(uint8_t));
    if(!*dataBuffer){
        cout << "Could not allocate the buffer memory!" << endl;
        return -1;
    }

    // Success
    return 1;
}

// Initialize and transfer data to AVFrame
int initializeAVFrame(uint8_t** dataBuffer, int width, int height, int pixelFormat, AVFrame** frame){
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

    // Change pixel format if it is fundamentally the same
    if(pixelFormat == AV_PIX_FMT_YUV422PNORM)
        pixelFormat = AV_PIX_FMT_YUV422P;

    // Fill frame->data and frame->linesize pointers
    if(pixelFormat == AV_PIX_FMT_V210){
        (*frame)->data[0] = (*dataBuffer);
    } else{
        if(avpicture_fill((AVPicture*) *frame, *dataBuffer, (AVPixelFormat) pixelFormat, width, height) < 0){
            av_frame_free(&(*frame));
            cout << "Could not initialize frame!" << endl;
            return -1;
        }
    }

    // Success
    return 0;
}

// Constructor
ImageClass::ImageClass(string fileName, int width, int height, int pixelFormat){
    isInitialized = false;

    this->fileName = fileName;
    this->width = width;
    this->height = height;
    this->pixelFormat = pixelFormat;
}

// Load image into avframe
void ImageClass::loadImage(){
    if(isInitialized){
        av_frame_free(&frame);
        cudaFreeHost(frameBuffer);
    } else
        isInitialized = true;

    // Read image from a file
    if(readImageFromFile(fileName, &frameBuffer) < 0)
        return;

    // Initialize frame
    if(initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame) < 0){
        cudaFreeHost(frameBuffer);
    }
}

// Create frame
void ImageClass::initFrame(){
    if(isInitialized){
        av_frame_free(&frame);
        cudaFreeHost(frameBuffer);
    }else
        isInitialized = true;

    // Prepare to initialize frame
    if(createImageDataBuffer(width, height, pixelFormat, &frameBuffer) < 0)
        return;

    // Initialize frame
    if(initializeAVFrame(&frameBuffer, width, height, pixelFormat, &frame) < 0){
        cudaFreeHost(frameBuffer);
    }
}

// Write image into a file
void ImageClass::writeImage(){
    // Write image to file
    writeImageToFile(fileName, &frame);
}

// Free image resources
void ImageClass::freeResources(){
    av_frame_free(&frame);
    cudaFreeHost(frameBuffer);
}