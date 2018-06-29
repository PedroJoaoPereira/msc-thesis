#include "Tests.h"

// Facilitate writing operations
string pixelFormatToString(int format){
    // Return string name of format
    switch(format){
        case AV_PIX_FMT_RGB24:
            return "RGB24";
        case AV_PIX_FMT_GBRP:
            return "GBRP";
        case AV_PIX_FMT_YUV444P:
            return "YUV444P";
        case AV_PIX_FMT_YUV422P:
            return "YUV422P";
        case AV_PIX_FMT_YUV420P:
            return "YUV420P";
        case AV_PIX_FMT_UYVY422:
            return "UYVY422";
        case AV_PIX_FMT_NV12:
            return "NV12";
        case AV_PIX_FMT_V210:
            return "V210";
        case AV_PIX_FMT_YUV422PNORM:
            return "YUV422PNORM";
    }

    // Insuccess
    return "";
}

string operationToString(int operation){
    // Return string name of format
    switch(operation){
        case SWS_POINT:
            return "NN";
        case SWS_BILINEAR:
            return "Lin";
        case SWS_BICUBIC:
            return "Cub";
        case SWS_LANCZOS:
            return "Lcz";
    }

    // Insuccess
    return "";
}

// Test ffmpeg procedure
int testFFMPEGSingle(ImageClass &inImg, ImageClass &outImg, int operation){
    // Prepare output frame
    inImg.loadImage();
    outImg.initFrame();

    // Resample and scale
    int executionTime = ffmpeg_resample(inImg.frame, outImg.frame, operation);
    if(executionTime < 0){
        cout << "[FFMPEG] Scale has failed with image: " << inImg.fileName << endl;
        cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
        cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
        cout << "\t\tOperation: " << operationToString(operation) << endl << endl;
        return -1;
    }

    // Success
    return executionTime;
}

int testFFMPEGAverage(ImageClass &inImg, ImageClass outImg, int operation, int nTimes){
    // Temporary variable
    long long acc = 0;

    // Repeat nTimes
    for(int ithTime = 0; ithTime < nTimes; ithTime++){
        int tempExecutionTime = testFFMPEGSingle(inImg, outImg, operation);
        if(tempExecutionTime < 0)
            return -1;

        // Increment execution time accumulator
        acc += tempExecutionTime;
    }

    // Average execution time
    int avgExecutionTime = acc / nTimes;

    // Display results
    //cout << "[FFMPEG] Processed image x" << nTimes << " time(s): " << inImg.fileName << endl;
    //cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
    //cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
    //cout << "\t\tOperation: " << operationToString(operation) << endl;
    //cout << "\tExecution Time ==> " << avgExecutionTime / 1000. << " ms" << endl << endl;

    cout << operationToString(operation) << ",";
    cout << pixelFormatToString(inImg.pixelFormat) << "," << pixelFormatToString(outImg.pixelFormat) << ",";
    cout << outImg.width << "x" << outImg.height << ",";
    cout << avgExecutionTime / 1000. << ",";

    // Write image to file
    outImg.fileName += "[FFMPEG]" + operationToString(operation) + "-" + pixelFormatToString(inImg.pixelFormat) + "-" + pixelFormatToString(outImg.pixelFormat);
    outImg.fileName += "-" + to_string(inImg.width) + "x" + to_string(inImg.height) + "-" + to_string(outImg.width) + "x" + to_string(outImg.height) + ".yuv";
    outImg.writeImage();

    outImg.freeResources();

    // Success
    return avgExecutionTime;
}

void testFFMPEG(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes){
    // For each operation
    for(int indexOp = 0; indexOp < operations.size(); indexOp++){
        // For each output image
        for(int indexOut = 0; indexOut < outImgs.size(); indexOut++){
            // For each input image
            for(int indexIn = 0; indexIn < inImgs.size(); indexIn++){
                if((*inImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_V210 || (*outImgs.at(indexOut)).pixelFormat == AV_PIX_FMT_V210)
                    continue;

                if(testFFMPEGAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}

// Test cuda procedure
int testCUDASingle(ImageClass &inImg, ImageClass &outImg, int operation, double* &times){
    // Prepare output frame
    inImg.loadImage();
    outImg.initFrame();

    // Resample and scale
    times = (double*) malloc(3 * sizeof(double));
    int executionTime = cuda_resample(inImg.frame, outImg.frame, operation, times);
    if(executionTime < 0){
        cout << "[CUDA] Scale has failed with image: " << inImg.fileName << endl;
        cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
        cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
        cout << "\t\tOperation: " << operationToString(operation) << endl << endl;
        return -1;
    }

    // Success
    return 1;
}

int testCUDAAverage(ImageClass &inImg, ImageClass outImg, int operation, int nTimes){
    // Temporary variable
    high_resolution_clock::time_point initTime, stopTime;
    long long initAcc = 0, finishAcc = 0;
    long long firstConvertion = 0, transferAndResample = 0, secondConversion = 0;

    inImg.loadImage();
    outImg.initFrame();

    // Initializes memory in device
    cuda_init(inImg.frame, outImg.frame, operation);

    // Repeat nTimes
    double* times;
    for(int ithTime = 0; ithTime < nTimes; ithTime++){
        int tempExecutionTime = testCUDASingle(inImg, outImg, operation, times);
        if(tempExecutionTime < 0)
            return -1;

        // Increment execution time accumulator
        firstConvertion += times[0];
        transferAndResample += times[1];
        secondConversion += times[2];
    }

    // Write image to file
    outImg.fileName += "[CUDA]" + operationToString(operation) + "-" + pixelFormatToString(inImg.pixelFormat) + "-" + pixelFormatToString(outImg.pixelFormat);
    outImg.fileName += "-" + to_string(inImg.width) + "x" + to_string(inImg.height) + "-" + to_string(outImg.width) + "x" + to_string(outImg.height) + ".yuv";
    outImg.writeImage();

    // Free used resources by device
    cuda_finish();

    if(!(inImg.height == outImg.height && inImg.width == outImg.width))
        for(int ithTime = 0; ithTime < nTimes; ithTime++){
            initTime = high_resolution_clock::now();
            cuda_init(inImg.frame, outImg.frame, operation);
            stopTime = high_resolution_clock::now();
            initAcc += duration_cast<microseconds>(stopTime - initTime).count();

            initTime = high_resolution_clock::now();
            cuda_finish();
            stopTime = high_resolution_clock::now();
            finishAcc += duration_cast<microseconds>(stopTime - initTime).count();
        }

    outImg.freeResources();

    // Display results
    //cout << "[CUDA] Processed image x" << nTimes << " time(s): " << inImg.fileName << endl;
    //cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
    //cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
    //cout << "\t\tOperation: " << operationToString(operation) << endl;
    //cout << "\tExecution Time ==> " << (1. * firstConvertion / nTimes + transferAndResample / nTimes + secondConversion / nTimes) / 1000. << " ms" << endl;
    //cout << "\t\t1st Conversion Time ==> " << (1. * firstConvertion / nTimes) / 1000. << " ms" << endl;
    //cout << "\t\tData and Resample Time ==> " << (transferAndResample / nTimes) / 1000. << " ms" << endl;
    //cout << "\t\t2nd Conversion Time ==> " << (secondConversion / nTimes) / 1000. << " ms" << endl;
    //cout << "\tInit time: " << (initAcc / nTimes) / 1000. << " ms" << endl;
    //cout << "\tFinish time: " << (finishAcc / nTimes) / 1000. << " ms" << endl << endl;

    cout << (firstConvertion / nTimes + transferAndResample / nTimes + secondConversion / nTimes) / 1000. << ",";
    cout << (initAcc / nTimes) / 1000. << ",";
    cout << (finishAcc / nTimes) / 1000. << ",";
    cout << (firstConvertion / nTimes) / 1000. << ",";
    cout << (secondConversion / nTimes) / 1000. << ",";
    cout << (transferAndResample / nTimes) / 1000. << "," << " ," << " ," << " ," << endl;

    // Success
    return 1;
}

void testCUDA(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes){
    // For each operation
    for(int indexOp = 0; indexOp < operations.size(); indexOp++){
        // For each output image
        for(int indexOut = 0; indexOut < outImgs.size(); indexOut++){
            // For each input image
            for(int indexIn = 0; indexIn < inImgs.size(); indexIn++){
                if(testCUDAAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}

// Test all procedures
void testAll(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes){
    // For each input image
    for(int indexIn = 0; indexIn < inImgs.size(); indexIn++){
        // For each operation
        for(int indexOp = 0; indexOp < operations.size(); indexOp++){
            // For each output image
            for(int indexOut = 0; indexOut < outImgs.size(); indexOut++){
                if(!((*inImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_V210 || (*outImgs.at(indexOut)).pixelFormat == AV_PIX_FMT_V210))
                    if(testFFMPEGAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                        return;

                if(testCUDAAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}