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
        case AV_PIX_FMT_NONE:
            return "V210";
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
int testFFMPEGSingle(ImageInfo &inImg, ImageInfo &outImg, int operation){
    // Prepare output frame
    outImg.initFrame();

    // Resample and scale
    int executionTime = ffmpeg_scale(inImg.frame, outImg.frame, operation);
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

int testFFMPEGAverage(ImageInfo &inImg, ImageInfo outImg, int operation, int nTimes){
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
    cout << "[FFMPEG] Processed image x" << nTimes << " time(s): " << inImg.fileName << endl;
    cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
    cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
    cout << "\t\tOperation: " << operationToString(operation) << endl;
    cout << "\tExecution Time ==> " << avgExecutionTime / 1000. << " ms" << endl << endl;

    // Write image to file
    outImg.fileName += "[FFMPEG]" + operationToString(operation) + "-" + pixelFormatToString(inImg.pixelFormat) + "-" + pixelFormatToString(outImg.pixelFormat);
    outImg.fileName += "-" + to_string(inImg.width) + "x" + to_string(inImg.height) + "-" + to_string(outImg.width) + "x" + to_string(outImg.height) + ".yuv";
    outImg.writeImage();

    // Success
    return avgExecutionTime;
}

void testFFMPEG(vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes){
    // For each operation
    for(int indexOp = 0; indexOp < operations.size(); indexOp++){
        // For each output image
        for(int indexOut = 0; indexOut < outImgs.size(); indexOut++){
            // For each input image
            for(int indexIn = 0; indexIn < inImgs.size(); indexIn++){
                if((*inImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_NONE || (*outImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_NONE)
                    continue;

                if(testFFMPEGAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}

// Test simulator procedure
int testSimulatorSingle(ImageInfo &inImg, ImageInfo &outImg, int operation){
    // Prepare output frame
    outImg.initFrame();

    // Resample and scale
    int executionTime = simulator_scale(inImg.frame, outImg.frame, operation);
    if(executionTime < 0){
        cout << "[SIMULATOR] Scale has failed with image: " << inImg.fileName << endl;
        cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
        cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
        cout << "\t\tOperation: " << operationToString(operation) << endl << endl;
        return -1;
    }

    // Success
    return executionTime;
}

int testSimulatorAverage(ImageInfo &inImg, ImageInfo outImg, int operation, int nTimes){
    // Temporary variable
    long long acc = 0;

    // Repeat nTimes
    for(int ithTime = 0; ithTime < nTimes; ithTime++){
        int tempExecutionTime = testSimulatorSingle(inImg, outImg, operation);
        if(tempExecutionTime < 0)
            return -1;

        // Increment execution time accumulator
        acc += tempExecutionTime;
    }

    // Average execution time
    int avgExecutionTime = acc / nTimes;

    // Display results
    cout << "[SIMULATOR] Processed image x" << nTimes << " time(s): " << inImg.fileName << endl;
    cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
    cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
    cout << "\t\tOperation: " << operationToString(operation) << endl;
    cout << "\tExecution Time ==> " << avgExecutionTime / 1000. << " ms" << endl << endl;

    // Write image to file
    outImg.fileName += "[SIMULATOR]" + operationToString(operation) + "-" + pixelFormatToString(inImg.pixelFormat) + "-" + pixelFormatToString(outImg.pixelFormat);
    outImg.fileName += "-" + to_string(inImg.width) + "x" + to_string(inImg.height) + "-" + to_string(outImg.width) + "x" + to_string(outImg.height) + ".yuv";
    outImg.writeImage();

    // Success
    return avgExecutionTime;
}

void testSimulator(vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes){
    // For each operation
    for(int indexOp = 0; indexOp < operations.size(); indexOp++){
        // For each output image
        for(int indexOut = 0; indexOut < outImgs.size(); indexOut++){
            // For each input image
            for(int indexIn = 0; indexIn < inImgs.size(); indexIn++){
                if(!(((*inImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_YUV444P || (*inImgs.at(indexIn)).pixelFormat == AV_PIX_FMT_GBRP) &&
                    ((*outImgs.at(indexOut)).pixelFormat == AV_PIX_FMT_YUV444P || (*outImgs.at(indexOut)).pixelFormat == AV_PIX_FMT_GBRP)))
                    continue;

                if(testSimulatorAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}

// Test sequential procedure
int testSequentialSingle(ImageInfo &inImg, ImageInfo &outImg, int operation){
    // Prepare output frame
    outImg.initFrame();

    // Resample and scale
    int executionTime = sequential_scale(inImg.frame, outImg.frame, operation);
    if (executionTime < 0) {
        cout << "[SEQUENTIAL] Scale has failed with image: " << inImg.fileName << endl;
        cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
        cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
        cout << "\t\tOperation: " << operationToString(operation) << endl << endl;
        return -1;
    }

    // Success
    return executionTime;
}

int testSequentialAverage(ImageInfo &inImg, ImageInfo outImg, int operation, int nTimes){
    // Temporary variable
    long long acc = 0;

    // Repeat nTimes
    for (int ithTime = 0; ithTime < nTimes; ithTime++) {
        int tempExecutionTime = testSequentialSingle(inImg, outImg, operation);
        if (tempExecutionTime < 0)
            return -1;

        // Increment execution time accumulator
        acc += tempExecutionTime;
    }

    // Average execution time
    int avgExecutionTime = acc / nTimes;

    // Display results
    cout << "[SEQUENTIAL] Processed image x" << nTimes << " time(s): " << inImg.fileName << endl;
    cout << "\t\tDimensions: " << inImg.width << "x" << inImg.height << "\tTo: " << outImg.width << "x" << outImg.height << endl;
    cout << "\t\tFormats: " << pixelFormatToString(inImg.pixelFormat) << "\tTo: " << pixelFormatToString(outImg.pixelFormat) << endl;
    cout << "\t\tOperation: " << operationToString(operation) << endl;
    cout << "\tExecution Time ==> " << avgExecutionTime / 1000. << " ms" << endl << endl;

    // Write image to file
    outImg.fileName += "[SEQUENTIAL]" + operationToString(operation) + "-" + pixelFormatToString(inImg.pixelFormat) + "-" + pixelFormatToString(outImg.pixelFormat);
    outImg.fileName += "-" + to_string(inImg.width) + "x" + to_string(inImg.height) + "-" + to_string(outImg.width) + "x" + to_string(outImg.height) + ".yuv";
    outImg.writeImage();

    // Success
    return avgExecutionTime;
}

void testSequential(vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes){
    // For each operation
    for (int indexOp = 0; indexOp < operations.size(); indexOp++) {
        // For each output image
        for (int indexOut = 0; indexOut < outImgs.size(); indexOut++) {
            // For each input image
            for (int indexIn = 0; indexIn < inImgs.size(); indexIn++) {
                if (testSequentialAverage((*inImgs.at(indexIn)), (*outImgs.at(indexOut)), operations.at(indexOp), nTimes) < 0)
                    return;
            }
        }
    }
}

// Test all procedures
void testAll(bool isTestFFMPEG, bool isTestSimulator, bool isTestSequential, bool isTestOpenMP, vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes){
    if(isTestFFMPEG)
        testFFMPEG(inImgs, outImgs, operations, nTimes);
    if (isTestSimulator)
        testSimulator(inImgs, outImgs, operations, nTimes);
    if (isTestSequential)
        testSequential(inImgs, outImgs, operations, nTimes);
}