#include <iostream>
#include <vector>

#include "ImageClass.h"
#include "Tests.h"

extern "C"{
    #define __STDC_CONSTANT_MACROS
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
}

using namespace std;

int main(){

    // IMAGE RESOLUTIONS -----------------------------
    // FUHD   - 7680 x 4320
    // UHD    - 3840 x 2160
    // HD1080 - 1920 x 1080
    // HD720  - 1280 x 720

    // Real test images
    ImageClass* img_uyvy422_1920x1080 = new ImageClass("imgs/bbb-uyvy422-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_UYVY422);
    ImageClass* img_yuv422p_1920x1080 = new ImageClass("imgs/bbb-yuv422p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV422P);
    ImageClass* img_yuv420p_1920x1080 = new ImageClass("imgs/bbb-yuv420p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV420P);
    ImageClass* img_nv12_1920x1080 = new ImageClass("imgs/bbb-nv12-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_NV12);
    ImageClass* img_v210_1920x1080 = new ImageClass("imgs/bbb-v210-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_V210);
    ImageClass* img_yuv422pnorm_1920x1080 = new ImageClass("imgs/bbb-yuv422pnorm-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV422PNORM);

    // Initialize execution
    cout << "[MAIN] Program just started!" << endl;

    // Initialize ffmpeg
    av_register_all();

    // Initialize GPU activity
    high_resolution_clock::time_point initTime, stopTime;
    initTime = high_resolution_clock::now();
    cudaFree(0);
    stopTime = high_resolution_clock::now();

    // Display how much time took GPU initializaion
    cout << "[CUDA] GPU initialization took " << duration_cast<milliseconds>(stopTime - initTime).count() << " ms!" << endl << endl;

    // Import all images into a vector
    vector<ImageClass*> allImgs = vector<ImageClass*>();
    //allImgs.push_back(img_uyvy422_1920x1080);
    //allImgs.push_back(img_yuv422p_1920x1080);
    //allImgs.push_back(img_yuv420p_1920x1080);
    //allImgs.push_back(img_nv12_1920x1080);
    //allImgs.push_back(img_v210_1920x1080);
    //allImgs.push_back(img_yuv422pnorm_1920x1080);

    // Load all images
    for(int index = 0; index < allImgs.size(); index++)
        (*allImgs.at(index)).loadImage();

    // Debug variables
    bool isTestFFMPEG = true;
    bool isTestSequential = false;
    bool isTestOpenMP = false;
    bool isTestCUDA = true;

    int nTimes = 1;

    // TEST FORMAT CONVERSION ---------------------------------------

    // Create format conversions output images
    vector<ImageClass*> formatConversionOutImgs = vector<ImageClass*>();
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_UYVY422));
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422P));
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV420P));
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_NV12));
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_V210));
    //formatConversionOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422PNORM));

    // Create format conersions operations
    vector<int> formatConversionOperations = vector<int>();
    formatConversionOperations.push_back(SWS_POINT);

    //testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, allImgs, formatConversionOutImgs, formatConversionOperations, nTimes);

    // TEST SCALING -------------------------------------------------

    // Create scale input images
    vector<ImageClass*> scaleInImgs = vector<ImageClass*>();
    //scaleInImgs.push_back(img_uyvy422_1920x1080);
    scaleInImgs.push_back(img_yuv422p_1920x1080);

    // Load all images
    for(int index = 0; index < scaleInImgs.size(); index++)
        (*scaleInImgs.at(index)).loadImage();

    // Create scale output images
    vector<ImageClass*> scaleOutImgs = vector<ImageClass*>();
    //scaleOutImgs.push_back(new ImageClass("imgs/results/", 1280, 720, AV_PIX_FMT_UYVY422));
    //scaleOutImgs.push_back(new ImageClass("imgs/results/", 3840, 2160, AV_PIX_FMT_UYVY422));
    //scaleOutImgs.push_back(new ImageClass("imgs/results/", 7680, 4320, AV_PIX_FMT_UYVY422));

    //scaleOutImgs.push_back(new ImageClass("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422P));
    scaleOutImgs.push_back(new ImageClass("imgs/results/", 3840, 2160, AV_PIX_FMT_YUV422P));

    // Create scaling operations
    vector<int> scaleOperations = vector<int>();
    //scaleOperations.push_back(SWS_POINT);
    //scaleOperations.push_back(SWS_BILINEAR);
    //scaleOperations.push_back(SWS_BICUBIC);
    scaleOperations.push_back(SWS_LANCZOS);

    // Test procedures
    testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, scaleInImgs, scaleOutImgs, scaleOperations, nTimes);

    // Success
    return 0;
}