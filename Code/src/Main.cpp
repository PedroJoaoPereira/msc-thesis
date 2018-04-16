#include <iostream>
#include <vector>

#include "ImageInfo.h"
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

    // INFO OF IMAGES USED IN TESTS ------------------
    ImageInfo* img_test_yuv444p_8x8 = new ImageInfo("imgs/test-yuv444p-8x8.yuv", 8, 8, AV_PIX_FMT_YUV444P);
    ImageInfo* img_obj_yuv444p_40x40 = new ImageInfo("imgs/obj-yuv444p-40x40.yuv", 40, 40, AV_PIX_FMT_YUV444P);
    ImageInfo* img_col_rgb24_8x8 = new ImageInfo("imgs/col-rgb24-8x8.yuv", 8, 8, AV_PIX_FMT_RGB24);
    ImageInfo* img_col_yuv444p_8x8 = new ImageInfo("imgs/col-yuv444p-8x8.yuv", 8, 8, AV_PIX_FMT_YUV444P);
    ImageInfo* img_col_uyvy422_8x8 = new ImageInfo("imgs/col-uyvy422-8x8.yuv", 8, 8, AV_PIX_FMT_UYVY422);

    // Real test images
    ImageInfo* img_rgb24_1920x1080 = new ImageInfo("imgs/bbb-rgb24-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_RGB24);
    ImageInfo* img_gbrp_1920x1080 = new ImageInfo("imgs/bbb-gbrp-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_GBRP);
    ImageInfo* img_yuv444p_1920x1080 = new ImageInfo("imgs/bbb-yuv444p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV444P);
    ImageInfo* img_yuv422p_1920x1080 = new ImageInfo("imgs/bbb-yuv422p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV422P);
    ImageInfo* img_yuv420p_1920x1080 = new ImageInfo("imgs/bbb-yuv420p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV420P);
    ImageInfo* img_uyvy422_1920x1080 = new ImageInfo("imgs/bbb-uyvy422-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_UYVY422);
    ImageInfo* img_v210_1920x1080 = new ImageInfo("imgs/bbb-v210-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_NONE);
    ImageInfo* img_nv12_1920x1080 = new ImageInfo("imgs/bbb-nv12-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_NV12);

    // Initialize ffmpeg
    av_register_all();

    // Import all images into a vector
    vector<ImageInfo*> inImgs = vector<ImageInfo*>();
    inImgs.push_back(img_yuv422p_1920x1080);
    
    // Load all images
    for(int index = 0; index < inImgs.size(); index++)
        (*inImgs.at(index)).loadImage();

    // Create output images
    vector<ImageInfo*> outImgs = vector<ImageInfo*>();
    //outImgs.push_back(new ImageInfo("imgs/results/", 20, 20, AV_PIX_FMT_YUV444P));
    //outImgs.push_back(new ImageInfo("imgs/results/", 80, 80, AV_PIX_FMT_YUV444P));
    outImgs.push_back(new ImageInfo("imgs/results/", 3840, 2160, AV_PIX_FMT_YUV422P));

    // create operations
    vector<int> operations = vector<int>();
    //operations.push_back(SWS_POINT);
    //operations.push_back(SWS_BILINEAR);
    //operations.push_back(SWS_BICUBIC);
    operations.push_back(SWS_LANCZOS);

    // Debug variables
    bool isTestFFMPEG = true;
    bool isTestSimulator = false;
    bool isTestSequential = true;
    bool isTestOpenMP = false;

    int nTimes = 1;

    // Test procedures
    testAll(isTestFFMPEG, isTestSimulator, isTestSequential, isTestOpenMP, inImgs, outImgs, operations, nTimes);

    // Success
    cout << endl;
    //system("pause");
    return 0;
}