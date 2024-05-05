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
    ImageInfo* img_obj_yuv444p_40x40 = new ImageInfo("imgs/obj-yuv444p-40x40.yuv", 40, 40, AV_PIX_FMT_YUV444P);
    ImageInfo* img_obj_uyvy422_40x40 = new ImageInfo("imgs/obj-uyvy422-40x40.yuv", 40, 40, AV_PIX_FMT_UYVY422);
    ImageInfo* img_obj_uyvy422_48x48 = new ImageInfo("imgs/obj-uyvy422-48x48.yuv", 48, 48, AV_PIX_FMT_UYVY422);
    ImageInfo* img_obj_yuv444p_48x48 = new ImageInfo("imgs/obj-yuv444p-48x48.yuv", 48, 48, AV_PIX_FMT_YUV444P);
    ImageInfo* img_obj_v210_48x48 = new ImageInfo("imgs/obj-v210-48x48.yuv", 48, 48, AV_PIX_FMT_V210);
    ImageInfo* img_col_yuv444p_8x8 = new ImageInfo("imgs/col-yuv444p-8x8.yuv", 8, 8, AV_PIX_FMT_YUV444P);

    // Real test images
    ImageInfo* img_yuv444p_1920x1080 = new ImageInfo("imgs/bbb-yuv444p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV444P);
    ImageInfo* img_yuv422p_1920x1080 = new ImageInfo("imgs/bbb-yuv422p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV422P);
    ImageInfo* img_yuv420p_1920x1080 = new ImageInfo("imgs/bbb-yuv420p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV420P);
    ImageInfo* img_uyvy422_1920x1080 = new ImageInfo("imgs/bbb-uyvy422-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_UYVY422);
    ImageInfo* img_v210_1920x1080 = new ImageInfo("imgs/bbb-v210-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_V210);

    // Initialize ffmpeg
    av_register_all();

    // Import all images into a vector
    vector<ImageInfo*> allImgs = vector<ImageInfo*>();
    allImgs.push_back(img_v210_1920x1080);
    allImgs.push_back(img_uyvy422_1920x1080);
    allImgs.push_back(img_yuv422p_1920x1080);

    allImgs.push_back(img_obj_yuv444p_48x48);
    
    // Load all images
    for(int index = 0; index < allImgs.size(); index++)
        (*allImgs.at(index)).loadImage();

    // Debug variables
    bool isTestFFMPEG = true;
    bool isTestSequential = true;
    bool isTestOpenMP = false;
    bool isTestCUDA = false;

    int nTimes = 1;

    // Create resample operations
    vector<int> resampleOperations = vector<int>();
    resampleOperations.push_back(SWS_POINT);

    // RESAMPLE UYVY422 ---------------------------------------------
    vector<ImageInfo*> uyvy422ResampleInImgs = vector<ImageInfo*>();
    uyvy422ResampleInImgs.push_back(img_uyvy422_1920x1080);
    vector<ImageInfo*> uyvy422ResampleOutImgs = vector<ImageInfo*>();
    //uyvy422ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422P));
    //uyvy422ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV420P));
    //uyvy422ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_NV12));
    //uyvy422ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_V210));

    //testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, uyvy422ResampleInImgs, uyvy422ResampleOutImgs, resampleOperations, nTimes);

    // RESAMPLE YUV422P ---------------------------------------------
    vector<ImageInfo*> yuv422pResampleInImgs = vector<ImageInfo*>();
    yuv422pResampleInImgs.push_back(img_yuv422p_1920x1080);
    vector<ImageInfo*> yuv422pResampleOutImgs = vector<ImageInfo*>();
    //yuv422pResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_UYVY422));
    //yuv422pResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV420P));    
    //yuv422pResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_NV12));
    //yuv422pResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_V210));

    //testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, yuv422pResampleInImgs, yuv422pResampleOutImgs, resampleOperations, nTimes);

    // RESAMPLE V210 ------------------------------------------------
    vector<ImageInfo*> v210ResampleInImgs = vector<ImageInfo*>();
    v210ResampleInImgs.push_back(img_v210_1920x1080);
    vector<ImageInfo*> v210ResampleOutImgs = vector<ImageInfo*>();
    //v210ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_UYVY422));
    //v210ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422P));
    //v210ResampleOutImgs.push_back(new ImageInfo("imgs/results/", 1920, 1080, AV_PIX_FMT_YUV422PNORM));

    //testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, v210ResampleInImgs, v210ResampleOutImgs, resampleOperations, nTimes);

    // TEST SCALING -------------------------------------------------

    // Create scaling operations
    vector<int> scaleOperations = vector<int>();
    //scaleOperations.push_back(SWS_POINT);
    scaleOperations.push_back(SWS_BILINEAR);
    //scaleOperations.push_back(SWS_BICUBIC);
    //scaleOperations.push_back(SWS_LANCZOS);

    // Test uyvy422 resample
    vector<ImageInfo*> scaleInImgs = vector<ImageInfo*>();
    //scaleInImgs.push_back(img_uyvy422_1920x1080);
    scaleInImgs.push_back(img_obj_yuv444p_48x48);
    vector<ImageInfo*> scaleOutImgs = vector<ImageInfo*>();
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 24, 24, AV_PIX_FMT_YUV444P));
    scaleOutImgs.push_back(new ImageInfo("imgs/results/", 32, 32, AV_PIX_FMT_YUV444P));
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 72, 72, AV_PIX_FMT_YUV444P));
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 144, 144, AV_PIX_FMT_YUV444P));
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 1280, 720, AV_PIX_FMT_UYVY422));
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 3840, 2160, AV_PIX_FMT_UYVY422));
    //scaleOutImgs.push_back(new ImageInfo("imgs/results/", 7680, 4320, AV_PIX_FMT_UYVY422));

    // Test procedures
    testAll(isTestFFMPEG, isTestSequential, isTestOpenMP, isTestCUDA, scaleInImgs, scaleOutImgs, scaleOperations, nTimes);

    // Success
    return 0;
}