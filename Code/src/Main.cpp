#include <iostream>

#include "ImageInfo.h"
#include "FFMPEG_Scale.h"
#include "Sequential_Scale.h"

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
    // -----------------------------------------------
    // INFO OF IMAGES USED IN TESTS ------------------
    ImageInfo img01("imgs/uyvy422-7680x4320.yuv", 7680, 4320, AV_PIX_FMT_UYVY422);
    ImageInfo img02("imgs/yuv420p-7680x4320.yuv", 7680, 4320, AV_PIX_FMT_YUV420P);
    ImageInfo img03("imgs/yuv422p-7680x4320.yuv", 7680, 4320, AV_PIX_FMT_YUV422P);
    // -----------------------------------------------
    // DEBUG VARIABLES -------------------------------
    int nTimes = 15;
    int operation = SWS_BICUBIC;
    ImageInfo outImg("imgs/output.yuv", 7680, 4320, AV_PIX_FMT_YUV420P);
    //ImageInfo outImg("imgs/output.yuv", 1920, 1080, AV_PIX_FMT_YUV422P);
    // -----------------------------------------------
    // SCALING OPERATIONS ----------------------------
    // Initialize ffmpeg
    av_register_all();

    // Apply the operations
    while(nTimes > 0){
        //int executionTime = ffmpeg_scale(img01, outImg, operation);
        int executionTime = sequential_scale(img01, outImg, operation);
        if(executionTime < 0){
            cerr << "Could not execute the scaling method!" << endl;
            system("pause");
            return -1;
        }

        cout << "Execution time was " << executionTime << " ms!" << endl;

        nTimes--;
    }

    system("pause");
    return 0;
}