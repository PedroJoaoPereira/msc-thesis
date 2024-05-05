#include <iostream>
#include <vector>

#include "ImageInfo.h"
#include "FFMPEG_Scale.h"
#include "FFMPEG_Sim_Scale.h"
#include "Sequential_Scale.h"
#include "OpenMP_Scale.h"

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
    ImageInfo img_test_yuv444p_8x8("imgs/test-yuv444p-8x8.yuv", 8, 8, AV_PIX_FMT_YUV444P);
    img_test_yuv444p_8x8.loadImage();

    ImageInfo img_col_rgb24_8x8("imgs/col-rgb24-8x8.yuv", 8, 8, AV_PIX_FMT_RGB24);
    img_col_rgb24_8x8.loadImage();

    ImageInfo img_col_yuv444p_8x8("imgs/col-yuv444p-8x8.yuv", 8, 8, AV_PIX_FMT_YUV444P);
    img_col_yuv444p_8x8.loadImage();

    // Real test images
	ImageInfo img_rgb24_1920x1080("imgs/bbb-rgb24-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_RGB24);
	img_rgb24_1920x1080.loadImage();

	ImageInfo img_yuv444p_1920x1080("imgs/bbb-yuv444p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV444P);
	img_yuv444p_1920x1080.loadImage();

	ImageInfo img_yuv422p_1920x1080("imgs/bbb-yuv422p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV422P);
	img_yuv422p_1920x1080.loadImage();

	ImageInfo img_yuv420p_1920x1080("imgs/bbb-yuv420p-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_YUV420P);
	img_yuv420p_1920x1080.loadImage();

	ImageInfo img_uyvy422_1920x1080("imgs/bbb-uyvy422-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_UYVY422);
	img_uyvy422_1920x1080.loadImage();

    ImageInfo img_nv12_1920x1080("imgs/bbb-nv12-1920x1080.yuv", 1920, 1080, AV_PIX_FMT_NV12);
    img_nv12_1920x1080.loadImage();

    // DEBUG VARIABLES -------------------------------
    ImageInfo inImg = img_nv12_1920x1080;
    int dstWidth = 1920;
    int dstHeight = 1080;
    AVPixelFormat dstFormat = AV_PIX_FMT_NV12;
    int operation = SWS_BICUBIC;

    int maxTestTimes = 1;
    bool testAverage = false;
    bool testFfmpeg = true;
    bool testFfmpegSim = true;
    bool testSequential = false;
    bool testOpenmp = false;

    int avgAcc;

    ImageInfo outImgFFmpeg("imgs/!ffmpeg-out.yuv", dstWidth, dstHeight, dstFormat);
    ImageInfo outImgFFmpegSim("imgs/!ffmpeg-sim-out.yuv", dstWidth, dstHeight, dstFormat);
    ImageInfo outImgSequential("imgs/!sequential-out.yuv", dstWidth, dstHeight, dstFormat);
    ImageInfo outImgOpenMP("imgs/!openmp-out.yuv", dstWidth, dstHeight, dstFormat);

    // SCALING OPERATIONS ----------------------------
    // Initialize ffmpeg
    av_register_all();
	
	// It is resample and resizing operations
	cout << endl << ">> Resample and Resizing Operations:" << endl << endl;

    // Operate with ffmpeg
    if(testFfmpeg){
        // Prepares the average accumulator
        if(testAverage){
            avgAcc = 0;
            cout << "[FFMPEG] Average execution started!" << endl;
        }

        for(int testTimes = maxTestTimes; testTimes > 0; testTimes--){
            // Reset output frame
            outImgFFmpeg.initFrame();

            // Resample and scale
            int executionTime = ffmpeg_scale(inImg, outImgFFmpeg, operation);
            if(executionTime < 0){
                system("pause");
                return -1;
            }

            // Display test results
            if(!testAverage)
                cout << "[FFMPEG] Execution time was " << executionTime << " ms!" << endl;
            else
                avgAcc += executionTime;
        }

        // Display averaged results
        if(testAverage)
            cout << "[FFMPEG] Average execution time was " << avgAcc / static_cast<float>(maxTestTimes) << " ms!" << endl;
    }

    // Operate with ffmpeg simulator
    if(testFfmpegSim){
        // Prepares the average accumulator
        if(testAverage){
            avgAcc = 0;
            cout << "[SIMULATOR] Average execution started!" << endl;
        }

        for(int testTimes = maxTestTimes; testTimes > 0; testTimes--){
            // Reset output frame
            outImgFFmpegSim.initFrame();

            // Resample and scale
            int executionTime = ffmpeg_sim_scale(inImg, outImgFFmpegSim, operation);
            if(executionTime < 0){
                system("pause");
                return -1;
            }

            // Display test results
            if(!testAverage)
                cout << "[SIMULATOR] Execution time was " << executionTime << " ms!" << endl;
            else
                avgAcc += executionTime;
        }

        // Display averaged results
        if(testAverage)
            cout << "[SIMULATOR] Average execution time was " << avgAcc / static_cast<float>(maxTestTimes) << " ms!" << endl;
    }

    // Operate with sequential process
    if(testSequential){
        // Prepares the average accumulator
        if(testAverage){
            avgAcc = 0;
            cout << "[SEQUENTIAL] Average execution started!" << endl;
        }

        for(int testTimes = maxTestTimes; testTimes > 0; testTimes--){
            // Reset output frame
            outImgSequential.initFrame();

            // Resample and scale
            int executionTime = sequential_scale(inImg, outImgSequential, operation);
            if(executionTime < 0){
                system("pause");
                return -1;
            }

            // Display test results
            if(!testAverage)
                cout << "[SEQUENTIAL] Execution time was " << executionTime << " ms!" << endl;
            else
                avgAcc += executionTime;
        }

        // Display averaged results
        if(testAverage)
            cout << "[SEQUENTIAL] Average execution time was " << avgAcc / static_cast<float>(maxTestTimes) << " ms!" << endl;
    }

    // Operate with openmp process
    if(testOpenmp){
        // Prepares the average accumulator
        if(testAverage){
            avgAcc = 0;
            cout << "[OPENMP] Average execution started!" << endl;
        }

        for(int testTimes = maxTestTimes; testTimes > 0; testTimes--){
            // Reset output frame
            outImgOpenMP.initFrame();

            // Resample and scale
            int executionTime = openmp_scale(inImg, outImgOpenMP, operation);
            if(executionTime < 0){
                system("pause");
                return -1;
            }

            // Display test results
            if(!testAverage)
                cout << "[OPENMP] Execution time was " << executionTime << " ms!" << endl;
            else
                avgAcc += executionTime;
        }

        // Display averaged results
        if(testAverage)
            cout << "[OPENMP] Average execution time was " << avgAcc / static_cast<float>(maxTestTimes) << " ms!" << endl;
    }

    // Write results
    cout << endl << ">> Writing images to files!" << endl << endl;
    if(testFfmpeg)
        outImgFFmpeg.writeImage();
    if(testFfmpegSim)
        outImgFFmpegSim.writeImage();
    if(testSequential)
        outImgSequential.writeImage();
    if(testOpenmp)
        outImgOpenMP.writeImage();

	// Success
    system("pause");
    return 0;
}