#include <iostream>
#include <vector>

#include "ImageInfo.h"
#include "Reference_Image.h"
#include "FFMPEG_Scale.h"
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
    ImageInfo inImg = img_rgb24_1920x1080;
    int dstWidth = 3840;
    int dstHeight = 2160;
	AVPixelFormat dstFormat = AV_PIX_FMT_YUV444P;
	int operation = SWS_BILINEAR;

	bool prepareTemporaryReferenceImages = false;
	bool createReferenceImages = true;

    int maxTestTimes = 1;
    bool testAverage = false;
    bool testFfmpeg = true;
    bool testSequential = true;
    bool testOpenmp = false;

    int avgAcc;

    ImageInfo outImgFFmpeg("imgs/!ffmpeg-out.yuv", dstWidth, dstHeight, dstFormat);
    ImageInfo outImgSequential("imgs/!sequential-out.yuv", dstWidth, dstHeight, dstFormat);
    ImageInfo outImgOpenMP("imgs/!openmp-out.yuv", dstWidth, dstHeight, dstFormat);

    // SCALING OPERATIONS ----------------------------
    // Initialize ffmpeg
    av_register_all();

	// Checks if is any error measurement operation
	if (prepareTemporaryReferenceImages || CreateReferenceImage) {

		// Tools name vector holder
		vector<string> tools = vector<string>();
		tools.push_back("custom-");
		tools.push_back("ffmpeg-");
		tools.push_back("gimp-");
		tools.push_back("opencv-");
		tools.push_back("ps-");

		// Sufix name of the test files
		vector<string> tests = vector<string>();
		tests.push_back("bilinear-1280x720-yuv444p.yuv");
		tests.push_back("bicubic-1280x720-yuv444p.yuv");
		tests.push_back("bilinear-3840x2160-yuv444p.yuv");
		tests.push_back("bicubic-3840x2160-yuv444p.yuv");

		// Vector width test images dimensions
		vector<int> widths = vector<int>();
		widths.push_back(1280);
		widths.push_back(1280);
		widths.push_back(3840);
		widths.push_back(3840);

		// Vector height test images dimensions
		vector<int> heights = vector<int>();
		heights.push_back(720);
		heights.push_back(720);
		heights.push_back(2160);
		heights.push_back(2160);

		// Checks if it is creating layers for reference images
		if (prepareTemporaryReferenceImages) {
			cout << endl << ">> Creating Layers For Reference Image:" << endl << endl;

			// Images obtained from tools directory
			string toolsImagesDir = "imgs/tools_images/";
			// Output directory for layers
			string layersDir = "imgs/tools_images/layers/";

			// For each test
			for (int testIndex = 0; testIndex < tests.size(); testIndex++) {
				// For each tool
				for (int toolIndex = 0; toolIndex < tools.size(); toolIndex++) {
					// Name of the file to be loaded
					string fileName = tools.at(toolIndex) + tests.at(testIndex);

					// Displays which image is processing
					cout << "Processing image: " << fileName << endl;

					// Load image
					ImageInfo inputImage(toolsImagesDir + fileName, widths.at(testIndex), heights.at(testIndex), AV_PIX_FMT_YUV444P);
					inputImage.loadImage();

					// Creates layer for a given image
					PrepareReferenceImage(inputImage, layersDir + "temp-" + fileName + ".txt", tools.size());
				}
			}

			// Success
			system("pause");
			return 0;
		}

		// Checks if it is creating reference images
		if (createReferenceImages) {
			cout << endl << ">> Creating Reference Images:" << endl << endl;

			// Layers directory
			string layersDir = "imgs/tools_images/layers/";
			// Output directory for references
			string referencesDir = "imgs/tools_images/references/";

			// For each test
			for (int testIndex = 0; testIndex < tests.size(); testIndex++) {
				// Name of the reference to be created
				string fileName = tests.at(testIndex);
				// Displays which image is processing
				cout << "Creating reference image: " << fileName << endl;

				// Create reference image
				ImageInfo outputImage(referencesDir + fileName, widths.at(testIndex), heights.at(testIndex), AV_PIX_FMT_YUV444P);
				outputImage.initFrame();

				// Create vector of directories of the layers
				vector<string> inputDirs = vector<string>();
				// For each tool
				for (int toolIndex = 0; toolIndex < tools.size(); toolIndex++) {
					// Push the directory for each tool layer
					inputDirs.push_back(layersDir + "temp-" + tools.at(toolIndex) + fileName + ".txt");
				}

				// Create reference image
				CreateReferenceImage(outputImage, inputDirs);

				// Write reference image to a file
				outputImage.writeImage();
			}

			// Success
			system("pause");
			return 0;
		}
	}
	
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
    if(testSequential)
        outImgSequential.writeImage();
    if(testOpenmp)
        outImgOpenMP.writeImage();

	// Success
    system("pause");
    return 0;
}