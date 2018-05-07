#ifndef TESTS_H
#define TESTS_H

#include <string>
#include <vector>

#include "FFMPEG_Resample.h"
#include "CUDA_Resample.h"

using namespace std;

// Facilitate writing operations
string pixelFormatToString(int format);
string operationToString(int operation);

// Test ffmpeg procedure
int testFFMPEGSingle(ImageClass &inImg, ImageClass &outImg, int operation);
int testFFMPEGAverage(ImageClass &inImg, ImageClass outImg, int operation, int nTimes);
void testFFMPEG(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes);

// Test cuda procedure
int testCUDASingle(ImageClass &inImg, ImageClass &outImg, int operation);
int testCUDAAverage(ImageClass &inImg, ImageClass outImg, int operation, int nTimes);
void testCUDA(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes);

// Test all procedures
void testAll(vector<ImageClass*> &inImgs, vector<ImageClass*> &outImgs, vector<int> &operations, int nTimes);

#endif
