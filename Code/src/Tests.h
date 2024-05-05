#ifndef TESTS_H
#define TESTS_H

#include <string>
#include <vector>

#include "FFMPEG_Scale.h"
#include "Simulator_Scale.h"

using namespace std;

// Facilitate writing operations
string pixelFormatToString(int format);
string operationToString(int operation);

// Test ffmpeg procedure
int testFFMPEGSingle(ImageInfo &inImg, ImageInfo &outImg, int operation);
int testFFMPEGAverage(ImageInfo &inImg, ImageInfo outImg, int operation, int nTimes);
void testFFMPEG(vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes);

// Test simulator procedure
int testSimulatorSingle(ImageInfo &inImg, ImageInfo &outImg, int operation);
int testSimulatorAverage(ImageInfo &inImg, ImageInfo outImg, int operation, int nTimes);
void testSimulator(vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes);

// Test all procedures
void testAll(bool isTestFFMPEG, bool isTestSimulator, bool isTestSequential, bool isTestOpenMP, vector<ImageInfo*> &inImgs, vector<ImageInfo*> &outImgs, vector<int> &operations, int nTimes);

#endif
