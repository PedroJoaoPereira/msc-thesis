#ifndef REFERENCE_IMAGE_H
#define REFERENCE_IMAGE_H

#include <fstream>
#include <vector>

#include "ImageInfo.h"

using namespace std;

// ImageInfo srcImage	- image to create layer from
// string outputDir		- directory of the output layer
// int numOfTestedTools	- number of tested tools
// Creates a layer of an image
void PrepareReferenceImage(ImageInfo srcImage, string outputDir, int numOfTestedTools);

// ImageInfo referenceImage	- reference image being created
// vector<string> inputDirs	- vector of directories of the input layers
// Creates an image from layers
void CreateReferenceImage(ImageInfo referenceImage, vector<string> inputDirs);

#endif