
#include "ImageInfo.h"

ImageInfo::ImageInfo(string fileName, int width, int height, AVPixelFormat pixelFormat){
    this->fileName = fileName;
    this->width = width;
    this->height = height;
    this->pixelFormat = pixelFormat;
}