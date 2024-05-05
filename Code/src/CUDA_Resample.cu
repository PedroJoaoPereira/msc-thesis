#include "CUDA_Resample.h"

// Allocate image channels data buffers depending of the pixel format
void cudaAllocBuffers(uint8_t** &buffer, int* &bufferSize, int width, int height, int pixelFormat){
    // Allocate channel buffer size
    bufferSize = static_cast<int*>(malloc(3 * sizeof(int)));

    // Calculate once
    int wxh = width * height;
    int wxhDi2 = wxh / 2;
    int wxhDi4 = wxh / 4;

    // Calculate buffer sizes for each pixel format
    switch(pixelFormat){
    case AV_PIX_FMT_YUV444P:
        bufferSize[0] = wxh;
        bufferSize[1] = wxh;
        bufferSize[2] = wxh;
        break;
    case AV_PIX_FMT_YUV422P:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDi2;
        bufferSize[2] = wxhDi2;
        break;
    case AV_PIX_FMT_YUV422PNORM:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDi2;
        bufferSize[2] = wxhDi2;
        break;
    case AV_PIX_FMT_YUV420P:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDi4;
        bufferSize[2] = wxhDi4;
        break;
    case AV_PIX_FMT_UYVY422:
        bufferSize[0] = wxh * 2;
        bufferSize[1] = 0;
        bufferSize[2] = 0;
        break;
    case AV_PIX_FMT_NV12:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDi2;
        bufferSize[2] = 0;
        break;
    case AV_PIX_FMT_V210:
        bufferSize[0] = height * 128 * ((width + 47) / 48);
        bufferSize[1] = 0;
        bufferSize[2] = 0;
        break;
    }

    // Allocate buffer memory
    buffer = static_cast<uint8_t**>(malloc(3 * sizeof(uint8_t*)));

    // Allocate buffer in the GPU memory
    cudaMalloc((void **) &buffer[0], bufferSize[0]);
    if(bufferSize[1] != 0)
        cudaMalloc((void **) &buffer[1], bufferSize[1]);
    if(bufferSize[2] != 0)
        cudaMalloc((void **) &buffer[2], bufferSize[2]);
}

// Free used GPU memory
void freeCudaMemory(uint8_t** &buffer){
    // Iterate each channel and free memory
    for(int i = 0; i < 3; i++)
        cudaFree(buffer[i]);

    // Free host memory
    free(buffer);
}

// Copy data from host to device
void cudaCopyBuffersToGPU(uint8_t* srcBuffer[], uint8_t* gpuBuffer[], int* &bufferSize){
    // First channel
    cudaMemcpy(gpuBuffer[0], srcBuffer[0], bufferSize[0], cudaMemcpyHostToDevice);

    // Copy chroma channels if they exist
    if(bufferSize[1] != 0)
        cudaMemcpy(gpuBuffer[1], srcBuffer[1], bufferSize[1], cudaMemcpyHostToDevice);
    if(bufferSize[2] != 0)
        cudaMemcpy(gpuBuffer[2], srcBuffer[2], bufferSize[2], cudaMemcpyHostToDevice);
}

// Copy data from device to host
void cudaCopyBuffersFromGPU(uint8_t* targetBuffer[], uint8_t* gpuBuffer[], int* &bufferSize){
    // First channel
    cudaMemcpy(targetBuffer[0], gpuBuffer[0], bufferSize[0], cudaMemcpyDeviceToHost);

    // Copy chroma channels if they exist
    if(bufferSize[1] != 0)
        cudaMemcpy(targetBuffer[1], gpuBuffer[1], bufferSize[1], cudaMemcpyDeviceToHost);
    if(bufferSize[2] != 0)
        cudaMemcpy(targetBuffer[2], gpuBuffer[2], bufferSize[2], cudaMemcpyDeviceToHost);
}

// Calculate launch parameters of format conversion kernel
pair<dim3, dim3> calculateConversionLP(int width, int height, int srcPixelFormat, int dstPixelFormat){
    // Variable with result launch parameters
    pair<dim3, dim3> result;

    // Discover dimensions value depending of the conversion
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width * 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width * 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width * 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width * 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width * 2 / 3, height);
    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width / 2, height);

    // Calculate thread size
    int hDivisor = greatestDivisor(result.first.x, 16);
    int vDivisor = greatestDivisor(result.first.y, 16);

    // Assign thread size
    result.second = dim3(hDivisor, vDivisor);

    // Calculate block size
    result.first.x /= hDivisor;
    result.first.y /= vDivisor;

    return result;
}

// Convert the pixel format of the image
template <class PrecisionType>
__global__ void cuda_formatConversion(int width, int height,
    int srcPixelFormat, uint8_t* srcSlice0, uint8_t* srcSlice1, uint8_t* srcSlice2,
    int dstPixelFormat, uint8_t* dstSlice0, uint8_t* dstSlice1, uint8_t* dstSlice2){

    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int offset = lin * width * 2 + col;

        // Calculate source index
        auto srcBuffer = srcSlice0 + offset;
        auto dstBuffer = dstSlice0 + offset;

        *dstBuffer = *srcBuffer;

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int whichCol = threadIdx.x % 4;
        int linMulWidth = lin * width;

        // Calculate source index
        auto srcBuffer = srcSlice0 + linMulWidth * 2 + col;

        // Discover which component pixel is
        if(whichCol == 0){
            auto dstBuffer = dstSlice1 + linMulWidth / 2 + col / 4;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 1){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 2){
            auto dstBuffer = dstSlice2 + linMulWidth / 2 + col / 4;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 3){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int whichCol = threadIdx.x % 4;
        int linMulWidth = lin * width;

        // Calculate source index
        auto srcBuffer = srcSlice0 + linMulWidth * 2 + col;

        // Discover which component pixel is
        if(whichCol == 0 && lin % 2 == 0){
            auto dstBuffer = dstSlice1 + linMulWidth / 4 + col / 4;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 1){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 2 && lin % 2 == 0){
            auto dstBuffer = dstSlice2 + linMulWidth / 4 + col / 4;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 3){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int whichCol = threadIdx.x % 4;
        int linMulWidth = lin * width;

        // Calculate source index
        auto srcBuffer = srcSlice0 + linMulWidth * 2 + col;

        // Discover which component pixel is
        if(whichCol == 0 && lin % 2 == 0){
            auto dstBuffer = dstSlice1 + linMulWidth / 2 + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 1){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 2 && lin % 2 == 0){
            auto dstBuffer = dstSlice1 + linMulWidth / 2 + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
        if(whichCol == 3){
            auto dstBuffer = dstSlice0 + linMulWidth + col / 2;
            *dstBuffer = *srcBuffer;
            return;
        }
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int linMulWidth = lin * width;

        // Calculate source index
        auto srcBuffer = srcSlice0 + linMulWidth * 2 + col * 3;
        auto dstBuffer = reinterpret_cast<uint32_t*>(dstSlice0);
        dstBuffer += linMulWidth / 6 * 4 + col;

        // Upsample to 10 bits
        auto valA10b = *srcBuffer++ << 2U;
        auto valB10b = *srcBuffer++ << 2U;
        auto valC10b = *srcBuffer++ << 2U;

        // Assign value
        *dstBuffer = (valC10b << 20U) | (valB10b << 10U) | valA10b;

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Calculate pixel location
        int lin = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Calculate once
        int linMulWidth = lin * width;
        int linMulWidthDiv2 = linMulWidth / 2;

        // Calculate source index
        auto srcBuffer = srcSlice0 + linMulWidth + col * 2;
        auto srcBufferChromaU = srcSlice1 + linMulWidthDiv2 + col;
        auto srcBufferChromaV = srcSlice2 + linMulWidthDiv2 + col;
        auto dstBuffer = dstSlice0 + linMulWidth * 2 + col * 4;

        // Assigne values to dst buffer
        *dstBuffer++ = *srcBufferChromaU; // U0
        *dstBuffer++ = *srcBuffer++; // Y0
        *dstBuffer++ = *srcBufferChromaV; // V0
        *dstBuffer++ = *srcBuffer; // Y1

        return;
    }
}

// Precalculate coefficients
template <class PrecisionType>
int cuda_preCalculateCoefficients(int srcSize, int dstSize, int operation,
    int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType* &preCalculatedCoefs){

    // Calculate size ratio
    PrecisionType sizeRatio = static_cast<PrecisionType>(dstSize) / static_cast<PrecisionType>(srcSize);

    // Calculate once
    PrecisionType pixelSupportDiv2 = pixelSupport / static_cast<PrecisionType>(2.);
    bool isDownScale = sizeRatio < static_cast<PrecisionType>(1.);
    PrecisionType regionRadius = isDownScale ? pixelSupportDiv2 / sizeRatio : pixelSupportDiv2;
    PrecisionType filterStep = isDownScale && operation != SWS_POINT ? static_cast<PrecisionType>(1.) / sizeRatio : static_cast<PrecisionType>(1.);
    int numCoefficients = isDownScale ? ceil(pixelSupport / sizeRatio) : pixelSupport;
    int numCoefficientsDiv2 = numCoefficients / 2;

    // Calculate number of lines of coefficients
    int preCalcCoefSize = isDownScale ? dstSize : lcm(srcSize, dstSize) / min<int>(srcSize, dstSize);

    // Initialize array
    preCalculatedCoefs = static_cast<PrecisionType*>(malloc(preCalcCoefSize * numCoefficients * sizeof(PrecisionType)));

    // For each necessary line of coefficients
    for(int col = 0; col < preCalcCoefSize; col++){
        // Calculate once
        int indexOffset = col * numCoefficients;

        // Original line index coordinate
        PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(.5)) / sizeRatio;

        // Discover source limit pixels
        PrecisionType nearPixel = colOriginal - filterStep;
        PrecisionType leftPixel = colOriginal - regionRadius;

        // Discover offset to pixel of filter start
        PrecisionType offset = round(leftPixel) + static_cast<PrecisionType>(.5) - leftPixel;
        // Calculate maximum distance to normalize distances
        PrecisionType maxDistance = colOriginal - nearPixel;
        // Calculate where filtering will start
        PrecisionType startPosition = leftPixel + offset;

        // Calculate coefficients
        PrecisionType coefAcc = static_cast<PrecisionType>(0.);
        for(int index = 0; index < numCoefficients; index++){
            PrecisionType coefHolder = coefFunc((colOriginal - (startPosition + index)) / maxDistance);
            coefAcc += coefHolder;
            preCalculatedCoefs[indexOffset + index] = coefHolder;
        }

        // Avoid lines of coefficients without valid values
        if(operation == SWS_POINT){
            if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == preCalculatedCoefs[indexOffset + numCoefficientsDiv2]){
                if(isDownScale){
                    if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == static_cast<PrecisionType>(0.) && numCoefficients % 2 != 0)
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] = static_cast<PrecisionType>(1.);
                    else
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
                } else
                    preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
            }
        }

        // Normalizes coefficients except on Nearest Neighbor interpolation
        if(operation != SWS_POINT)
            for(int index = 0; index < numCoefficients; index++)
                preCalculatedCoefs[indexOffset + index] /= coefAcc;
    }

    // Success
    return preCalcCoefSize;
}

// Change the image dimension
template <class PrecisionType>
void cuda_resize(int srcWidth, int srcHeight, uint8_t* srcData,
    int dstWidth, int dstHeight, uint8_t* dstData,
    int operation, int pixelSupport,
    int vCoefsSize, PrecisionType* &vCoefs, int hCoefsSize, PrecisionType* &hCoefs,
    int colorChannel){

    // Get scale ratios
    PrecisionType scaleHeightRatio = static_cast<PrecisionType>(dstHeight) / static_cast<PrecisionType>(srcHeight);
    PrecisionType scaleWidthRatio = static_cast<PrecisionType>(dstWidth) / static_cast<PrecisionType>(srcWidth);

    // Calculate once
    PrecisionType pixelSupportDiv2 = pixelSupport / static_cast<PrecisionType>(2.);
    bool isDownScaleV = scaleHeightRatio < static_cast<PrecisionType>(1.);
    bool isDownScaleH = scaleWidthRatio < static_cast<PrecisionType>(1.);
    PrecisionType regionVRadius = isDownScaleV ? pixelSupportDiv2 / scaleHeightRatio : pixelSupportDiv2;
    PrecisionType regionHRadius = isDownScaleH ? pixelSupportDiv2 / scaleWidthRatio : pixelSupportDiv2;
    int numVCoefs = isDownScaleV ? ceil(pixelSupport / scaleHeightRatio) : pixelSupport;
    int numHCoefs = isDownScaleH ? ceil(pixelSupport / scaleWidthRatio) : pixelSupport;

    // Iterate through each line of the scaled image
    for(int lin = 0; lin < dstHeight; lin++){
        // Calculate once the target line coordinate
        int targetLine = lin * dstWidth;
        // Calculate once the line index of coefficients
        int indexLinOffset = (lin % vCoefsSize) * numVCoefs;

        // Original line index coordinate
        PrecisionType linOriginal = (static_cast<PrecisionType>(lin) + static_cast<PrecisionType>(.5)) / scaleHeightRatio;

        // Discover source limit pixels
        PrecisionType upperPixel = linOriginal - regionVRadius;
        // Discover offset to pixel of filter start
        PrecisionType offsetV = round(upperPixel) + static_cast<PrecisionType>(.5) - upperPixel;

        // Calculate once
        PrecisionType startLinPosition = upperPixel + offsetV;

        // Iterate through each column of the scaled image
        for(int col = 0; col < dstWidth; col++){
            // Calculate once the column index of coefficients
            int indexColOffset = (col % hCoefsSize) * numHCoefs;

            // Original line index coordinate
            PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(.5)) / scaleWidthRatio;

            // Discover source limit pixels
            PrecisionType leftPixel = colOriginal - regionHRadius;
            // Discover offset to pixel of filter start
            PrecisionType offsetH = round(leftPixel) + static_cast<PrecisionType>(.5) - leftPixel;

            // Calculate once
            PrecisionType startColPosition = leftPixel + offsetH;

            // Temporary variables used in the interpolation
            PrecisionType result = static_cast<PrecisionType>(0.);
            // Calculate resulting color from coefficients
            for(int indexV = 0; indexV < numVCoefs; indexV++){
                // Access once the memory
                PrecisionType vCoef = vCoefs[indexLinOffset + indexV];

                for(int indexH = 0; indexH < numHCoefs; indexH++){
                    // Access once the memory
                    PrecisionType hCoef = hCoefs[indexColOffset + indexH];

                    // Get pixel from source data
                    uint8_t colorHolder = getPixel(startLinPosition + indexV, startColPosition + indexH, srcWidth, srcHeight, srcData);

                    // Calculate pixel color weight
                    PrecisionType weight = vCoef * hCoef;

                    // Weights neighboring pixel and add it to the result
                    result += static_cast<PrecisionType>(colorHolder) * weight;
                }
            }

            // Clamp value to avoid undershooting and overshooting
            if(colorChannel == 0)
                clamp<PrecisionType>(result, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(235.));
            else
                clamp<PrecisionType>(result, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(240.));
            // Assign calculated color to destiantion data
            dstData[targetLine + col] = roundTo<uint8_t, PrecisionType>(result);
        }
    }
}

// Prepares the resample operation
template <class PrecisionType>
int cuda_resample_aux(AVFrame* src, AVFrame* dst, int operation){
    // Return value of this method
    int returnValue = 0;

    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    int srcFormat = src->format;
    int dstWidth = dst->width, dstHeight = dst->height;
    int dstFormat = dst->format;

    // Check if is only a format conversion
    bool isOnlyFormatConversion = srcWidth == dstWidth && srcHeight == dstHeight;

    // Initialize needed variables if it is a scaling operation
    int scalingSupportedFormat = getTempScaleFormat(srcFormat, dstFormat);

    // Last format conversion variables
    uint8_t** lastConversion;
    int* lastConversionSizes;
    int lastConversionPixelFormat = srcFormat;
    // Allocate buffer
    cudaAllocBuffers(lastConversion, lastConversionSizes, srcWidth, srcHeight, srcFormat);
    // Copy source data to GPU
    cudaCopyBuffersToGPU(src->data, lastConversion, lastConversionSizes);

    // Temporary buffers used in intermediate operations
    uint8_t** formatConversionBuffer, **resizeBuffer;

    // Rescaling operation branch
    if(!isOnlyFormatConversion){
        /*
        // Allocate temporary buffers
        cudaAllocBuffers(formatConversionBuffer, srcWidth, srcHeight, scalingSupportedFormat);
        cudaAllocBuffers(resizeBuffer, dstWidth, dstHeight, scalingSupportedFormat);

        // Resamples image to a supported format
        if(cuda_formatConversion<PrecisionType>(srcWidth, srcHeight,
            srcFormat, src->data,
            scalingSupportedFormat, formatConversionBuffer) < 0){
            returnValue = -1;
            goto END;
        }

        // Needed resources for coefficients calculations
        PrecisionType(*coefFunc)(PrecisionType) = getCoefMethod<PrecisionType>(operation);
        int pixelSupport = getPixelSupport(operation);

        // Variables for precalculated coefficients
        PrecisionType* vCoefs;
        int vCoefsSize = cuda_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefs);
        PrecisionType* hCoefs;
        int hCoefsSize = cuda_preCalculateCoefficients<PrecisionType>(srcWidth, dstWidth, operation, pixelSupport, coefFunc, hCoefs);

        // Chroma size discovery
        float widthPerc = 1.f;
        float heightPerc = 1.f;
        if(scalingSupportedFormat == AV_PIX_FMT_YUV422P ||
            scalingSupportedFormat == AV_PIX_FMT_YUV420P ||
            scalingSupportedFormat == AV_PIX_FMT_YUV422PNORM)
            widthPerc = 0.5f;
        if(scalingSupportedFormat == AV_PIX_FMT_YUV420P)
            heightPerc = 0.5f;

        // Apply the resizing operation to luma channel
        cuda_resize<PrecisionType>(srcWidth, srcHeight, formatConversionBuffer[0],
            dstWidth, dstHeight, resizeBuffer[0], operation,
            pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, 0);

        // Apply the resizing operation to chroma channels
        int srcWidthChroma = static_cast<int>(srcWidth * widthPerc);
        int srcHeightChroma = static_cast<int>(srcHeight * heightPerc);
        int dstWidthChroma = static_cast<int>(dstWidth * widthPerc);
        int dstHeightChroma = static_cast<int>(dstHeight * heightPerc);
        for(int colorChannel = 1; colorChannel < 3; colorChannel++){
            cuda_resize<PrecisionType>(srcWidthChroma, srcHeightChroma, formatConversionBuffer[colorChannel],
                dstWidthChroma, dstHeightChroma, resizeBuffer[colorChannel], operation,
                pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, colorChannel);
        }


        // Assign correct values to apply last resample
        lastFormatConversionBuffer = resizeBuffer;
        lastFormatConversionPixelFormat = scalingSupportedFormat;

        // Free used resources
        free(vCoefs);
        free(hCoefs);
        */
    }

    // Result data buffer
    uint8_t** finalData;
    int* finalDataSizes;
    // Allocate buffer
    cudaAllocBuffers(finalData, finalDataSizes, dstWidth, dstHeight, dstFormat);

    // Create launch parameters of format conversion kernel
    pair<dim3, dim3> formatConversionLP = calculateConversionLP(dstWidth, dstHeight, lastConversionPixelFormat, dstFormat);
    // Resamples image to a target format
    cuda_formatConversion<PrecisionType> << <formatConversionLP.first, formatConversionLP.second>> > (dstWidth, dstHeight,
        lastConversionPixelFormat, lastConversion[0], lastConversion[1], lastConversion[2],
        dstFormat, finalData[0], finalData[1], finalData[2]);

    // Copy resulting data to CPU
    cudaCopyBuffersFromGPU(dst->data, finalData, finalDataSizes);

    // Free used resources
    freeCudaMemory(lastConversion);
    freeCudaMemory(finalData);

    END:
    // Free used resources
    if(!isOnlyFormatConversion){
        free2dBuffer<uint8_t>(formatConversionBuffer, 3);
        free2dBuffer<uint8_t>(resizeBuffer, 3);
    }

    // Return negative if insuccess
    return returnValue;
}

// Wrapper for the cuda resample operation method
int cuda_resample(AVFrame* src, AVFrame* dst, int operation){
    // Variables used
    int duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Verify valid frames
    if(src == nullptr || dst == nullptr){
        cerr << "[CUDA] One or both input frames are null!" << endl;
        return -1;
    }

    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid input dimensions
    if(src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0){
        cerr << "[CUDA] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid resize
    if((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)){
        cerr << "[CUDA] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }
    // Verify valid input data
    if(!src->data || !src->linesize || !dst->data || !dst->linesize){
        cerr << "[CUDA] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if(!isSupportedFormat(srcFormat) || !isSupportedFormat(dstFormat)){
        cerr << "[CUDA] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if can convert a 10 bit format
    if((src->width % 12 != 0 && srcFormat == AV_PIX_FMT_V210) || (dst->width % 12 != 0 && dstFormat == AV_PIX_FMT_V210)){
        cerr << "[CUDA] Can not handle 10 bit format because data is not aligned!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if(!isSupportedOperation(operation)){
        cerr << "[CUDA] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(cuda_resample_aux<double>(src, dst, operation) < 0){
        // Display error
        cerr << "[CUDA] Operation could not be done (resample - conversion not supported)!" << endl;

        // Insuccess
        return -1;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}