#include "CUDA_Resample.h"

texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texY;
texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texU;
texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texV;

// Allocate image channels data buffers depending of the pixel format
void cudaAllocBuffers(uint8_t* &buffer, int* &bufferSize, int width, int height, int pixelFormat){
    // Calculate once
    int wxh = width * height;
    int wxhDiv2 = wxh / 2;
    int wxhDiv4 = wxh / 4;

    // Allocate channel buffer size
    bufferSize = static_cast<int*>(malloc(3 * sizeof(int)));
    // Calculate buffer sizes for each pixel format
    switch(pixelFormat){
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV422PNORM:
            bufferSize[0] = wxh;
            bufferSize[1] = wxhDiv2;
            bufferSize[2] = wxhDiv2;
            break;
        case AV_PIX_FMT_YUV420P:
            bufferSize[0] = wxh;
            bufferSize[1] = wxhDiv4;
            bufferSize[2] = wxhDiv4;
            break;
    }

    // Allocate buffer memory in device
    cudaMalloc((void **) &buffer, bufferSize[0] + bufferSize[1] + bufferSize[2]);
}

// Calculate launch parameters of resize kernel
pair<dim3, dim3> calculateResizeLP(int width, int height){
    // Define the maximum number of thread dim1 size
    int maxNumThreads = 32;

    // Find the thead size
    int vThreadSize = min(maxNumThreads, greatestDivisor(height));
    int hThreadSize = min(maxNumThreads, greatestDivisor(width));

    // Calculate the block size
    int vBlockSize = height / vThreadSize;
    int hBlockSize = width / hThreadSize;

    // Return valid launch parameters
    return pair<dim3, dim3>(dim3(hBlockSize, vBlockSize), dim3(hThreadSize, vThreadSize));
}

// ------------------------------------------------------------------

uint8_t* pinnedHost;
cudaChannelFormatDesc channelDesc;
cudaArray *ySrc, *uSrc, *vSrc;
uint8_t* scaledDevice;
int* scaledDeviceSizes;
cudaStream_t streamY, streamU, streamV;

// Initializes data
void cuda_init(AVFrame* src, AVFrame* dst, int operation){
    // Get standard supported pixel format in scaling
    int scaleFormat = getScaleFormat(src->format, dst->format);

    // Calculate the size of the chroma components
    int srcHeightChroma = src->height;
    int srcWidthChroma = src->width;
    int dstHeightChroma = dst->height;
    int dstWidthChroma = dst->width;
    if(scaleFormat == AV_PIX_FMT_YUV422P || scaleFormat == AV_PIX_FMT_YUV420P || scaleFormat == AV_PIX_FMT_YUV422PNORM){
        srcWidthChroma /= 2;
        dstWidthChroma /= 2;
    }
    if(scaleFormat == AV_PIX_FMT_YUV420P){
        srcHeightChroma /= 2;
        dstHeightChroma /= 2;
    }

    // Allocate host pinned memory
    cudaMallocHost((void **) &pinnedHost, src->height * src->width + 2 * srcHeightChroma * srcWidthChroma + dst->height * dst->width + 2 * dstHeightChroma * dstWidthChroma);

    // Create channel texture descriptor
    channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // Set configurations of texture memory
    texY.addressMode[0] = cudaAddressModeClamp;
    texY.addressMode[1] = cudaAddressModeClamp;
    texY.normalized = false;
    texU.addressMode[0] = cudaAddressModeClamp;
    texU.addressMode[1] = cudaAddressModeClamp;
    texU.normalized = false;
    texV.addressMode[0] = cudaAddressModeClamp;
    texV.addressMode[1] = cudaAddressModeClamp;
    texV.normalized = false;

    // Set interpolation method
    if(operation == SWS_BILINEAR){
        texY.filterMode = cudaFilterModeLinear;
        texU.filterMode = cudaFilterModeLinear;
        texV.filterMode = cudaFilterModeLinear;
    } else{
        texY.filterMode = cudaFilterModePoint;
        texU.filterMode = cudaFilterModePoint;
        texV.filterMode = cudaFilterModePoint;
    }

    // Create a 2d cuda array for each source component
    cudaMallocArray(&ySrc, &channelDesc, src->width, src->height);
    cudaMallocArray(&uSrc, &channelDesc, srcWidthChroma, srcHeightChroma);
    cudaMallocArray(&vSrc, &channelDesc, srcWidthChroma, srcHeightChroma);

    // Bind textures to device memory
    cudaBindTextureToArray(&texY, ySrc, &channelDesc);
    cudaBindTextureToArray(&texU, uSrc, &channelDesc);
    cudaBindTextureToArray(&texV, vSrc, &channelDesc);

    // Allocate source buffer in device
    cudaAllocBuffers(scaledDevice, scaledDeviceSizes, dst->width, dst->height, scaleFormat);

    // Create cuda streams for concurrent execution of kernels
    cudaStreamCreate(&streamY);
    cudaStreamCreate(&streamU);
    cudaStreamCreate(&streamV);
}

// Free resources
void cuda_finish(){
    // Delete cuda arrays
    cudaFreeArray(ySrc);
    cudaFreeArray(uSrc);
    cudaFreeArray(vSrc);

    // Free used resources
    cudaFree(scaledDevice);
    free(scaledDeviceSizes);

    cudaStreamDestroy(streamY);
    cudaStreamDestroy(streamU);
    cudaStreamDestroy(streamV);

    // Deallocate host pinned memory
    cudaFreeHost(pinnedHost);
}

// ------------------------------------------------------------------

// Nearest neighbor and bilinear hardware interpolation for tex y
__global__ void scaleTexY(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(tex2D(texY, colOriginal, linOriginal) * 255.f));
}

// Nearest neighbor and bilinear hardware interpolation for tex u
__global__ void scaleTexU(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData, const int dstOffset){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col + dstOffset] = uint8_t(roundf(tex2D(texU, colOriginal, linOriginal) * 255.f));
}

// Nearest neighbor and bilinear hardware interpolation for tex v
__global__ void scaleTexV(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData, const int dstOffset){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col + dstOffset] = uint8_t(roundf(tex2D(texV, colOriginal, linOriginal) * 255.f));
}

// Calculate coefficient of cubic interpolation
inline __device__ float cubicFilter(const float x, const float c0, const float c1, const float c2, const float c3){
    // Resulting color is the sum of all weighted colors
    float result = c0 * (-.6f * x * (x * (x - 2.f) + 1.f));
    result += c1 * (x * x * (1.4f * x - 2.4f) + 1.f);
    result += c2 * (x * (x * (-1.4f * x + 1.8f) + .6f));
    result += c3 * (.6f * x * x * (x - 1.f));
    return result;
}

// Bicubic interpolation for tex y
__global__ void cubicScaleY(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio - .5f;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio - .5f;

    // Calculate nearest source sample
    const float pixLin = floorf(linOriginal);
    const float pixCol = floorf(colOriginal);

    // Calculate distance to the source sample
    const float distLin = linOriginal - pixLin;
    const float distCol = colOriginal - pixCol;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(255.f * cubicFilter(distLin,
        cubicFilter(distCol, tex2D(texY, pixCol - 1, pixLin - 1), tex2D(texY, pixCol, pixLin - 1), tex2D(texY, pixCol + 1, pixLin - 1), tex2D(texY, pixCol + 2, pixLin - 1)),
        cubicFilter(distCol, tex2D(texY, pixCol - 1, pixLin), tex2D(texY, pixCol, pixLin), tex2D(texY, pixCol + 1, pixLin), tex2D(texY, pixCol + 2, pixLin)),
        cubicFilter(distCol, tex2D(texY, pixCol - 1, pixLin + 1), tex2D(texY, pixCol, pixLin + 1), tex2D(texY, pixCol + 1, pixLin + 1), tex2D(texY, pixCol + 2, pixLin + 1)),
        cubicFilter(distCol, tex2D(texY, pixCol - 1, pixLin + 2), tex2D(texY, pixCol, pixLin + 2), tex2D(texY, pixCol + 1, pixLin + 2), tex2D(texY, pixCol + 2, pixLin + 2)))));
}

// Bicubic interpolation for tex u
__global__ void cubicScaleU(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData, const int dstOffset){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio - .5f;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio - .5f;

    // Calculate nearest source sample
    const float pixLin = floorf(linOriginal);
    const float pixCol = floorf(colOriginal);

    // Calculate distance to the source sample
    const float distLin = linOriginal - pixLin;
    const float distCol = colOriginal - pixCol;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col + dstOffset] = uint8_t(roundf(255.f * cubicFilter(distLin,
        cubicFilter(distCol, tex2D(texU, pixCol - 1, pixLin - 1), tex2D(texU, pixCol, pixLin - 1), tex2D(texU, pixCol + 1, pixLin - 1), tex2D(texU, pixCol + 2, pixLin - 1)),
        cubicFilter(distCol, tex2D(texU, pixCol - 1, pixLin), tex2D(texU, pixCol, pixLin), tex2D(texU, pixCol + 1, pixLin), tex2D(texU, pixCol + 2, pixLin)),
        cubicFilter(distCol, tex2D(texU, pixCol - 1, pixLin + 1), tex2D(texU, pixCol, pixLin + 1), tex2D(texU, pixCol + 1, pixLin + 1), tex2D(texU, pixCol + 2, pixLin + 1)),
        cubicFilter(distCol, tex2D(texU, pixCol - 1, pixLin + 2), tex2D(texU, pixCol, pixLin + 2), tex2D(texU, pixCol + 1, pixLin + 2), tex2D(texU, pixCol + 2, pixLin + 2)))));
}

// Bicubic interpolation for tex v
__global__ void cubicScaleV(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData, const int dstOffset){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = ((float) lin + .5f) / scaleHeightRatio - .5f;
    const float colOriginal = ((float) col + .5f) / scaleWidthRatio - .5f;

    // Calculate nearest source sample
    const float pixLin = floorf(linOriginal);
    const float pixCol = floorf(colOriginal);

    // Calculate distance to the source sample
    const float distLin = linOriginal - pixLin;
    const float distCol = colOriginal - pixCol;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col + dstOffset] = uint8_t(roundf(255.f * cubicFilter(distLin,
        cubicFilter(distCol, tex2D(texV, pixCol - 1, pixLin - 1), tex2D(texV, pixCol, pixLin - 1), tex2D(texV, pixCol + 1, pixLin - 1), tex2D(texV, pixCol + 2, pixLin - 1)),
        cubicFilter(distCol, tex2D(texV, pixCol - 1, pixLin), tex2D(texV, pixCol, pixLin), tex2D(texV, pixCol + 1, pixLin), tex2D(texV, pixCol + 2, pixLin)),
        cubicFilter(distCol, tex2D(texV, pixCol - 1, pixLin + 1), tex2D(texV, pixCol, pixLin + 1), tex2D(texV, pixCol + 1, pixLin + 1), tex2D(texV, pixCol + 2, pixLin + 1)),
        cubicFilter(distCol, tex2D(texV, pixCol - 1, pixLin + 2), tex2D(texV, pixCol, pixLin + 2), tex2D(texV, pixCol + 1, pixLin + 2), tex2D(texV, pixCol + 2, pixLin + 2)))));
}

// Prepares the resample operation
void cuda_resample_aux(AVFrame* src, AVFrame* dst, int operation,
    int srcWidth, int srcHeight, int dstWidth, int dstHeight,
    int srcWidthChroma, int srcHeightChroma, int dstWidthChroma, int dstHeightChroma,
    int srcFormat, int dstFormat, int scaleFormat,     
    uint8_t* &pinnedHost, cudaChannelFormatDesc &channelDesc, cudaArray* &ySrc, cudaArray* &uSrc, cudaArray* &vSrc,
    uint8_t* &scaledDevice, int* &scaledDeviceSizes, cudaStream_t &streamY, cudaStream_t &streamU, cudaStream_t &streamV,
    double* &times){

    // Get scale ratios
    float scaleHeightRatio = static_cast<float>(dstHeight) / static_cast<float>(srcHeight);
    float scaleWidthRatio = static_cast<float>(dstWidth) / static_cast<float>(srcWidth);

    // Buffers for first format conversion
    uint8_t** toScalePtrs = static_cast<uint8_t**>(malloc(3 * sizeof(uint8_t*)));
    toScalePtrs[0] = pinnedHost;
    toScalePtrs[1] = toScalePtrs[0] + srcHeight * srcWidth;
    toScalePtrs[2] = toScalePtrs[1] + srcHeightChroma * srcWidthChroma;
    uint8_t** fromScalePtrs = static_cast<uint8_t**>(malloc(3 * sizeof(uint8_t*)));
    fromScalePtrs[0] = toScalePtrs[2] + srcHeightChroma * srcWidthChroma;
    fromScalePtrs[1] = fromScalePtrs[0] + dstHeight * dstWidth;
    fromScalePtrs[2] = fromScalePtrs[1] + dstHeightChroma * dstWidthChroma;

    // Time variables
    high_resolution_clock::time_point initTime, stopTime;

    // Format conversion operation
    initTime = high_resolution_clock::now();
    omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, scaleFormat, toScalePtrs);
    stopTime = high_resolution_clock::now();
    times[0] = duration_cast<microseconds>(stopTime - initTime).count() * 1.;

    // Calculate launch parameters
    pair<dim3, dim3> lumaLP = calculateResizeLP(dstWidth, dstHeight);
    pair<dim3, dim3> chromaLP = calculateResizeLP(dstWidthChroma, dstHeightChroma);

    // Calculate once the offsets
    int offsetFrom0 = scaledDeviceSizes[0];
    int offsetFrom1 = offsetFrom0 + scaledDeviceSizes[1];

    // Scale each component
    initTime = high_resolution_clock::now();
    if(operation == SWS_POINT || operation == SWS_BILINEAR){
        cudaMemcpyToArrayAsync(ySrc, 0, 0, toScalePtrs[0], srcHeight * srcWidth, cudaMemcpyHostToDevice, streamY);
        scaleTexY << <lumaLP.first, lumaLP.second, 0, streamY >> > (srcWidth, srcHeight, dstWidth, dstHeight, scaleWidthRatio, scaleHeightRatio, scaledDevice);
        cudaMemcpyAsync(fromScalePtrs[0], scaledDevice, dstHeight * dstWidth, cudaMemcpyDeviceToHost, streamY);

        cudaMemcpyToArrayAsync(uSrc, 0, 0, toScalePtrs[1], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice, streamU);
        scaleTexU << <chromaLP.first, chromaLP.second, 0, streamU >> > (srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice, offsetFrom0);
        cudaMemcpyAsync(fromScalePtrs[1], scaledDevice + offsetFrom0, dstHeightChroma * dstWidthChroma, cudaMemcpyDeviceToHost, streamU);

        cudaMemcpyToArrayAsync(vSrc, 0, 0, toScalePtrs[2], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice, streamV);
        scaleTexV << <chromaLP.first, chromaLP.second, 0, streamV >> > (srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice, offsetFrom1);
        cudaMemcpyAsync(fromScalePtrs[2], scaledDevice + offsetFrom1, dstHeightChroma * dstWidthChroma, cudaMemcpyDeviceToHost, streamV);
    } else if(operation == SWS_BICUBIC){
        cudaMemcpyToArrayAsync(ySrc, 0, 0, toScalePtrs[0], srcHeight * srcWidth, cudaMemcpyHostToDevice, streamY);
        cubicScaleY << <lumaLP.first, lumaLP.second, 0, streamY >> > (srcWidth, srcHeight, dstWidth, dstHeight, scaleWidthRatio, scaleHeightRatio, scaledDevice);
        cudaMemcpyAsync(fromScalePtrs[0], scaledDevice, dstHeight * dstWidth, cudaMemcpyDeviceToHost, streamY);

        cudaMemcpyToArrayAsync(uSrc, 0, 0, toScalePtrs[1], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice, streamU);
        cubicScaleU << <chromaLP.first, chromaLP.second, 0, streamU >> > (srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice, offsetFrom0);
        cudaMemcpyAsync(fromScalePtrs[1], scaledDevice + offsetFrom0, dstHeightChroma * dstWidthChroma, cudaMemcpyDeviceToHost, streamU);

        cudaMemcpyToArrayAsync(vSrc, 0, 0, toScalePtrs[2], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice, streamV);
        cubicScaleV << <chromaLP.first, chromaLP.second, 0, streamV >> > (srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice, offsetFrom1);
        cudaMemcpyAsync(fromScalePtrs[2], scaledDevice + offsetFrom1, dstHeightChroma * dstWidthChroma, cudaMemcpyDeviceToHost, streamV);
    }

    // Synchronize device
    cudaDeviceSynchronize();
    stopTime = high_resolution_clock::now();
    times[1] = duration_cast<microseconds>(stopTime - initTime).count() * 1.;

    // Free used resources
    free(toScalePtrs);

    // Format conversion operation
    initTime = high_resolution_clock::now();
    omp_formatConversion(dstWidth, dstHeight, scaleFormat, fromScalePtrs, dstFormat, dst->data);
    stopTime = high_resolution_clock::now();
    times[2] = duration_cast<microseconds>(stopTime - initTime).count() * 1.;

    // Free used resources    
    free(fromScalePtrs);

    // Sucess
    return;
}

// Initializes memory if needed and prepares variables
void cuda_resampleStarter(AVFrame* src, AVFrame* dst, int operation,
    int srcWidth, int srcHeight, int dstWidth, int dstHeight, int srcFormat, int dstFormat,
    double* &times){

    // Check if is only a format conversion
    if(srcWidth == dstWidth && srcHeight == dstHeight){
        // Format conversion operation
        omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, dstFormat, dst->data);
        // End resample operation
        return;
    }

    // Get standard supported pixel format in scaling
    int scaleFormat = getScaleFormat(srcFormat, dstFormat);

    // Calculate the size of the chroma components
    int srcHeightChroma = srcHeight;
    int srcWidthChroma = srcWidth;
    int dstHeightChroma = dstHeight;
    int dstWidthChroma = dstWidth;
    if(scaleFormat == AV_PIX_FMT_YUV422P || scaleFormat == AV_PIX_FMT_YUV420P || scaleFormat == AV_PIX_FMT_YUV422PNORM){
        srcWidthChroma /= 2;
        dstWidthChroma /= 2;
    }
    if(scaleFormat == AV_PIX_FMT_YUV420P){
        srcHeightChroma /= 2;
        dstHeightChroma /= 2;
    }

    // Apply the scaling operation
    cuda_resample_aux(src, dst, operation,
        srcWidth, srcHeight, dstWidth, dstHeight,
        srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma,
        srcFormat, dstFormat, scaleFormat,
        pinnedHost, channelDesc, ySrc, uSrc, vSrc,
        scaledDevice, scaledDeviceSizes, streamY, streamU, streamV,
        times);
}

// Wrapper for the cuda resample operation method
int cuda_resample(AVFrame* src, AVFrame* dst, int operation, double* &times){
    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    int srcFormat = src->format;
    int dstWidth = dst->width, dstHeight = dst->height;
    int dstFormat = dst->format;

    // Verify valid frames
    if(src == nullptr || dst == nullptr){
        cerr << "[CUDA] One or both input frames are null!" << endl;
        return -1;
    }

    // Verify valid input data
    if(!src->data || !src->linesize || !dst->data || !dst->linesize){
        cerr << "[CUDA] Frame data buffers can not be null!" << endl;
        return -1;
    }

    // Verify valid input dimensions
    if(srcWidth < 0 || srcHeight < 0 || dstWidth < 0 || dstHeight < 0){
        cerr << "[CUDA] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }

    // Verify if data is aligned
    if(srcFormat == AV_PIX_FMT_UYVY422 || dstFormat == AV_PIX_FMT_UYVY422){
        if((srcWidth % 4 != 0) || (dstWidth % 4 != 0)){
            cerr << "[CUDA] Can not handle unaligned data!" << endl;
            return -1;
        }
        if((srcHeight % 4 != 0) || (dstHeight % 4 != 0)){
            cerr << "[CUDA] Can not handle unaligned data!" << endl;
            return -1;
        }
    }

    if(srcFormat == AV_PIX_FMT_V210 || dstFormat == AV_PIX_FMT_V210){
        if((srcWidth % 12 != 0) || (dstWidth % 12 != 0)){
            cerr << "[CUDA] Can not handle unaligned data!" << endl;
            return -1;
        }
        if((srcHeight % 12 != 0) || (dstHeight % 12 != 0)){
            cerr << "[CUDA] Can not handle unaligned data!" << endl;
            return -1;
        }
    }

    // Verify valid resize
    if((srcWidth < dstWidth && srcHeight > dstHeight) ||
        (srcWidth > dstWidth && srcHeight < dstHeight)){
        cerr << "[CUDA] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }

    // Verify if supported conversion
    if(!hasSupportedConversion(srcFormat, dstFormat)){
        cerr << "[CUDA] Pixel format conversion is not supported!" << endl;
        return -1;
    }

    // Verify if supported scaling operation
    if(!isSupportedOperation(operation)){
        cerr << "[CUDA] Scaling operation is not supported" << endl;
        return -1;
    }

    // Apply operation
    cuda_resampleStarter(src, dst, operation, srcWidth, srcHeight, dstWidth, dstHeight, srcFormat, dstFormat, times);

    // Return execution time of the scaling operation
    return 0;
}