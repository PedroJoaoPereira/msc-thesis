#include "CUDA_Resample.h"

texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texY;
texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texU;
texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> texV;

// Allocate image channels data buffers depending of the pixel format
void cudaAllocBuffers(uint8_t** &buffer, int* &bufferSize, int width, int height, int pixelFormat){
    // Allocate channel buffer size
    bufferSize = static_cast<int*>(malloc(3 * sizeof(int)));

    // Calculate once
    int wxh = width * height;
    int wxhDiv2 = wxh / 2;
    int wxhDiv4 = wxh / 4;

    // Calculate buffer sizes for each pixel format
    switch(pixelFormat){
    case AV_PIX_FMT_UYVY422:
        bufferSize[0] = wxh * 2;
        bufferSize[1] = 0;
        bufferSize[2] = 0;
        break;
    case AV_PIX_FMT_YUV422P:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDiv2;
        bufferSize[2] = wxhDiv2;
        break;
    case AV_PIX_FMT_YUV420P:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDiv4;
        bufferSize[2] = wxhDiv4;
        break;
    case AV_PIX_FMT_NV12:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDiv2;
        bufferSize[2] = 0;
        break;
    case AV_PIX_FMT_V210:
        bufferSize[0] = height * 128 * ((width + 47) / 48);
        bufferSize[1] = 0;
        bufferSize[2] = 0;
        break;
    case AV_PIX_FMT_YUV422PNORM:
        bufferSize[0] = wxh;
        bufferSize[1] = wxhDiv2;
        bufferSize[2] = wxhDiv2;
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

__host__ __device__
float w0(float a){
    //return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    //return (1.0f / 6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized

    return -0.6f*a*a*a + 1.2f*a*a - 0.6f*a;
}

__host__ __device__
float w1(float a){
    //return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    //return (1.0f / 6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);

    return 1.4f*a*a*a - 2.4f*a*a + 1.0f;
}

__host__ __device__
float w2(float a){
    //return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    //return (1.0f / 6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);

    return -1.4f*a*a*a + 1.8f*a*a + 0.6f*a;
}

__host__ __device__
float w3(float a){
    //return (1.0f / 6.0f)*(a*a*a);

    return a*a*a*0.6f - 0.6f*a*a;
}

__device__ float g0(float a){
    return w0(a) + w1(a);
}

__device__ float g1(float a){
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a){
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + .5f;
}

__device__ float h1(float a){
    return 1.0f + w3(a) / (w2(a) + w3(a)) + .5f;
}

__device__
float cubicFilter(float x, float c0, float c1, float c2, float c3){
    float r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

__global__ void cubicScaleY(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    float linOriginal = (lin + .5f) / scaleHeightRatio;
    float colOriginal = (col + .5f) / scaleWidthRatio;
    
    linOriginal -= .5f;
    colOriginal -= .5f;

    float px = floor(colOriginal);
    float py = floor(linOriginal);
    float fx = colOriginal - px;
    float fy = linOriginal - py;

    float r = cubicFilter(fy,
        cubicFilter(fx, tex2D(texY, px - 1, py - 1), tex2D(texY, px, py - 1), tex2D(texY, px + 1, py - 1), tex2D(texY, px + 2, py - 1)),
        cubicFilter(fx, tex2D(texY, px - 1, py), tex2D(texY, px, py), tex2D(texY, px + 1, py), tex2D(texY, px + 2, py)),
        cubicFilter(fx, tex2D(texY, px - 1, py + 1), tex2D(texY, px, py + 1), tex2D(texY, px + 1, py + 1), tex2D(texY, px + 2, py + 1)),
        cubicFilter(fx, tex2D(texY, px - 1, py + 2), tex2D(texY, px, py + 2), tex2D(texY, px + 1, py + 2), tex2D(texY, px + 2, py + 2))
        );

    /*float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float r = g0(fy) * (g0x * tex2D(texY, px + h0x, py + h0y) +
        g1x * tex2D(texY, px + h1x, py + h0y)) +
        g1(fy) * (g0x * tex2D(texY, px + h0x, py + h1y) +
            g1x * tex2D(texY, px + h1x, py + h1y));*/

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(r * 255.f));
}

__global__ void rescaleY(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = (lin + .5f) / scaleHeightRatio;
    const float colOriginal = (col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(tex2D(texY, colOriginal, linOriginal) * 255.f));
}

__global__ void rescaleU(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = (lin + .5f) / scaleHeightRatio;
    const float colOriginal = (col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(tex2D(texU, colOriginal, linOriginal) * 255.f));
}

__global__ void rescaleV(const int srcWidth, const int srcHeight, const int dstWidth, const int dstHeight,
    const float scaleWidthRatio, const float scaleHeightRatio, uint8_t* dstData){

    // Calculate pixel location
    const int lin = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const int col = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    // Original index coordinates
    const float linOriginal = (lin + .5f) / scaleHeightRatio;
    const float colOriginal = (col + .5f) / scaleWidthRatio;

    // Assign color
    dstData[__mul24(lin, dstWidth) + col] = uint8_t(roundf(tex2D(texV, colOriginal, linOriginal) * 255.f));
}

// Prepares the resample operation
void cuda_resample_aux(AVFrame* src, AVFrame* dst, int operation){
    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    int srcFormat = src->format;
    int dstWidth = dst->width, dstHeight = dst->height;
    int dstFormat = dst->format;

    // Check if is only a format conversion
    bool isOnlyFormatConversion = srcWidth == dstWidth && srcHeight == dstHeight;
    // Changes image pixel format only
    if(isOnlyFormatConversion){
        // Format conversion operation
        omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, dstFormat, dst->data);
        // End resample operation
        return;
    }

    // Get scale ratios
    float scaleHeightRatio = static_cast<float>(dstHeight) / static_cast<float>(srcHeight);
    float scaleWidthRatio = static_cast<float>(dstWidth) / static_cast<float>(srcWidth);

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

    // Temporary buffer
    uint8_t** scaleFormatConverted;
    // Allocate channel buffer pointers
    allocBuffers(scaleFormatConverted, srcWidth, srcHeight, scaleFormat);

    // Format conversion operation
    omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, scaleFormat, scaleFormatConverted);

    // Create channel texture descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // Create a 2d cuda array for each component
    cudaArray *yArray, *uArray, *vArray;
    cudaMallocArray(&yArray, &channelDesc, srcWidth, srcHeight);
    cudaMallocArray(&uArray, &channelDesc, srcWidthChroma, srcHeightChroma);
    cudaMallocArray(&vArray, &channelDesc, srcWidthChroma, srcHeightChroma);

    // Copy components to cuda arrays
    cudaMemcpyToArray(yArray, 0, 0, scaleFormatConverted[0], srcHeight * srcWidth, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(uArray, 0, 0, scaleFormatConverted[1], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(vArray, 0, 0, scaleFormatConverted[2], srcHeightChroma * srcWidthChroma, cudaMemcpyHostToDevice);

    // Bind textures to device memory
    cudaBindTextureToArray(&texY, yArray, &channelDesc);
    cudaBindTextureToArray(&texU, uArray, &channelDesc);
    cudaBindTextureToArray(&texV, vArray, &channelDesc);

    // Set interpolation method
    if(operation == SWS_POINT){
        texY.filterMode = cudaFilterModePoint;
        texU.filterMode = cudaFilterModePoint;
        texV.filterMode = cudaFilterModePoint;
    } else{
        texY.filterMode = cudaFilterModeLinear;
        texU.filterMode = cudaFilterModeLinear;
        texV.filterMode = cudaFilterModeLinear;
    }

    texY.filterMode = cudaFilterModePoint;
    texU.filterMode = cudaFilterModePoint;
    texV.filterMode = cudaFilterModePoint;

    // Free source data
    free2dBuffer(scaleFormatConverted, 3);

    // Create target buffer in device
    uint8_t** scaledDevice;
    int* scaledDeviceSizes;
    // Allocate source buffer in device
    cudaAllocBuffers(scaledDevice, scaledDeviceSizes, dstWidth, dstHeight, scaleFormat);

    // Calculate launch parameters
    pair<dim3, dim3> lumaLP = calculateResizeLP(dstWidth, dstHeight);
    pair<dim3, dim3> chromaLP = calculateResizeLP(dstWidthChroma, dstHeightChroma);

    // Scale each component
    if(operation != SWS_BICUBIC){
        rescaleY << <lumaLP.first, lumaLP.second >> >(srcWidth, srcHeight, dstWidth, dstHeight, scaleWidthRatio, scaleHeightRatio, scaledDevice[0]);
        rescaleU << <chromaLP.first, chromaLP.second >> >(srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice[1]);
        rescaleV << <chromaLP.first, chromaLP.second >> >(srcWidthChroma, srcHeightChroma, dstWidthChroma, dstHeightChroma, scaleWidthRatio, scaleHeightRatio, scaledDevice[2]);
    } else{
        cubicScaleY << <lumaLP.first, lumaLP.second >> >(srcWidth, srcHeight, dstWidth, dstHeight, scaleWidthRatio, scaleHeightRatio, scaledDevice[0]);
    }

    // Free cuda arrays
    cudaFreeArray(yArray);
    cudaFreeArray(uArray);
    cudaFreeArray(vArray);

    // Temporary buffer
    uint8_t** scaleFormatResized;
    // Allocate channel buffer pointers
    allocBuffers(scaleFormatResized, dstWidth, dstHeightChroma, scaleFormat);

    // Copy resulting data from device
    cudaCopyBuffersFromGPU(scaleFormatResized, scaledDevice, scaledDeviceSizes);

    // Free device memory
    freeCudaMemory(scaledDevice);
    free(scaledDeviceSizes);

    // Format conversion operation
    omp_formatConversion(dstWidth, dstHeight, scaleFormat, scaleFormatResized, dstFormat, dst->data);

    // Free last buffer resources
    free2dBuffer(scaleFormatResized, 3);

    // Sucess
    return;
}

// Wrapper for the cuda resample operation method
int cuda_resample(AVFrame* src, AVFrame* dst, int operation){
    // Access once
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

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
    if(src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0){
        cerr << "[CUDA] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }

    // Verify if data is aligned
    if(((src->width % 4 != 0 && srcFormat == AV_PIX_FMT_UYVY422) || (dst->width % 4 != 0 && dstFormat == AV_PIX_FMT_UYVY422)) &&
        ((src->width % 12 != 0 && srcFormat == AV_PIX_FMT_V210) || (dst->width % 12 != 0 && dstFormat == AV_PIX_FMT_V210))){
        cerr << "[CUDA] Can not handle unaligned data!" << endl;
        return -1;
    }

    // Verify valid resize
    if((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)){
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

    // Variables used
    int duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    cuda_resample_aux(src, dst, operation);

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}