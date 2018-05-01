#include "CUDA_Resample.h"

// global declaration of 2D float texture (visible for host and device code)
texture<uint8_t, cudaTextureType2D, cudaReadModeElementType> tex;

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

// Calculate launch parameters of format conversion kernel
pair<dim3, dim3> calculateConversionLP(int width, int height, int srcPixelFormat, int dstPixelFormat){
    // Variable with result launch parameters
    pair<dim3, dim3> result;

    // Discover dimensions value depending of the conversion
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width * 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width / 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width * 2 / 12, height);

    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width / 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width / 2, height);
    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width / 6, height);

    else if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width / 6, height / 2);

    else if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width / 2, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width / 6, height / 2);

    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_UYVY422)
        result.first = dim3(width / 6, height);
    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422P)
        result.first = dim3(width / 6, height);
    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV420P)
        result.first = dim3(width / 6, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_NV12)
        result.first = dim3(width / 6, height / 2);
    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width / 3, height);
    else if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422PNORM)
        result.first = dim3(width / 6, height);

    else if(srcPixelFormat == AV_PIX_FMT_YUV422PNORM && dstPixelFormat == AV_PIX_FMT_V210)
        result.first = dim3(width / 6, height);

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

// Calculate launch parameters of resize kernel
pair<dim3, dim3> calculateResizeLP(int width, int height, int initDivisor){
    // Variable with result launch parameters
    pair<dim3, dim3> result;

    // Dimensions are always the same because only deal with planar formats
    result.first = dim3(width, height);

    // Calculate thread size
    int hDivisor = greatestDivisor(result.first.x, initDivisor);
    int vDivisor = greatestDivisor(result.first.y, initDivisor);

    // Assign thread size
    result.second = dim3(hDivisor, vDivisor);

    // Calculate block size
    result.first.x /= hDivisor;
    result.first.y /= vDivisor;

    return result;
}

// ------------------------------------------------------------------

// Convert the pixel format of the image
void cuda_omp_formatConversion(int width, int height, int srcPixelFormat, uint8_t* srcSlice[], int dstPixelFormat, uint8_t* dstSlice[]){
    #pragma region UYVY422
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideUYVY422 * hStrideUYVY422);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideUYVY422 / 4; hIndex++){
                *dstU++ = *srcB++; // U0
                *dstB++ = *srcB++; // Y0
                *dstV++ = *srcB++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422 * 2;
            auto srcBb = srcB + hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV420P * 2;
            auto dstBb = dstB + hStrideYUV420P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

            for(int hIndex = 0; hIndex < hStrideUYVY422 / 4; hIndex++){
                // Get above line
                uint8_t u0 = *srcB++; // U0
                uint8_t y0 = *srcB++; // Y0
                uint8_t v0 = *srcB++; // V0
                uint8_t y1 = *srcB++; // Y1

                                      // Get below line
                *srcBb++; // U0
                uint8_t y0b = *srcBb++; // Y0
                *srcBb++; // V0
                uint8_t y1b = *srcBb++; // Y1

                                        // Assign above luma values
                *dstB++ = y0;
                *dstB++ = y1;

                // Assign below luma values
                *dstBb++ = y0b;
                *dstBb++ = y1b;

                // Assigne chroma values
                *dstU++ = u0;
                *dstV++ = v0;
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422 * 2;
            auto srcBb = srcB + hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideNV12 * 2;
            auto dstBb = dstB + hStrideNV12;
            auto dstC = dstSlice[1] + vIndex * hStrideNV12;

            for(int hIndex = 0; hIndex < hStrideUYVY422 / 4; hIndex++){
                // Get above line
                uint8_t u0 = *srcB++; // U0
                uint8_t y0 = *srcB++; // Y0
                uint8_t v0 = *srcB++; // V0
                uint8_t y1 = *srcB++; // Y1

                                      // Get below line
                *srcBb++; // U0
                uint8_t y0b = *srcBb++; // Y0
                *srcBb++; // V0
                uint8_t y1b = *srcBb++; // Y1

                                        // Assign above luma values
                *dstB++ = y0;
                *dstB++ = y1;

                // Assign below luma values
                *dstBb++ = y0b;
                *dstBb++ = y1b;

                // Assigne chroma values
                *dstC++ = u0;
                *dstC++ = v0;
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 1x12 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;

            for(int hIndex = 0; hIndex < hStrideUYVY422 / 12; hIndex++){
                // Get components from source
                auto u0 = *srcB++ << 2U; // U0
                auto y0 = *srcB++ << 2U; // Y0
                auto v0 = *srcB++ << 2U; // V0
                auto y1 = *srcB++ << 2U; // Y1

                auto u1 = *srcB++ << 2U; // U1
                auto y2 = *srcB++ << 2U; // Y2
                auto v1 = *srcB++ << 2U; // V1
                auto y3 = *srcB++ << 2U; // Y3

                auto u2 = *srcB++ << 2U; // U2
                auto y4 = *srcB++ << 2U; // Y4
                auto v2 = *srcB++ << 2U; // V2
                auto y5 = *srcB++ << 2U; // Y5

                                         // Assign value
                *dstB++ = (v0 << 20U) | (y0 << 10U) | u0;
                *dstB++ = (y2 << 20U) | (u1 << 10U) | y1;
                *dstB++ = (u2 << 20U) | (y3 << 10U) | v1;
                *dstB++ = (y5 << 20U) | (v2 << 10U) | y4;
            }
        }

        return;
    }
    #pragma endregion

    #pragma region YUV422P
    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 1x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = dstSlice[0] + vIndex * hStrideUYVY422;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV422P / 2; hIndex++){
                *dstB++ = *srcU++; // U0
                *dstB++ = *srcB++; // Y0
                *dstB++ = *srcV++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);
        memcpy(dstSlice[1], srcSlice[1], vStrideYUV422P * hStrideYUV422P / 2);
        memcpy(dstSlice[2], srcSlice[2], vStrideYUV422P * hStrideYUV422P / 2);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

            // Iterate blocks of 2x1 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcV = srcSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcUb = srcU + hStrideYUV422PChroma;
                auto srcVb = srcV + hStrideYUV422PChroma;
                auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
                auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

                for(int hIndex = 0; hIndex < hStrideYUV422PChroma; hIndex++){
                    // Get above chroma values
                    uint8_t u = *srcU++; // U0
                    uint8_t v = *srcV++; // V0

                                         // Get below chroma values
                    uint8_t ub = *srcUb++; // U1
                    uint8_t vb = *srcVb++; // V1

                                           // Assign values
                    *dstU++ = uint8_t(roundFast((static_cast<double>(u) + static_cast<double>(ub)) / 2.));
                    *dstV++ = uint8_t(roundFast((static_cast<double>(v) + static_cast<double>(vb)) / 2.));
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

            // Iterate blocks of 2x1 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcV = srcSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcUb = srcU + hStrideYUV422PChroma;
                auto srcVb = srcV + hStrideYUV422PChroma;
                auto dstC = dstSlice[1] + vIndex * hStrideNV12;

                for(int hIndex = 0; hIndex < hStrideYUV422PChroma; hIndex++){
                    // Get above chroma values
                    uint8_t u = *srcU++; // U0
                    uint8_t v = *srcV++; // V0

                                         // Get below chroma values
                    uint8_t ub = *srcUb++; // U1
                    uint8_t vb = *srcVb++; // V1

                                           // Assign values
                    *dstC++ = uint8_t(roundFast((static_cast<double>(u) + static_cast<double>(ub)) / 2.));
                    *dstC++ = uint8_t(roundFast((static_cast<double>(v) + static_cast<double>(vb)) / 2.));
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 1x6 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV422P / 6; hIndex++){
                // Get components from source
                auto u0 = *srcU++ << 2U; // U0
                auto y0 = *srcB++ << 2U; // Y0
                auto v0 = *srcV++ << 2U; // V0
                auto y1 = *srcB++ << 2U; // Y1

                auto u1 = *srcU++ << 2U; // U1
                auto y2 = *srcB++ << 2U; // Y2
                auto v1 = *srcV++ << 2U; // V1
                auto y3 = *srcB++ << 2U; // Y3

                auto u2 = *srcU++ << 2U; // U2
                auto y4 = *srcB++ << 2U; // Y4
                auto v2 = *srcV++ << 2U; // V2
                auto y5 = *srcB++ << 2U; // Y5

                                         // Assign value
                *dstB++ = (v0 << 20U) | (y0 << 10U) | u0;
                *dstB++ = (y2 << 20U) | (u1 << 10U) | y1;
                *dstB++ = (u2 << 20U) | (y3 << 10U) | v1;
                *dstB++ = (y5 << 20U) | (v2 << 10U) | y4;
            }
        }

        return;
    }
    #pragma endregion

    #pragma region YUV420P
    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto srcBb = srcB + hStrideYUV420P;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideUYVY422;
            auto dstBb = dstB + hStrideUYVY422;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                // Get chroma values
                uint8_t u = *srcU++; // U
                uint8_t v = *srcV++; // V

                                     // Assign above line values
                *dstB++ = u; // U0
                *dstB++ = *srcB++; // Y0
                *dstB++ = v; // V0
                *dstB++ = *srcB++; // Y1

                                   // Assign below line values
                *dstBb++ = u; // U0
                *dstBb++ = *srcBb++; // Y0
                *dstBb++ = v; // V0
                *dstBb++ = *srcBb++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
                auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;
                auto dstU = dstSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstV = dstSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstUb = dstU + hStrideYUV422PChroma;
                auto dstVb = dstV + hStrideYUV422PChroma;

                for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                    // Get chroma values
                    uint8_t u = *srcU++; // U
                    uint8_t v = *srcV++; // V

                                         // Assign values dupicated
                    *dstU++ = u;
                    *dstV++ = v;

                    *dstUb++ = u;
                    *dstVb++ = v;
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);
        memcpy(dstSlice[1], srcSlice[1], vStrideYUV420P * hStrideYUV420P / 4);
        memcpy(dstSlice[2], srcSlice[2], vStrideYUV420P * hStrideYUV420P / 4);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
                auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;
                auto dstC = dstSlice[1] + vIndex * hStrideNV12;

                for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                    *dstC++ = *srcU++; // U
                    *dstC++ = *srcV++; // V
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto srcBb = srcB + hStrideYUV420P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * 2 * hStrideV210;
            auto dstBb = dstB + hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV420P / 6; hIndex++){
                // Get lumas from above line
                auto y0 = *srcB++ << 2U;
                auto y1 = *srcB++ << 2U;
                auto y2 = *srcB++ << 2U;
                auto y3 = *srcB++ << 2U;
                auto y4 = *srcB++ << 2U;
                auto y5 = *srcB++ << 2U;

                // Get lumas from below line
                auto y0b = *srcBb++ << 2U;
                auto y1b = *srcBb++ << 2U;
                auto y2b = *srcBb++ << 2U;
                auto y3b = *srcBb++ << 2U;
                auto y4b = *srcBb++ << 2U;
                auto y5b = *srcBb++ << 2U;

                // Get chroma U
                auto u0 = *srcU++ << 2U;
                auto u1 = *srcU++ << 2U;
                auto u2 = *srcU++ << 2U;

                // Get chroma V
                auto v0 = *srcV++ << 2U;
                auto v1 = *srcV++ << 2U;
                auto v2 = *srcV++ << 2U;

                // Assign above line
                *dstB++ = (v0 << 20U) | (y0 << 10U) | u0;
                *dstB++ = (y2 << 20U) | (u1 << 10U) | y1;
                *dstB++ = (u2 << 20U) | (y3 << 10U) | v1;
                *dstB++ = (y5 << 20U) | (v2 << 10U) | y4;

                // Assign below line
                *dstBb++ = (v0 << 20U) | (y0b << 10U) | u0;
                *dstBb++ = (y2b << 20U) | (u1 << 10U) | y1b;
                *dstBb++ = (u2 << 20U) | (y3b << 10U) | v1;
                *dstBb++ = (y5b << 20U) | (v2 << 10U) | y4b;
            }
        }

        return;
    }
    #pragma endregion

    #pragma region NV12
    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideNV12;
            auto srcBb = srcB + hStrideNV12;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideUYVY422;
            auto dstBb = dstB + hStrideUYVY422;
            auto srcC = srcSlice[1] + vIndex * hStrideNV12;

            for(int hIndex = 0; hIndex < hStrideNV12 / 2; hIndex++){
                // Get chroma values
                uint8_t u = *srcC++; // U
                uint8_t v = *srcC++; // V

                                     // Assign above line values
                *dstB++ = u; // U0
                *dstB++ = *srcB++; // Y0
                *dstB++ = v; // V0
                *dstB++ = *srcB++; // Y1

                                   // Assign below line values
                *dstBb++ = u; // U0
                *dstBb++ = *srcBb++; // Y0
                *dstBb++ = v; // V0
                *dstBb++ = *srcBb++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
                // Discover buffer pointers
                auto srcC = srcSlice[1] + vIndex * hStrideNV12;
                auto dstU = dstSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstV = dstSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstUb = dstU + hStrideYUV422PChroma;
                auto dstVb = dstV + hStrideYUV422PChroma;

                for(int hIndex = 0; hIndex < hStrideNV12 / 2; hIndex++){
                    // Get chroma values
                    uint8_t u = *srcC++; // U
                    uint8_t v = *srcC++; // V

                                         // Assign values dupicated
                    *dstU++ = u;
                    *dstV++ = v;

                    *dstUb++ = u;
                    *dstVb++ = v;
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideNV12; vIndex++){
                // Discover buffer pointers
                auto srcC = srcSlice[1] + vIndex * hStrideNV12 / 2;
                auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 4;
                auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 4;

                for(int hIndex = 0; hIndex < hStrideNV12 / 4; hIndex++){
                    *dstU++ = *srcC++; // U
                    *dstV++ = *srcC++; // V
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);
        memcpy(dstSlice[1], srcSlice[1], vStrideNV12 * hStrideNV12 / 2);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideNV12;
            auto srcBb = srcB + hStrideNV12;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * 2 * hStrideV210;
            auto dstBb = dstB + hStrideV210;
            auto srcC = srcSlice[1] + vIndex * hStrideNV12;

            for(int hIndex = 0; hIndex < hStrideNV12 / 6; hIndex++){
                // Get lumas from above line
                auto y0 = *srcB++ << 2U;
                auto y1 = *srcB++ << 2U;
                auto y2 = *srcB++ << 2U;
                auto y3 = *srcB++ << 2U;
                auto y4 = *srcB++ << 2U;
                auto y5 = *srcB++ << 2U;

                // Get lumas from below line
                auto y0b = *srcBb++ << 2U;
                auto y1b = *srcBb++ << 2U;
                auto y2b = *srcBb++ << 2U;
                auto y3b = *srcBb++ << 2U;
                auto y4b = *srcBb++ << 2U;
                auto y5b = *srcBb++ << 2U;

                // Get chroma U and V
                auto u0 = *srcC++ << 2U;
                auto v0 = *srcC++ << 2U;
                auto u1 = *srcC++ << 2U;
                auto v1 = *srcC++ << 2U;
                auto u2 = *srcC++ << 2U;
                auto v2 = *srcC++ << 2U;

                // Assign above line
                *dstB++ = (v0 << 20U) | (y0 << 10U) | u0;
                *dstB++ = (y2 << 20U) | (u1 << 10U) | y1;
                *dstB++ = (u2 << 20U) | (y3 << 10U) | v1;
                *dstB++ = (y5 << 20U) | (v2 << 10U) | y4;

                // Assign below line
                *dstBb++ = (v0 << 20U) | (y0b << 10U) | u0;
                *dstBb++ = (y2b << 20U) | (u1 << 10U) | y1b;
                *dstBb++ = (u2 << 20U) | (y3b << 10U) | v1;
                *dstBb++ = (y5b << 20U) | (v2 << 10U) | y4b;
            }
        }

        return;
    }
    #pragma endregion

    #pragma region V210
    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideUYVY422;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                auto u0 = (*srcB >> 2U) & 0xFF; // U0
                auto y0 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y0
                auto v0 = ((*srcB >> 2U) >> 20U) & 0xFF; // V0
                *srcB++;

                auto y1 = (*srcB >> 2U) & 0xFF; // Y1
                auto u1 = ((*srcB >> 2U) >> 10U) & 0xFF; // U1
                auto y2 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y2
                *srcB++;

                auto v1 = (*srcB >> 2U) & 0xFF; // V1
                auto y3 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y3
                auto u2 = ((*srcB >> 2U) >> 20U) & 0xFF; // U2
                *srcB++;

                auto y4 = (*srcB >> 2U) & 0xFF; // Y4
                auto v2 = ((*srcB >> 2U) >> 10U) & 0xFF; // V2
                auto y5 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y5
                *srcB++;

                *(dstB++) = u0;
                *(dstB++) = y0;
                *(dstB++) = v0;
                *(dstB++) = y1;

                *(dstB++) = u1;
                *(dstB++) = y2;
                *(dstB++) = v1;
                *(dstB++) = y3;

                *(dstB++) = u2;
                *(dstB++) = y4;
                *(dstB++) = v2;
                *(dstB++) = y5;
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                auto u0 = (*srcB >> 2U) & 0xFF; // U0
                auto y0 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y0
                auto v0 = ((*srcB >> 2U) >> 20U) & 0xFF; // V0
                *srcB++;

                auto y1 = (*srcB >> 2U) & 0xFF; // Y1
                auto u1 = ((*srcB >> 2U) >> 10U) & 0xFF; // U1
                auto y2 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y2
                *srcB++;

                auto v1 = (*srcB >> 2U) & 0xFF; // V1
                auto y3 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y3
                auto u2 = ((*srcB >> 2U) >> 20U) & 0xFF; // U2
                *srcB++;

                auto y4 = (*srcB >> 2U) & 0xFF; // Y4
                auto v2 = ((*srcB >> 2U) >> 10U) & 0xFF; // V2
                auto y5 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y5
                *srcB++;

                *(dstU++) = u0;
                *(dstB++) = y0;
                *(dstV++) = v0;
                *(dstB++) = y1;

                *(dstU++) = u1;
                *(dstB++) = y2;
                *(dstV++) = v1;
                *(dstB++) = y3;

                *(dstU++) = u2;
                *(dstB++) = y4;
                *(dstV++) = v2;
                *(dstB++) = y5;
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * 2 * hStrideV210;
            auto srcBb = srcB + hStrideV210;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto dstBb = dstB + hStrideYUV420P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                // Get above line
                auto u0 = (*srcB >> 2U) & 0xFF; // U0
                auto y0 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y0
                auto v0 = ((*srcB >> 2U) >> 20U) & 0xFF; // V0
                *srcB++;

                auto y1 = (*srcB >> 2U) & 0xFF; // Y1
                auto u1 = ((*srcB >> 2U) >> 10U) & 0xFF; // U1
                auto y2 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y2
                *srcB++;

                auto v1 = (*srcB >> 2U) & 0xFF; // V1
                auto y3 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y3
                auto u2 = ((*srcB >> 2U) >> 20U) & 0xFF; // U2
                *srcB++;

                auto y4 = (*srcB >> 2U) & 0xFF; // Y4
                auto v2 = ((*srcB >> 2U) >> 10U) & 0xFF; // V2
                auto y5 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y5
                *srcB++;

                // Get below line
                auto u0b = (*srcBb >> 2U) & 0xFF; // U0
                auto y0b = ((*srcBb >> 2U) >> 10U) & 0xFF; // Y0
                auto v0b = ((*srcBb >> 2U) >> 20U) & 0xFF; // V0
                *srcBb++;

                auto y1b = (*srcBb >> 2U) & 0xFF; // Y1
                auto u1b = ((*srcBb >> 2U) >> 10U) & 0xFF; // U1
                auto y2b = ((*srcBb >> 2U) >> 20U) & 0xFF; // Y2
                *srcBb++;

                auto v1b = (*srcBb >> 2U) & 0xFF; // V1
                auto y3b = ((*srcBb >> 2U) >> 10U) & 0xFF; // Y3
                auto u2b = ((*srcBb >> 2U) >> 20U) & 0xFF; // U2
                *srcBb++;

                auto y4b = (*srcBb >> 2U) & 0xFF; // Y4
                auto v2b = ((*srcBb >> 2U) >> 10U) & 0xFF; // V2
                auto y5b = ((*srcBb >> 2U) >> 20U) & 0xFF; // Y5
                *srcBb++;

                // Assign above luma values
                *dstB++ = y0;
                *dstB++ = y1;
                *dstB++ = y2;
                *dstB++ = y3;
                *dstB++ = y4;
                *dstB++ = y5;

                // Assign below luma values
                *dstBb++ = y0b;
                *dstBb++ = y1b;
                *dstBb++ = y2b;
                *dstBb++ = y3b;
                *dstBb++ = y4b;
                *dstBb++ = y5b;

                // Assign chroma values
                *dstU++ = uint8_t(roundFast((static_cast<double>(u0) + static_cast<double>(u0b)) / 2.));
                *dstU++ = uint8_t(roundFast((static_cast<double>(u1) + static_cast<double>(u1b)) / 2.));
                *dstU++ = uint8_t(roundFast((static_cast<double>(u2) + static_cast<double>(u2b)) / 2.));

                *dstV++ = uint8_t(roundFast((static_cast<double>(v0) + static_cast<double>(v0b)) / 2.));
                *dstV++ = uint8_t(roundFast((static_cast<double>(v1) + static_cast<double>(v1b)) / 2.));
                *dstV++ = uint8_t(roundFast((static_cast<double>(v2) + static_cast<double>(v2b)) / 2.));
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * 2 * hStrideV210;
            auto srcBb = srcB + hStrideV210;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideNV12;
            auto dstBb = dstB + hStrideNV12;
            auto dstC = dstSlice[1] + vIndex * hStrideNV12;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                // Get above line
                auto u0 = (*srcB >> 2U) & 0xFF; // U0
                auto y0 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y0
                auto v0 = ((*srcB >> 2U) >> 20U) & 0xFF; // V0
                *srcB++;

                auto y1 = (*srcB >> 2U) & 0xFF; // Y1
                auto u1 = ((*srcB >> 2U) >> 10U) & 0xFF; // U1
                auto y2 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y2
                *srcB++;

                auto v1 = (*srcB >> 2U) & 0xFF; // V1
                auto y3 = ((*srcB >> 2U) >> 10U) & 0xFF; // Y3
                auto u2 = ((*srcB >> 2U) >> 20U) & 0xFF; // U2
                *srcB++;

                auto y4 = (*srcB >> 2U) & 0xFF; // Y4
                auto v2 = ((*srcB >> 2U) >> 10U) & 0xFF; // V2
                auto y5 = ((*srcB >> 2U) >> 20U) & 0xFF; // Y5
                *srcB++;

                // Get below line
                auto u0b = (*srcBb >> 2U) & 0xFF; // U0
                auto y0b = ((*srcBb >> 2U) >> 10U) & 0xFF; // Y0
                auto v0b = ((*srcBb >> 2U) >> 20U) & 0xFF; // V0
                *srcBb++;

                auto y1b = (*srcBb >> 2U) & 0xFF; // Y1
                auto u1b = ((*srcBb >> 2U) >> 10U) & 0xFF; // U1
                auto y2b = ((*srcBb >> 2U) >> 20U) & 0xFF; // Y2
                *srcBb++;

                auto v1b = (*srcBb >> 2U) & 0xFF; // V1
                auto y3b = ((*srcBb >> 2U) >> 10U) & 0xFF; // Y3
                auto u2b = ((*srcBb >> 2U) >> 20U) & 0xFF; // U2
                *srcBb++;

                auto y4b = (*srcBb >> 2U) & 0xFF; // Y4
                auto v2b = ((*srcBb >> 2U) >> 10U) & 0xFF; // V2
                auto y5b = ((*srcBb >> 2U) >> 20U) & 0xFF; // Y5
                *srcBb++;

                // Assign above luma values
                *dstB++ = y0;
                *dstB++ = y1;
                *dstB++ = y2;
                *dstB++ = y3;
                *dstB++ = y4;
                *dstB++ = y5;

                // Assign below luma values
                *dstBb++ = y0b;
                *dstBb++ = y1b;
                *dstBb++ = y2b;
                *dstBb++ = y3b;
                *dstBb++ = y4b;
                *dstBb++ = y5b;

                // Assign chroma values
                *dstC++ = uint8_t(roundFast((static_cast<double>(u0) + static_cast<double>(u0b)) / 2.));
                *dstC++ = uint8_t(roundFast((static_cast<double>(v0) + static_cast<double>(v0b)) / 2.));
                *dstC++ = uint8_t(roundFast((static_cast<double>(u1) + static_cast<double>(u1b)) / 2.));
                *dstC++ = uint8_t(roundFast((static_cast<double>(v1) + static_cast<double>(v1b)) / 2.));
                *dstC++ = uint8_t(roundFast((static_cast<double>(u2) + static_cast<double>(u2b)) / 2.));
                *dstC++ = uint8_t(roundFast((static_cast<double>(v2) + static_cast<double>(v2b)) / 2.));
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideV210 * hStrideV210 * sizeof(uint32_t));

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422PNORM){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Create const for normalization
        double constLuma = 219. / 1023.;
        double constChroma = 224. / 1023.;
        double const16 = 16.;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                auto u0 = *srcB & 0x3FF; // U0
                auto y0 = (*srcB >> 10U) & 0x3FF; // Y0
                auto v0 = (*srcB >> 20U) & 0x3FF; // V0
                *srcB++;

                auto y1 = *srcB & 0x3FF; // Y1
                auto u1 = (*srcB >> 10U) & 0x3FF; // U1
                auto y2 = (*srcB >> 20U) & 0x3FF; // Y2
                *srcB++;

                auto v1 = *srcB & 0x3FF; // V1
                auto y3 = (*srcB >> 10U) & 0x3FF; // Y3
                auto u2 = (*srcB >> 20U) & 0x3FF; // U2
                *srcB++;

                auto y4 = *srcB & 0x3FF; // Y4
                auto v2 = (*srcB >> 10U) & 0x3FF; // V2
                auto y5 = (*srcB >> 20U) & 0x3FF; // Y5
                *srcB++;

                *dstU++ = uint8_t(roundFast(static_cast<double>(u0) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y0) * constLuma + const16));
                *dstV++ = uint8_t(roundFast(static_cast<double>(v0) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y1) * constLuma + const16));

                *dstU++ = uint8_t(roundFast(static_cast<double>(u1) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y2) * constLuma + const16));
                *dstV++ = uint8_t(roundFast(static_cast<double>(v1) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y3) * constLuma + const16));

                *dstU++ = uint8_t(roundFast(static_cast<double>(u2) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y4) * constLuma + const16));
                *dstV++ = uint8_t(roundFast(static_cast<double>(v2) * constChroma + const16));
                *dstB++ = uint8_t(roundFast(static_cast<double>(y5) * constLuma + const16));
            }
        }

        return;
    }
    #pragma endregion

    #pragma region YUV422PNORM
    if(srcPixelFormat == AV_PIX_FMT_YUV422PNORM && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Create const for normalization
        double const16 = 16.;
        double constLuma = 1023. / 219.;
        double constChroma = 1023. / 224.;

        // Iterate blocks of 1x6 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV422P / 6; hIndex++){
                // Get components from source
                auto u0n = *srcU++; // U0
                auto y0n = *srcB++; // Y0
                auto v0n = *srcV++; // V0
                auto y1n = *srcB++; // Y1

                auto u1n = *srcU++; // U1
                auto y2n = *srcB++; // Y2
                auto v1n = *srcV++; // V1
                auto y3n = *srcB++; // Y3

                auto u2n = *srcU++; // U2
                auto y4n = *srcB++; // Y4
                auto v2n = *srcV++; // V2
                auto y5n = *srcB++; // Y5

                                    // Denormalize values
                auto v0 = uint16_t(roundFast((static_cast<double>(v0n) - const16) * constChroma)) & 0x3FF;
                auto y0 = uint16_t(roundFast((static_cast<double>(y0n) - const16) * constLuma)) & 0x3FF;
                auto u0 = uint16_t(roundFast((static_cast<double>(u0n) - const16) * constChroma)) & 0x3FF;
                auto y2 = uint16_t(roundFast((static_cast<double>(y2n) - const16) * constLuma)) & 0x3FF;

                auto u1 = uint16_t(roundFast((static_cast<double>(u1n) - const16) * constChroma)) & 0x3FF;
                auto y1 = uint16_t(roundFast((static_cast<double>(y1n) - const16) * constLuma)) & 0x3FF;
                auto u2 = uint16_t(roundFast((static_cast<double>(u2n) - const16) * constChroma)) & 0x3FF;
                auto y3 = uint16_t(roundFast((static_cast<double>(y3n) - const16) * constLuma)) & 0x3FF;

                auto v1 = uint16_t(roundFast((static_cast<double>(v1n) - const16) * constChroma)) & 0x3FF;
                auto y5 = uint16_t(roundFast((static_cast<double>(y5n) - const16) * constLuma)) & 0x3FF;
                auto v2 = uint16_t(roundFast((static_cast<double>(v2n) - const16) * constChroma)) & 0x3FF;
                auto y4 = uint16_t(roundFast((static_cast<double>(y4n) - const16) * constLuma)) & 0x3FF;

                // Assign value
                *dstB++ = (v0 << 20U) | (y0 << 10U) | u0;
                *dstB++ = (y2 << 20U) | (u1 << 10U) | y1;
                *dstB++ = (u2 << 20U) | (y3 << 10U) | v1;
                *dstB++ = (y5 << 20U) | (v2 << 10U) | y4;
            }
        }

        return;
    }
    #pragma endregion
}

// Precalculate coefficients
int cuda_omp_preCalculateCoefficients(int srcSize, int dstSize, int operation, int pixelSupport, double(*coefFunc)(double), float* &preCalculatedCoefs){
    // Calculate size ratio
    double sizeRatio = static_cast<double>(dstSize) / static_cast<double>(srcSize);

    // Calculate once
    double pixelSupportDiv2 = pixelSupport / 2.;
    bool isDownScale = sizeRatio < 1.;
    double regionRadius = isDownScale ? pixelSupportDiv2 / sizeRatio : pixelSupportDiv2;
    double filterStep = isDownScale && operation != SWS_POINT ? 1. / sizeRatio : 1.;
    int numCoefficients = isDownScale ? ceil(pixelSupport / sizeRatio) : pixelSupport;
    int numCoefficientsDiv2 = numCoefficients / 2;

    // Calculate number of lines of coefficients
    int preCalcCoefSize = isDownScale ? (lcm(srcSize, dstSize) / min(srcSize, dstSize)) * (static_cast<double>(srcSize) / static_cast<double>(dstSize)) : lcm(srcSize, dstSize) / min(srcSize, dstSize);

    // Initialize array
    preCalculatedCoefs = static_cast<float*>(malloc(preCalcCoefSize * numCoefficients * sizeof(float)));

    // For each necessary line of coefficients
    for(int col = 0; col < preCalcCoefSize; col++){
        // Calculate once
        int indexOffset = col * numCoefficients;

        // Original line index coordinate
        double colOriginal = (static_cast<double>(col) + .5) / sizeRatio;

        // Discover source limit pixels
        double nearPixel = colOriginal - filterStep;
        double leftPixel = colOriginal - regionRadius;

        // Discover offset to pixel of filter start
        double offset = round(leftPixel) + .5 - leftPixel;
        // Calculate maximum distance to normalize distances
        double maxDistance = colOriginal - nearPixel;
        // Calculate where filtering will start
        double startPosition = leftPixel + offset;

        // Calculate coefficients
        float coefAcc = 0.f;
        for(int index = 0; index < numCoefficients; index++){
            float coefHolder = static_cast<float>(coefFunc((colOriginal - (startPosition + index)) / maxDistance));
            coefAcc += coefHolder;
            preCalculatedCoefs[indexOffset + index] = coefHolder;
        }

        // Avoid lines of coefficients without valid values
        if(operation == SWS_POINT){
            if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == preCalculatedCoefs[indexOffset + numCoefficientsDiv2]){
                if(isDownScale){
                    if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == 0.f && numCoefficients % 2 != 0)
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] = 1.f;
                    else
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = 1.f;
                } else
                    preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = 1.f;
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
__global__ void cuda_resize(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
    float scaleWidthRatio, float scaleHeightRatio, uint8_t* srcData, uint8_t* dstData, float regionHRadius, float regionVRadius, int colorChannel,
    int vCoefsSize, int numVCoefs, float* vCoefs, int hCoefsSize, int numHCoefs, float* hCoefs){

    // Calculate pixel location
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate once coefficients index
    int linCoefOffset = (lin % vCoefsSize) * numVCoefs;
    int colCoefOffset = (col % hCoefsSize) * numHCoefs;

    // Original index coordinates
    float linOriginal = (static_cast<float>(lin) + .5f) / scaleHeightRatio;
    float colOriginal = (static_cast<float>(col) + .5f) / scaleWidthRatio;

    // Discover source limit pixels
    float upperPixel = linOriginal - regionVRadius;
    float leftPixel = colOriginal - regionHRadius;

    // Discover offset to pixel of filter start
    float offsetV = roundf(upperPixel) + .5f - upperPixel;
    float offsetH = roundf(leftPixel) + .5f - leftPixel;

    // Calculate once
    float startLinPosition = upperPixel + offsetV;
    float startColPosition = leftPixel + offsetH;

    // Color accumulator
    float acc = 0.f;
    // Calculate resulting color from coefficients
    for(int indexV = 0; indexV < numVCoefs; indexV++){
        // Get vertical coefficient
        float vCoef = vCoefs[linCoefOffset + indexV];

        // Calculate source pixel line index
        int srcLinIndex = startLinPosition + indexV;
        // Clamp coords
        if(srcLinIndex < 0)
            srcLinIndex = 0;
        else if(srcLinIndex > srcHeight - 1)
            srcLinIndex = srcHeight - 1;

        // Calculate once
        int srcLinIndexOffset = srcLinIndex * srcWidth;

        for(int indexH = 0; indexH < numHCoefs; indexH++){
            // Get horizontal coefficient
            float hCoef = hCoefs[colCoefOffset + indexH];

            // Calculate source pixel column index
            int srcColIndex = startColPosition + indexH;
            // Clamp coords
            if(srcColIndex < 0)
                srcColIndex = 0;
            else if(srcColIndex > srcWidth - 1)
                srcColIndex = srcWidth - 1;

            // Get neighbor pixel color
            uint8_t colorHolder = srcData[srcLinIndexOffset + srcColIndex];

            // Calculate pixel color weight
            float weight = vCoef * hCoef;

            // Weighted color
            acc += colorHolder * weight;
        }
    }

    // Clamp value to avoid undershooting and overshooting
    if(colorChannel == 0){
        if(acc < 16.f)
            acc = 16.f;
        else if(acc > 235.f)
            acc = 235.f;
    } else{
        if(acc < 16.f)
            acc = 16.f;
        else if(acc > 240.f)
            acc = 240.f;
    }

    // Assign calculated color to destiantion data
    dstData[lin * dstWidth + col] = uint8_t(lroundf(acc));
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
    if(isOnlyFormatConversion && false){
        // Format conversion operation
        cuda_omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, dstFormat, dst->data);
        // End resample operation
        return;
    }

    // Get standard supported pixel format in scaling
    int scaleFormat = getScaleFormat(srcFormat, dstFormat);

    // Get scale ratios
    float scaleHeightRatio = static_cast<float>(dstHeight) / static_cast<float>(srcHeight);
    float scaleWidthRatio = static_cast<float>(dstWidth) / static_cast<float>(srcWidth);

    // Needed resources for coefficients calculations
    double(*coefFunc)(double) = getCoefMethod(operation);
    int pixelSupport = getPixelSupport(operation);

    // Calculate once
    float pixelSupportDiv2 = pixelSupport / 2.f;
    bool isDownScaleV = scaleHeightRatio < 1.f;
    bool isDownScaleH = scaleWidthRatio < 1.f;
    float regionVRadius = isDownScaleV ? pixelSupportDiv2 / scaleHeightRatio : pixelSupportDiv2;
    float regionHRadius = isDownScaleH ? pixelSupportDiv2 / scaleWidthRatio : pixelSupportDiv2;
    int numVCoefs = isDownScaleV ? ceil(pixelSupport / scaleHeightRatio) : pixelSupport;
    int numHCoefs = isDownScaleH ? ceil(pixelSupport / scaleWidthRatio) : pixelSupport;

    // Chroma size discovery
    float widthPerc = 1.f;
    float heightPerc = 1.f;
    if(scaleFormat == AV_PIX_FMT_YUV422P || scaleFormat == AV_PIX_FMT_YUV420P || scaleFormat == AV_PIX_FMT_YUV422PNORM)
        widthPerc = .5f;
    if(scaleFormat == AV_PIX_FMT_YUV420P)
        heightPerc = .5f;

    // Precalculate coefficients
    float* vCoefsHost;
    int vCoefsSize = cuda_omp_preCalculateCoefficients(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefsHost);
    float* hCoefsHost;
    int hCoefsSize = cuda_omp_preCalculateCoefficients(srcWidth, dstWidth, operation, pixelSupport, coefFunc, hCoefsHost);

    // Allocate coefficients buffer in device
    float *vCoefsDevice, *hCoefsDevice;
    cudaMalloc((void **) &vCoefsDevice, vCoefsSize * numVCoefs * sizeof(float));
    cudaMalloc((void **) &hCoefsDevice, hCoefsSize * numHCoefs * sizeof(float));

    // Copy coefficients to device
    cudaMemcpy(vCoefsDevice, vCoefsHost, vCoefsSize * numVCoefs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(hCoefsDevice, hCoefsHost, hCoefsSize * numHCoefs * sizeof(float), cudaMemcpyHostToDevice);

    // Free host coefficients
    free(vCoefsHost);
    free(hCoefsHost);

    // Temporary buffer
    uint8_t** forScalePointersHost;
    // Allocate channel buffer pointers
    allocBuffers(forScalePointersHost, srcWidth, srcHeight, scaleFormat);

    // Resamples image to a supported format
    cuda_omp_formatConversion(srcWidth, srcHeight, srcFormat, src->data, scaleFormat, forScalePointersHost);

    // Create target buffer in device
    uint8_t** forScalePointersDevice;
    int* forScalePointersDeviceSizes;
    // Allocate source buffer in device
    cudaAllocBuffers(forScalePointersDevice, forScalePointersDeviceSizes, srcWidth, srcHeight, scaleFormat);

    // Copy source data to device
    cudaCopyBuffersToGPU(forScalePointersHost, forScalePointersDevice, forScalePointersDeviceSizes);

    // Free host memory
    free2dBuffer(forScalePointersHost, 3);

    // Create target buffer in device
    uint8_t** fromScalePointersDevice;
    int* fromScalePointersDeviceSizes;
    // Allocate source buffer in device
    cudaAllocBuffers(fromScalePointersDevice, fromScalePointersDeviceSizes, dstWidth, dstHeight, scaleFormat);

    // Create launch parameters of resize kernel
    pair<dim3, dim3> resizeLP = calculateResizeLP(dstWidth, dstHeight, 32);
    // Recalculate launch parameters for chromas
    int heightChroma = static_cast<int>(dstHeight * heightPerc);
    int widthChroma = static_cast<int>(dstWidth * widthPerc);
    pair<dim3, dim3> resizeChromaLP = calculateResizeLP(widthChroma, heightChroma, 32);

    // Apply the resizing operation to luma channel
    cuda_resize << <resizeLP.first, resizeLP.second>> > (srcWidth, srcHeight, dstWidth, dstHeight,
        scaleWidthRatio, scaleHeightRatio, forScalePointersDevice[0], fromScalePointersDevice[0], regionHRadius, regionVRadius, 0,
        vCoefsSize, numVCoefs, vCoefsDevice, hCoefsSize, numHCoefs, hCoefsDevice);

    // Apply the resizing operation to U chroma channel
    cuda_resize << <resizeChromaLP.first, resizeChromaLP.second>> > (static_cast<int>(srcWidth * widthPerc), static_cast<int>(srcHeight * heightPerc), widthChroma, heightChroma,
        scaleWidthRatio, scaleHeightRatio, forScalePointersDevice[1], fromScalePointersDevice[1], regionHRadius, regionVRadius, 1,
        vCoefsSize, numVCoefs, vCoefsDevice, hCoefsSize, numHCoefs, hCoefsDevice);

    // Apply the resizing operation to V chroma channel
    cuda_resize << <resizeChromaLP.first, resizeChromaLP.second >> > (static_cast<int>(srcWidth * widthPerc), static_cast<int>(srcHeight * heightPerc), widthChroma, heightChroma,
        scaleWidthRatio, scaleHeightRatio, forScalePointersDevice[2], fromScalePointersDevice[2], regionHRadius, regionVRadius, 2,
        vCoefsSize, numVCoefs, vCoefsDevice, hCoefsSize, numHCoefs, hCoefsDevice);

    // Synchronize GPU
    cudaDeviceSynchronize();

    // Free used data resources
    freeCudaMemory(forScalePointersDevice);
    free(forScalePointersDeviceSizes);

    // Temporary buffer
    uint8_t** fromScalePointersHost;
    // Allocate channel buffer pointers
    allocBuffers(fromScalePointersHost, dstWidth, dstHeight, scaleFormat);

    // Copy resulting data from device
    cudaCopyBuffersFromGPU(fromScalePointersHost, fromScalePointersDevice, fromScalePointersDeviceSizes);

    // Free used data resources
    freeCudaMemory(fromScalePointersDevice);
    free(fromScalePointersDeviceSizes);

    // Free coefficients in device
    cudaFree(vCoefsDevice);
    cudaFree(hCoefsDevice);

    // Resamples image to target format
    cuda_omp_formatConversion(dstWidth, dstHeight, scaleFormat, fromScalePointersHost, dstFormat, dst->data);

    // Free used resources
    free2dBuffer(fromScalePointersHost, 3);

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