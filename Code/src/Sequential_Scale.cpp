#include "Sequential_Scale.h"

// Modify the color model of the image
template <class PrecisionType>
int sequential_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
    int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]) {

    // If same formats no need to resample
    if (srcPixelFormat == dstPixelFormat) {
        // Copy data between buffers
        if(srcPixelFormat == AV_PIX_FMT_NONE) // V210
            memcpy(dstSlice[0], srcSlice[0], ((srcWidth + 47) / 48) * 128 * srcHeight);
        else
            memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight);
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight);
        memcpy(dstSlice[3], srcSlice[3], srcStride[3] * srcHeight);

        // Success
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Number of elements
        long numElements = srcStride[0] * srcHeight / 4;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto dstBuffer = dstSlice[0];
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            PrecisionType u0 = static_cast<PrecisionType>(*srcBuffer++); // U0
            PrecisionType y0 = static_cast<PrecisionType>(*srcBuffer++); // Y0
            PrecisionType v0 = static_cast<PrecisionType>(*srcBuffer++); // V0
            PrecisionType y1 = static_cast<PrecisionType>(*srcBuffer++); // Y1

            *dstBuffer++ = y0;
            *dstBuffer++ = y1;

            *dstBufferChromaU++ = u0;
            *dstBufferChromaV++ = v0;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Access once
        int stride = srcStride[0];

        // Calculate once
        int heightDiv2 = srcHeight / 2;
        int strideDiv4 = stride / 4;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto srcBufferBelow = srcBuffer + stride;
        auto dstBuffer = dstSlice[0];
        auto dstBufferBelow = dstBuffer + srcWidth;
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Loop through each pixel
        for(int lin = 0; lin < heightDiv2; lin++){
            for(int col = 0; col < strideDiv4; col++){
                PrecisionType u0 = static_cast<PrecisionType>(*srcBuffer++); // U0
                PrecisionType y0 = static_cast<PrecisionType>(*srcBuffer++); // Y0
                PrecisionType v0 = static_cast<PrecisionType>(*srcBuffer++); // V0
                PrecisionType y1 = static_cast<PrecisionType>(*srcBuffer++); // Y1

                srcBufferBelow++;
                PrecisionType y2 = static_cast<PrecisionType>(*srcBufferBelow++); // Y2
                srcBufferBelow++;
                PrecisionType y3 = static_cast<PrecisionType>(*srcBufferBelow++); // Y3

                *dstBuffer++ = y0;
                *dstBuffer++ = y1;

                *dstBufferBelow++ = y2;
                *dstBufferBelow++ = y3;

                *dstBufferChromaU++ = u0;
                *dstBufferChromaV++ = v0;
            }

            srcBuffer += stride;
            srcBufferBelow += stride;

            dstBuffer += srcWidth;
            dstBufferBelow += srcWidth;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Access once
        int stride = srcStride[0];

        // Calculate once
        int heightDiv2 = srcHeight / 2;
        int strideDiv4 = stride / 4;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto srcBufferBelow = srcBuffer + stride;
        auto dstBuffer = dstSlice[0];
        auto dstBufferBelow = dstBuffer + srcWidth;
        auto dstBufferChroma = dstSlice[1];

        // Loop through each pixel
        for(int lin = 0; lin < heightDiv2; lin++){
            for(int col = 0; col < strideDiv4; col++){
                PrecisionType u0 = static_cast<PrecisionType>(*srcBuffer++); // U0
                PrecisionType y0 = static_cast<PrecisionType>(*srcBuffer++); // Y0
                PrecisionType v0 = static_cast<PrecisionType>(*srcBuffer++); // V0
                PrecisionType y1 = static_cast<PrecisionType>(*srcBuffer++); // Y1

                srcBufferBelow++;
                PrecisionType y2 = static_cast<PrecisionType>(*srcBufferBelow++); // Y2
                srcBufferBelow++;
                PrecisionType y3 = static_cast<PrecisionType>(*srcBufferBelow++); // Y3

                *dstBuffer++ = y0;
                *dstBuffer++ = y1;

                *dstBufferBelow++ = y2;
                *dstBufferBelow++ = y3;

                *dstBufferChroma++ = u0;
                *dstBufferChroma++ = v0;
            }

            srcBuffer += stride;
            srcBufferBelow += stride;

            dstBuffer += srcWidth;
            dstBufferBelow += srcWidth;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NONE){ // V210
        // Number of elements
        long numElements = srcStride[0] * srcHeight / 12;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto dstBuffer = reinterpret_cast<uint32_t*>(dstSlice[0]);

        // Assign once
        enum{ SHIFT_8TO10B = 2U, SHIFT_LEFT = 20U, SHIFT_MIDDLE = 10U, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0 = *srcBuffer++ << SHIFT_8TO10B; // U0
            auto y0 = *srcBuffer++ << SHIFT_8TO10B; // Y0
            auto v0 = *srcBuffer++ << SHIFT_8TO10B; // V0
            auto y1 = *srcBuffer++ << SHIFT_8TO10B; // Y1

            auto u1 = *srcBuffer++ << SHIFT_8TO10B; // U1
            auto y2 = *srcBuffer++ << SHIFT_8TO10B; // Y2
            auto v1 = *srcBuffer++ << SHIFT_8TO10B; // V1
            auto y3 = *srcBuffer++ << SHIFT_8TO10B; // Y3

            auto u2 = *srcBuffer++ << SHIFT_8TO10B; // U2
            auto y4 = *srcBuffer++ << SHIFT_8TO10B; // Y4
            auto v2 = *srcBuffer++ << SHIFT_8TO10B; // V2
            auto y5 = *srcBuffer++ << SHIFT_8TO10B; // Y5

            *dstBuffer++ = (v0 << SHIFT_LEFT) | (y0 << SHIFT_MIDDLE) | u0;
            *dstBuffer++ = (y2 << SHIFT_LEFT) | (u1 << SHIFT_MIDDLE) | y1;
            *dstBuffer++ = (u2 << SHIFT_LEFT) | (y3 << SHIFT_MIDDLE) | v1;
            *dstBuffer++ = (y5 << SHIFT_LEFT) | (v2 << SHIFT_MIDDLE) | y4;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Number of elements
        long numElements = srcStride[0] * srcHeight / 2;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto srcBufferChromaU = srcSlice[1];
        auto srcBufferChromaV = srcSlice[2];
        auto dstBuffer = dstSlice[0];

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            *dstBuffer++ = *srcBufferChromaU++; // U0
            *dstBuffer++ = *srcBuffer++; // Y0
            *dstBuffer++ = *srcBufferChromaV++; // V0
            *dstBuffer++ = *srcBuffer++; // Y1
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Access once
        int stride = srcStride[0];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], stride * srcHeight);

        // Calculate once
        int heightDiv2 = srcHeight / 2;
        int strideDiv2 = stride / 2;

        // Buffer pointers
        auto srcBufferChromaU = srcSlice[1];
        auto srcBufferChromaV = srcSlice[2];
        auto srcBufferChromaUBelow = srcBufferChromaU + strideDiv2;
        auto srcBufferChromaVBelow = srcBufferChromaV + strideDiv2;
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Loop through each pixel
        for(int lin = 0; lin < heightDiv2; lin++){
            for(int col = 0; col < strideDiv2; col++){
                PrecisionType u0 = static_cast<PrecisionType>(*srcBufferChromaU++); // U0
                PrecisionType v0 = static_cast<PrecisionType>(*srcBufferChromaV++); // V0

                PrecisionType u1 = static_cast<PrecisionType>(*srcBufferChromaUBelow++); // U1
                PrecisionType v1 = static_cast<PrecisionType>(*srcBufferChromaVBelow++); // V1

                *dstBufferChromaU++ = roundTo<PrecisionType>((u0 + u1) / static_cast<PrecisionType>(2.));
                *dstBufferChromaV++ = roundTo<PrecisionType>((v0 + v1) / static_cast<PrecisionType>(2.));
            }

            srcBufferChromaU += strideDiv2;
            srcBufferChromaV += strideDiv2;
            srcBufferChromaUBelow += strideDiv2;
            srcBufferChromaVBelow += strideDiv2;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Access once
        int stride = srcStride[0];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], stride * srcHeight);

        // Calculate once
        int heightDiv2 = srcHeight / 2;
        int strideDiv2 = stride / 2;

        // Buffer pointers
        auto srcBufferChromaU = srcSlice[1];
        auto srcBufferChromaV = srcSlice[2];
        auto srcBufferChromaUBelow = srcBufferChromaU + strideDiv2;
        auto srcBufferChromaVBelow = srcBufferChromaV + strideDiv2;
        auto dstBufferChroma = dstSlice[1];

        // Loop through each pixel
        for(int lin = 0; lin < heightDiv2; lin++){
            for(int col = 0; col < strideDiv2; col++){
                PrecisionType u0 = static_cast<PrecisionType>(*srcBufferChromaU++); // U0
                PrecisionType v0 = static_cast<PrecisionType>(*srcBufferChromaV++); // V0

                PrecisionType u1 = static_cast<PrecisionType>(*srcBufferChromaUBelow++); // U1
                PrecisionType v1 = static_cast<PrecisionType>(*srcBufferChromaVBelow++); // V1

                *dstBufferChroma++ = roundTo<PrecisionType>((u0 + u1) / static_cast<PrecisionType>(2.));
                *dstBufferChroma++ = roundTo<PrecisionType>((v0 + v1) / static_cast<PrecisionType>(2.));
            }

            srcBufferChromaU += strideDiv2;
            srcBufferChromaV += strideDiv2;
            srcBufferChromaUBelow += strideDiv2;
            srcBufferChromaVBelow += strideDiv2;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NONE){ // V210
        // Number of elements
        long numElements = srcStride[0] * srcHeight / 6;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto srcBufferChromaU = srcSlice[1];
        auto srcBufferChromaV = srcSlice[2];
        auto dstBuffer = reinterpret_cast<uint32_t*>(dstSlice[0]);

        // Assign once
        enum{ SHIFT_8TO10B = 2U, SHIFT_LEFT = 20U, SHIFT_MIDDLE = 10U, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0 = *srcBufferChromaU++ << SHIFT_8TO10B; // U0
            auto y0 = *srcBuffer++ << SHIFT_8TO10B; // Y0
            auto v0 = *srcBufferChromaV++ << SHIFT_8TO10B; // V0
            auto y1 = *srcBuffer++ << SHIFT_8TO10B; // Y1

            auto u1 = *srcBufferChromaU++ << SHIFT_8TO10B; // U1
            auto y2 = *srcBuffer++ << SHIFT_8TO10B; // Y2
            auto v1 = *srcBufferChromaV++ << SHIFT_8TO10B; // V1
            auto y3 = *srcBuffer++ << SHIFT_8TO10B; // Y3

            auto u2 = *srcBufferChromaU++ << SHIFT_8TO10B; // U2
            auto y4 = *srcBuffer++ << SHIFT_8TO10B; // Y4
            auto v2 = *srcBufferChromaV++ << SHIFT_8TO10B; // V2
            auto y5 = *srcBuffer++ << SHIFT_8TO10B; // Y5

            *dstBuffer++ = (v0 << SHIFT_LEFT) | (y0 << SHIFT_MIDDLE) | u0;
            *dstBuffer++ = (y2 << SHIFT_LEFT) | (u1 << SHIFT_MIDDLE) | y1;
            *dstBuffer++ = (u2 << SHIFT_LEFT) | (y3 << SHIFT_MIDDLE) | v1;
            *dstBuffer++ = (y5 << SHIFT_LEFT) | (v2 << SHIFT_MIDDLE) | y4;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_NONE && dstPixelFormat == AV_PIX_FMT_UYVY422){ // V210
        // Number of elements
        long numElements = ((srcWidth + 47) / 48) * 128 * srcHeight / 16;

        // Buffer pointers
        auto srcBuffer = reinterpret_cast<uint32_t*>(srcSlice[0]);
        auto dstBuffer = dstSlice[0];

        // Assign once
        enum{ SHIFT_10TO8B = 2U, SHIFT_RIGHT = 20U, SHIFT_MIDDLE = 10U, XFF = 0xFF, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // U0
            auto y0 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // Y0
            auto v0 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // V0
            *srcBuffer++;

            auto y1 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // Y1
            auto u1 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // U1
            auto y2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // Y2
            *srcBuffer++;

            auto v1 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // V1
            auto y3 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // Y3
            auto u2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // U2
            *srcBuffer++;

            auto y4 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // Y4
            auto v2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // V2
            auto y5 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // Y5
            *srcBuffer++;

            *(dstBuffer++) = u0;
            *(dstBuffer++) = y0;
            *(dstBuffer++) = v0;
            *(dstBuffer++) = y1;

            *(dstBuffer++) = u1;
            *(dstBuffer++) = y2;
            *(dstBuffer++) = v1;
            *(dstBuffer++) = y3;

            *(dstBuffer++) = u2;
            *(dstBuffer++) = y4;
            *(dstBuffer++) = v2;
            *(dstBuffer++) = y5;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_NONE && dstPixelFormat == AV_PIX_FMT_YUV422P){ // V210
        // Number of elements
        long numElements = ((srcWidth + 47) / 48) * 128 * srcHeight / 16;

        // Buffer pointers
        auto srcBuffer = reinterpret_cast<uint32_t*>(srcSlice[0]);
        auto dstBuffer = dstSlice[0];
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Assign once
        enum{ SHIFT_10TO8B = 2U, SHIFT_RIGHT = 20U, SHIFT_MIDDLE = 10U, XFF = 0xFF, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // U0
            auto y0 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // Y0
            auto v0 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // V0
            *srcBuffer++;

            auto y1 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // Y1
            auto u1 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // U1
            auto y2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // Y2
            *srcBuffer++;

            auto v1 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // V1
            auto y3 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // Y3
            auto u2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // U2
            *srcBuffer++;

            auto y4 = (*srcBuffer >> SHIFT_10TO8B) & XFF; // Y4
            auto v2 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_MIDDLE) & XFF; // V2
            auto y5 = ((*srcBuffer >> SHIFT_10TO8B) >> SHIFT_RIGHT) & XFF; // Y5
            *srcBuffer++;

            *(dstBufferChromaU++) = u0;
            *(dstBuffer++) = y0;
            *(dstBufferChromaV++) = v0;
            *(dstBuffer++) = y1;

            *(dstBufferChromaU++) = u1;
            *(dstBuffer++) = y2;
            *(dstBufferChromaV++) = v1;
            *(dstBuffer++) = y3;

            *(dstBufferChromaU++) = u2;
            *(dstBuffer++) = y4;
            *(dstBufferChromaV++) = v2;
            *(dstBuffer++) = y5;
        }

        // Success
        return 0;
    }

    // No conversion was supported
    return -1;
}

// Precalculate coefficients
template <class PrecisionType>
int sequential_preCalculateCoefficients(int srcSize, int dstSize, int operation, int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType** &preCalculatedCoefs) {
    // Calculate size ratio
    PrecisionType sizeRatio = static_cast<PrecisionType>(dstSize) / static_cast<PrecisionType>(srcSize);

    // Check if is downscale or upscale
    bool isDownScale = sizeRatio < static_cast<PrecisionType>(1.);
    int pixelSupportDiv2 = pixelSupport / 2;
    PrecisionType filterStep = static_cast<PrecisionType>(1.);
    if(isDownScale && operation != SWS_POINT){
        filterStep = 1. / (ceil((pixelSupport / 2.) / sizeRatio) / (pixelSupport / 2.));
    }
    
    // Calculate number of lines of coefficients
    int preCalcCoefSize = lcm(srcSize, dstSize) / min<int>(srcSize, dstSize);

    // Initialize 2d array
    preCalculatedCoefs = static_cast<PrecisionType**>(malloc(preCalcCoefSize * sizeof(PrecisionType*)));
    for(int index = 0; index < preCalcCoefSize; index++)
        preCalculatedCoefs[index] = static_cast<PrecisionType*>(malloc(pixelSupport * sizeof(PrecisionType)));

    // For each necessary line of coefficients
    for (int lin = 0; lin < preCalcCoefSize; lin++) {
        // Original line index coordinate
        PrecisionType linOriginal = (static_cast<PrecisionType>(lin) + static_cast<PrecisionType>(0.5)) / sizeRatio - static_cast<PrecisionType>(0.5);
        // Calculate nearest original position
        int linNearest = floor(linOriginal);

        // Calculate distance to left nearest pixel
        PrecisionType dist = linOriginal - static_cast<PrecisionType>(linNearest);
        // Calculate distance to original pixels
        PrecisionType upperCoef = dist;
        PrecisionType bottomtCoef = static_cast<PrecisionType>(1.) - dist;

        // Calculate coefficients
        for (int index = 0; index < pixelSupportDiv2; index++) {
            preCalculatedCoefs[lin][pixelSupportDiv2 - index - 1] = coefFunc((upperCoef + index) * filterStep);
            preCalculatedCoefs[lin][index + pixelSupportDiv2] = coefFunc((bottomtCoef + index) * filterStep);

            if(sizeRatio < static_cast<PrecisionType>(1.) && operation == SWS_POINT)
                if(preCalculatedCoefs[lin][pixelSupportDiv2 - index - 1] == preCalculatedCoefs[lin][index + pixelSupportDiv2])
                    preCalculatedCoefs[lin][index + pixelSupportDiv2] = static_cast<PrecisionType>(1.);
        }
    }

    // Success
    return preCalcCoefSize;
}

// Apply resizing operation
template <class PrecisionType>
void sequential_resize(int srcWidth, int srcHeight, uint8_t* srcData,
    int dstWidth, int dstHeight, uint8_t* dstData, int operation,
    int pixelSupport, int vCoefsSize, PrecisionType** vCoefs, int hCoefsSize, PrecisionType** hCoefs) {

    // Get scale ratios
    PrecisionType scaleHeightRatio = static_cast<PrecisionType>(dstHeight) / static_cast<PrecisionType>(srcHeight);
    PrecisionType scaleWidthRatio = static_cast<PrecisionType>(dstWidth) / static_cast<PrecisionType>(srcWidth);

    // Calculate once
    int pixelSupportDiv2 = pixelSupport / 2;

    // Iterate through each line of the scaled image
    for (int lin = 0; lin < dstHeight; lin++) {
        // Calculate once the target line coordinate
        int targetLine = lin * dstWidth;

        // Original line index coordinate
        PrecisionType linOriginal = (static_cast<PrecisionType>(lin) + static_cast<PrecisionType>(0.5)) / scaleHeightRatio - static_cast<PrecisionType>(0.5);

        // Calculate nearest original position
        int linNearest = floor(linOriginal);
        // Calculate limit positions
        int linStart = linNearest - pixelSupportDiv2 + 1;
        int linStop = linStart + pixelSupport - 1;

        // Iterate through each column of the scaled image
        for (int col = 0; col < dstWidth; col++) {
            // Original column index coordinate
            PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(0.5)) / scaleWidthRatio - static_cast<PrecisionType>(0.5);

            // Calculate nearest original position
            int colNearest = floor(colOriginal);
            // Calculate limit positions
            int colStart = colNearest - pixelSupportDiv2 + 1;
            int colStop = colStart + pixelSupport - 1;

            // Temporary variables used in the interpolation
            PrecisionType colorAcc = static_cast<PrecisionType>(0.);
            PrecisionType weightAcc = static_cast<PrecisionType>(0.);

            // Calculate resulting color from coefficients
            for (int linTemp = linStart; linTemp <= linStop; linTemp++) {
                // Access once the memory
                PrecisionType vCoef = vCoefs[lin % vCoefsSize][linTemp - linStart];

                for (int colTemp = colStart; colTemp <= colStop; colTemp++) {
                    // Access once the memory
                    PrecisionType hCoef = hCoefs[col % hCoefsSize][colTemp - colStart];

                    // Get pixel from source data
                    uint8_t colorHolder = getPixel(linTemp, colTemp, srcWidth, srcHeight, srcData);

                    // Calculate pixel color weight
                    PrecisionType weight = vCoef * hCoef;

                    // Weights neighboring pixel and add it to the result
                    colorAcc += static_cast<PrecisionType>(colorHolder) * weight;
                    weightAcc += weight;
                }
            }

            // Calculate resulting color
            PrecisionType result = colorAcc / weightAcc;
            // Clamp value to avoid undershooting and overshooting
            clamp<PrecisionType>(result, static_cast<PrecisionType>(0.), static_cast<PrecisionType>(255.));
            // Assign calculated color to destiantion data
            dstData[targetLine + col] = roundTo<PrecisionType>(result);
        }
    }
}

// Prepares the scaling operation
template <class PrecisionType>
int sequential_scale_aux(AVFrame* src, AVFrame* dst, int operation) {

    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    int dstWidth = dst->width, dstHeight = dst->height;
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Check if is only a resample operation
    bool isOnlyResample = false;
    if (srcWidth == dstWidth && srcHeight == dstHeight)
        isOnlyResample = true;

    // Initialize needed variables if it is a scaling operation
    AVPixelFormat scalingSupportedFormat = getTempScaleFormat(srcFormat);

#pragma region INITIALIZE TEMPORARY FRAMES
    // Temporary frames used in intermediate operations
    uint8_t* resampleBuffer, *scaleBuffer;
    AVFrame* resampleFrame, *scaleFrame;

    // Only initializes frames if is not only a resample opreration
    if (!isOnlyResample) {
        // Initialize temporary frame buffers
        if (createImageDataBuffer(srcWidth, srcHeight, scalingSupportedFormat, &resampleBuffer) < 0)
            return -1;
        if (createImageDataBuffer(dstWidth, dstHeight, scalingSupportedFormat, &scaleBuffer) < 0) {
            free(resampleBuffer);
            return -1;
        }

        // Initialize temporary frames
        if (initializeAVFrame(&resampleBuffer, srcWidth, srcHeight, scalingSupportedFormat, &resampleFrame) < 0) {
            free(resampleBuffer);
            free(scaleBuffer);
            return -1;
        }
        if (initializeAVFrame(&scaleBuffer, dstWidth, dstHeight, scalingSupportedFormat, &scaleFrame) < 0) {
            av_frame_free(&resampleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
            return -1;
        }
    }
#pragma endregion

    // Last resample frame
    AVFrame* lastResampleFrame = src;
    AVPixelFormat lastResamplePixelFormat = srcFormat;

#pragma region RESIZE OPERATION
    // Verify if is not only a resample operation
    if (!isOnlyResample) {
        // Resamples image to a supported format
        if (sequential_resampler<PrecisionType>(srcWidth, srcHeight, srcFormat, src->data, src->linesize,
            srcWidth, srcHeight, scalingSupportedFormat, resampleFrame->data, resampleFrame->linesize) < 0) {
            av_frame_free(&resampleFrame);
            av_frame_free(&scaleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
            return -2;
        }

        // Temporary variables for precalculation of coefficients
        PrecisionType(*coefFunc)(PrecisionType) = getCoefMethod<PrecisionType>(operation);
        int pixelSupportV = getPixelSupport(operation, max<int>(round((static_cast<PrecisionType>(srcHeight) / static_cast<PrecisionType>(dstHeight)) / static_cast<PrecisionType>(2.)) * 2, 1));
        int pixelSupportH = getPixelSupport(operation, max<int>(round((static_cast<PrecisionType>(srcWidth) / static_cast<PrecisionType>(dstWidth)) / static_cast<PrecisionType>(2.)) * 2, 1));
        int pixelSupport = max<int>(pixelSupportV, pixelSupportH);

        // Create variables for precalculated coefficients
        PrecisionType** vCoefs;
        int vCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefs);
        PrecisionType** hCoefs;
        int hCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcWidth, dstWidth, operation, pixelSupport, coefFunc, hCoefs);

        // Calculate the chroma size depending on the source data pixel format
        float tempWidthRatio = 1.f;
        float tempHeightRatio = 1.f;
        if (scalingSupportedFormat == AV_PIX_FMT_YUV422P || scalingSupportedFormat == AV_PIX_FMT_YUV420P)
            tempWidthRatio = 0.5f;
        if (scalingSupportedFormat == AV_PIX_FMT_YUV420P)
            tempHeightRatio = 0.5f;

        // Apply the resizing operation to luma channel
        sequential_resize<PrecisionType>(srcWidth, srcHeight, resampleFrame->data[0],
            dstWidth, dstHeight, scaleFrame->data[0], operation,
            pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs);

        // Apply the resizing operation to chroma channels
        for (int colorChannel = 1; colorChannel < 3; colorChannel++) {
            sequential_resize<PrecisionType>(static_cast<int>(srcWidth * tempWidthRatio), static_cast<int>(srcHeight * tempHeightRatio), resampleFrame->data[colorChannel],
                static_cast<int>(dstWidth * tempWidthRatio), static_cast<int>(dstHeight * tempHeightRatio), scaleFrame->data[colorChannel], operation,
                pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs);
        }

        // Free used resources
        av_frame_free(&resampleFrame);
        free(resampleBuffer);
        for (int i = 0; i < vCoefsSize; i++)
            free(vCoefs[i]);
        for (int i = 0; i < hCoefsSize; i++)
            free(hCoefs[i]);
        free(vCoefs);
        free(hCoefs);

        // Assign correct values to apply last resample
        lastResampleFrame = scaleFrame;
        lastResamplePixelFormat = scalingSupportedFormat;
    }
#pragma endregion

#pragma region LAST RESAMPLE
    // Last resample to destination frame
    if (sequential_resampler<PrecisionType>(dstWidth, dstHeight, lastResamplePixelFormat, lastResampleFrame->data, lastResampleFrame->linesize,
        dstWidth, dstHeight, dstFormat, dst->data, dst->linesize) < 0) {
        if (!isOnlyResample) {
            av_frame_free(&scaleFrame);
            free(scaleBuffer);
        }
        
        return -2;
    }
#pragma endregion

    // Free used resources
    if (!isOnlyResample) {
        av_frame_free(&scaleFrame);
        free(scaleBuffer);
    }

    //Success
    return 0;
}

// Wrapper for the sequential scale operation method
int sequential_scale(AVFrame* src, AVFrame* dst, int operation) {
    // Variables used
    int retVal = -1, duration = -1;
    high_resolution_clock::time_point initTime, stopTime;
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid input dimensions
    if (src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0) {
        cerr << "[SEQUENTIAL] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid resize
    if ((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)) {
        cerr << "[SEQUENTIAL] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }
    // Verify valid input data
    if (!src->data || !src->linesize || !dst->data || !dst->linesize) {
        cerr << "[SEQUENTIAL] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if (!isSupportedFormat(srcFormat) || !isSupportedFormat(dstFormat)) {
        cerr << "[SEQUENTIAL] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if (!isSupportedOperation(operation)) {
        cerr << "[SEQUENTIAL] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    retVal = sequential_scale_aux<double>(src, dst, operation);
    if (retVal < 0) {
        string error = "[SEQUENTIAL] Operation could not be done (";

        if (retVal == -1)
            error += "error initializing frames)!";

        if (retVal == -2)
            error += "resample - conversion not supported)!";

        // Display error
        cerr << error << endl;

        // Insuccess
        return retVal;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}