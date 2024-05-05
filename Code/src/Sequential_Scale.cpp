#include "Sequential_Scale.h"

// Modify the color model of the image
template <class PrecisionType>
int sequential_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
    int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]) {

    // If same formats no need to resample
    if (srcPixelFormat == dstPixelFormat) {
        // Copy data between buffers
        memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight);
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight);
        memcpy(dstSlice[3], srcSlice[3], srcStride[3] * srcHeight);

        // Success
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
    if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_YUV422P) {
        // Number of elements
        long numElements = srcWidth * srcHeight;

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], numElements);

        // Loop through each pixel
        for (int index = 0; index < numElements; index += 2) {
            // Calculate once
            int indexDiv2 = index / 2;

            dstSlice[1][indexDiv2] = srcSlice[1][index]; // U
            dstSlice[2][indexDiv2] = srcSlice[2][index]; // V
        }

        // Success
        return 0;
    }

    // No conversion was supported
    return -1;
}

// Precalculate coefficients
template <class PrecisionType>
int sequential_preCalculateCoefficients(int srcSize, int dstSize, int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType** &preCalculatedCoefs) {
    // Calculate number of lines of coefficients
    int preCalcCoefSize = lcm(srcSize, dstSize) / min(srcSize, dstSize);

    // Initialize 2d array
    preCalculatedCoefs = static_cast<PrecisionType**>(malloc(preCalcCoefSize * sizeof(PrecisionType*)));
    for (int index = 0; index < preCalcCoefSize; index++)
        preCalculatedCoefs[index] = static_cast<PrecisionType*>(malloc(pixelSupport * sizeof(PrecisionType)));

    // Calculate size ratio
    PrecisionType sizeRatio = static_cast<PrecisionType>(dstSize) / static_cast<PrecisionType>(srcSize);
    // Calculate once
    int pixelSupportDiv2 = pixelSupport / 2;
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
            preCalculatedCoefs[lin][pixelSupportDiv2 - index - 1] = coefFunc(upperCoef + index * sizeRatio);
            preCalculatedCoefs[lin][index + pixelSupportDiv2] = coefFunc(bottomtCoef + index * sizeRatio);
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
        int pixelSupport = getPixelSupport(operation);

        // Create variables for precalculated coefficients
        PrecisionType** vCoefs;
        int vCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, pixelSupport, coefFunc, vCoefs);
        PrecisionType** hCoefs;
        int hCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcWidth, dstWidth, pixelSupport, coefFunc, hCoefs);

        // Apply the resizing operation to each color channel
        for (int colorChannel = 0; colorChannel < 3; colorChannel++) {
            sequential_resize<PrecisionType>(srcWidth, srcHeight, resampleFrame->data[colorChannel],
                dstWidth, dstHeight, scaleFrame->data[colorChannel], operation,
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