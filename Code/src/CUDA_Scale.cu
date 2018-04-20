#include "CUDA_Scale.cuh"

// Modify the color model of the image
template <class PrecisionType>
__global__ void cuda_resampler(int srcWidth, int srcHeight, int srcPixelFormat,
    uint8_t* srcSlice0, uint8_t* srcSlice1, uint8_t* srcSlice2, uint8_t* srcSlice3,
    int srcStride0, int srcStride1, int srcStride2, int srcStride3,
    int dstWidth, int dstHeight, int dstPixelFormat,
    uint8_t* dstSlice0, uint8_t* dstSlice1, uint8_t* dstSlice2, uint8_t* dstSlice3,
    int dstStride0, int dstStride1, int dstStride2, int dstStride3) {

    // REORGANIZE COMPONENTS -------------------------
    /*if (srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P) {
        // Number of elements
        long numElements = srcStride[0] * srcHeight / 4;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto dstBuffer = dstSlice[0];
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Loop through each pixel
        for (int index = 0; index < numElements; index++) {
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
    }*/

    // Insuccess
    return;
}

// Precalculate coefficients
template <class PrecisionType>
int cuda_preCalculateCoefficients(int srcSize, int dstSize, int operation, int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType** &preCalculatedCoefs) {
    // Calculate size ratio
    PrecisionType sizeRatio = static_cast<PrecisionType>(dstSize) / static_cast<PrecisionType>(srcSize);

    // Check if is downscale or upscale
    bool isDownScale = sizeRatio < static_cast<PrecisionType>(1.);
    int pixelSupportDiv2 = pixelSupport / 2;
    PrecisionType filterStep = static_cast<PrecisionType>(1.);
    if (isDownScale && operation != SWS_POINT) {
        filterStep = 1. / (ceil((pixelSupport / 2.) / sizeRatio) / (pixelSupport / 2.));
    }

    // Calculate number of lines of coefficients
    int preCalcCoefSize = lcm(srcSize, dstSize) / min<int>(srcSize, dstSize);

    // Initialize 2d array
    preCalculatedCoefs = static_cast<PrecisionType**>(malloc(preCalcCoefSize * sizeof(PrecisionType*)));
    for (int index = 0; index < preCalcCoefSize; index++)
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

            if (sizeRatio < static_cast<PrecisionType>(1.) && operation == SWS_POINT)
                if (preCalculatedCoefs[lin][pixelSupportDiv2 - index - 1] == preCalculatedCoefs[lin][index + pixelSupportDiv2])
                    preCalculatedCoefs[lin][index + pixelSupportDiv2] = static_cast<PrecisionType>(1.);
        }
    }

    // Success
    return preCalcCoefSize;
}


// Prepares the scaling operation
template <class PrecisionType>
int cuda_scale_aux(AVFrame* src, AVFrame* dst, int operation) {

    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    int srcFormat = src->format;
    int dstWidth = dst->width, dstHeight = dst->height;
    int dstFormat = dst->format;

    // Check if is only a resample operation
    bool isOnlyResample = false;
    if (srcWidth == dstWidth && srcHeight == dstHeight)
        isOnlyResample = true;

    // Initialize needed variables if it is a scaling operation
    int scalingSupportedFormat;
    if (srcFormat == AV_PIX_FMT_V210 && dstFormat == AV_PIX_FMT_V210)
        scalingSupportedFormat = AV_PIX_FMT_YUV422PNORM;
    else
        scalingSupportedFormat = getTempScaleFormat(srcFormat);

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
    int lastResamplePixelFormat = srcFormat;

    #pragma region RESIZE OPERATION
    // Verify if is not only a resample operation
    if (!isOnlyResample) {
        // Resamples image to a supported format
        /*if (cuda_resampler<PrecisionType>(srcWidth, srcHeight, srcFormat, src->data, src->linesize,
            srcWidth, srcHeight, scalingSupportedFormat, resampleFrame->data, resampleFrame->linesize) < 0) {
            av_frame_free(&resampleFrame);
            av_frame_free(&scaleFrame);
            free(resampleBuffer);
            free(scaleBuffer);
            return -2;
        }*/

        // Temporary variables for precalculation of coefficients
        PrecisionType(*coefFunc)(PrecisionType) = getCoefMethod<PrecisionType>(operation);
        int pixelSupportV = getPixelSupport(operation, max<int>(round((static_cast<PrecisionType>(srcHeight) / static_cast<PrecisionType>(dstHeight)) / static_cast<PrecisionType>(2.)) * 2, 1));
        int pixelSupportH = getPixelSupport(operation, max<int>(round((static_cast<PrecisionType>(srcWidth) / static_cast<PrecisionType>(dstWidth)) / static_cast<PrecisionType>(2.)) * 2, 1));
        int pixelSupport = max<int>(pixelSupportV, pixelSupportH);

        // Create variables for precalculated coefficients
        PrecisionType** vCoefs;
        int vCoefsSize = cuda_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefs);
        PrecisionType** hCoefs;
        int hCoefsSize = cuda_preCalculateCoefficients<PrecisionType>(srcWidth, dstWidth, operation, pixelSupport, coefFunc, hCoefs);

        // Calculate the chroma size depending on the source data pixel format
        float tempWidthRatio = 1.f;
        float tempHeightRatio = 1.f;
        if (scalingSupportedFormat == AV_PIX_FMT_YUV422P || scalingSupportedFormat == AV_PIX_FMT_YUV420P || scalingSupportedFormat == AV_PIX_FMT_YUV422PNORM)
            tempWidthRatio = 0.5f;
        if (scalingSupportedFormat == AV_PIX_FMT_YUV420P)
            tempHeightRatio = 0.5f;

        /*
        // Apply the resizing operation to luma channel
        sequential_resize<PrecisionType>(srcWidth, srcHeight, resampleFrame->data[0],
            dstWidth, dstHeight, scaleFrame->data[0], operation,
            pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, 0);

        // Apply the resizing operation to chroma channels
        for (int colorChannel = 1; colorChannel < 3; colorChannel++) {
            sequential_resize<PrecisionType>(static_cast<int>(srcWidth * tempWidthRatio), static_cast<int>(srcHeight * tempHeightRatio), resampleFrame->data[colorChannel],
                static_cast<int>(dstWidth * tempWidthRatio), static_cast<int>(dstHeight * tempHeightRatio), scaleFrame->data[colorChannel], operation,
                pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, colorChannel);
        }
        */

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
    /*if (cuda_resampler<PrecisionType>(dstWidth, dstHeight, lastResamplePixelFormat, lastResampleFrame->data, lastResampleFrame->linesize,
        dstWidth, dstHeight, dstFormat, dst->data, dst->linesize) < 0) {
        if (!isOnlyResample) {
            av_frame_free(&scaleFrame);
            free(scaleBuffer);
        }
        return -2;
    }*/
    #pragma endregion

    // Free used resources
    if (!isOnlyResample) {
        av_frame_free(&scaleFrame);
        free(scaleBuffer);
    }

    //Success
    return 0;
}

// Wrapper for the cuda scale operation method
int cuda_scale(AVFrame* src, AVFrame* dst, int operation) {
    // Variables used
    int retVal = -1, duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Verify valid frames
    if (src == nullptr || dst == nullptr) {
        cerr << "[CUDA] One or both input frames are null!" << endl;
        return -1;
    }

    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid input dimensions
    if (src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0) {
        cerr << "[CUDA] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid resize
    if ((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)) {
        cerr << "[CUDA] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }
    // Verify valid input data
    if (!src->data || !src->linesize || !dst->data || !dst->linesize) {
        cerr << "[CUDA] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if (!isSupportedFormat(srcFormat) || !isSupportedFormat(dstFormat)) {
        cerr << "[CUDA] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if can convert a 10 bit format
    if ((src->width % 12 != 0 && srcFormat == AV_PIX_FMT_V210) || (dst->width % 12 != 0 && dstFormat == AV_PIX_FMT_V210)) {
        cerr << "[CUDA] Can not handle 10 bit format because data is not aligned!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if (!isSupportedOperation(operation)) {
        cerr << "[CUDA] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    retVal = cuda_scale_aux<double>(src, dst, operation);
    if (retVal < 0) {
        string error = "[CUDA] Operation could not be done (";

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