#include "Sequential_Scale.h"

// Resampler sequential method
int sequential_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]);

// Sequential scale method
int sequential_scale(int srcWidth, int srcHeight, uint8_t* srcSlice,
                  int dstWidth, int dstHeight, uint8_t* dstSlice,
                  int operation);

// Prepares the scaling operation
int sequential_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation);

int sequential_scale(ImageInfo src, ImageInfo dst, int operation){
    // Variables used
    int retVal = -1, duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Verify input parameters
    if(src.width < 0 || src.height < 0 || dst.width < 0 || dst.height < 0){
        cerr << "[SEQUENTIAL] One of input dimensions is negative!" << endl;
        return -1;
    }
    if(src.width % 2 != 0 || src.height % 2 != 0 || dst.width % 2 != 0 || dst.height % 2 != 0){
        cerr << "[SEQUENTIAL] One of the input dimensions is not divisible by 2!" << endl;
        return -1;
    }
    if(!src.frame->data || !src.frame->linesize || !dst.frame->data || !dst.frame->linesize){
        cerr << "[SEQUENTIAL] One of the input parameters is null!" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Verify if it is only a resample operation
    if(src.width == dst.width && src.height == dst.height){
        if(sequential_resampler(src.width, src.height, src.pixelFormat, src.frame->data, src.frame->linesize,
                         dst.width, dst.height, dst.pixelFormat, dst.frame->data, dst.frame->linesize) < 0)
            return -1;
    } else{
        // Apply the scaling operation
        if(sequential_scale_aux(src.width, src.height, src.pixelFormat, src.frame->data, src.frame->linesize,
                                dst.width, dst.height, dst.pixelFormat, dst.frame->data, dst.frame->linesize, operation) < 0)
            return -1;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<milliseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}

int sequential_resampler(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
                  int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        memcpy(dstSlice[0], srcSlice[0], srcStride[0] * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcStride[1] * srcHeight);
        memcpy(dstSlice[2], srcSlice[2], srcStride[2] * srcHeight);
        memcpy(dstSlice[3], srcSlice[3], srcStride[3] * srcHeight);
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Number of elements
        long numElements = srcStride[0] * srcHeight;

        // Loop through each pixel
        for(int index = 0; index < numElements; index += 4){
			// Calculate once
			int indexDiv2 = index / 2;
			int indexDiv4 = index / 4;

            dstSlice[0][indexDiv2] = srcSlice[0][index + 1];        // Ya
            dstSlice[0][indexDiv2 + 1] = srcSlice[0][index + 3];    // Yb

            dstSlice[1][indexDiv4] = srcSlice[0][index];            // U
            dstSlice[2][indexDiv4] = srcSlice[0][index + 2];        // V
        }

        // Success
        return 0;
    }

	if (srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;

		// Loop through each pixel
		for (int index = 0; index < numElements; index += 2) {
			// Calculate once
			int indexMul2 = index * 2;
			int indexDiv2 = index / 2;

			dstSlice[0][indexMul2 + 1] = srcSlice[0][index];        // Ya
			dstSlice[0][indexMul2 + 3] = srcSlice[0][index + 1];    // Yb

			dstSlice[0][indexMul2] = srcSlice[1][indexDiv2];        // U
			dstSlice[0][indexMul2 + 2] = srcSlice[2][indexDiv2];    // V
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;
		long numElementsDiv2 = numElements / 2;

		// Luma Plane is the same
		memcpy(dstSlice[0], srcSlice[0], numElements);

		// Loop through each pixel chroma 
		for (int index = 0; index < numElementsDiv2; index += 2) {
			// Calculate once
			int indexDiv2 = index / 2;

			dstSlice[1][indexDiv2] = srcSlice[1][index];        // U
			dstSlice[2][indexDiv2] = srcSlice[1][index + 1];    // V
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_NV12) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;
		long numElementsDiv4 = numElements / 4;

		// Luma Plane is the same
		memcpy(dstSlice[0], srcSlice[0], numElements);

		// Loop through each pixel chroma 
		for (int index = 0; index < numElementsDiv4; index++) {
			// Calculate once
			int indexMul2 = index * 2;

			dstSlice[1][indexMul2] = srcSlice[1][index];        // U
			dstSlice[1][indexMul2 + 1] = srcSlice[2][index];    // V
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_RGB24 && dstPixelFormat == AV_PIX_FMT_YUV444P) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;

		// Loop through each pixel
		for (int index = 0; index < numElements; index++) {
			// Calculate once
			int indexDiv3 = index / 3;

			float r = static_cast<float>(srcSlice[0][index]);	// R
			float g = static_cast<float>(srcSlice[0][++index]);	// G
			float b = static_cast<float>(srcSlice[0][++index]);	// B

			dstSlice[0][indexDiv3] = float2uint8_t(0.299f * r + 0.587f * g + 0.114f * b);			// Y
			dstSlice[1][indexDiv3] = float2uint8_t(-0.169f * r - 0.331f * g + 0.499f * b + 128.f);	// U
			dstSlice[2][indexDiv3] = float2uint8_t(0.499f * r - 0.418f * g - 0.0813f * b + 128.f);	// V
		}

		// Success
		return 0;
	}

    cerr << "Conversion not supported" << endl;
    return -1;
}

int sequential_scale(int srcWidth, int srcHeight, uint8_t* srcSlice,
              int dstWidth, int dstHeight, uint8_t* dstSlice,
              int operation){

    // Get scale ratios
    float scaleHeightRatio = static_cast<float>(dstHeight) / srcHeight;
    float scaleWidthRatio = static_cast<float>(dstWidth) / srcWidth;

    if(operation == SWS_BILINEAR){
        // Iterate through each line of the scaled image
        for(int lin = 0; lin < dstHeight; lin++){
            // Scaled image line coordinates in the original image
            float linOriginal = (static_cast<float>(lin) + 0.5f) / scaleHeightRatio - 0.5f;
            // Original line index coordinate
            float linOriginalIndex = floor(linOriginal);
            int linOriginalIndexRounded = float2int(linOriginalIndex);

            // Calculate original line coordinates of the pixels to interpolate
            int linThresholdMax = srcHeight - 1;
            int linMin = linOriginalIndexRounded;
            clampPixel(linMin, 0, linThresholdMax);
            int linMax = linOriginalIndexRounded + 1;
            clampPixel(linMax, 0, linThresholdMax);

            // Calculate distance of the scaled coordinate to the original
            float verticalDistance = linOriginal - static_cast<float>(linMin);

            // Calculate the weight of original pixels
            float linMinDistance = 1.f - verticalDistance;
            float linMaxDistance = verticalDistance;

            // Iterate through each column of the scaled image
            for(int col = 0; col < dstWidth; col++){
                // Scaled image column coordinates in the original image
                float colOriginal = (static_cast<float>(col) + 0.5f) / scaleWidthRatio - 0.5f;
                // Original column index coordinate
                float colOriginalIndex = floor(colOriginal);
                int colOriginalIndexRounded = float2int(colOriginalIndex);

                // Calculate original column coordinates of the pixels to interpolate
                int colThresholdMax = srcWidth - 1;
                int colMin = colOriginalIndexRounded;
                clampPixel(colMin, 0, colThresholdMax);
                int colMax = colOriginalIndexRounded + 1;
                clampPixel(colMax, 0, colThresholdMax);

                // Calculate distance of the scaled coordinate to the original
                float horizontalDistance = colOriginal - static_cast<float>(colMin);

                // Calculate the weight of original pixels
                float colMinDistance = 1.f - horizontalDistance;
                float colMaxDistance = horizontalDistance;

                // Temporary variables used in the bilinear interpolation
                uint8_t colorTopLeft, colorTopRight, colorBottomLeft, colorBottomRight;
                // Retrieve pixel from data buffer
                getPixel(srcSlice, srcWidth, srcHeight, linMin, colMin, &colorTopLeft);
                getPixel(srcSlice, srcWidth, srcHeight, linMin, colMax, &colorTopRight);
                getPixel(srcSlice, srcWidth, srcHeight, linMax, colMin, &colorBottomLeft);
                getPixel(srcSlice, srcWidth, srcHeight, linMax, colMax, &colorBottomRight);
                // Interpolate and store value
                dstSlice[lin * dstWidth + col] = float2uint8_t(
                    (static_cast<float>(colorTopLeft) * colMinDistance + static_cast<float>(colorTopRight) * colMaxDistance) * linMinDistance +
                    (static_cast<float>(colorBottomLeft) * colMinDistance + static_cast<float>(colorBottomRight) * colMaxDistance) * linMaxDistance);
            }
        }

        // Success
        return 0;
    }

    if(operation == SWS_BICUBIC){
        // Iterate through each line of the scaled image
        for(int lin = 0; lin < dstHeight; lin++){
            // Scaled image line coordinates in the original image
            float linOriginal = (static_cast<float>(lin) + 0.5f) / scaleHeightRatio;
            // Original line index coordinate
            float linOriginalIndex = floor(linOriginal);
            int linOriginalIndexRounded = float2int(linOriginalIndex);

            // Calculate original line coordinates of the pixels to interpolate
            int linMin = linOriginalIndexRounded - 1;
            int linMax = linOriginalIndexRounded + 2;

            // Iterate through each column of the scaled image
            for(int col = 0; col < dstWidth; col++){
                // Scaled image column coordinates in the original image
                float colOriginal = (static_cast<float>(col) + 0.5f) / scaleWidthRatio;
                // Original column index coordinate
                float colOriginalIndex = floor(colOriginal);
                int colOriginalIndexRounded = float2int(colOriginalIndex);

                // Calculate original column coordinates of the pixels to interpolate
                int colMin = colOriginalIndexRounded - 1;
                int colMax = colOriginalIndexRounded + 2;

                // Temporary variables used in the bicubic interpolation
                uint8_t colorHolder;
                float sum = 0.f, wSum = 0.f, weight;
                // Iterate through each row of neighboring pixels
                for(int linTemp = linMin; linTemp <= linMax; linTemp++){
                    // Iterate through each of the neighboring pixels
                    for(int colTemp = colMin; colTemp <= colMax; colTemp++){
                        // Retrieve pixel from data buffer
                        getPixel(srcSlice, srcWidth, srcHeight, linTemp, colTemp, &colorHolder);
                        // Calculate weight of pixel in the bicubic interpolation
                        weight = getBicubicCoef(abs(linOriginal - (static_cast<float>(linTemp) + 0.5f)))
                            * getBicubicCoef(abs(colOriginal - (static_cast<float>(colTemp) + 0.5f)));
                        // Sum weighted color values
                        sum += static_cast<float>(colorHolder) * weight;
                        // Sum weights
                        wSum += weight;
                    }
                }

                // Calculate resulting color
                float result = sum / wSum;
                // Clamp value to avoid color undershooting and overshooting
                clamp(result, 0.0f, 255.0f);
                // Store the result value
                dstSlice[lin * dstWidth + col] = float2uint8_t(result);
            }
        }

        // Success
        return 0;
    }

    cerr << "Operation not supported" << endl;
    return -1;
}

int sequential_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation){

    // Variables used
    int retVal = -1;
    AVPixelFormat scalingSupportedFormat = AV_PIX_FMT_YUV422P;
    uint8_t* resampleTempFrameBuffer,* scaleTempFrameBuffer;
    AVFrame* resampleTempFrame,* scaleTempFrame;

    // Prepare to initialize resampleTempFrame
    retVal = createImageDataBuffer(srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrameBuffer);
    if(retVal < 0)
        return retVal;

    // Initialize resampleTempFrame
    retVal = initializeAVFrame(&resampleTempFrameBuffer, srcWidth, srcHeight, scalingSupportedFormat, &resampleTempFrame);
    if(retVal < 0){
        free(resampleTempFrameBuffer);
        return retVal;
    }

    // Resamples image to a supported format
    retVal = sequential_resampler(srcWidth, srcHeight, srcPixelFormat, srcSlice, srcStride,
                           srcWidth, srcHeight, scalingSupportedFormat, resampleTempFrame->data, resampleTempFrame->linesize);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        return retVal;
    }

    // Prepare to initialize scaleTempFrame
    retVal = createImageDataBuffer(dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrameBuffer);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        return retVal;
    }

    // Initialize scaleTempFrame
    retVal = initializeAVFrame(&scaleTempFrameBuffer, dstWidth, dstHeight, scalingSupportedFormat, &scaleTempFrame);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        free(scaleTempFrameBuffer);
        return retVal;
    }

    // Apply the scaling operation
    for(int colorChannel = 0; colorChannel < 3; colorChannel++){
        if(colorChannel == 0){
            retVal = sequential_scale(srcWidth, srcHeight, resampleTempFrame->data[0],
                               dstWidth, dstHeight, scaleTempFrame->data[0],
                               operation);
        } else{
            retVal = sequential_scale(srcWidth / 2, srcHeight, resampleTempFrame->data[colorChannel],
                               dstWidth / 2, dstHeight, scaleTempFrame->data[colorChannel],
                               operation);
        }

        if(retVal < 0){
            av_frame_free(&resampleTempFrame);
            free(resampleTempFrameBuffer);
            av_frame_free(&scaleTempFrame);
            free(scaleTempFrameBuffer);
            return retVal;
        }
    }

    // Resamples results to the desired one
    retVal = sequential_resampler(dstWidth, dstHeight, scalingSupportedFormat, scaleTempFrame->data, scaleTempFrame->linesize,
                           dstWidth, dstHeight, dstPixelFormat, dstSlice, dstStride);
    if(retVal < 0){
        av_frame_free(&resampleTempFrame);
        free(resampleTempFrameBuffer);
        av_frame_free(&scaleTempFrame);
        free(scaleTempFrameBuffer);
        return retVal;
    }

    // Free used resources
    av_frame_free(&resampleTempFrame);
    free(resampleTempFrameBuffer);
    av_frame_free(&scaleTempFrame);
    free(scaleTempFrameBuffer);

    return 0;
}
