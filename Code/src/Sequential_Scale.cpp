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
		// Calculate the chroma size depending on the source data pixel format
		float tempWidthRatio = 1.f;
		float tempHeightRatio = 1.f;
		if (srcPixelFormat == AV_PIX_FMT_YUV422P || srcPixelFormat == AV_PIX_FMT_YUV420P)
			tempWidthRatio = 0.5f;
		if (srcPixelFormat == AV_PIX_FMT_YUV420P)
			tempHeightRatio = 0.5f;

		// Copy data between buffers
        memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);
        memcpy(dstSlice[1], srcSlice[1], srcWidth * srcHeight * tempWidthRatio * tempHeightRatio);
		memcpy(dstSlice[2], srcSlice[2], srcWidth * srcHeight * tempWidthRatio * tempHeightRatio);

		// Success
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
	if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_RGB24) {
		// Number of elements
		long numElements = srcWidth * srcHeight;

		// Loop through each pixel
		for (int index = 0; index < numElements; index++) {
			// Calculate once
			int indexMul3 = index * 3;

			float y = static_cast<float>(srcSlice[0][index]);	// Y
			float u = static_cast<float>(srcSlice[1][index]);	// U
			float v = static_cast<float>(srcSlice[2][index]);	// V

			y -= 16.f;
			u -= 128.f;
			v -= 128.f;

			float r = 1.164f * y + 0.f * u + 1.596f * v;
			float g = 1.164f * y - 0.392f * u - 0.813f * v;
			float b = 1.164f * y + 2.017f * u + 0.f * v;

			// Clamp values to avoid overshooting and undershooting
			clamp(r, 0.f, 255.f);
			clamp(g, 0.f, 255.f);
			clamp(b, 0.f, 255.f);

			dstSlice[0][indexMul3] = roundTo<uint8_t, float>(r);		// R
			dstSlice[0][indexMul3 + 1] = roundTo<uint8_t, float>(g);	// G
			dstSlice[0][indexMul3 + 2] = roundTo<uint8_t, float>(b);	// B
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_YUV422P) {
		// Number of elements
		long numElements = srcWidth * srcHeight;

		// Luma plane is the same
		memcpy(dstSlice[0], srcSlice[0], numElements);

		// Loop through each pixel
		for (int index = 0; index < numElements; index += 2) {
			// Calculate once
			int indexAdd1 = index + 1;
			int indexDiv2 = index / 2;

			float u1 = static_cast<float>(srcSlice[1][index]);		// U1
			float v1 = static_cast<float>(srcSlice[2][index]);		// V1

			float u2 = static_cast<float>(srcSlice[1][indexAdd1]);	// U2
			float v2 = static_cast<float>(srcSlice[2][indexAdd1]);	// V2

			dstSlice[1][indexDiv2] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
			dstSlice[2][indexDiv2] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_YUV420P) {
		// Luma plane is the same
		memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);

		// Loop through each pixel
		for (int lin = 0; lin < srcHeight; lin += 2) {
			// Calculate once
			int linIndexTop = lin * srcWidth;
			int linIndexBottom = linIndexTop + srcWidth;

			for (int col = 0; col < srcWidth; col += 2) {
				// Calculate once
				int colIndexLeft = col;
				int colIndexRight = colIndexLeft + 1;

				int index1 = linIndexTop + colIndexLeft;
				float u1 = static_cast<float>(srcSlice[1][index1]);	// U1
				float v1 = static_cast<float>(srcSlice[2][index1]);	// V1

				int index2 = linIndexTop + colIndexRight;
				float u2 = static_cast<float>(srcSlice[1][index2]);	// U2
				float v2 = static_cast<float>(srcSlice[2][index2]);	// V2

				int index3 = linIndexBottom + colIndexLeft;
				float u3 = static_cast<float>(srcSlice[1][index3]);	// U3
				float v3 = static_cast<float>(srcSlice[2][index3]);	// V3

				int index4 = linIndexBottom + colIndexRight;
				float u4 = static_cast<float>(srcSlice[1][index4]);	// U4
				float v4 = static_cast<float>(srcSlice[2][index4]);	// V4

				int indexFinal = (lin / 2) * (srcWidth / 2) + (col / 2);
				dstSlice[1][indexFinal] = roundTo<uint8_t, float>((u1 + u2 + u3 + u4) / 4.f);
				dstSlice[2][indexFinal] = roundTo<uint8_t, float>((v1 + v2 + v3 + v4) / 4.f);
			}
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_UYVY422) {
		// Number of elements
		long numElements = srcWidth * srcHeight;

		// Loop through each pixel
		for (int index = 0; index < numElements; index += 2) {
			// Calculate once
			int indexAdd1 = index + 1;
			int indexMul2 = index * 2;

			float u1 = static_cast<float>(srcSlice[1][index]);		// U1
			float v1 = static_cast<float>(srcSlice[2][index]);		// V1

			float u2 = static_cast<float>(srcSlice[1][indexAdd1]);	// U2
			float v2 = static_cast<float>(srcSlice[2][indexAdd1]);	// V2

			dstSlice[0][indexMul2] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
			dstSlice[0][indexMul2 + 1] = srcSlice[0][index];
			dstSlice[0][indexMul2 + 2] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
			dstSlice[0][indexMul2 + 3] = srcSlice[0][indexAdd1];
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_NV12) {
		// Luma plane is the same
		memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);

		// Calculate once
		int widthDiv2 = srcWidth / 2;

		// Loop through each pixel
		for (int lin = 0; lin < srcHeight; lin += 2) {
			// Calculate once
			int linIndexTop = lin * srcWidth;
			int linIndexBottom = linIndexTop + srcWidth;

			for (int col = 0; col < srcWidth; col += 2) {
				// Calculate once
				int colIndexLeft = col;
				int colIndexRight = colIndexLeft + 1;

				int index1 = linIndexTop + colIndexLeft;
				float u1 = static_cast<float>(srcSlice[1][index1]);	// U1
				float v1 = static_cast<float>(srcSlice[2][index1]);	// V1

				int index2 = linIndexTop + colIndexRight;
				float u2 = static_cast<float>(srcSlice[1][index2]);	// U2
				float v2 = static_cast<float>(srcSlice[2][index2]);	// V2

				int index3 = linIndexBottom + colIndexLeft;
				float u3 = static_cast<float>(srcSlice[1][index3]);	// U3
				float v3 = static_cast<float>(srcSlice[2][index3]);	// V3

				int index4 = linIndexBottom + colIndexRight;
				float u4 = static_cast<float>(srcSlice[1][index4]);	// U4
				float v4 = static_cast<float>(srcSlice[2][index4]);	// V4

				int indexFinal = lin * widthDiv2 + col;
				dstSlice[1][indexFinal] = roundTo<uint8_t, float>((u1 + u2 + u3 + u4) / 4.f);
				dstSlice[1][indexFinal + 1] = roundTo<uint8_t, float>((v1 + v2 + v3 + v4) / 4.f);
			}
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P) {
		// Calculate once
		int widthDiv2 = srcWidth / 2;

		// Luma plane is the same
		memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);

		// Loop through each pixel
		for (int lin = 0; lin < srcHeight; lin += 2) {
			// Calculate once
			int linIndexTop = lin * widthDiv2;
			int linIndexBottom = linIndexTop + widthDiv2;

			for (int col = 0; col < widthDiv2; col++) {
				int index1 = linIndexTop + col;
				float u1 = static_cast<float>(srcSlice[1][index1]);	// U1
				float v1 = static_cast<float>(srcSlice[2][index1]);	// V1

				int index2 = linIndexBottom + col;
				float u2 = static_cast<float>(srcSlice[1][index2]);	// U2
				float v2 = static_cast<float>(srcSlice[2][index2]);	// V2

				int indexFinal = (lin / 2) * widthDiv2 + col;
				dstSlice[1][indexFinal] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
				dstSlice[2][indexFinal] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
			}
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422) {
		// Number of elements
		long numElements = srcWidth * srcHeight;

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

	if (srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12) {
		// Calculate once
		int widthDiv2 = srcWidth / 2;

		// Luma plane is the same
		memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);

		// Loop through each pixel
		for (int lin = 0; lin < srcHeight; lin += 2) {
			// Calculate once
			int linIndexTop = lin * widthDiv2;
			int linIndexBottom = linIndexTop + widthDiv2;

			for (int col = 0; col < widthDiv2; col++) {
				int index1 = linIndexTop + col;
				float u1 = static_cast<float>(srcSlice[1][index1]);	// U1
				float v1 = static_cast<float>(srcSlice[2][index1]);	// V1

				int index2 = linIndexBottom + col;
				float u2 = static_cast<float>(srcSlice[1][index2]);	// U2
				float v2 = static_cast<float>(srcSlice[2][index2]);	// V2

				int indexFinal = lin * widthDiv2 + col * 2;
				dstSlice[1][indexFinal] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
				dstSlice[1][indexFinal + 1] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
			}
		}

		// Success
		return 0;
	}

	if (srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;

		// Loop through each pixel
		for (int index = 0; index < numElements; index += 4) {
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

	if (srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P) {
		// Loop through each pixel
		for (int lin = 0; lin < srcHeight; lin += 2) {
			// Calculate once
			int linIndexTop = lin * srcStride[0];
			int linIndexBottom = linIndexTop + srcStride[0];

			for (int col = 0; col < srcStride[0]; col += 4) {
                // Calculate once
                int colDiv2 = col / 2;

				int index1 = linIndexTop + col;
				float u1 = static_cast<float>(srcSlice[0][index1]);			// U1
				float ya1 = static_cast<float>(srcSlice[0][index1 + 1]);	// Ya1
				float v1 = static_cast<float>(srcSlice[0][index1 + 2]);		// V1
				float yb1 = static_cast<float>(srcSlice[0][index1 + 3]);	// Yb1

				int index2 = linIndexBottom + col;
				float u2 = static_cast<float>(srcSlice[0][index2]);			// U2
				float ya2 = static_cast<float>(srcSlice[0][index2 + 1]);	// Ya2
				float v2 = static_cast<float>(srcSlice[0][index2 + 2]);		// V2
				float yb2 = static_cast<float>(srcSlice[0][index2 + 3]);	// Yb2

				int indexFinalTop = linIndexTop / 2 + colDiv2;
				int indexFinalBottom = linIndexBottom / 2 + colDiv2;
				dstSlice[0][indexFinalTop] = ya1;
				dstSlice[0][indexFinalTop + 1] = yb1;
				dstSlice[0][indexFinalBottom] = ya2;
				dstSlice[0][indexFinalBottom + 1] = yb2;

                int indexFinalTopChroma = linIndexTop / 8 + col / 4;
                dstSlice[1][indexFinalTopChroma] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
                dstSlice[2][indexFinalTopChroma] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
			}
		}

		// Success
		return 0;
	}

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Loop through each pixel
        for(int lin = 0; lin < srcHeight; lin += 2){
            // Calculate once
            int linIndexTop = lin * srcStride[0];
            int linIndexBottom = linIndexTop + srcStride[0];

            for(int col = 0; col < srcStride[0]; col += 4){
                // Calculate once
                int colDiv2 = col / 2;

                int index1 = linIndexTop + col;
                float u1 = static_cast<float>(srcSlice[0][index1]);			// U1
                float ya1 = static_cast<float>(srcSlice[0][index1 + 1]);	// Ya1
                float v1 = static_cast<float>(srcSlice[0][index1 + 2]);		// V1
                float yb1 = static_cast<float>(srcSlice[0][index1 + 3]);	// Yb1

                int index2 = linIndexBottom + col;
                float u2 = static_cast<float>(srcSlice[0][index2]);			// U2
                float ya2 = static_cast<float>(srcSlice[0][index2 + 1]);	// Ya2
                float v2 = static_cast<float>(srcSlice[0][index2 + 2]);		// V2
                float yb2 = static_cast<float>(srcSlice[0][index2 + 3]);	// Yb2

                int indexFinalTop = linIndexTop / 2 + colDiv2;
                int indexFinalBottom = linIndexBottom / 2 + colDiv2;
                dstSlice[0][indexFinalTop] = ya1;
                dstSlice[0][indexFinalTop + 1] = yb1;
                dstSlice[0][indexFinalBottom] = ya2;
                dstSlice[0][indexFinalBottom + 1] = yb2;

                int indexFinalTopChroma = linIndexTop / 4 + colDiv2;
                dstSlice[1][indexFinalTopChroma] = roundTo<uint8_t, float>((u1 + u2) / 2.f);
                dstSlice[1][indexFinalTopChroma + 1] = roundTo<uint8_t, float>((v1 + v2) / 2.f);
            }
        }

        // Success
        return 0;
    }



	

	if (srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P) {
		// Number of elements
		long numElements = srcStride[0] * srcHeight;
		long numElementsDiv2 = numElements / 2;

		// Luma plane is the same
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

		// Luma plane is the same
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

			float y = 0.257f * r + 0.504f*g + 0.098f*b + 16.f;	// Y
			float u = -0.148f*r - 0.291f*g + 0.439f*b + 128.f;	// U
			float v = 0.439f*r - 0.368f*g - 0.071f*b + 128.f;	// V

			dstSlice[0][indexDiv3] = roundTo<uint8_t, float>(y);	// Y
			dstSlice[1][indexDiv3] = roundTo<uint8_t, float>(u);	// U
			dstSlice[2][indexDiv3] = roundTo<uint8_t, float>(v);	// V
		}

		// Success
		return 0;
	}

    cerr << "[SEQUENTIAL] Conversion not supported" << endl;
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
            int linOriginalIndexRounded = roundTo<uint8_t, float>(linOriginalIndex);

            // Calculate original line coordinates of the pixels to interpolate
            int linThresholdMax = srcHeight - 1;
            int linMin = linOriginalIndexRounded;
            clamp<int>(linMin, 0, linThresholdMax);
            int linMax = linOriginalIndexRounded + 1;
            clamp<int>(linMax, 0, linThresholdMax);

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
                int colOriginalIndexRounded = roundTo<uint8_t, float>(colOriginalIndex);

                // Calculate original column coordinates of the pixels to interpolate
                int colThresholdMax = srcWidth - 1;
                int colMin = colOriginalIndexRounded;
                clamp<int>(colMin, 0, colThresholdMax);
                int colMax = colOriginalIndexRounded + 1;
                clamp<int>(colMax, 0, colThresholdMax);

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
                dstSlice[lin * dstWidth + col] = roundTo<uint8_t, float>(
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
            int linOriginalIndexRounded = roundTo<uint8_t, float>(linOriginalIndex);

            // Calculate original line coordinates of the pixels to interpolate
            int linMin = linOriginalIndexRounded - 1;
            int linMax = linOriginalIndexRounded + 2;

            // Iterate through each column of the scaled image
            for(int col = 0; col < dstWidth; col++){
                // Scaled image column coordinates in the original image
                float colOriginal = (static_cast<float>(col) + 0.5f) / scaleWidthRatio;
                // Original column index coordinate
                float colOriginalIndex = floor(colOriginal);
                int colOriginalIndexRounded = roundTo<uint8_t, float>(colOriginalIndex);

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
                dstSlice[lin * dstWidth + col] = roundTo<uint8_t, float>(result);
            }
        }

        // Success
        return 0;
    }

    cerr << "[SEQUENTIAL] Operation not supported" << endl;
    return -1;
}

int sequential_scale_aux(int srcWidth, int srcHeight, AVPixelFormat srcPixelFormat, uint8_t* srcSlice[], int srcStride[],
               int dstWidth, int dstHeight, AVPixelFormat dstPixelFormat, uint8_t* dstSlice[], int dstStride[],
               int operation){

    // Variables used
    int retVal = -1;
    uint8_t* resampleTempFrameBuffer,* scaleTempFrameBuffer;
    AVFrame* resampleTempFrame,* scaleTempFrame;

	// Retrieve the temporary scaling pixel format
	AVPixelFormat scalingSupportedFormat = getTempScaleFormat(srcPixelFormat);
	if (scalingSupportedFormat == AV_PIX_FMT_NONE) {
		cerr << "[SEQUENTIAL] Source pixel format is not supported" << endl;
		return -1;
	}

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

    // Apply the scaling operation to the luma component
	retVal = sequential_scale(srcWidth, srcHeight, resampleTempFrame->data[0],
		dstWidth, dstHeight, scaleTempFrame->data[0],
		operation);
	if (retVal < 0) {
		av_frame_free(&resampleTempFrame);
		free(resampleTempFrameBuffer);
		av_frame_free(&scaleTempFrame);
		free(scaleTempFrameBuffer);
		return retVal;
	}

	// Calculate the chroma size depending on the source data pixel format
	float tempWidthRatio = 1.f;
	float tempHeightRatio = 1.f;
	if (scalingSupportedFormat == AV_PIX_FMT_YUV422P || scalingSupportedFormat == AV_PIX_FMT_YUV420P)
		tempWidthRatio = 0.5f;
	if (scalingSupportedFormat == AV_PIX_FMT_YUV420P)
		tempHeightRatio = 0.5f;

	// Apply the scaling operation to the second chroma component
	retVal = sequential_scale(static_cast<int>(srcWidth * tempWidthRatio), static_cast<int>(srcHeight * tempHeightRatio), resampleTempFrame->data[1],
		static_cast<int>(dstWidth * tempWidthRatio), static_cast<int>(dstHeight * tempHeightRatio), scaleTempFrame->data[1],
		operation);
	if (retVal < 0) {
		av_frame_free(&resampleTempFrame);
		free(resampleTempFrameBuffer);
		av_frame_free(&scaleTempFrame);
		free(scaleTempFrameBuffer);
		return retVal;
	}

	// Apply the scaling operation to the third chroma component
	retVal = sequential_scale(static_cast<int>(srcWidth * tempWidthRatio), static_cast<int>(srcHeight * tempHeightRatio), resampleTempFrame->data[2],
		static_cast<int>(dstWidth * tempWidthRatio), static_cast<int>(dstHeight * tempHeightRatio), scaleTempFrame->data[2],
		operation);
	if (retVal < 0) {
		av_frame_free(&resampleTempFrame);
		free(resampleTempFrameBuffer);
		av_frame_free(&scaleTempFrame);
		free(scaleTempFrameBuffer);
		return retVal;
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
