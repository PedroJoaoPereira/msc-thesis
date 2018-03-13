// Iterate through each line of the scaled image
for(int lin = 0; lin < dstHeight; lin++){
	// Scaled image line coordinates in the original image
	float linOriginal = (static_cast<float>(lin) + 0.5f) / scaleHeightRatio - 0.5f;
	// Original line index coordinate
	float linOriginalIndex = floor(linOriginal);
	int linOriginalIndexRounded = float2int(linOriginalIndex);

	// Calculate original line coordinates of the pixels to interpolate
	int linThresholdMax = srcHeight - 1;
	int linMin = linOriginalIndex;
	clampPixel(linMin, 0, linThresholdMax);
	int linMax = linOriginalIndex + 1;
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
		int colMin = colOriginalIndex;
		clampPixel(colMin, 0, colThresholdMax);
		int colMax = colOriginalIndex + 1;
		clampPixel(colMax, 0, colThresholdMax);

		// Calculate distance of the scaled coordinate to the original
		float horizontalDistance = colOriginal - static_cast<float>(colMin);

		// Calculate the weight of original pixels
		float colMinDistance = 1.f - horizontalDistance;
		float colMaxDistance = horizontalDistance;

		// Temporary variables used in the bilinear interpolation
		uint8_t colorTopLeft, colorTopRight, colorBottomLeft, colorBottomRight;
		// Bilinear interpolation operation for each color channel
		for(int colorChannel = 0; colorChannel < 3; colorChannel++){
			// Retrieve pixel from data buffer
			getPixel(srcSlice, colorChannel, srcWidth, srcHeight, linMin, colMin, &colorTopLeft);
			getPixel(srcSlice, colorChannel, srcWidth, srcHeight, linMin, colMax, &colorTopRight);
			getPixel(srcSlice, colorChannel, srcWidth, srcHeight, linMax, colMin, &colorBottomLeft);
			getPixel(srcSlice, colorChannel, srcWidth, srcHeight, linMax, colMax, &colorBottomRight);
			// Interpolate and store value
			dstSlice[colorChannel][lin * dstWidth + col] = float2uint8_t(
				(static_cast<float>(colorTopLeft) * colMinDistance + static_cast<float>(colorTopRight) * colMaxDistance) * linMinDistance + 
				(static_cast<float>(colorBottomLeft) * colMinDistance + static_cast<float>(colorBottomRight) * colMaxDistance) * linMaxDistance);
		}
	}
}