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
				weight = bcoef(abs(linOriginal - (static_cast<float>(linTemp) + 0.5f)))
					* bcoef(abs(colOriginal - (static_cast<float>(colTemp) + 0.5f)));
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