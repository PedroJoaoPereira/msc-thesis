// -----------------------------------------------
// WILL PROBABLY NOT BE USED ---------------------
if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
	// Calculate once
	int stride = srcStride[0];
	long numElements = stride * srcHeight;
	int columnsByLine = stride / 2;

	// Loop through each pixel
	for(int index = 0; index < numElements; index += 4){
		dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
		dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

		int lineIndex = index / (stride * 2);
		if(lineIndex % 2 == 0){
			int columnIndex = (index / 4) % columnsByLine;
			int chromaIndex = lineIndex / 2 * columnsByLine + columnIndex;
			dstSlice[1][chromaIndex] = srcSlice[0][index];      // U
			dstSlice[2][chromaIndex] = srcSlice[0][index + 2];  // V
		}
	}

	// Success
	return 0;
}

if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV444P){
	// Calculate once
	long numElements = srcStride[0] * srcHeight;

	// Loop through each pixel
	for(int index = 0; index < numElements; index += 4){
		dstSlice[0][index / 2] = srcSlice[0][index + 1];        // Ya
		dstSlice[0][index / 2 + 1] = srcSlice[0][index + 3];    // Yb

		dstSlice[1][index / 2] = srcSlice[0][index];            // U
		dstSlice[1][index / 2 + 1] = srcSlice[0][index];

		dstSlice[2][index / 2] = srcSlice[0][index + 2];        // V
		dstSlice[2][index / 2 + 1] = srcSlice[0][index + 2];
	}

	// Success
	return 0;
}

if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV444P){
	// Calculate once
	long numElements = srcStride[0] * srcHeight;

	// Loop through each pixel
	for(int index = 0; index < numElements; index++){
		dstSlice[0][index] = srcSlice[0][index];        // Y
		dstSlice[1][index] = srcSlice[1][index / 2];    // U
		dstSlice[2][index] = srcSlice[2][index / 2];    // V
	}

	// Success
	return 0;
}

if(srcPixelFormat == AV_PIX_FMT_YUV444P && dstPixelFormat == AV_PIX_FMT_YUV422P){
	// Calculate once
	long numElements = srcStride[0] * srcHeight;

	// Loop through each pixel
	for(int index = 0; index < numElements; index++){
		dstSlice[0][index] = srcSlice[0][index];            // Y

		if(index % 2 == 0){
			dstSlice[1][index / 2] = srcSlice[1][index];    // U
			dstSlice[2][index / 2] = srcSlice[2][index];    // V
		}
	}

	// Success
	return 0;
}

if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_GBRP){
	// Calculate once
	long numElements = srcStride[0] * srcHeight;

	// Loop through each pixel
	for(int index = 0; index < numElements; index++){
		uint8_t y = srcSlice[0][index];         // Y
		uint8_t cb = srcSlice[1][index / 2];    // U
		uint8_t cr = srcSlice[2][index / 2];    // V

		double rd = static_cast<double>(y) + 1.402 * (static_cast<double>(cr) - 128);
		double gd = static_cast<double>(y) - 0.344 * (static_cast<double>(cb) - 128) - 0.714 * (static_cast<double>(cr) - 128);
		double bd = static_cast<double>(y) + 1.772 * (static_cast<double>(cb) - 128);

		//clamp(rd, 0.0, 255.0);
		//clamp(gd, 0.0, 255.0);
		//clamp(bd, 0.0, 255.0);

		dstSlice[0][index] = round(rd); // R
		dstSlice[1][index] = round(gd); // G
		dstSlice[2][index] = round(bd); // B
	}

	// Success
	return 0;
}

if(srcPixelFormat == AV_PIX_FMT_GBRP && dstPixelFormat == AV_PIX_FMT_YUV422P){
	// Calculate once
	long numElements = srcStride[0] * srcHeight;

	// Loop through each pixel
	for(int index = 0; index < numElements; index++){
		uint8_t r = srcSlice[0][index]; // R
		uint8_t g = srcSlice[1][index]; // G
		uint8_t b = srcSlice[2][index]; // B

		double y = 0.299 * static_cast<double>(r) + 0.587 * static_cast<double>(g) + 0.114 * static_cast<double>(b) + 0;
		double cb = -0.169 * static_cast<double>(r) - 0.331 * static_cast<double>(g) + 0.499 * static_cast<double>(b) + 128;
		double cr = 0.499 * static_cast<double>(r) - 0.418 * static_cast<double>(g) - 0.0813 * static_cast<double>(b) + 128;

		dstSlice[0][index] = round(y);          // Y

		if(index % 2 == 0){
			dstSlice[1][index / 2] = round(cb); // U
			dstSlice[2][index / 2] = round(cr); // V
		}
	}

	// Success
	return 0;
}