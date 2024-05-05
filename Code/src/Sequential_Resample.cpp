#include "Sequential_Resample.h"

// Convert the pixel format of the image
template <class PrecisionType>
int sequential_formatConversion(int srcWidth, int srcHeight,
    int srcPixelFormat, uint8_t* srcSlice[],
    int dstPixelFormat, uint8_t* dstSlice[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        // Copy data between buffers
        if(srcPixelFormat == AV_PIX_FMT_V210){
            memcpy(dstSlice[0], srcSlice[0], ((srcWidth + 47) / 48) * 128 * srcHeight);
        } else if(srcPixelFormat == AV_PIX_FMT_UYVY422){
            memcpy(dstSlice[0], srcSlice[0], srcWidth * 2 * srcHeight);
        } else{
            // Chroma size discovery
            float widthPerc = 1.f;
            float heightPerc = 1.f;
            if(srcPixelFormat == AV_PIX_FMT_YUV422P ||
                srcPixelFormat == AV_PIX_FMT_YUV420P ||
                srcPixelFormat == AV_PIX_FMT_YUV422PNORM)
                widthPerc = 0.5f;
            if(srcPixelFormat == AV_PIX_FMT_YUV420P)
                heightPerc = 0.5f;

            memcpy(dstSlice[0], srcSlice[0], srcWidth * srcHeight);
            memcpy(dstSlice[1], srcSlice[1], srcWidth * srcHeight * widthPerc * heightPerc);
            memcpy(dstSlice[2], srcSlice[2], srcWidth * srcHeight * widthPerc * heightPerc);
        }

        // Success
        return 0;
    }

    // REORGANIZE COMPONENTS -------------------------
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Number of elements
        long numElements = srcWidth * srcHeight / 2;

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
        int stride = srcWidth * 2;

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
        int stride = srcWidth * 2;

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

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210){
        // Number of elements
        long numElements = srcWidth * srcHeight / 6;

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
        long numElements = srcWidth * srcHeight / 2;

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
        int stride = srcWidth;

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

                *dstBufferChromaU++ = roundTo<uint8_t, PrecisionType>((u0 + u1) / static_cast<PrecisionType>(2.));
                *dstBufferChromaV++ = roundTo<uint8_t, PrecisionType>((v0 + v1) / static_cast<PrecisionType>(2.));
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
        int stride = srcWidth;

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

                *dstBufferChroma++ = roundTo<uint8_t, PrecisionType>((u0 + u1) / static_cast<PrecisionType>(2.));
                *dstBufferChroma++ = roundTo<uint8_t, PrecisionType>((v0 + v1) / static_cast<PrecisionType>(2.));
            }

            srcBufferChromaU += strideDiv2;
            srcBufferChromaV += strideDiv2;
            srcBufferChromaUBelow += strideDiv2;
            srcBufferChromaVBelow += strideDiv2;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_V210){
        // Number of elements
        long numElements = srcWidth * srcHeight / 6;

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

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_UYVY422){
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

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422P){
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

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422PNORM){
        // Number of elements
        long numElements = ((srcWidth + 47) / 48) * 128 * srcHeight / 16;

        // Buffer pointers
        auto srcBuffer = reinterpret_cast<uint32_t*>(srcSlice[0]);
        auto dstBuffer = dstSlice[0];
        auto dstBufferChromaU = dstSlice[1];
        auto dstBufferChromaV = dstSlice[2];

        // Calculate once
        PrecisionType valueConstLuma = static_cast<PrecisionType>(219.) / static_cast<PrecisionType>(1023.);
        PrecisionType valueConstChroma = static_cast<PrecisionType>(224.) / static_cast<PrecisionType>(1023.);
        PrecisionType value16 = static_cast<PrecisionType>(16.);

        // Assign once
        enum{ SHIFT_RIGHT = 20U, SHIFT_MIDDLE = 10U, X3FF = 0x3FF, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0 = *srcBuffer & X3FF; // U0
            auto y0 = (*srcBuffer >> SHIFT_MIDDLE) & X3FF; // Y0
            auto v0 = (*srcBuffer >> SHIFT_RIGHT) & X3FF; // V0
            *srcBuffer++;

            auto y1 = *srcBuffer & X3FF; // Y1
            auto u1 = (*srcBuffer >> SHIFT_MIDDLE) & X3FF; // U1
            auto y2 = (*srcBuffer >> SHIFT_RIGHT) & X3FF; // Y2
            *srcBuffer++;

            auto v1 = *srcBuffer & X3FF; // V1
            auto y3 = (*srcBuffer >> SHIFT_MIDDLE) & X3FF; // Y3
            auto u2 = (*srcBuffer >> SHIFT_RIGHT) & X3FF; // U2
            *srcBuffer++;

            auto y4 = *srcBuffer & X3FF; // Y4
            auto v2 = (*srcBuffer >> SHIFT_MIDDLE) & X3FF; // V2
            auto y5 = (*srcBuffer >> SHIFT_RIGHT) & X3FF; // Y5
            *srcBuffer++;

            *(dstBufferChromaU++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(u0) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y0) * valueConstLuma + value16);
            *(dstBufferChromaV++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(v0) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y1) * valueConstLuma + value16);

            *(dstBufferChromaU++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(u1) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y2) * valueConstLuma + value16);
            *(dstBufferChromaV++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(v1) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y3) * valueConstLuma + value16);

            *(dstBufferChromaU++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(u2) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y4) * valueConstLuma + value16);
            *(dstBufferChromaV++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(v2) * valueConstChroma + value16);
            *(dstBuffer++) = roundTo<uint8_t, PrecisionType>(static_cast<PrecisionType>(y5) * valueConstLuma + value16);
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422PNORM && dstPixelFormat == AV_PIX_FMT_V210){
        // Number of elements
        long numElements = srcWidth * srcHeight / 6;

        // Buffer pointers
        auto srcBuffer = srcSlice[0];
        auto srcBufferChromaU = srcSlice[1];
        auto srcBufferChromaV = srcSlice[2];
        auto dstBuffer = reinterpret_cast<uint32_t*>(dstSlice[0]);

        // Calculate once
        PrecisionType value16 = static_cast<PrecisionType>(16.);
        PrecisionType valueConstLuma = static_cast<PrecisionType>(1023.) / static_cast<PrecisionType>(219.);
        PrecisionType valueConstChroma = static_cast<PrecisionType>(1023.) / static_cast<PrecisionType>(224.);

        // Assign once
        enum{ SHIFT_LEFT = 20U, SHIFT_MIDDLE = 10U, X3FF = 0x3FF, };

        // Loop through each pixel
        for(int index = 0; index < numElements; index++){
            auto u0bpp8 = *srcBufferChromaU++; // U0
            auto y0bpp8 = *srcBuffer++; // Y0
            auto v0bpp8 = *srcBufferChromaV++; // V0
            auto y1bpp8 = *srcBuffer++; // Y1

            auto u1bpp8 = *srcBufferChromaU++; // U1
            auto y2bpp8 = *srcBuffer++; // Y2
            auto v1bpp8 = *srcBufferChromaV++; // V1
            auto y3bpp8 = *srcBuffer++; // Y3

            auto u2bpp8 = *srcBufferChromaU++; // U2
            auto y4bpp8 = *srcBuffer++; // Y4
            auto v2bpp8 = *srcBufferChromaV++; // V2
            auto y5bpp8 = *srcBuffer++; // Y5

            auto v0 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(v0bpp8) - value16) * valueConstChroma) & X3FF;
            auto y0 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y0bpp8) - value16) * valueConstLuma) & X3FF;
            auto u0 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(u0bpp8) - value16) * valueConstChroma) & X3FF;
            auto y2 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y2bpp8) - value16) * valueConstLuma) & X3FF;

            auto u1 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(u1bpp8) - value16) * valueConstChroma) & X3FF;
            auto y1 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y1bpp8) - value16) * valueConstLuma) & X3FF;
            auto u2 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(u2bpp8) - value16) * valueConstChroma) & X3FF;
            auto y3 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y3bpp8) - value16) * valueConstLuma) & X3FF;

            auto v1 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(v1bpp8) - value16) * valueConstChroma) & X3FF;
            auto y5 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y5bpp8) - value16) * valueConstLuma) & X3FF;
            auto v2 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(v2bpp8) - value16) * valueConstChroma) & X3FF;
            auto y4 = roundTo<uint16_t, PrecisionType>((static_cast<PrecisionType>(y4bpp8) - value16) * valueConstLuma) & X3FF;

            *dstBuffer++ = (v0 << SHIFT_LEFT) | (y0 << SHIFT_MIDDLE) | u0;
            *dstBuffer++ = (y2 << SHIFT_LEFT) | (u1 << SHIFT_MIDDLE) | y1;
            *dstBuffer++ = (u2 << SHIFT_LEFT) | (y3 << SHIFT_MIDDLE) | v1;
            *dstBuffer++ = (y5 << SHIFT_LEFT) | (v2 << SHIFT_MIDDLE) | y4;
        }

        // Success
        return 0;
    }

    // No conversion was supported
    return -1;
}

// Precalculate coefficients
template <class PrecisionType>
int sequential_preCalculateCoefficients(int srcSize, int dstSize, int operation,
    int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType** &preCalculatedCoefs){

    // Calculate size ratio
    PrecisionType sizeRatio = static_cast<PrecisionType>(dstSize) / static_cast<PrecisionType>(srcSize);

    // Calculate once
    PrecisionType pixelSupportDiv2 = pixelSupport / static_cast<PrecisionType>(2.);
    bool isDownScale = sizeRatio < static_cast<PrecisionType>(1.);
    PrecisionType regionRadius = isDownScale ? pixelSupportDiv2 / sizeRatio : pixelSupportDiv2;
    PrecisionType filterStep = isDownScale && operation != SWS_POINT ? static_cast<PrecisionType>(1.) / sizeRatio : static_cast<PrecisionType>(1.);
    int numCoefficients = isDownScale ? ceil(pixelSupport / sizeRatio) : pixelSupport;
    int numCoefficientsDiv2 = numCoefficients / 2;

    // Calculate number of lines of coefficients
    int preCalcCoefSize = isDownScale ? dstSize : lcm(srcSize, dstSize) / min<int>(srcSize, dstSize);

    // Initialize 2d array
    preCalculatedCoefs = static_cast<PrecisionType**>(malloc(preCalcCoefSize * sizeof(PrecisionType*)));
    for(int index = 0; index < preCalcCoefSize; index++)
        preCalculatedCoefs[index] = static_cast<PrecisionType*>(malloc(numCoefficients * sizeof(PrecisionType)));

    // For each necessary line of coefficients
    for(int col = 0; col < preCalcCoefSize; col++){
        // Original line index coordinate
        PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(.5)) / sizeRatio;

        // Discover source limit pixels
        PrecisionType nearPixel = colOriginal - filterStep;
        PrecisionType leftPixel = colOriginal - regionRadius;

        // Discover offset to pixel of filter start
        PrecisionType offset = round(leftPixel) + static_cast<PrecisionType>(.5) - leftPixel;
        // Calculate maximum distance to normalize distances
        PrecisionType maxDistance = colOriginal - nearPixel;
        // Calculate where filtering will start
        PrecisionType startPosition = leftPixel + offset;

        // Calculate coefficients
        PrecisionType coefAcc = static_cast<PrecisionType>(0.);
        for(int index = 0; index < numCoefficients; index++){
            PrecisionType coefHolder = coefFunc((colOriginal - (startPosition + index)) / maxDistance);
            coefAcc += coefHolder;
            preCalculatedCoefs[col][index] = coefHolder;
        }

        // Avoid lines of coefficients without valid values
        if(operation == SWS_POINT){
            if(preCalculatedCoefs[col][numCoefficientsDiv2 - 1] == preCalculatedCoefs[col][numCoefficientsDiv2]){
                if(isDownScale){
                    if(preCalculatedCoefs[col][numCoefficientsDiv2 - 1] == static_cast<PrecisionType>(0.) && numCoefficients % 2 != 0)
                        preCalculatedCoefs[col][numCoefficientsDiv2 - 1] = static_cast<PrecisionType>(1.);
                    else
                        preCalculatedCoefs[col][numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
                } else
                    preCalculatedCoefs[col][numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
            }
        }

        // Normalizes coefficients except on Nearest Neighbor interpolation
        if(operation != SWS_POINT)
            for(int index = 0; index < numCoefficients; index++)
                preCalculatedCoefs[col][index] /= coefAcc;
    }

    // Success
    return preCalcCoefSize;
}

// Change the image dimension
template <class PrecisionType>
void sequential_resize(int srcWidth, int srcHeight, uint8_t* srcData,
    int dstWidth, int dstHeight, uint8_t* dstData,
    int operation, int pixelSupport,
    int vCoefsSize, PrecisionType** vCoefs, int hCoefsSize, PrecisionType** hCoefs,
    int colorChannel){

    // Get scale ratios
    PrecisionType scaleHeightRatio = static_cast<PrecisionType>(dstHeight) / static_cast<PrecisionType>(srcHeight);
    PrecisionType scaleWidthRatio = static_cast<PrecisionType>(dstWidth) / static_cast<PrecisionType>(srcWidth);

    // Calculate once
    PrecisionType pixelSupportDiv2 = pixelSupport / static_cast<PrecisionType>(2.);
    bool isDownScaleV = scaleHeightRatio < static_cast<PrecisionType>(1.);
    bool isDownScaleH = scaleWidthRatio < static_cast<PrecisionType>(1.);
    PrecisionType regionVRadius = isDownScaleV ? pixelSupportDiv2 / scaleHeightRatio : pixelSupportDiv2;
    PrecisionType regionHRadius = isDownScaleH ? pixelSupportDiv2 / scaleWidthRatio : pixelSupportDiv2;
    int numVCoefs = isDownScaleV ? ceil(pixelSupport / scaleHeightRatio) : pixelSupport;
    int numHCoefs = isDownScaleH ? ceil(pixelSupport / scaleWidthRatio) : pixelSupport;

    // Iterate through each line of the scaled image
    for(int lin = 0; lin < dstHeight; lin++){
        // Calculate once the target line coordinate
        int targetLine = lin * dstWidth;
        // Calculate once the line index of coefficients
        int indexLinCoef = lin % vCoefsSize;

        // Original line index coordinate
        PrecisionType linOriginal = (static_cast<PrecisionType>(lin) + static_cast<PrecisionType>(.5)) / scaleHeightRatio;

        // Discover source limit pixels
        PrecisionType upperPixel = linOriginal - regionVRadius;
        // Discover offset to pixel of filter start
        PrecisionType offsetV = round(upperPixel) + static_cast<PrecisionType>(.5) - upperPixel;

        // Calculate once
        PrecisionType startLinPosition = upperPixel + offsetV;

        // Iterate through each column of the scaled image
        for(int col = 0; col < dstWidth; col++){
            // Calculate once the column index of coefficients
            int indexColCoef = col % hCoefsSize;

            // Original line index coordinate
            PrecisionType colOriginal = (static_cast<PrecisionType>(col) + static_cast<PrecisionType>(.5)) / scaleWidthRatio;

            // Discover source limit pixels
            PrecisionType leftPixel = colOriginal - regionHRadius;
            // Discover offset to pixel of filter start
            PrecisionType offsetH = round(leftPixel) + static_cast<PrecisionType>(.5) - leftPixel;

            // Calculate once
            PrecisionType startColPosition = leftPixel + offsetH;

            // Temporary variables used in the interpolation
            PrecisionType result = static_cast<PrecisionType>(0.);
            // Calculate resulting color from coefficients
            for(int indexV = 0; indexV < numVCoefs; indexV++){
                // Access once the memory
                PrecisionType vCoef = vCoefs[indexLinCoef][indexV];

                for(int indexH = 0; indexH < numHCoefs; indexH++){
                    // Access once the memory
                    PrecisionType hCoef = hCoefs[indexColCoef][indexH];

                    // Get pixel from source data
                    uint8_t colorHolder = getPixel(startLinPosition + indexV, startColPosition + indexH, srcWidth, srcHeight, srcData);

                    // Calculate pixel color weight
                    PrecisionType weight = vCoef * hCoef;

                    // Weights neighboring pixel and add it to the result
                    result += static_cast<PrecisionType>(colorHolder) * weight;
                }
            }

            // Clamp value to avoid undershooting and overshooting
            if(colorChannel == 0)
                clamp<PrecisionType>(result, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(235.));
            else
                clamp<PrecisionType>(result, static_cast<PrecisionType>(16.), static_cast<PrecisionType>(240.));
            // Assign calculated color to destiantion data
            dstData[targetLine + col] = roundTo<uint8_t, PrecisionType>(result);
        }
    }
}

// Prepares the resample operation
template <class PrecisionType>
int sequential_resample_aux(AVFrame* src, AVFrame* dst, int operation){
    // Return value of this method
    int returnValue = 0;

    // Access once
    int srcWidth = src->width, srcHeight = src->height;
    int srcFormat = src->format;
    int dstWidth = dst->width, dstHeight = dst->height;
    int dstFormat = dst->format;

    // Check if is only a format conversion
    bool isOnlyFormatConversion = srcWidth == dstWidth && srcHeight == dstHeight;

    // Initialize needed variables if it is a scaling operation
    int scalingSupportedFormat = getTempScaleFormat(srcFormat, dstFormat);

    // Temporary buffers used in intermediate operations
    uint8_t** formatConversionBuffer, **resizeBuffer;

    // Last format conversion buffers
    uint8_t** lastFormatConversionBuffer = src->data;
    int lastFormatConversionPixelFormat = srcFormat;

    // Rescaling operation branch
    if(!isOnlyFormatConversion){
        // Allocate temporary buffers
        allocBuffers(formatConversionBuffer, srcWidth, srcHeight, scalingSupportedFormat);
        allocBuffers(resizeBuffer, dstWidth, dstHeight, scalingSupportedFormat);

        // Resamples image to a supported format
        if(sequential_formatConversion<PrecisionType>(srcWidth, srcHeight,
            srcFormat, src->data,
            scalingSupportedFormat, formatConversionBuffer) < 0){
            returnValue = -1;
            goto END;
        }

        // Needed resources for coefficients calculations
        PrecisionType(*coefFunc)(PrecisionType) = getCoefMethod<PrecisionType>(operation);
        int pixelSupport = getPixelSupport(operation);

        // Variables for precalculated coefficients
        PrecisionType** vCoefs;
        int vCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefs);
        PrecisionType** hCoefs;
        int hCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcWidth, dstWidth, operation, pixelSupport, coefFunc, hCoefs);

        // Chroma size discovery
        float widthPerc = 1.f;
        float heightPerc = 1.f;
        if(scalingSupportedFormat == AV_PIX_FMT_YUV422P ||
            scalingSupportedFormat == AV_PIX_FMT_YUV420P ||
            scalingSupportedFormat == AV_PIX_FMT_YUV422PNORM)
            widthPerc = 0.5f;
        if(scalingSupportedFormat == AV_PIX_FMT_YUV420P)
            heightPerc = 0.5f;

        // Apply the resizing operation to luma channel
        sequential_resize<PrecisionType>(srcWidth, srcHeight, formatConversionBuffer[0],
            dstWidth, dstHeight, resizeBuffer[0], operation,
            pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, 0);

        // Apply the resizing operation to chroma channels
        int srcWidthChroma = static_cast<int>(srcWidth * widthPerc);
        int srcHeightChroma = static_cast<int>(srcHeight * heightPerc);
        int dstWidthChroma = static_cast<int>(dstWidth * widthPerc);
        int dstHeightChroma = static_cast<int>(dstHeight * heightPerc);
        for(int colorChannel = 1; colorChannel < 3; colorChannel++){
            sequential_resize<PrecisionType>(srcWidthChroma, srcHeightChroma, formatConversionBuffer[colorChannel],
                dstWidthChroma, dstHeightChroma, resizeBuffer[colorChannel], operation,
                pixelSupport, vCoefsSize, vCoefs, hCoefsSize, hCoefs, colorChannel);
        }


        // Assign correct values to apply last resample
        lastFormatConversionBuffer = resizeBuffer;
        lastFormatConversionPixelFormat = scalingSupportedFormat;

        // Free used resources
        free2dBuffer<PrecisionType>(vCoefs, vCoefsSize);
        free2dBuffer<PrecisionType>(hCoefs, hCoefsSize);
    }

    // Resamples image to a target format
    if(sequential_formatConversion<PrecisionType>(dstWidth, dstHeight,
        lastFormatConversionPixelFormat, lastFormatConversionBuffer,
        dstFormat, dst->data) < 0){
        returnValue = -1;
        goto END;
    }

    END:
    // Free used resources
    if(!isOnlyFormatConversion){
        free2dBuffer<uint8_t>(formatConversionBuffer, 3);
        free2dBuffer<uint8_t>(resizeBuffer, 3);
    }

    // Return negative if insuccess
    return returnValue;
}

// Wrapper for the sequential resample operation method
int sequential_resample(AVFrame* src, AVFrame* dst, int operation){
    // Variables used
    int duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

    // Verify valid frames
    if(src == nullptr || dst == nullptr){
        cerr << "[SEQUENTIAL] One or both input frames are null!" << endl;
        return -1;
    }

    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid input dimensions
    if(src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0){
        cerr << "[SEQUENTIAL] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }
    // Verify valid resize
    if((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)){
        cerr << "[SEQUENTIAL] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }
    // Verify valid input data
    if(!src->data || !src->linesize || !dst->data || !dst->linesize){
        cerr << "[SEQUENTIAL] Frame data buffers can not be null!" << endl;
        return -1;
    }
    // Verify if supported pixel formats
    if(!isSupportedFormat(srcFormat) || !isSupportedFormat(dstFormat)){
        cerr << "[SEQUENTIAL] Frame pixel format is not supported!" << endl;
        return -1;
    }
    // Verify if can convert a 10 bit format
    if((src->width % 12 != 0 && srcFormat == AV_PIX_FMT_V210) || (dst->width % 12 != 0 && dstFormat == AV_PIX_FMT_V210)){
        cerr << "[SEQUENTIAL] Can not handle 10 bit format because data is not aligned!" << endl;
        return -1;
    }
    // Verify if supported scaling operation
    if(!isSupportedOperation(operation)){
        cerr << "[SEQUENTIAL] Scaling operation is not supported" << endl;
        return -1;
    }

    // Start counting operation execution time
    initTime = high_resolution_clock::now();

    // Apply the scaling operation
    if(sequential_resample_aux<double>(src, dst, operation) < 0){
        // Display error
        cerr << "[SEQUENTIAL] Operation could not be done (resample - conversion not supported)!" << endl;

        // Insuccess
        return -1;
    }

    // Stop counting operation execution time
    stopTime = high_resolution_clock::now();

    // Calculate the execution time
    duration = duration_cast<microseconds>(stopTime - initTime).count();

    // Return execution time of the scaling operation
    return duration;
}