#include "Sequential_Resample.h"

// Convert the pixel format of the image
template <class PrecisionType>
int sequential_formatConversion(int width, int height,
    int srcPixelFormat, uint8_t* srcSlice[],
    int dstPixelFormat, uint8_t* dstSlice[]){

    // If same formats no need to resample
    if(srcPixelFormat == dstPixelFormat){
        // Copy data between buffers
        if(srcPixelFormat == AV_PIX_FMT_V210){
            memcpy(dstSlice[0], srcSlice[0], ((width + 47) / 48) * 128 * height);
        } else if(srcPixelFormat == AV_PIX_FMT_UYVY422){
            memcpy(dstSlice[0], srcSlice[0], width * 2 * height);
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

            memcpy(dstSlice[0], srcSlice[0], width * height);
            memcpy(dstSlice[1], srcSlice[1], width * height * widthPerc * heightPerc);
            memcpy(dstSlice[2], srcSlice[2], width * height * widthPerc * heightPerc);
        }

        // Success
        return 0;
    }

    #pragma region UYVY422
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Discover buffer pointers
        auto srcB = srcSlice[0];

        auto dstB = dstSlice[0];
        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Iterate blocks of 1x4 channel points
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
            for(int hIndex = 0; hIndex < hStrideUYVY422 / 4; hIndex++){
                *dstU++ = *srcB++; // U0
                *dstB++ = *srcB++; // Y0
                *dstV++ = *srcB++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcBb = srcB + hStrideUYVY422;

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideYUV420P;

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Iterate blocks of 2x4 channel points
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
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

            // At the end of each line of block 2x4 corrects pointers
            srcB = srcBb;
            srcBb += hStrideUYVY422;

            dstB = dstBb;
            dstBb += hStrideYUV420P;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcBb = srcB + hStrideUYVY422;

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideNV12;

        auto dstC = dstSlice[1];

        // Iterate blocks of 2x4 channel points
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
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

            // At the end of each line of block 2x4 corrects pointers
            srcB = srcBb;
            srcBb += hStrideUYVY422;

            dstB = dstBb;
            dstBb += hStrideNV12;
        }
        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Discover buffer pointers
        auto srcB = srcSlice[0];

        auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]);

        // Iterate blocks of 1x12 channel points
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
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

        // Success
        return 0;
    }
    #pragma endregion

    #pragma region YUV422P
    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto dstB = dstSlice[0];

        // Iterate blocks of 1x2 channel points
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            for(int hIndex = 0; hIndex < hStrideYUV422P / 2; hIndex++){
                *dstB++ = *srcU++; // U0
                *dstB++ = *srcB++; // Y0
                *dstB++ = *srcV++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        // Discover buffer pointers
        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto srcUb = srcU + hStrideYUV422PChroma;
        auto srcVb = srcV + hStrideYUV422PChroma;

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

        // Iterate blocks of 2x1 channel points
        for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
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

            // At the end of each line of block 2x1 corrects pointers
            srcU = srcUb;
            srcUb += hStrideYUV422PChroma;

            srcV = srcVb;
            srcVb += hStrideYUV422PChroma;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        // Discover buffer pointers
        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto srcUb = srcU + hStrideYUV422PChroma;
        auto srcVb = srcV + hStrideYUV422PChroma;

        auto dstC = dstSlice[1];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

        // Iterate blocks of 2x1 channel points
        for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
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

            // At the end of each line of block 2x1 corrects pointers
            srcU = srcUb;
            srcUb += hStrideYUV422PChroma;

            srcV = srcVb;
            srcVb += hStrideYUV422PChroma;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Discover buffer pointers
        auto srcB = srcSlice[0];

        auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]);

        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        // Iterate blocks of 1x6 channel points
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
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

        // Success
        return 0;
    }
    #pragma endregion

    #pragma region YUV420P
    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcBb = srcB + hStrideYUV420P;

        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideUYVY422;

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
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

            // At the end of each line of block 2x2 corrects pointers
            srcB = srcBb;
            srcBb += width;

            dstB = dstBb;
            dstBb += hStrideUYVY422;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        // Discover buffer pointers
        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        auto dstUb = dstU + hStrideYUV422PChroma;
        auto dstVb = dstV + hStrideYUV422PChroma;

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            for(int hIndex = 0; hIndex < hStrideYUV422PChroma; hIndex++){
                // Get chroma values
                uint8_t u = *srcU++; // U
                uint8_t v = *srcV++; // V

                // Assign values dupicated
                *dstU++ = u;
                *dstV++ = v;

                *dstUb++ = u;
                *dstVb++ = v;
            }

            // At the end of each line of block 2x2 corrects pointers
            dstU = dstUb;
            dstUb += hStrideYUV422PChroma;

            dstV = dstVb;
            dstVb += hStrideYUV422PChroma;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Discover buffer pointers
        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        auto dstC = dstSlice[1];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                *dstC++ = *srcU++; // U
                *dstC++ = *srcV++; // V
            }
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Discover buffer pointers
        auto srcY = srcSlice[0];
        auto srcYb = srcY + hStrideYUV420P;

        auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]);
        auto dstBb = dstB + hStrideV210;

        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            for(int hIndex = 0; hIndex < hStrideYUV420P / 6; hIndex++){
                // Get lumas from above line
                auto y0 = *srcY++ << 2U;
                auto y1 = *srcY++ << 2U;
                auto y2 = *srcY++ << 2U;
                auto y3 = *srcY++ << 2U;
                auto y4 = *srcY++ << 2U;
                auto y5 = *srcY++ << 2U;

                // Get lumas from below line
                auto y0b = *srcYb++ << 2U;
                auto y1b = *srcYb++ << 2U;
                auto y2b = *srcYb++ << 2U;
                auto y3b = *srcYb++ << 2U;
                auto y4b = *srcYb++ << 2U;
                auto y5b = *srcYb++ << 2U;

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

            // At the end of each line of block 2x2 corrects pointers
            srcY = srcYb;
            srcYb += hStrideYUV420P;

            dstB = dstBb;
            dstBb += hStrideV210;
        }

        // Success
        return 0;
    }
    #pragma endregion

    #pragma region NV12
    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcBb = srcB + hStrideNV12;

        auto srcC = srcSlice[1];

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideUYVY422;

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
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

            // At the end of each line of block 2x2 corrects pointers
            srcB = srcBb;
            srcBb += width;

            dstB = dstBb;
            dstBb += hStrideUYVY422;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        // Discover buffer pointers
        auto srcC = srcSlice[1];

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        auto dstUb = dstU + hStrideYUV422PChroma;
        auto dstVb = dstV + hStrideYUV422PChroma;

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
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

            // At the end of each line of block 2x2 corrects pointers
            dstU = dstUb;
            dstUb += hStrideYUV422PChroma;

            dstV = dstVb;
            dstVb += hStrideYUV422PChroma;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Discover buffer pointers
        auto srcC = srcSlice[1];

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Luma plane is the same
        memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideNV12; vIndex++){
            for(int hIndex = 0; hIndex < hStrideNV12 / 4; hIndex++){
                *dstU++ = *srcC++; // U
                *dstV++ = *srcC++; // V
            }
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Discover buffer pointers
        auto srcB = srcSlice[0];
        auto srcBb = srcB + hStrideNV12;

        auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]);
        auto dstBb = dstB + hStrideV210;

        auto srcC = srcSlice[1];

        // Iterate blocks of 2x2 channel points
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
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

            // At the end of each line of block 2x2 corrects pointers
            srcB = srcBb;
            srcBb += hStrideNV12;

            dstB = dstBb;
            dstBb += hStrideV210;
        }

        // Success
        return 0;
    }
    #pragma endregion

    #pragma region V210
    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Discover buffer pointers
        auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]);

        auto dstB = dstSlice[0];

        // Iterate blocks of 1x4 channel points
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
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

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Discover buffer pointers
        auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]);

        auto dstB = dstSlice[0];

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Iterate blocks of 1x4 channel points
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
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

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Discover buffer pointers
        auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]);
        auto srcBb = srcB + hStrideV210;

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideYUV420P;

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Iterate blocks of 2x4 channel points
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
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

            // At the end of each line of block 2x4 corrects pointers
            srcB = srcBb;
            srcBb += hStrideV210;

            dstB = dstBb;
            dstBb += hStrideYUV420P;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Discover buffer pointers
        auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]);
        auto srcBb = srcB + hStrideV210;

        auto dstB = dstSlice[0];
        auto dstBb = dstB + hStrideNV12;

        auto dstC = dstSlice[1];

        // Iterate blocks of 2x4 channel points
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
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

            // At the end of each line of block 2x4 corrects pointers
            srcB = srcBb;
            srcBb += hStrideV210;

            dstB = dstBb;
            dstBb += hStrideNV12;
        }

        // Success
        return 0;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422PNORM){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Discover buffer pointers
        auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]);

        auto dstB = dstSlice[0];

        auto dstU = dstSlice[1];
        auto dstV = dstSlice[2];

        // Create const for normalization
        double constLuma = 219. / 1023.;
        double constChroma = 224. / 1023.;
        double const16 = 16.;

        // Iterate blocks of 1x4 channel points
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
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

        // Success
        return 0;
    }
    #pragma endregion

    #pragma region YUV422PNORM
    if(srcPixelFormat == AV_PIX_FMT_YUV422PNORM && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Discover buffer pointers
        auto srcB = srcSlice[0];

        auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]);

        auto srcU = srcSlice[1];
        auto srcV = srcSlice[2];

        // Create const for normalization
        double const16 = 16.;
        double constLuma = 1023. / 219.;
        double constChroma = 1023. / 224.;

        // Iterate blocks of 1x6 channel points
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            for(int hIndex = 0; hIndex < hStrideYUV422P / 6; hIndex++){
                // Get components from source
                auto u0n = *srcU++ << 2U; // U0
                auto y0n = *srcB++ << 2U; // Y0
                auto v0n = *srcV++ << 2U; // V0
                auto y1n = *srcB++ << 2U; // Y1

                auto u1n = *srcU++ << 2U; // U1
                auto y2n = *srcB++ << 2U; // Y2
                auto v1n = *srcV++ << 2U; // V1
                auto y3n = *srcB++ << 2U; // Y3

                auto u2n = *srcU++ << 2U; // U2
                auto y4n = *srcB++ << 2U; // Y4
                auto v2n = *srcV++ << 2U; // V2
                auto y5n = *srcB++ << 2U; // Y5

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

        // Success
        return 0;
    }
    #pragma endregion

    // No conversion was supported
    return -1;
}

// Precalculate coefficients
template <class PrecisionType>
int sequential_preCalculateCoefficients(int srcSize, int dstSize, int operation,
    int pixelSupport, PrecisionType(*coefFunc)(PrecisionType), PrecisionType* &preCalculatedCoefs){

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

    // Initialize array
    preCalculatedCoefs = static_cast<PrecisionType*>(malloc(preCalcCoefSize * numCoefficients * sizeof(PrecisionType)));

    // For each necessary line of coefficients
    for(int col = 0; col < preCalcCoefSize; col++){
        // Calculate once
        int indexOffset = col * numCoefficients;

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
            preCalculatedCoefs[indexOffset + index] = coefHolder;
        }

        // Avoid lines of coefficients without valid values
        if(operation == SWS_POINT){
            if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == preCalculatedCoefs[indexOffset + numCoefficientsDiv2]){
                if(isDownScale){
                    if(preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] == static_cast<PrecisionType>(0.) && numCoefficients % 2 != 0)
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2 - 1] = static_cast<PrecisionType>(1.);
                    else
                        preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
                } else
                    preCalculatedCoefs[indexOffset + numCoefficientsDiv2] = static_cast<PrecisionType>(1.);
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
template <class PrecisionType>
void sequential_resize(int srcWidth, int srcHeight, uint8_t* srcData,
    int dstWidth, int dstHeight, uint8_t* dstData,
    int operation, int pixelSupport,
    int vCoefsSize, PrecisionType* &vCoefs, int hCoefsSize, PrecisionType* &hCoefs,
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
        int indexLinOffset = (lin % vCoefsSize) * numVCoefs;

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
            int indexColOffset = (col % hCoefsSize) * numHCoefs;

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
                PrecisionType vCoef = vCoefs[indexLinOffset + indexV];

                for(int indexH = 0; indexH < numHCoefs; indexH++){
                    // Access once the memory
                    PrecisionType hCoef = hCoefs[indexColOffset + indexH];

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
        PrecisionType* vCoefs;
        int vCoefsSize = sequential_preCalculateCoefficients<PrecisionType>(srcHeight, dstHeight, operation, pixelSupport, coefFunc, vCoefs);
        PrecisionType* hCoefs;
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
        free(vCoefs);
        free(hCoefs);
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
    // Access once
    AVPixelFormat srcFormat = static_cast<AVPixelFormat>(src->format);
    AVPixelFormat dstFormat = static_cast<AVPixelFormat>(dst->format);

    // Verify valid frames
    if(src == nullptr || dst == nullptr){
        cerr << "[SEQUENTIAL] One or both input frames are null!" << endl;
        return -1;
    }

    // Verify valid input data
    if(!src->data || !src->linesize || !dst->data || !dst->linesize){
        cerr << "[SEQUENTIAL] Frame data buffers can not be null!" << endl;
        return -1;
    }

    // Verify valid input dimensions
    if(src->width < 0 || src->height < 0 || dst->width < 0 || dst->height < 0){
        cerr << "[SEQUENTIAL] Frame dimensions can not be a negative number!" << endl;
        return -1;
    }

    // Verify if data is aligned
    if(((src->width % 4 != 0 && srcFormat == AV_PIX_FMT_UYVY422) || (dst->width % 4 != 0 && dstFormat == AV_PIX_FMT_UYVY422)) &&
        ((src->width % 12 != 0 && srcFormat == AV_PIX_FMT_V210) || (dst->width % 12 != 0 && dstFormat == AV_PIX_FMT_V210))){
        cerr << "[SEQUENTIAL] Can not handle unaligned data!" << endl;
        return -1;
    }

    // Verify valid resize
    if((src->width < dst->width && src->height > dst->height) ||
        (src->width > dst->width && src->height < dst->height)){
        cerr << "[SEQUENTIAL] Can not upscale in an orientation and downscale another!" << endl;
        return -1;
    }

    // Verify if supported conversion
    if(!hasSupportedConversion(srcFormat, dstFormat)){
        cerr << "[SEQUENTIAL] Pixel format conversion is not supported!" << endl;
        return -1;
    }

    // Verify if supported scaling operation
    if(!isSupportedOperation(operation)){
        cerr << "[SEQUENTIAL] Scaling operation is not supported" << endl;
        return -1;
    }

    // Variables used
    int duration = -1;
    high_resolution_clock::time_point initTime, stopTime;

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