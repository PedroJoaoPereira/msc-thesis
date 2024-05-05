#include "OpenMP_Utils.h"

// Convert the pixel format of the image
void omp_formatConversion(int width, int height, int srcPixelFormat, uint8_t* srcSlice[], int dstPixelFormat, uint8_t* dstSlice[]){
    #pragma region UYVY422
    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideUYVY422 * hStrideUYVY422);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideUYVY422 / 4; hIndex++){
                *dstU++ = *srcB++; // U0
                *dstB++ = *srcB++; // Y0
                *dstV++ = *srcB++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422 * 2;
            auto srcBb = srcB + hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV420P * 2;
            auto dstBb = dstB + hStrideYUV420P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422 * 2;
            auto srcBb = srcB + hStrideUYVY422;
            auto dstB = dstSlice[0] + vIndex * hStrideNV12 * 2;
            auto dstBb = dstB + hStrideNV12;
            auto dstC = dstSlice[1] + vIndex * hStrideNV12;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_UYVY422 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 1x12 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideUYVY422; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideUYVY422;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;

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

        return;
    }
    #pragma endregion

    #pragma region YUV422P
    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 1x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = dstSlice[0] + vIndex * hStrideUYVY422;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV422P / 2; hIndex++){
                *dstB++ = *srcU++; // U0
                *dstB++ = *srcB++; // Y0
                *dstB++ = *srcV++; // V0
                *dstB++ = *srcB++; // Y1
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);
        memcpy(dstSlice[1], srcSlice[1], vStrideYUV422P * hStrideYUV422P / 2);
        memcpy(dstSlice[2], srcSlice[2], vStrideYUV422P * hStrideYUV422P / 2);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

            // Iterate blocks of 2x1 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcV = srcSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcUb = srcU + hStrideYUV422PChroma;
                auto srcVb = srcV + hStrideYUV422PChroma;
                auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
                auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

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
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV422P * hStrideYUV422P);

            // Iterate blocks of 2x1 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV422P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcV = srcSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto srcUb = srcU + hStrideYUV422PChroma;
                auto srcVb = srcV + hStrideYUV422PChroma;
                auto dstC = dstSlice[1] + vIndex * hStrideNV12;

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
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV422P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 1x6 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

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

        return;
    }
    #pragma endregion

    #pragma region YUV420P
    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto srcBb = srcB + hStrideYUV420P;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideUYVY422;
            auto dstBb = dstB + hStrideUYVY422;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
                auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;
                auto dstU = dstSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstV = dstSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstUb = dstU + hStrideYUV422PChroma;
                auto dstVb = dstV + hStrideYUV422PChroma;

                for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                    // Get chroma values
                    uint8_t u = *srcU++; // U
                    uint8_t v = *srcV++; // V

                                         // Assign values dupicated
                    *dstU++ = u;
                    *dstV++ = v;

                    *dstUb++ = u;
                    *dstVb++ = v;
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);
        memcpy(dstSlice[1], srcSlice[1], vStrideYUV420P * hStrideYUV420P / 4);
        memcpy(dstSlice[2], srcSlice[2], vStrideYUV420P * hStrideYUV420P / 4);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideYUV420P * hStrideYUV420P);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
                // Discover buffer pointers
                auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
                auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;
                auto dstC = dstSlice[1] + vIndex * hStrideNV12;

                for(int hIndex = 0; hIndex < hStrideYUV420P / 2; hIndex++){
                    *dstC++ = *srcU++; // U
                    *dstC++ = *srcV++; // V
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_YUV420P && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV420P / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto srcBb = srcB + hStrideYUV420P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * 2 * hStrideV210;
            auto dstBb = dstB + hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV420P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV420P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV420P / 6; hIndex++){
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
        }

        return;
    }
    #pragma endregion

    #pragma region NV12
    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideNV12;
            auto srcBb = srcB + hStrideNV12;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideUYVY422;
            auto dstBb = dstB + hStrideUYVY422;
            auto srcC = srcSlice[1] + vIndex * hStrideNV12;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        int hStrideYUV422PChroma = hStrideYUV422P / 2;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
                // Discover buffer pointers
                auto srcC = srcSlice[1] + vIndex * hStrideNV12;
                auto dstU = dstSlice[1] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstV = dstSlice[2] + vIndex * 2 * hStrideYUV422PChroma;
                auto dstUb = dstU + hStrideYUV422PChroma;
                auto dstVb = dstV + hStrideYUV422PChroma;

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
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        #pragma omp parallel
        {
            // Luma plane is the same
            #pragma omp single nowait
            memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);

            // Iterate blocks of 2x2 channel points
            #pragma omp for schedule(static)
            for(int vIndex = 0; vIndex < vStrideNV12; vIndex++){
                // Discover buffer pointers
                auto srcC = srcSlice[1] + vIndex * hStrideNV12 / 2;
                auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 4;
                auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 4;

                for(int hIndex = 0; hIndex < hStrideNV12 / 4; hIndex++){
                    *dstU++ = *srcC++; // U
                    *dstV++ = *srcC++; // V
                }
            }
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideNV12 * hStrideNV12);
        memcpy(dstSlice[1], srcSlice[1], vStrideNV12 * hStrideNV12 / 2);

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_NV12 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideNV12 = height;
        int hStrideNV12 = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Iterate blocks of 2x2 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideNV12 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * 2 * hStrideNV12;
            auto srcBb = srcB + hStrideNV12;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * 2 * hStrideV210;
            auto dstBb = dstB + hStrideV210;
            auto srcC = srcSlice[1] + vIndex * hStrideNV12;

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
        }

        return;
    }
    #pragma endregion

    #pragma region V210
    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_UYVY422){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideUYVY422 = height;
        int hStrideUYVY422 = width * 2;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideUYVY422;

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

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

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

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV420P){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV420P = height;
        int hStrideYUV420P = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * 2 * hStrideV210;
            auto srcBb = srcB + hStrideV210;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideYUV420P;
            auto dstBb = dstB + hStrideYUV420P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV420P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV420P / 2;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_NV12){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideNV12 = height;
        int hStrideNV12 = width;

        // Iterate blocks of 2x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210 / 2; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * 2 * hStrideV210;
            auto srcBb = srcB + hStrideV210;
            auto dstB = dstSlice[0] + vIndex * 2 * hStrideNV12;
            auto dstBb = dstB + hStrideNV12;
            auto dstC = dstSlice[1] + vIndex * hStrideNV12;

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
        }

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Copy data
        memcpy(dstSlice[0], srcSlice[0], vStrideV210 * hStrideV210 * sizeof(uint32_t));

        return;
    }

    if(srcPixelFormat == AV_PIX_FMT_V210 && dstPixelFormat == AV_PIX_FMT_YUV422PNORM){
        // Used metrics
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;

        // Create const for normalization
        double constLuma = 219. / 1023.;
        double constChroma = 224. / 1023.;
        double const16 = 16.;

        // Iterate blocks of 1x4 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideV210; vIndex++){
            // Discover buffer pointers
            auto srcB = reinterpret_cast<uint32_t*>(srcSlice[0]) + vIndex * hStrideV210;
            auto dstB = dstSlice[0] + vIndex * hStrideYUV422P;
            auto dstU = dstSlice[1] + vIndex * hStrideYUV422P / 2;
            auto dstV = dstSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideV210 / 4; hIndex++){
                auto u0 = *srcB & 0x3FF; // U0
                auto y0 = (*srcB >> 10U) & 0x3FF; // Y0
                auto v0 = (*srcB >> 20U) & 0x3FF; // V0
                *srcB++;

                auto y1 = *srcB & 0x3FF; // Y1
                auto u1 = (*srcB >> 10U) & 0x3FF; // U1
                auto y2 = (*srcB >> 20U) & 0x3FF; // Y2
                *srcB++;

                auto v1 = *srcB & 0x3FF; // V1
                auto y3 = (*srcB >> 10U) & 0x3FF; // Y3
                auto u2 = (*srcB >> 20U) & 0x3FF; // U2
                *srcB++;

                auto y4 = *srcB & 0x3FF; // Y4
                auto v2 = (*srcB >> 10U) & 0x3FF; // V2
                auto y5 = (*srcB >> 20U) & 0x3FF; // Y5
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

        return;
    }
    #pragma endregion

    #pragma region YUV422PNORM
    if(srcPixelFormat == AV_PIX_FMT_YUV422PNORM && dstPixelFormat == AV_PIX_FMT_V210){
        // Used metrics
        int vStrideYUV422P = height;
        int hStrideYUV422P = width;
        int vStrideV210 = height;
        int hStrideV210 = width / 6 * 4;

        // Create const for normalization
        double const16 = 16.;
        double constLuma = 1023. / 219.;
        double constChroma = 1023. / 224.;

        // Iterate blocks of 1x6 channel points
        #pragma omp parallel for schedule(static)
        for(int vIndex = 0; vIndex < vStrideYUV422P; vIndex++){
            // Discover buffer pointers
            auto srcB = srcSlice[0] + vIndex * hStrideYUV422P;
            auto dstB = reinterpret_cast<uint32_t*>(dstSlice[0]) + vIndex * hStrideV210;
            auto srcU = srcSlice[1] + vIndex * hStrideYUV422P / 2;
            auto srcV = srcSlice[2] + vIndex * hStrideYUV422P / 2;

            for(int hIndex = 0; hIndex < hStrideYUV422P / 6; hIndex++){
                // Get components from source
                auto u0n = *srcU++; // U0
                auto y0n = *srcB++; // Y0
                auto v0n = *srcV++; // V0
                auto y1n = *srcB++; // Y1

                auto u1n = *srcU++; // U1
                auto y2n = *srcB++; // Y2
                auto v1n = *srcV++; // V1
                auto y3n = *srcB++; // Y3

                auto u2n = *srcU++; // U2
                auto y4n = *srcB++; // Y4
                auto v2n = *srcV++; // V2
                auto y5n = *srcB++; // Y5

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

        return;
    }
    #pragma endregion
}
