#include "MOG/MPL/FFmpeg/pch.h"

#include "MOG/MPL/FFmpeg/Filters/VideoResampler/V210ToUyvy.h"
#include "MOG/Core/Platform/CPUInfo.h"
#include "MOG/Core/Time/HighResTimer.h"

#include "immintrin.h"

using namespace MOG;
using namespace MOG::Core;
using namespace MOG::MPL;
using namespace MOG::MPL::FFmpeg;
using namespace MOG::MPL::Essence;
using namespace MOG::MPL::Memory;

namespace
{
    const int32_t V210_PIXEL_PER_PACK = 6;
    const int32_t UYVY_BYTES_PER_PACK = 12;

    inline int32_t calculate_v210_stride(int32_t width)
    {
        return ((width + 47) / 48) * 128;
    }

    template<size_t NThreads = 0, typename IndexType, typename Fn>
    void parallel_for(IndexType start, IndexType end, Fn fn)
    {
        size_t max_threads = std::thread::hardware_concurrency();
        size_t threads = NThreads > 0 ? std::min<size_t>(NThreads, max_threads) : max_threads;

        IndexType delta = end - start;
        IndexType number_blocks = static_cast<IndexType>(threads);

        auto div = std::div(delta, number_blocks);
        IndexType chunk_size = div.quot + (div.rem > 0 ? 1 : 0);

        concurrency::parallel_for(start, end, fn, concurrency::simple_partitioner(chunk_size));
    }

    #pragma region SSE
#   define ENABLE_SSE3
#   if defined(ENABLE_SSE3)
    inline bool has_sse3_support()
    {
        return Platform::CPUInfo::SSE3();
    }

    const auto THREADS_SSE = 4;

    const __m128i mask_1st_pack = _mm_set_epi32(0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF);
    const __m128i mask_2nd_pack = _mm_set_epi32(0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00);
    const __m128i mask_3rd_pack = _mm_set_epi32(0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000);

    //Requires a shuffle with a shuffle mask because
    //                    0  1   2   3     4   5  6    7     8  9  10   11   12  13 14   15
    //Layout after sse  CB0 Y0 CR0 NULL   Y1 CB1 Y2 NULL   CR1 Y3 CB2 NULL   Y4 CR2 Y5 NULL
    //                    0  1   2    4  5   6     8  9  10   12 13  14     15   15   15   15
    //Layout in UYVY    CB0 Y0 CR0   Y1 CB1 Y2   CR1 Y3 CB2   Y4 CR2 Y5   NULL NULL NULL NULL
    const __m128i shuffle_epi   = _mm_set_epi8(15, 15, 15, 15, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

    inline void convert_line_v210_to_uyvy_sse3(int32_t width, int32_t /*height*/, const uint8_t* input, uint8_t* output_p, int32_t output_linesize, size_t line_y, size_t input_stride)
    {
        uint8_t *output = output_p + line_y * output_linesize;
        const uint32_t *p = reinterpret_cast<const uint32_t*>(input + line_y * input_stride);

        for (auto x = 0; x < width; x += V210_PIXEL_PER_PACK)
        {
            // Scalar logic:
            //output[0] = p[0] >>  2 & 0xFF;    // Cb0
            //output[1] = p[0] >> 12 & 0xFF;    // Y0
            //output[2] = p[0] >> 22 & 0xFF;    // Cr0
            //output[3] = p[1] >>  2 & 0xFF;    // Y1
            //output[4] = p[1] >> 12 & 0xFF;    // Cb1
            //output[5] = p[1] >> 22 & 0xFF;    // Y2
            //output[6] = p[2] >>  2 & 0xFF;    // Cr1
            //output[7] = p[2] >> 12 & 0xFF;    // Y3
            //output[8] = p[2] >> 22 & 0xFF;    // Cb2
            //output[9]  = p[3] >>  2 & 0xFF;    // Y4
            //output[10] = p[3] >> 12 & 0xFF;    // Cr2
            //output[11] = p[3] >> 22 & 0xFF;    // Y5

            __m128i m = _mm_lddqu_si128((__m128i*)p);

            __m128i shifted_1st_pack = _mm_srli_epi32(m, 2);
            __m128i shifted_2nd_pack = _mm_srli_epi32(m, 4);
            __m128i shifted_3rd_pack = _mm_srli_epi32(m, 6);

            __m128i p1 = _mm_and_si128(shifted_1st_pack, mask_1st_pack);
            __m128i p2 = _mm_and_si128(shifted_2nd_pack, mask_2nd_pack);
            __m128i p3 = _mm_and_si128(shifted_3rd_pack, mask_3rd_pack);

            __m128i unpacked = _mm_or_si128(_mm_or_si128(p1, p2), p3);
            __m128i packed = _mm_shuffle_epi8(unpacked, shuffle_epi);

            _mm_storeu_si128((__m128i*)output, packed);

            p += sizeof(uint32_t);
            output += UYVY_BYTES_PER_PACK;
        }
    }

    void convert_v210_to_uyvy_sse3(int32_t width, int32_t height, const uint8_t* input, uint8_t* output, int32_t output_linesize)
    {
        const int32_t input_stride = calculate_v210_stride(width);

        parallel_for<THREADS_SSE>(0, height, [=](auto y)
        {
            convert_line_v210_to_uyvy_sse3(width, height, input, output, output_linesize, y, input_stride);
        });
    }
    #endif
    #pragma endregion

    #pragma region AVX2
//TODO: undefined because of potential Alignment issues if the buffers are not 256bits aligned on the end.
//#   define ENABLE_AVX2
#   if defined(ENABLE_AVX2)
    inline bool has_avx2_support()
    {
        return Platform::CPUInfo::AVX2();
    }

    const auto THREADS_AVX2 = 4;

    const __m256i avx_mask_1st_pack = _mm256_set_epi32(0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF);
    const __m256i avx_mask_2nd_pack = _mm256_set_epi32(0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00, 0x0000FF00);
    const __m256i avx_mask_3rd_pack = _mm256_set_epi32(0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000, 0x00FF0000);

    const __m256i avx_shuffle_epi = _mm256_set_epi8(31, 31, 31, 31, 31, 31, 31, 31,30, 29, 28, 26, 25, 24, 22, 21, 20, 18, 17, 16, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

    inline void convert_line_v210_to_uyvy_avx2(int32_t width, int32_t /*height*/, const uint8_t* input, uint8_t* output_p, int32_t output_linesize, size_t line_y, size_t input_stride)
    {
        const auto AVX_SOURCE_STEP = V210_PIXEL_PER_PACK * 2;
        const auto AVX_BYTES_READ = sizeof(uint32_t) * 2;
        const auto AVX_BYTES_WRITTEN= UYVY_BYTES_PER_PACK * 2;
        uint8_t *output = output_p + line_y * output_linesize;
        const uint32_t *p = reinterpret_cast<const uint32_t*>(input + line_y * input_stride);

        for (auto x = 0; x < width; x += AVX_SOURCE_STEP)
        {
            __m256i m = _mm256_lddqu_si256((__m256i*)p);

            __m256i shifted_1st_pack = _mm256_srli_epi32(m, 2);
            __m256i shifted_2nd_pack = _mm256_srli_epi32(m, 4);
            __m256i shifted_3rd_pack = _mm256_srli_epi32(m, 6);

            __m256i p1 = _mm256_and_si256(shifted_1st_pack, avx_mask_1st_pack);
            __m256i p2 = _mm256_and_si256(shifted_2nd_pack, avx_mask_2nd_pack);
            __m256i p3 = _mm256_and_si256(shifted_3rd_pack, avx_mask_3rd_pack);

            __m256i unpacked = _mm256_or_si256(_mm256_or_si256(p1, p2), p3);
            __m256i packed = _mm256_shuffle_epi8(unpacked, avx_shuffle_epi);

            _mm256_storeu_si256((__m256i*)output, packed);

            p += AVX_BYTES_READ;
            output += AVX_BYTES_WRITTEN;
        }
    }

    void convert_v210_to_uyvy_avx2(int32_t width, int32_t height, const uint8_t* input, uint8_t* output, int32_t output_linesize)
    {
        const int32_t input_stride = calculate_v210_stride(width);

        parallel_for<THREADS_AVX2>(0, height, [=](auto y)
        {
            convert_line_v210_to_uyvy_avx2(width, height, input, output, output_linesize, y, input_stride);
        });
    }
#   endif
    #pragma endregion

    #pragma region CPU Normal
    const auto THREADS_NORMAL = 0; // All

    inline void convert_line_v210_to_uyvy(int32_t width, int32_t /*height*/, const uint8_t* input, uint8_t* output_p, int32_t output_linesize, size_t line_y, size_t input_stride)
    {
        uint8_t *output = output_p + line_y * output_linesize;
        const uint32_t *p = reinterpret_cast<const uint32_t*>(input + line_y * input_stride);

        for (auto x = 0; x < width; x += V210_PIXEL_PER_PACK)
        {
            *(output++) = (*p >> 2)        & 0xFF;    // Cb0
            *(output++) = (*p >> 12)       & 0xFF;    // Y0
            *(output++) = (*p >> 22)       & 0xFF;    // Cr0

            *(output++) = (*(p + 1) >> 2)  & 0xFF;    // Y1
            *(output++) = (*(p + 1) >> 12) & 0xFF;    // Cb2
            *(output++) = (*(p + 1) >> 22) & 0xFF;    // Y2

            *(output++) = (*(p + 2) >> 2)  & 0xFF;    // Cr2
            *(output++) = (*(p + 2) >> 12) & 0xFF;    // Y3
            *(output++) = (*(p + 2) >> 22) & 0xFF;    // Cb4

            *(output++) = (*(p + 3) >> 2)  & 0xFF;    // Y4
            *(output++) = (*(p + 3) >> 12) & 0xFF;    // Cr4
            *(output++) = (*(p + 3) >> 22) & 0xFF;    // Y5

            p += 4;
        }
    }

    void convert_v210_to_uyvy(int32_t width, int32_t height, const uint8_t* input, uint8_t* output, int32_t output_linesize)
    {
        const int32_t input_stride = calculate_v210_stride(width);

        parallel_for<THREADS_NORMAL>(0, height, [=](auto y)
        {
            convert_line_v210_to_uyvy(width, height, input, output, output_linesize, y, input_stride);
        });
    }
    #pragma endregion
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

V210ToUyvy::V210ToUyvy(UncompressedPictureDescriptor_cptr input_descriptor, UncompressedPictureDescriptor_ptr output_descriptor, const VideoResampler::Padding& padding)
    : input_descriptor_(input_descriptor)
{
    MOG_ENFORCE(input_descriptor->storedWidth.get() == output_descriptor->storedWidth.get(), Exceptions::InvalidArgument, ("Scaling not supported for this pixel format"));
    MOG_ENFORCE(input_descriptor->storedHeight.get() == output_descriptor->storedHeight.get(), Exceptions::InvalidArgument, ("Scaling not supported for this pixel format"));
    MOG_ENFORCE(padding.bottom + padding.left + padding.right + padding.top == UInt32(0), Exceptions::InvalidArgument, ("Padding not supported for this pixel format"));
    set_uncompressed_parameters(*output_descriptor);

    out_format_.width = output_descriptor->storedWidth.get();
    out_format_.height = output_descriptor->storedHeight.get();

#   if defined(ENABLE_AVX2)
    if (has_avx2_support())
    {
        processor_ = convert_v210_to_uyvy_avx2;
        return
    }
#   endif
#   if defined(ENABLE_SSE3)
    if (has_sse3_support())
    {
        processor_ = convert_v210_to_uyvy_sse3;
        return;
    }
#   endif

    processor_ = convert_v210_to_uyvy;
}

void V210ToUyvy::process(AVFrame* in_frame, AVFrame* out_frame)
{
    MOG_ENFORCE(in_frame != nullptr, Exceptions::InvalidArgument, ("Null input frame"));
    MOG_ENFORCE(out_frame != nullptr, Exceptions::InvalidArgument, ("Null output frame"));
    MOG_ASSERT(out_frame->linesize[0] != 0);

    out_frame->width = out_format_.width;
    out_frame->height = out_format_.height;
    out_frame->format = AV_PIX_FMT_UYVY422;

    static auto logger = Logging::get_logger(L"");
    static double time_total = 0.0;
    static size_t counter = 0;
    MOG::Core::HighResTimer t;

    MOG_ASSERT(processor_);
    processor_(out_frame->width, out_frame->height, in_frame->data[0], out_frame->data[0], out_frame->linesize[0]);

    auto time = t.elapsed();
    time_total += time;
    counter++;
    OutputDebugString((L"V210ToUyvy::process: avg: " + std::to_wstring(time_total / counter) + L" this: " + std::to_wstring(time) + L"\n").c_str());
}

bool V210ToUyvy::supports_input_format(PixelFormatId id)
{
    return id == PIXEL_FORMAT_V210;
}

bool V210ToUyvy::supports_output_format(PixelFormatId id)
{
    return id == PIXEL_FORMAT_UYVY422;
}
