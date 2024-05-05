#include "MOG/MPL/FFmpeg/pch.h"

#include "MOG/MPL/FFmpeg/Filters/VideoResampler/UyvyToV210Impl.h"
using namespace MOG;
using namespace MOG::Core;
using namespace MOG::MPL;
using namespace MOG::MPL::FFmpeg;
using namespace MOG::MPL::Essence;
using namespace MOG::MPL::Memory;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
    enum
    {
        PIXELS_PER_BLOCK = 48,
        PIXELS_PER_SUB_BLOCK = 6,
        UYVY_PIXEL_SIZE = 2,
    };

    inline void resample_block_6(uint32_t*& target, uint8_t*& source)
    {
        enum { SHIFT_8TO10B = 2U, SHIFT_LEFT = 20U, SHIFT_MIDLE = 10U,  };

        auto u0 = *source++ << SHIFT_8TO10B;
        auto y0 = *source++ << SHIFT_8TO10B;
        auto v0 = *source++ << SHIFT_8TO10B;
        auto y1 = *source++ << SHIFT_8TO10B;

        auto u1 = *source++ << SHIFT_8TO10B;
        auto y2 = *source++ << SHIFT_8TO10B;
        auto v1 = *source++ << SHIFT_8TO10B;
        auto y3 = *source++ << SHIFT_8TO10B;

        auto u2 = *source++ << SHIFT_8TO10B;
        auto y4 = *source++ << SHIFT_8TO10B;
        auto v2 = *source++ << SHIFT_8TO10B;
        auto y5 = *source++ << SHIFT_8TO10B;

        *target++ = (v0 << SHIFT_LEFT) | (y0 << SHIFT_MIDLE) | u0;
        *target++ = (y2 << SHIFT_LEFT) | (u1 << SHIFT_MIDLE) | y1;
        *target++ = (u2 << SHIFT_LEFT) | (y3 << SHIFT_MIDLE) | v1;
        *target++ = (y5 << SHIFT_LEFT) | (v2 << SHIFT_MIDLE) | y4;
    }

    inline void resample_block_48_direct(uint32_t*& target, uint8_t*& source)
    {
        for (auto i = 0; i < PIXELS_PER_BLOCK; i += PIXELS_PER_SUB_BLOCK)
        {
            resample_block_6(target, source);
        }
    }

    inline void resample_block_48_copy(uint32_t*& target, uint8_t*& source, int block_size)
    {
        uint8_t buffer[PIXELS_PER_BLOCK * UYVY_PIXEL_SIZE];

        MOG_ASSERT(block_size > 0);
        MOG_ASSERT(block_size <= 48);

        auto block_in_bytes = block_size * UYVY_PIXEL_SIZE;

        std::memset(buffer, 0, sizeof(buffer));
        std::memcpy(buffer, source, block_in_bytes);

        uint8_t* buffer_ptr = buffer;
        resample_block_48_direct(target, buffer_ptr);

        source += block_in_bytes;
    }

    inline void resample_block_48(uint32_t*& target, uint8_t*& source, int block_size)
    {
        if (block_size < PIXELS_PER_BLOCK)
        {
            MOG_ASSERT(block_size > 0);
            resample_block_48_copy(target, source, block_size);
        }
        else
        {
            resample_block_48_direct(target, source);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UyvyToV210Impl::UyvyToV210Impl(UncompressedPictureDescriptor_cptr input_descriptor,
    UncompressedPictureDescriptor_ptr output_descriptor,
    const VideoResampler::Padding& padding)
{
    MOG_ENFORCE(input_descriptor->storedWidth == output_descriptor->storedWidth, Exceptions::InvalidArgument, (L"Scaling not supported for this pixel format"));
    MOG_ENFORCE(input_descriptor->storedHeight == output_descriptor->storedHeight, Exceptions::InvalidArgument, (L"Scaling not supported for this pixel format"));
    MOG_ENFORCE(padding.bottom + padding.left + padding.right + padding.top == 0, Exceptions::InvalidArgument, (L"Padding not supported for this pixel format"));

    auto cdci = std::make_shared<CDCISubDescriptor>();
    cdci->horizontalSubsampling = 2;
    cdci->verticalSubsampling = 1;
    output_descriptor->sub_descriptor = cdci;

    set_uncompressed_parameters(*output_descriptor);

    out_format_.width = output_descriptor->storedWidth.get();
    out_format_.height = output_descriptor->storedHeight.get();
}

void UyvyToV210Impl::process(AVFrame* in_frame, AVFrame* out_frame)
{
    MOG_ENFORCE(in_frame != nullptr, Exceptions::InvalidArgument, (L"Null input frame"));
    MOG_ENFORCE(out_frame != nullptr, Exceptions::InvalidArgument, (L"Null output frame"));

    MOG_ASSERT(out_frame->linesize[0] != 0);

    out_frame->width = out_format_.width;
    out_frame->height = out_format_.height;
    out_frame->format = AV_PIX_FMT_NONE;
    out_frame->linesize[0] = Essence::get_line_size_in_bytes(0, out_frame->width, PixelFormatId::PIXEL_FORMAT_V210);

    auto input_p = in_frame->data[0];
    auto output_p = reinterpret_cast<uint32_t*>(out_frame->data[0]);

    for (auto y = 0; y < out_frame->height; ++y)
    {
        for (auto x = 0; x < out_frame->width; x += PIXELS_PER_BLOCK)
        {
            resample_block_48(output_p, input_p, out_frame->width - x);
        }
    }
}

bool UyvyToV210Impl::supports_input_format(PixelFormatId id)
{
    return id == PIXEL_FORMAT_UYVY422;
}

bool UyvyToV210Impl::supports_output_format(PixelFormatId id)
{
    return id == PIXEL_FORMAT_V210;
}
