namespace NdlOcrExample.Ocr;

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

internal static class PixelNormalizer
{
    // Converts RGBA8888 pixel bytes to a 3-channel planar float tensor with normalization.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Normalize(
        ReadOnlySpan<byte> pixels,
        Span<float> tensorBuffer,
        int pixelCount,
        int ch0Byte,
        int ch1Byte,
        int ch2Byte,
        float ch0Scale,
        float ch0Offset,
        float ch1Scale,
        float ch1Offset,
        float ch2Scale,
        float ch2Offset)
    {
        fixed (byte* srcBase = pixels)
        {
            fixed (float* dstBase = tensorBuffer)
            {
                var ch0 = dstBase;
                var ch1 = dstBase + pixelCount;
                var ch2 = dstBase + (2 * pixelCount);

                var i = 0;

                if (Vector256.IsHardwareAccelerated)
                {
                    i = NormalizeVector256(srcBase, ch0, ch1, ch2, pixelCount, ch0Byte, ch1Byte, ch2Byte, ch0Scale, ch0Offset, ch1Scale, ch1Offset, ch2Scale, ch2Offset);
                }
                else if (Vector128.IsHardwareAccelerated)
                {
                    i = NormalizeVector128(srcBase, ch0, ch1, ch2, pixelCount, ch0Byte, ch1Byte, ch2Byte, ch0Scale, ch0Offset, ch1Scale, ch1Offset, ch2Scale, ch2Offset);
                }

                for (; i < pixelCount; i++)
                {
                    var p = srcBase + (i * 4);
                    ch0[i] = (p[ch0Byte] * ch0Scale) + ch0Offset;
                    ch1[i] = (p[ch1Byte] * ch1Scale) + ch1Offset;
                    ch2[i] = (p[ch2Byte] * ch2Scale) + ch2Offset;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int NormalizeVector256(
        byte* srcBase,
        float* ch0,
        float* ch1,
        float* ch2,
        int pixelCount,
        int ch0Byte,
        int ch1Byte,
        int ch2Byte,
        float ch0Scale,
        float ch0Offset,
        float ch1Scale,
        float ch1Offset,
        float ch2Scale,
        float ch2Offset)
    {
        var vCh0Scale = Vector256.Create(ch0Scale);
        var vCh1Scale = Vector256.Create(ch1Scale);
        var vCh2Scale = Vector256.Create(ch2Scale);
        var vCh0Offset = Vector256.Create(ch0Offset);
        var vCh1Offset = Vector256.Create(ch1Offset);
        var vCh2Offset = Vector256.Create(ch2Offset);
        var count = Vector256<float>.Count;

        var i = 0;
        for (; i <= pixelCount - count; i += count)
        {
            var p = srcBase + (i * 4);

            var v0 = Vector256.Create(p[ch0Byte], p[4 + ch0Byte], p[8 + ch0Byte], p[12 + ch0Byte], p[16 + ch0Byte], p[20 + ch0Byte], p[24 + ch0Byte], (float)p[28 + ch0Byte]);
            var v1 = Vector256.Create(p[ch1Byte], p[4 + ch1Byte], p[8 + ch1Byte], p[12 + ch1Byte], p[16 + ch1Byte], p[20 + ch1Byte], p[24 + ch1Byte], (float)p[28 + ch1Byte]);
            var v2 = Vector256.Create(p[ch2Byte], p[4 + ch2Byte], p[8 + ch2Byte], p[12 + ch2Byte], p[16 + ch2Byte], p[20 + ch2Byte], p[24 + ch2Byte], (float)p[28 + ch2Byte]);

            v0 = (v0 * vCh0Scale) + vCh0Offset;
            v1 = (v1 * vCh1Scale) + vCh1Offset;
            v2 = (v2 * vCh2Scale) + vCh2Offset;

            v0.Store(ch0 + i);
            v1.Store(ch1 + i);
            v2.Store(ch2 + i);
        }

        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int NormalizeVector128(
        byte* srcBase,
        float* ch0,
        float* ch1,
        float* ch2,
        int pixelCount,
        int ch0Byte,
        int ch1Byte,
        int ch2Byte,
        float ch0Scale,
        float ch0Offset,
        float ch1Scale,
        float ch1Offset,
        float ch2Scale,
        float ch2Offset)
    {
        var vCh0Scale = Vector128.Create(ch0Scale);
        var vCh1Scale = Vector128.Create(ch1Scale);
        var vCh2Scale = Vector128.Create(ch2Scale);
        var vCh0Offset = Vector128.Create(ch0Offset);
        var vCh1Offset = Vector128.Create(ch1Offset);
        var vCh2Offset = Vector128.Create(ch2Offset);
        var count = Vector128<float>.Count;

        var i = 0;
        for (; i <= pixelCount - count; i += count)
        {
            var p = srcBase + (i * 4);

            var v0 = Vector128.Create(p[ch0Byte], p[4 + ch0Byte], p[8 + ch0Byte], (float)p[12 + ch0Byte]);
            var v1 = Vector128.Create(p[ch1Byte], p[4 + ch1Byte], p[8 + ch1Byte], (float)p[12 + ch1Byte]);
            var v2 = Vector128.Create(p[ch2Byte], p[4 + ch2Byte], p[8 + ch2Byte], (float)p[12 + ch2Byte]);

            v0 = (v0 * vCh0Scale) + vCh0Offset;
            v1 = (v1 * vCh1Scale) + vCh1Offset;
            v2 = (v2 * vCh2Scale) + vCh2Offset;

            v0.Store(ch0 + i);
            v1.Store(ch1 + i);
            v2.Store(ch2 + i);
        }

        return i;
    }
}
