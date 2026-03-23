namespace NdlOcrExample.Ocr;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using SkiaSharp;

internal sealed class ParseqRecognizer : IDisposable
{
    private readonly InferenceSession session;

    private readonly string inputName;

    private readonly int inputHeight;
    private readonly int inputWidth;

    private readonly char[] charset;

    public ParseqRecognizer(string modelPath, char[] charsetData, bool useGpu = false)
    {
        using var options = OnnxSessionFactory.CreateOptions(useGpu);
        session = new InferenceSession(modelPath, options);

        var input = session.InputMetadata.First();
        inputName = input.Key;

        var shape = input.Value.Dimensions;
        inputHeight = shape[2];
        inputWidth = shape[3];

        charset = charsetData;
    }

    public string Read(SKBitmap lineImage)
    {
        var inputTensor = PreProcess(lineImage);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);

        return Decode(results);
    }

    public void Dispose()
    {
        session.Dispose();
    }

    #pragma warning disable CA2000
    private static SKBitmap RotateImage(SKBitmap image)
    {
        var rotated = new SKBitmap(image.Height, image.Width, SKColorType.Rgba8888, SKAlphaType.Premul);
        try
        {
            using var canvas = new SKCanvas(rotated);
            canvas.Translate(image.Height, 0);
            canvas.RotateDegrees(90);
            canvas.DrawBitmap(image, 0, 0);
            canvas.Flush();
            return rotated;
        }
        catch
        {
            rotated.Dispose();
            throw;
        }
    }
#pragma warning restore CA2000

    private DenseTensor<float> PreProcess(SKBitmap image)
    {
        using var rotated = image.Height > image.Width ? RotateImage(image) : null;
        var processImage = rotated ?? image;

        using var resized = new SKBitmap(inputWidth, inputHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
        using (var canvas = new SKCanvas(resized))
        {
            canvas.DrawBitmap(processImage, new SKRect(0, 0, inputWidth, inputHeight));
            canvas.Flush();
        }

        // PARSeq BGR normalization: 2*(pixel/255) - 1 = pixel * (2/255) + (-1)
        const float scale = 2f / 255f;
        const float offset = -1f;

        var tensor = new DenseTensor<float>([1, 3, inputHeight, inputWidth]);
        PixelNormalizer.Normalize(
            resized.GetPixelSpan(),
            tensor.Buffer.Span,
            inputHeight * inputWidth,
            ch0Byte: 2,
            ch1Byte: 1,
            ch2Byte: 0,
            ch0Scale: scale,
            ch0Offset: offset,
            ch1Scale: scale,
            ch1Offset: offset,
            ch2Scale: scale,
            ch2Offset: offset);

        return tensor;
    }

    private string Decode(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var logits = results[0].AsTensor<float>();
        var seqLength = logits.Dimensions[1];
        var numClasses = logits.Dimensions[2];

        var chars = new List<char>();

        for (var t = 0; t < seqLength; t++)
        {
            var maxIndex = 0;
            var maxValue = logits[0, t, 0];
            for (var c = 1; c < numClasses; c++)
            {
                var val = logits[0, t, c];
                if (val > maxValue)
                {
                    maxValue = val;
                    maxIndex = c;
                }
            }

            if (maxIndex == 0)
            {
                break;
            }

            var charIndex = maxIndex - 1;
            if ((charIndex >= 0) && (charIndex < charset.Length))
            {
                chars.Add(charset[charIndex]);
            }
        }

        return new string(chars.ToArray());
    }
}
