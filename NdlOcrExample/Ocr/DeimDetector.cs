namespace NdlOcrExample.Ocr;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using SkiaSharp;

internal sealed class DeimDetector : IDisposable
{
    private const float NmsIouThreshold = 0.2f;

    private readonly InferenceSession session;

    private readonly string inputImageName;
    private readonly string inputSizeName;

    private readonly int inputHeight;
    private readonly int inputWidth;

    private readonly Dictionary<int, string> classNames;
    private readonly float confThreshold;

    public DeimDetector(string modelPath, Dictionary<int, string> classes, float confThreshold = 0.25f, bool useGpu = false)
    {
        classNames = classes;
        this.confThreshold = confThreshold;

        using var options = OnnxSessionFactory.CreateOptions(useGpu);
        session = new InferenceSession(modelPath, options);

        var inputs = session.InputMetadata.ToList();
        inputImageName = inputs[0].Key;
        inputSizeName = inputs[1].Key;

        var imageShape = inputs[0].Value.Dimensions;
        inputHeight = imageShape[2];
        inputWidth = imageShape[3];
    }

    public void Dispose()
    {
        session.Dispose();
    }

    public List<DetectResult> Detect(SKBitmap bitmap)
    {
        var inputTensor = PreProcess(bitmap, out var paddedSize);
        var sizeTensor = new DenseTensor<long>(new long[] { inputHeight, inputWidth }, [1, 2]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputImageName, inputTensor),
            NamedOnnxValue.CreateFromTensor(inputSizeName, sizeTensor)
        };

        using var results = session.Run(inputs);

        return PostProcess(results, bitmap.Width, bitmap.Height, paddedSize);
    }

    private DenseTensor<float> PreProcess(SKBitmap bitmap, out int paddedSize)
    {
        paddedSize = Math.Max(bitmap.Width, bitmap.Height);

        using var resized = new SKBitmap(inputWidth, inputHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(resized);
        canvas.Clear(SKColors.Black);

        var scaleX = (float)inputWidth / paddedSize;
        var scaleY = (float)inputHeight / paddedSize;
        var destRect = new SKRect(0, 0, bitmap.Width * scaleX, bitmap.Height * scaleY);
        canvas.DrawBitmap(bitmap, destRect);
        canvas.Flush();

        // ImageNet normalization: (pixel/255 - mean) / std = pixel * (1/(255*std)) + (-mean/std)
        const float rMean = 0.485f, gMean = 0.456f, bMean = 0.406f;
        const float rStd = 0.229f, gStd = 0.224f, bStd = 0.225f;

        var tensor = new DenseTensor<float>([1, 3, inputHeight, inputWidth]);
        PixelNormalizer.Normalize(
            resized.GetPixelSpan(),
            tensor.Buffer.Span,
            inputHeight * inputWidth,
            ch0Byte: 0,
            ch1Byte: 1,
            ch2Byte: 2,
            ch0Scale: 1f / (255f * rStd),
            ch0Offset: -rMean / rStd,
            ch1Scale: 1f / (255f * gStd),
            ch1Offset: -gMean / gStd,
            ch2Scale: 1f / (255f * bStd),
            ch2Offset: -bMean / bStd);

        return tensor;
    }

    private List<DetectResult> PostProcess(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, int origWidth, int origHeight, int paddedSize)
    {
        var classIds = results[0].AsTensor<long>();
        var boxes = results[1].AsTensor<float>();
        var scores = results[2].AsTensor<float>();

        var numDetections = scores.Dimensions[1];

        var scaleX = (float)paddedSize / inputWidth;
        var scaleY = (float)paddedSize / inputHeight;

        var detections = new List<DetectResult>();

        for (var i = 0; i < numDetections; i++)
        {
            var score = scores[0, i];
            if (score <= confThreshold)
            {
                continue;
            }

            var classIndex = (int)classIds[0, i] - 1;
            if ((classIndex < 0) || !classNames.TryGetValue(classIndex, out var className))
            {
                continue;
            }

            if (!className.StartsWith("line_", StringComparison.Ordinal))
            {
                continue;
            }

            var bx = boxes[0, i, 0] * scaleX;
            var by = boxes[0, i, 1] * scaleY;
            var bw = boxes[0, i, 2] * scaleX;
            var bh = boxes[0, i, 3] * scaleY;

            var x1 = Math.Max(0, (int)bx);
            var y1 = Math.Max(0, (int)by);
            var x2 = Math.Min(origWidth, (int)bw);
            var y2 = Math.Min(origHeight, (int)bh);

            if ((x2 <= x1) || (y2 <= y1))
            {
                continue;
            }

            detections.Add(new DetectResult
            {
                ClassIndex = classIndex,
                ClassName = className,
                Confidence = score,
                X = x1,
                Y = y1,
                Width = x2 - x1,
                Height = y2 - y1
            });
        }

        return ApplyNms(detections);
    }

    private static List<DetectResult> ApplyNms(List<DetectResult> detections)
    {
        if (detections.Count <= 1)
        {
            return detections;
        }

        var sorted = detections.OrderByDescending(d => d.Confidence).ToList();
        var suppressed = new bool[sorted.Count];
        var kept = new List<DetectResult>();

        for (var i = 0; i < sorted.Count; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            kept.Add(sorted[i]);

            for (var j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j])
                {
                    continue;
                }

                if (ComputeIoU(sorted[i], sorted[j]) >= NmsIouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    private static float ComputeIoU(DetectResult a, DetectResult b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        if (x2 <= x1 || y2 <= y1)
        {
            return 0f;
        }

        var intersection = (float)((x2 - x1) * (y2 - y1));
        var union = ((float)a.Width * a.Height) + ((float)b.Width * b.Height) - intersection;

        return intersection / union;
    }
}
