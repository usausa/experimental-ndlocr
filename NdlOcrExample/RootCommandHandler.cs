namespace NdlOcrExample;

using System.Diagnostics;

using NdlOcrExample.Ocr;

using SkiaSharp;

using Smart.CommandLine.Hosting;

public sealed class RootCommandHandler : ICommandHandler
{
    private const string DetModelFile = "deim-s-1024x1024.onnx";
    private const string DetClassesFile = "ndl.yaml";

    private const string RecModelFile = "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx";
    private const string RecClassesFile = "NDLmoji.yaml";

    private const float DetConfThreshold = 0.5f;

    [Option<string>("--file", "-f", Description = "Image file path", Required = true)]
    public string ImageFile { get; set; } = default!;

    [Option("--gpu", "-g", Description = "Use gpu device")]
    public bool UseGpu { get; set; }

    public ValueTask ExecuteAsync(CommandContext context)
    {
        var assetsDir = Path.Combine(AppContext.BaseDirectory, "Assets");
        var modelDir = Path.Combine(assetsDir, "model");
        var configDir = Path.Combine(assetsDir, "config");

        var detWeights = Path.Combine(modelDir, DetModelFile);
        var detClasses = Path.Combine(configDir, DetClassesFile);
        var recWeights = Path.Combine(modelDir, RecModelFile);
        var recClasses = Path.Combine(configDir, RecClassesFile);

        Console.WriteLine($"Device: {(UseGpu ? "GPU (DirectML)" : "CPU")}");

        var totalWatch = Stopwatch.StartNew();

        //--------------------------------------------------------------------------------
        // Load configuration files
        //--------------------------------------------------------------------------------
        Console.WriteLine("Loading configuration files...");
        var phaseWatch = Stopwatch.StartNew();

        var classNames = YamlConfigLoader.LoadClassNames(detClasses);
        var charset = YamlConfigLoader.LoadCharset(recClasses);

        Console.WriteLine($"Configuration files loaded (charset: {charset.Length} chars, {phaseWatch.ElapsedMilliseconds}ms)");

        //--------------------------------------------------------------------------------
        // Initialize detection model
        //--------------------------------------------------------------------------------
        Console.WriteLine("Initializing detection model...");
        phaseWatch.Restart();

        using var detector = new DeimDetector(detWeights, classNames, confThreshold: DetConfThreshold, useGpu: false);

        Console.WriteLine($"Detection model initialized ({phaseWatch.ElapsedMilliseconds}ms)");

        //--------------------------------------------------------------------------------
        // Initialize recognition model
        //--------------------------------------------------------------------------------
        Console.WriteLine("Initializing recognition model...");
        phaseWatch.Restart();

        using var recognizer = new ParseqRecognizer(recWeights, charset, UseGpu);

        Console.WriteLine($"Recognition model initialized (100-char model, {phaseWatch.ElapsedMilliseconds}ms)");

        //--------------------------------------------------------------------------------
        // Load image
        //--------------------------------------------------------------------------------
        Console.WriteLine($"Loading image: {ImageFile}");
        phaseWatch.Restart();

        using var bitmap = SKBitmap.Decode(ImageFile);
        if (bitmap is null)
        {
            Console.WriteLine("Error: Failed to decode image.");
            return ValueTask.CompletedTask;
        }

        Console.WriteLine($"Image loaded (size: {bitmap.Width} x {bitmap.Height}, {phaseWatch.ElapsedMilliseconds}ms)");

        //--------------------------------------------------------------------------------
        // Layout detection
        //--------------------------------------------------------------------------------
        Console.WriteLine("Running layout detection...");
        phaseWatch.Restart();
        var textLines = detector.Detect(bitmap);
        Console.WriteLine($"Layout detection complete: {textLines.Count} text lines detected ({phaseWatch.ElapsedMilliseconds}ms)");

        if (textLines.Count == 0)
        {
            Console.WriteLine("No text lines detected.");
            return ValueTask.CompletedTask;
        }

        //--------------------------------------------------------------------------------
        // Character recognition
        //--------------------------------------------------------------------------------
        Console.WriteLine("Running character recognition...");
        phaseWatch.Restart();

        var results = new List<OcrResult>();

        foreach (var det in textLines)
        {
            var x = Math.Max(0, det.X);
            var y = Math.Max(0, det.Y);
            var w = Math.Min(det.Width, bitmap.Width - x);
            var h = Math.Min(det.Height, bitmap.Height - y);

            if (w <= 0 || h <= 0)
            {
                continue;
            }

#pragma warning disable CA2000
            using var lineImage = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);
#pragma warning restore CA2000
            using (var canvas = new SKCanvas(lineImage))
            {
                canvas.DrawBitmap(bitmap, new SKRect(x, y, x + w, y + h), new SKRect(0, 0, w, h));
                canvas.Flush();
            }

            var text = recognizer.Read(lineImage);
            results.Add(new OcrResult { Detection = det, Text = text });
        }

        Console.WriteLine($"Character recognition complete: {results.Count} lines recognized ({phaseWatch.ElapsedMilliseconds}ms)");

        totalWatch.Stop();
        Console.WriteLine($"Total processing time: {totalWatch.ElapsedMilliseconds}ms");

        //--------------------------------------------------------------------------------
        // Output results
        //--------------------------------------------------------------------------------
        var sortedResults = results.OrderByDescending(r => r.Detection.Confidence).ToList();

        Console.WriteLine("========== OCR Results ==========");

        foreach (var result in sortedResults)
        {
            var det = result.Detection;
            Console.WriteLine($"[{det.ClassName}] score={det.Confidence:F3} pos=({det.X},{det.Y}) size={det.Width}x{det.Height}");
            Console.WriteLine($"  {result.Text}");
        }

        return ValueTask.CompletedTask;
    }

    private sealed class OcrResult
    {
        public required DetectResult Detection { get; init; }

        public required string Text { get; init; }
    }
}
