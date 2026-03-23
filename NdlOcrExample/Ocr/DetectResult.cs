namespace NdlOcrExample.Ocr;

internal sealed class DetectResult
{
    public int ClassIndex { get; set; }

    public string ClassName { get; set; } = string.Empty;

    public float Confidence { get; set; }

    public int X { get; set; }

    public int Y { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }
}
