namespace NdlOcrExample.Ocr;

using Microsoft.ML.OnnxRuntime;

internal static class OnnxSessionFactory
{
#pragma warning disable CA2000
    public static SessionOptions CreateOptions(bool useGpu, int gpuDeviceId = 0, int intraOpThreads = 0)
    {
        var threads = intraOpThreads > 0 ? intraOpThreads : Environment.ProcessorCount / 2;
        var options = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = threads
        };
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");

        if (useGpu)
        {
            options.EnableMemoryPattern = false;
            options.EnableCpuMemArena = false;
            options.AppendExecutionProvider_DML(gpuDeviceId);
        }
        else
        {
            options.EnableMemoryPattern = true;
            options.EnableCpuMemArena = true;
        }

        return options;
    }
#pragma warning restore CA2000
}
