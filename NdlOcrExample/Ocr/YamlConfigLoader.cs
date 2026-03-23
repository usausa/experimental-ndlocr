namespace NdlOcrExample.Ocr;

using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

internal static class YamlConfigLoader
{
    public static Dictionary<int, string> LoadClassNames(string yamlPath)
    {
        var yaml = File.ReadAllText(yamlPath);
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();
        var doc = deserializer.Deserialize<Dictionary<string, object>>(yaml);

        var names = new Dictionary<int, string>();
        if (doc.TryGetValue("names", out var namesObj) && namesObj is Dictionary<object, object> namesDict)
        {
            foreach (var kv in namesDict)
            {
                if (Int32.TryParse(kv.Key.ToString(), out var id))
                {
                    names[id] = kv.Value.ToString() ?? string.Empty;
                }
            }
        }

        return names;
    }

    public static char[] LoadCharset(string yamlPath)
    {
        var yaml = File.ReadAllText(yamlPath);
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();
        var doc = deserializer.Deserialize<Dictionary<string, object>>(yaml);

        if (doc.TryGetValue("model", out var modelObj) && modelObj is Dictionary<object, object> modelDict)
        {
            if (modelDict.TryGetValue("charset_train", out var charsetObj))
            {
                var charsetStr = charsetObj.ToString() ?? string.Empty;
                return charsetStr.ToCharArray();
            }
        }

        throw new InvalidOperationException("charset_train not found in YAML file");
    }
}
