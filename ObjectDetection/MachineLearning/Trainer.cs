using ObjectDetection.MachineLearning.DataModel;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace ObjectDetection.MachineLearning
{
    public class Trainer
    {
        private MLContext _mlContext;

        public Trainer()
        {
            _mlContext = new MLContext();
        }

        public ITransformer BuildAndTrain(string yoloModelPath)
        {
            var pipeline = _mlContext.Transforms.ResizeImages(
                inputColumnName: "image",
                outputColumnName: "input_1:0",
                imageWidth: 416,
                imageHeight: 416,
                resizing: ResizingKind.IsoPad
            )
            .Append(_mlContext.Transforms.ExtractPixels(
                outputColumnName: "input_1:0",
                scaleImage: 1f / 255f,
                interleavePixelColors: true
            ))
            .Append(_mlContext.Transforms.ApplyOnnxModel(
                outputColumnNames: new[]
                {
                            "Identity:0",
                            "Identity_1:0",
                            "Identity_2:0"
                },
                inputColumnNames: new[]
                {
                            "input_1:0"
                },
                shapeDictionary: new Dictionary<string, int[]>()
                {
                            { "input_1:0", new[] { 1, 416, 416, 3 } },
                            { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
                            { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
                            { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
                },
                modelFile: yoloModelPath,
                gpuDeviceId: null,
                fallbackToCpu: true));

            return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ImageData>()));;
        }
    }
}