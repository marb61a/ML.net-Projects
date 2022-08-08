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
            );
        }
    }
}