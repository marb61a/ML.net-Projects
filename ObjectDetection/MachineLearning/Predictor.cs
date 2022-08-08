using ObjectDetection.MachineLearning.DataModel;

namespace ObjectDetection.MachineLearning
{
    public class Predictor
    {
        private MLContext _mlContext;
        private PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

        public Predictor(ITransformer trainedModel)
        {
            _mlContext = new MLContext();
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);
        }

        public ImagePrediction Predict(Bitmap image)
        {
            return _predictionEngine.Predict(new ImageData() { Image = image });
        }
    }
}
