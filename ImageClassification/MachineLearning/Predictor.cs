using ImageClassification.MachineLearning.DataModels;

namespace ImageClassification.MachineLearning
{
    public class Predictor
    {
        private static string ModelPath => Path.Combine(AppContext.BaseDirectory,"imageClassification.mdl");
        private MLContext _mlContext;
        private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;
        private ITransformer _trainedModel;
        private DataLoader _dataLoader;

        public Predictor(DataLoader dataLoader)
        {
            _mlContext = new MLContext(11);
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
            }

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                _trainedModel = _mlContext.Model.Load(stream, out _);
            }

            if (_trainedModel == null)
            {
                throw new Exception($"Failed to load Model");
            }

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);
            _dataLoader = dataLoader;

        }

        public IEnumerable<ModelOutput> MakeTestDatasetPredictions()
        {
            var predictionData = _trainedModel.Transform(_dataLoader.TestSet);
            return _mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
        }
    }
}