using Clustering.MachineLearning.DataModels;

namespace Clustering.MachineLearning.Predictors
{
    public class Predictor{
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "clustering.mdl");
        private readonly MLContext _mlContext;
        private ITransformer _model;
        public Predictor()
        {
            _mlContext = new MLContext(11);
        }

        public PalmerPenguinsPrediction Predict(PalmerPenguinsData newSample)
        {
            LoadModel();
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<PalmerPenguinsData, PalmerPenguinsPrediction>(_model);

            return predictionEngine.Predict(newSample);
        }

        public void LoadModel()
        {
            if(!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} does not exist");
            }

            using(var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                _model = _mlContext.Model.Load(stream, out _);
            }

            if(_model == null)
            {
                throw new Exception($"Failed to load Model");
            }
        }
    }
}