using RandomForestClassification.MachineLearning.DataModels;

namespace RandomForestClassification.MachineLearning.Predictor
{
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");
    private readonly MLContext _mlContext;

    private ITransformer _model;

    public Predictor()
    {
        _mlContext = new MLContext(11);
    }

    public PalmerPenguinsBinaryPrediction Predict(PalmerPenguinsBinaryData newSample)
    {
        LoadModel();

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<PalmerPenguinsBinaryData, PalmerPenguinsBinaryPrediction>(_model);
        return predictionEngine.Predict(newSample);
    }

    private void LoadModel()
    {
        if (!File.Exists(ModelPath))
        {
            throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
        }

        using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            _model = _mlContext.Model.Load(stream, out _);
        }

        if (_model == null)
        {
            throw new Exception($"Failed to load Model");
        }
    }

    }
}