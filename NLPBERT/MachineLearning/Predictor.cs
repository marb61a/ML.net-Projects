using NLPBERT.MachineLearning.DataModel;

namespace NLPBERT.MachineLearning
{
    public class Predictor
    {
        private MLContext _mlContext;
        private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

        public Predictor(ITransformer trainedModel)
        {
            _mlContext = new MLContext(11);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
        }

        public BertPredictions Predict(BertInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }
}