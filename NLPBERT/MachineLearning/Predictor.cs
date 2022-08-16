using NLPBERT.MachineLearning.DataModel;

namespace NLPBERT.MachineLearning
{
    public class predictor
    {
        private MLContext _mlContext;
        private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

        public Predictor(ITransformer trainedModel)
        {
            _mLContext = new MLContext(11);
            _predictionEngine = _mLContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
        }

        public BertPredictions Predict(BertInput encodedInput)
        {
            return _predictionEngine.Predict(encodedInput);
        }
    }
}