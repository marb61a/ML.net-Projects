using NLPBERT.Extensions;
using NLPBERT.MachineLearning;
using NLPBERT.MachineLearning.DataModel;

namespace NLPBERT
{
    public class Bert
    {
        private readonly BertUncasedBaseTokenizer _tokeniser;
        private Predictor _predictor;

        public Bert(string bertModelPath)
        {
            _tokeniser = new BertUncasedBaseTokenizer();
            var trainer = new Trainer();

            var trainedModel = trainer.BuildAndTrain(bertModelPath);
            _predictor = new Predictor(trainedModel);
        }

        public (List<string> tokens, float probablity) Predict(string context, string question)
        {
            var tokens = _tokeniser.Encode(256, question, context);
            var input = new BertInput()
            {
                InputId = tokens.Select(t => t.InputIds).ToArray(),
                SegmentIds = tokens.Select(t => t.TokenTypeIds).ToArray(),
                InputMask = tokens.Select(t => t.AttentionMask).ToArray(),
                UniqueIds = new long[] { 0 }
            };

            var predictions = _predictor.Predict(input);
            var contextStart = tokens.FindIndex(o => o.InputIds == 102);

            var (startIndex, endIndex, probability) = GetBestPrediction(predictions, contextStart, 20, 30);

            var predictedTokens = input.InputId
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _tokeniser.IdToToken((int)o))
                .ToList();

            var connectedTokens = _tokeniser.Untokenize(predictedTokens);
            return (connectedTokens, probability);
        }
    }
}