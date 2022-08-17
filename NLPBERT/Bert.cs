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

        private(int StartIndex, int EndIndex, float Probability) GetBestPrediction(BertPredictions result, int minIndex, int topN, int maxLength)
        {
            var bestStartLogits = result
                .StartLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(topN);
            
            var bestEndLogits = result
                .EndLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(topN);
            
            var bestResultsWithScore = bestStartLogits
                .SelectMany(startLogit => 
                    bestEndLogits.Select(endLogit => (
                        StartLogit: startLogit.Index,
                        EndLogit: endLogit.Index,
                        Score: startLogit.Logit + endLogit.Logit
                    ))
                )
                .Where(entry => !(
                    entry.EndLogit < entry.StartLogit ||
                    entry.EndLogit - entry.StartLogit > maxLength ||
                    entry.StartLogit == 0 && entry.EndLogit == 0 ||
                    entry.StartLogit < minIndex
                ))
                .Take(topN);
            
            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }
    }
}