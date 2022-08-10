namespace SentimentAnalysis.MachineLearning.DataModels
{
    // Model Sentiment Prediction
    public class SentimentPrediction: SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}