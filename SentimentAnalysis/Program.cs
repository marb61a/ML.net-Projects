using SentimentAnalysis.MachineLearning.Common;
using SentimentAnalysis.MachineLearning.DataModels;
using SentimentAnalysis.MachineLearning.Trainers;

var newSample = new SentimentData
{
    SentimentText = "This is awesome!"
};

var trainers = new List<ITrainerBase>
{
    new LbfgsLogisticRegressionTrainer(),
    new AveragedUPerceptronTrainer(),
    new PriorUTrainer(),
    new SdcaLogisticRegressionTrainer(),
    new SdcaNonCalibratedTrainer(),
    new SgdUCalibratedTrainer(),
    new SgdUNonCalibratedTrainer(),
    new DecisionTreeTrainer(5, 10),
    new DecisionTreeTrainer(5, 10, 0.1),
    new DecisionTreeTrainer(10, 20),
    new DecisionTreeTrainer(10, 20, 0.1),
    new GamTrainer(),
    new RandomForestTrainer(2, 5),
    new RandomForestTrainer(5, 10),
    new RandomForestTrainer(10, 20)
};

trainers.ForEach(t => TrainEvaluatePredict(t, newSample));

void TrainEvaluatePredict(ITrainerBase trainer, SentimentData newSample)
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    string fileName = "imdb_labelled.txt";
    string path = Path.Combine(Environment.CurrentDirectory, @"Data\", fileName);

    trainer.Fit(path.ToString());

    var modelMetrics = trainer.Evaluate();

    Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:0.##}{Environment.NewLine}" +
                      $"F1 Score: {modelMetrics.F1Score:#.##}{Environment.NewLine}" +
                      $"Positive Precision: {modelMetrics.PositivePrecision:#.##}{Environment.NewLine}" +
                      $"Negative Precision: {modelMetrics.NegativePrecision:0.##}{Environment.NewLine}" +
                      $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                      $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}" +
                      $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:#.##}{Environment.NewLine}");

    trainer.Save();

    var predictor = new Predictor();
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.Prediction:#.##}");
    Console.WriteLine($"Probability: {prediction.Probability:#.##}");
    Console.WriteLine("------------------------------");
}