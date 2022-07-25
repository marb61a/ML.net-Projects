using SVM.MachineLearning.Common;
using SVM.MachineLearning.DataModels;
using SVM.MachineLearning.Predictor;
using SVM.MachineLearning.Trainers;

var newSample = new PalmerPenguinsBinaryData
{
    CulmenDepth = 1.2f,
    CulmenLength = 1.1f
};

var trainers = new List<ITrainerBase>
{
    new LdSvmUTrainer(3),
    new LdSvmUTrainer(5),
    new LdSvmUTrainer(7),
    new LinearSvmUTrainer()
};

trainers.ForEach(t => TrainEvaluatePredict(t, newSample));

void TrainEvaluatePredict(ITrainerBase trainer, PalmerPenguinsBinaryData newSample)
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    string fileName = "penguins_size_binary.csv";
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
    Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
    Console.WriteLine("------------------------------");
}