using MulticlassClassification.MachineLearning.Common;
using MulticlassClassification.MachineLearning.DataModels;
using MulticlassClassification.MachineLearning.Predictors;
using MulticlassClassification.MachineLearning.Trainers;

var newSample = new PalmerPenguinsData
{
    Island = "Torgersen",
    CulmenDepth = 18.7f,
    CulmenLength = 39.3f,
    FliperLength = 180,
    BodyMass = 3700,
    Sex = "MALE"
};

var trainers = new List<ITrainerBase>
{
    new LbfgsUMaximumEntropyTrainer(),
    new NaiveUBayesTrainer(),
    new OneUVersusAllTrainer(),
    new SdcaUMaximumEntropyTrainer(),
    new SdcaUNonCalibratedTrainer()
};


trainers.ForEach(t => TrainEvaluatePredict(t, newSample));

void TrainEvaluatePredict(ITrainerBase trainer, PalmerPenguinsData newSample)
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    string fileName = "penguins_size.csv";
    string path = Path.Combine(Environment.CurrentDirectory, @"Data\", fileName);

    trainer.Fit(path.ToString());

    var modelMetrics = trainer.Evaluate();

    Console.WriteLine($"Macro Accuracy: {modelMetrics.MacroAccuracy:#.##}{Environment.NewLine}" +
                      $"Micro Accuracy: {modelMetrics.MicroAccuracy:#.##}{Environment.NewLine}" +
                      $"Log Loss: {modelMetrics.LogLoss:#.##}{Environment.NewLine}" +
                      $"Log Loss Reduction: {modelMetrics.LogLossReduction:#.##}{Environment.NewLine}");

    trainer.Save();

    var predictor = new Predictor();
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
    Console.WriteLine("------------------------------");
}