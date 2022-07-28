using Clustering.MachineLearning.Common;
using Clustering.MachineLearning.DataModels;
using Clustering.MachineLearning.Predictors;
using Clustering.MachineLearning.Trainers;

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
    new KMeansUTrainer(1),
    new KMeansUTrainer(2),
    new KMeansUTrainer(3),
    new KMeansUTrainer(4),
    new KMeansUTrainer(5),
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

    Console.WriteLine($"Average Distance: {modelMetrics.AverageDistance:#.##}{Environment.NewLine}" +
                      $"Davies Bouldin Index: {modelMetrics.DaviesBouldinIndex:#.##}{Environment.NewLine}" +
                      $"Normalized Mutual Information: {modelMetrics.NormalizedMutualInformation:#.##}{Environment.NewLine}");

    trainer.Save();

    var predictor = new Predictor();
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.PredictedClusterId:#.##}");
    Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
    Console.WriteLine("------------------------------");
}
