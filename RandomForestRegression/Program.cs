using RandomForestRegression.MachineLearning.Common;
using RandomForestRegression.MachineLearning.DataModels;
using RandomForestRegression.MachineLearning.Predictors;
using RandomForestRegression.MachineLearning.Trainers;

var newSample = new BostonHousingData
{
    Age = 65.2f,
    CrimeRate = 0.00632f,
    EmployCenterDistance = 4.0900f,
    HighwayAccecabilityRadius = 15.3f,
    NOConcetration = 0.538f,
    NumOfRoomsPerDwelling = 6.575f,
    Proportion = 1f,
    PTRatio = 15.3f,
    RiverCoast = 0f,
    TaxRate = 296f,
    Zoned = 23f
};

var trainers = new List<ITrainerBase>
{
    new RandomForestTrainer(2, 5),
    new RandomForestTrainer(5, 10),
    new RandomForestTrainer(10, 20)
};

trainers.ForEach(t => TrainEvaluatePredict(t, newSample));

void TrainEvaluatePredict(ITrainerBase trainer, BostonHousingData newSample)
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    string fileName = "boston_housing.csv";
    string path = Path.Combine(Environment.CurrentDirectory, @"Data\", fileName);

    trainer.Fit(path.ToString());

    var modelMetrics = trainer.Evaluate();

    Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
                      $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
                      $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
                      $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
                      $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");

    trainer.Save();

    var predictor = new Predictor();
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.MedianPrice:#.##}");
    Console.WriteLine("------------------------------");
}
