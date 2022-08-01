using RecommendationSystem.MachineLearning.Trainers;
using RecommendationSystem.MachineLearning.Common;
using RecommendationSystem.MachineLearning.DataModels;
using RecommendationSystem.MachineLearning.Predictors;

var newSample = new MovieRating
{
    UserId = 6,
    MovieId = 11
};

var trainers = new List<ITrainerBase>
{
    new MatrixFactorizationUTrainer(50, 50, 0.1),
    new MatrixFactorizationUTrainer(100, 50, 0.01),
    new MatrixFactorizationUTrainer(100, 50, 0.1),
    new MatrixFactorizationUTrainer(200, 100, 0.01),
    new MatrixFactorizationUTrainer(200, 100, 0.1)
};

trainers.ForEach(t => TrainEvaluatePredict(t, newSample));

static void TrainEvaluatePredict(ITrainerBase trainer, MovieRating newSample)
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    trainer.Fit(".\\Data\\netflix_subset.csv");

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
    Console.WriteLine($"Prediction: {prediction.Score:#.##}");
    Console.WriteLine("------------------------------");
}