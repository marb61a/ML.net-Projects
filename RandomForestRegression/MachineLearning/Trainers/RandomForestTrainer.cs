using RandomForestRegression.MachineLearning.Common;

namespace RandomForestRegression.MachineLearning.Trainers
{
    public sealed class RandomForestTrainer : TrainerBase<FastForestRegressionModelParameters>
    {
        public RandomForestTrainer(int numberOfLeaves, int numberOfTrees) : base()
        {
            Name = $"Random Forest: {numberOfLeaves}-{numberOfTrees}";
            _model = mlContext.Regression.Trainers.FastForest(
                numberOfLeaves: numberOfLeaves,
                numberOfTrees: numberOfTrees
            );
        }
    }

}