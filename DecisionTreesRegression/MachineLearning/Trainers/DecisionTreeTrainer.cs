using DecisionTreesRegression.MachineLearning.Common;

namespace DecisionTreesRegression.MachineLearning.Trainers
{
    public sealed class DecisionTreeTrainer : TrainerBase<FastTreeRegressionModelParameters>
    {
        public DecisionTreeTrainer(int numberOfLeaves, int numberOfTrees, double learningRate = 0.2): base()
        {
            Name = $"Decision Tree-{numberOfLeaves}-{numberOfTrees}-{learningRate}";
            _model = mlContext.Regression.Trainers.FastTree(
                numberOfLeaves: numberOfLeaves,
                numberOfTrees: numberOfTrees,
                learningRate: learningRate
            );
        }

    }
}