using DecisionTreesRegression.MachineLearning.Common;

namespace DecisionTreesRegression.MachineLearning.Trainers
{
    public sealed class FastTreeTweedieTrainer: TrainerBase<FastTreeTweedieModelParameters>
    {
        public FastTreeTweedieTrainer(int numberOfLeaves,int numberOfTrees,double learningRate = 0.2): base()
        {
            Name = $"Fast Tree Tweedie-{numberOfLeaves}-{numberOfTrees}-{learningRate}";
            _model = mlContext.Regression.Trainers.FastTreeTweedie(
                numberOfLeaves: numberOfLeaves,
                numberOfTrees: numberOfTrees,
                learningRate: learningRate
            );
        }

    }
}