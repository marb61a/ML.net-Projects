using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class DecisionTreeTrainer : TrainerBase<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>
    {
        public DecisionTreeTrainer(int numberOfLeaves, int numberOfTrees, double learningRate = 0.2): base()
        {
            Name = $"Decision Tree-{numberOfLeaves}-{numberOfTrees}-{learningRate}";
            _model = mlContext.BinaryClassification.Trainers.FastTree(
                numberOfLeaves: numberOfLeaves,
                numberOfTrees: numberOfTrees,
                learningRate: learningRate
            );
        }
    }

}