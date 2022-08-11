using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class RandomForestTrainer : TrainerBase<FastForestBinaryModelParameters>
    {
        public RandomForestTrainer(int numberOfLeaves, int numberOfTrees) : base()
        {
            Name = $"Random Forest: {numberOfLeaves}-{numberOfTrees}";
            _model = mlContext.BinaryClassification.Trainers.FastForest(
                                            numberOfLeaves: numberOfLeaves,
                                            numberOfTrees: numberOfTrees);
        }
    }
}