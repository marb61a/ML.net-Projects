using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class AveragedUPerceptronTrainer: TrainerBase<LinearBinaryModelParameters>
    {
        public AveragedUPerceptronTrainer(): base()
        {
            Name = "Averaged Perceptron";
            _model = mlContext
                        .BinaryClassification
                        .Trainers
                        .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
    
}