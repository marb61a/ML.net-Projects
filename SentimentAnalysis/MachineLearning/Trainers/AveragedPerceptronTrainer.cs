using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class AveragedPerceptronTrainer: TrainerBase<LinearBinaryModelParameters>
    {
        public AveragedPerceptronTrainer(): base()
        {
            Name = "Averaged Perceptron";
            _model = mlContext
                        .BinaryClassification
                        .Trainers
                        .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
    
}