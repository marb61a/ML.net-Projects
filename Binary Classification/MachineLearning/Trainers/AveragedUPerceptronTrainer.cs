using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class AveragedUPerceptronTrainer : TrainerBase<LinearBinaryModelParameters>
{
    public AveragedUPerceptronTrainer() : base()
    {
        Name = "Averaged Perceptron";
        _model = mlContext
                    .BinaryClassification
                    .Trainers
                    .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
    }
}