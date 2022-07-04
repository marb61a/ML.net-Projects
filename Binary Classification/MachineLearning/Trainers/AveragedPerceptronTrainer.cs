using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class AveragedPerceptronTrainer : TrainerBase<LinearBinaryModelParameters>
{
    public AveragedPerceptronTrainer() : base()
    {
        Name = "Averaged Perceptron";
        _model = mlContext
                    .BinaryClassification
                    .Trainers
                    .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
    }
}