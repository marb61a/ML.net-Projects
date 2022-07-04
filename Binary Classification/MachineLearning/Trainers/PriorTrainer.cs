using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class PriorUTrainer : TrainerBase<PriorModelParameters>
{
    public PriorUTrainer() : base()
    {
        Name = "Prior";
        _model =  mlContext
                    .BinaryClassification
                    .Trainers
                    .Prior(labelColumnName: "Label");
    }
}