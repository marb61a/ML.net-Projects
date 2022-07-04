using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class PriorTrainer : TrainerBase<PriorModelParameters>
{
    public PriorTrainer() : base()
    {
        Name = "Prior";
        _model =  mlContext
                    .BinaryClassification
                    .Trainers
                    .Prior(labelColumnName: "Label");
    }
}