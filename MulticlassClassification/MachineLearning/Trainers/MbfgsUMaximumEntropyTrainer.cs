using MulticlassClassification.MachineLearning.Common;

namespace MulticlassClassification.MachineLearning.Trainers;

public class LbfgsUMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
{
    public LbfgsUMaximumEntropyTrainer() : base()
    {
        Name = "LBFGS Maximum Entropy";
        _model = MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
    }
}