using MulticlassClassification.MachineLearning.Common;

namespace MulticlassClassification.MachineLearning.Trainers;

public class SdcaUMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
{
    public SdcaUMaximumEntropyTrainer() : base()
    {
        Name = "Sdca Maximum Entropy";
        _model = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
    }
}
