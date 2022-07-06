using MulticlassClassification.MachineLearning.Common;

namespace MulticlassClassification.MachineLearning.Trainers;

public class OneUVersusAllTrainer : TrainerBase<OneVersusAllModelParameters>
{
    public OneUVersusAllTrainer() : base()
    {
        Name = "One Versus All";
        _model = mlContext.MulticlassClassification.Trainers
                    .OneVersusAll(binaryEstimator: mlContext.BinaryClassification.Trainers.SgdCalibrated());
    }
}