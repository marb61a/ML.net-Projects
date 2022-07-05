namespace MulticlassClassification.MachineLearning.Trainers;

public class OneUVersusAllTrainer : TrainerBase<OneVersusAllModelParameters>
{
    public OneUVersusAllTrainer() : base()
    {
        Name = "One Versus All";
        _model = MlContext.MulticlassClassification.Trainers
                    .OneVersusAll(binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
    }
}