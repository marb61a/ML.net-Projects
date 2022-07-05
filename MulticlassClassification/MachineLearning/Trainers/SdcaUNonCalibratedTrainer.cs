namespace MulticlassClassification.MachineLearning.Trainers;

public class SdcaUNonCalibratedTrainer : TrainerBase<LinearMulticlassModelParameters>
{
    public SdcaUNonCalibratedTrainer() : base()
    {
        Name = "Sdca NonCalibrated";
        _model = MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}
