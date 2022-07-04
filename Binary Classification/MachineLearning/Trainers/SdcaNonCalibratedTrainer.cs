using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class SdcaNonCalibratedTrainer: TrainerBase<LinearBinaryModelParameters>
{
    public SdcaNonCalibratedTrainer() : base()
    {
        Name = "Sdca NonCalibrated";
        _model = mlContext
                    .BinaryClassification
                    .Trainers
                    .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}