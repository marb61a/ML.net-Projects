using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class SgdUNonCalibratedTrainer : TrainerBase<LinearBinaryModelParameters>
{
    public SgdUNonCalibratedTrainer() : base()
    {
        Name = "Sgd NonCalibrated";
        _model = mlContext
                    .BinaryClassification
                    .Trainers
                    .SgdNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}