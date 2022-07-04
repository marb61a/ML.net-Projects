using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers;

public class SdcaLogisticRegressionTrainer :
    TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SdcaLogisticRegressionTrainer() : base()
    {
        Name = "Sdca Logistic Regression";
        _model = mlContext
                    .BinaryClassification
                    .Trainers
                    .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
    }
}