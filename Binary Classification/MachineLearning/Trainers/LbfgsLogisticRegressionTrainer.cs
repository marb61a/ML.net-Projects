using BinaryClassification.MachineLearning.Common;

namespace BinaryClassification.MachineLearning.Trainers
{
    public class LbfgsLogisticRegressionTrainer:
        TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
         public LbfgsLogisticRegressionTrainer() : base()
        {
            Name = "LBFGS Logistic Regression";
            _model = mlContext
                .BinaryClassification
                .Trainers
                .LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        }

    }
}