using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SgdCalibratedTrainer: TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SgdCalibratedTrainer() : base()
        {
            Name = "Sgd Calibrated";
            _model = mlContext
                        .BinaryClassification
                        .Trainers
                        .SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}