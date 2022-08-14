using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SgdUCalibratedTrainer: TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SgdUCalibratedTrainer() : base()
        {
            Name = "Sgd Calibrated";
            _model = mlContext
                        .BinaryClassification
                        .Trainers
                        .SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}