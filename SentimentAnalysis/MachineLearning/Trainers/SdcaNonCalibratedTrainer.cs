using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
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
}