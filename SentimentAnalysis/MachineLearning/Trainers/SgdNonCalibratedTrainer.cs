using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SgdNonCalibratedTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public SgdNonCalibratedTrainer() : base()
        {
            Name = "Sgd NonCalibrated";
            _model = mlContext
                        .BinaryClassification
                        .Trainers
                        .SgdNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}