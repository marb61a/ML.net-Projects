using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
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
}