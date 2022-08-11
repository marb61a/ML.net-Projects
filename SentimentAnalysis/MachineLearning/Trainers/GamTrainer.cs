using SentimentAnalysis.MachineLearning.Common;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class GamTrainer : TrainerBase<CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
    {
        public GamTrainer() : base()
        {
            Name = "GAM";
            _model = mlContext.BinaryClassification.Trainers.Gam();
        }
    }
}