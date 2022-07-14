using DecisionTreesRegression.MachineLearning.Common;

namespace DecisionTreesRegression.MachineLearning.Trainers
{
    public sealed class GamTrainer : TrainerBase<GamRegressionModelParameters>
    {
        public GamTrainer(): base()
        {
            Name = $"GAM";
            _model = mlContext.Regression.Trainers.Gam();
        }
    }
}