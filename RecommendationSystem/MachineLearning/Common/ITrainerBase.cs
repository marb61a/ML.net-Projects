namespace RecommendationSystem.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; set; }
        void Fit(string trainingFileName);
        RegressionMetrics Evaluate();
        void Save();

    }
}