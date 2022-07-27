namespace Clustering.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        ClusteringMetrics Evaluate();
        void Save();

    }
}