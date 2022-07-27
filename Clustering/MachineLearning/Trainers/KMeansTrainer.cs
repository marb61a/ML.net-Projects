using Clustering.MachineLearning.Common;

namespace Clustering.MachineLearning.Trainers
{
    public class KMeansTrainer : TrainerBase<KMeansModelParameters>
    {
        public KMeansTrainer(int numberOfClusters) : base()
        {
            Name = $"K Means Clulstering - {numberOfClusters} Clusters";
            _model = mlContext.Clustering.Trainers
                    .KMeans(numberOfClusters: numberOfClusters, featureColumnName: "Features");
        }
    }
    
}