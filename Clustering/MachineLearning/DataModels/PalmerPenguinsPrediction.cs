namespace Clustering.MachineLearning.DataModels
{
    public class PalmerPenguinsPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}