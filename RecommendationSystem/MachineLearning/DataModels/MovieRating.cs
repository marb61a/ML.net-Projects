namespace RecommendationSystem.MachineLearning.DataModels
{
    public class MovieRating
    {
        [LoadColumn(1)]
        public int MovieId;

        [LoadColumn(2)]
        public int UserId;

        [LoadColumn(3)]
        public float Label;
    }
}