using RecommendationSystem.MachineLearning.Common;

namespace RecommendationSystem.MachineLearning.Trainers
{
    public class MatrixFactorizationTrainer: TrainerBase<MatrixFactorizationModelParameters>
    {
        public MatrixFactorizationTrainer(int numberOfIterations, int approximationRank, double learningRate) : base()
        {
            Name = $"Matrix Factorization {numberOfIterations} - {approximationRank}";
            _model = mlContext.Recommendation().Trainers.MatrixFactorization(
                labelColumnName: "Label",
                matrixColumnIndexColumnName: "UserIdEncoded",
                matrixRowIndexColumnName: "MovieIdEncoded",
                approximationRank: approximationRank,
                learningRate: learningRate,
                numberOfIterations: numberOfIterations
            );
        }
    }
}