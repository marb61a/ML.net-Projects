using RecommendationSystem.MachineLearning.Common;

namespace RecommendationSystem.MachineLearning.Trainers
{
    public class MatrixFactorizationUTrainer: TrainerBase<MatrixFactorizationModelParameters>
    {
        public MatrixFactorizationUTrainer(int numberOfIterations, int approximationRank, double learningRate) : base()
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