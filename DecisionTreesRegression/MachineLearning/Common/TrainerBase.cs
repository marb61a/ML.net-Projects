namespace DecisionTreesRegression.MachineLearning.Common
{
    public abstract class TrainerBase<TParameters>: ITrainerBase
        where TParameters: class
    {
        public string Name { get; protected set; }
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "regression.mdl");

        protected readonly MLContext mlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<RegressionPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;
    }
}