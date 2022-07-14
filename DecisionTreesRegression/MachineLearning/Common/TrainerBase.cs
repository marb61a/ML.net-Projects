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

        protected TrainerBase()
        {
            mlContext = new MLContext(11);
        }

        public void Fit(string trainingFileName)
        {
            if(!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);
            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);

        }

        // Evaluate the trained model
        public RegressionMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);
            return mlContext.Regression.Evaluate(testSetTransform);
        }

        // Save the model in a file
        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }
    }
}