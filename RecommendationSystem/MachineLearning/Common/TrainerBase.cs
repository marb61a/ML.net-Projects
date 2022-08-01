using RecommendationSystem.MachineLearning.DataModels;

namespace RecommendationSystem.MachineLearning.Common
{
    public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class{
        public string Name { get; set; }
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "recommendationsystem.mdl");   
        protected readonly MLContext mlContext;
        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<PredictionTransformerBase<TParameters>, TParameters> _model;
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

        public RegressionMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return mlContext.Recommendation().Evaluate(testSetTransform);
        }

        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        private EstimatorChain<ValueToKeyMappingTransformer>BuildDataProcessingPipeline()
        {
            return mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "UserId",
                    outputColumnName: "UserIdEncoded"
                )
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "MovieId",
                    outputColumnName: "MovieIdEncoded")
                )
                .AppendCacheCheckpoint(mlContext);
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingFileName, hasHeader: true, separatorChar: ',');
            return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}