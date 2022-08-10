using SentimentAnalysis.MachineLearning.DataModels;

namespace SentimentAnalysis.MachineLearning.Common
{
    // The base class for trainers
    public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");

        protected readonly MLContext mlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mlContext = new MLContext(11);
        }

        // Trains model on defined data
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        // Evaluates trained model
        public BinaryClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return mlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
        }

        // Save model in file
        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        // Data preprocessing and feature engineering
        private EstimatorChain<ITransformer> BuildDataProcessingPipeline()
        {
            return mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .AppendCacheCheckpoint(mlContext);
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<SentimentData>(trainingFileName, hasHeader: false);
            return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}