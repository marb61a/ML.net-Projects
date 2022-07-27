using Clustering.MachineLearning.DataModels;

namespace Clustering.MachineLearning.Common
{
    public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "clustering.mdl");

        protected readonly MLContext mlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<ClusteringPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mlContext = new MLContext(11);
        }

        public void Fit(string trainingFileName)
        {
            if(!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} does not exist");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);
            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        public ClusteringMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);
            return mlContext.Clustering.Evaluate(
                data: testSetTransform,
                labelColumnName: "PredictedLabel",
                scoreColumnName: "Score",
                featureColumnName: "Features"
            );
        }

        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        private EstimatorChain<ColumnConcatenatingTransformer>BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized")
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Island", outputColumnName: "IslandFeaturized"))
                .Append(MlContext.Transforms.Concatenate(
                    "Features",
                    "IslandFeaturized",
                    nameof(PalmerPenguinsData.CulmenLength),
                    nameof(PalmerPenguinsData.CulmenDepth),
                    nameof(PalmerPenguinsData.BodyMass),
                    nameof(PalmerPenguinsData.FliperLength),
                    "SexFeaturized"
                ))
                .AppendCacheCheckpoint(mlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<PalmerPenguinsData>(trainingFileName, hasHeader: true, separatorChar: ',');
            return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}