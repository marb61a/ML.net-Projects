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
    }
}