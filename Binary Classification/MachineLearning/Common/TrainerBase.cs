namespace BinaryClassification.MachineLearning.Common
{
    public class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }

        // Defines where the created model will be saved to on the file system
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");
        protected readonly MLContext mLContext;

        // Loads data and splits into test and train data
        protected DataOperationsCatalog.TrainTestData _dataSplit;

        // Uses the ML.Trainers package which is set in the usings file 
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mLContext = new MLContext(11);
        }

        public BinaryClassificationMetrics Evaluate()
        {
            throw new NotImplementedException();
        }

        // Check if file exists
        public void Fit(string trainingFileName)
        {
            if(!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} does not exist");
            }
        }

        public void Save()
        {
            throw new NotImplementedException();
        }
    }
}