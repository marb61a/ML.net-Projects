using SVM.MachineLearning.DataModels;

namespace SVM.MachineLearning.Common
{
    public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class
    {
        public string Name { get; set; }

        public static string ModelPath => Path.Combine(AppContext.BaseDirectory, "svmclassification.mbl");
        protected readonly MLContext mlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mlContext = new MLContext(11);
        }

        public void Fit()
        {

        }
    }
}