namespace BinaryClassification.MachineLearning.Common
{
    public class TrainerBase<TParameters> : ITrainerBase
    {
        public string Name => throw new NotImplementedException();

        public BinaryClassificationMetrics Evaluate()
        {
            throw new NotImplementedException();
        }

        public void Fit(string trainingFileName)
        {
            throw new NotImplementedException();
        }

        public void Save()
        {
            throw new NotImplementedException();
        }
    }
}