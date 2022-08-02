using ImageClassification.MachineLearning.DataModels;

namespace ImageClassification.MachineLearning
{
    public class DataLoader
    {
        private readonly MLContext _mlContext;
        private string _trainingFolder;

        public IDataView TrainSet { get; private set; }
        public IDataView ValidationSet { get; private set; }
        public IDataView TestSet { get; private set; }

        public DataLoader(string trainingFolder)
        {
            _trainingFolder = trainingFolder;
            _mlContext = new MLContext(11);

            var dataProcessPipeline = BuildDataProcessingPipeline();
            LoadAndPrepareData(dataProcessPipeline);
        }
        
    }
}