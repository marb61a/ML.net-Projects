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
        
        private EstimatorChain<ImageLoadingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey"
            )
            .Append(_mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: _trainingFolder,
                inputColumnName: "ImagePath"
            ));

            return dataProcessPipeline;
        }

        private void LoadAndPrepareData(EstimatorChain<ImageLoadingTransformer> dataProcessPipeline)
        {
            IEnumerable<ImageData> images = LoadImages(_trainingFolder);
            var imageData = _mlContext.Data.LoadFromEnumerable(images);
            var shuffledImageData = _mlContext.Data.ShuffleRows(imageData);

            var preparedData = dataProcessPipeline
                .Fit(shuffledImageData)
                .Transform(shuffledImageData);
            
            var trainSplit = _mlContext.Data.TrainTestSplit(data: preparedData, testFraction: 0.3);
            var validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            TrainSet = trainSplit.TrainSet;
            ValidationSet = validationTestSplit.TrainSet;
            TestSet = validationTestSplit.TestSet;
        }

        private IEnumerable<ImageData> LoadImages(string trainingFolder)
        {
            var files = Directory.GetFiles(trainingFolder, "*", searchOption: SearchOption.AllDirectories);

            foreach(var file in files)
            {
                if((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;
                
                var label = Path.GetFileName(file);
                label = label.Substring(0, 3);

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }
}