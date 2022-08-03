namespace ImageClassification.MachineLearning
{
    public class ImageClassifier
    {
        public static string ModelPath => Path.Combine(AppContext.BaseDirectory, "imageClassification.mdl");
        private readonly MLContext _mlContext;
        private ImageClassificationTrainer.Architecture _architecture;
        private DataLoader _dataLoader;

        private ITransformer _trainedModel;

        public ImageClassifier(ImageClassificationTrainer.Architecture architecture, DataLoader dataLoader)
        {
            _mlContext = new MLContext(111);
            _architecture = architecture;
            _dataLoader = dataLoader;
        }

        // Trains the model on the defined data 
        public void Fit()
        {
            var trainingPipeline = BuildTrainingPipeline();
            _trainedModel = trainingPipeline.Fit(_dataLoader.TrainSet);
        }

        // Save the model to file
        public void Save()
        {
            _mlContext.Model.Save(_trainedModel, _dataLoader.TrainSet.Schema, ModelPath);
        }

        // Build the training pipeline
        private EstimatorChain<KeyToValueMappingTransformer> BuildTrainingPipeline()
        {
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = _dataLoader.ValidationSet,
                Arch = _architecture,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                Epoch = 20
            };

            return _mlContext.MulticlassClassification
                .Trainers
                .ImageClassification(classifierOptions)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }
    }
}