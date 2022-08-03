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

    }
}